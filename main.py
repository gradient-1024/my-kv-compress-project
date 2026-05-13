import torch
import gc
import statistics
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from core.compressor import (
    StreamingLLMCompressor, 
    SnapKVCompressor, 
    H2OCompressor, 
)
from core.patcher import apply_compression_patch
from scripts.benchmark import Evaluator

class PG19StreamLoader:
    def __init__(self, split="test"):
        print(f"加载 PG-19 ({split}集) 数据")
        self.dataset = load_dataset(
            "emozilla/pg19",
            split=split, 
            streaming=True
        )

    def fetch_chunk_by_index(self, index, tokenizer, max_tokens=1500):
        try:
            sample = next(iter(self.dataset.skip(index)))
            raw_text = sample["text"]
            book_title = sample.get("short_book_title", f"Book_{index}")
            
            tokens = tokenizer.encode(raw_text)
            
            if len(tokens) > max_tokens:
                chunked_tokens = tokens[:max_tokens]
                processed_text = tokenizer.decode(chunked_tokens)
                real_len = max_tokens
            else:
                processed_text = raw_text
                real_len = len(tokens)
                
            print(f"加载文档: {book_title} | 截取长度: {real_len} Tokens")
            return processed_text
            
        except StopIteration:
            raise ValueError(f"索引 {index} 超出数据集范围。")
        

class WikitextLoader:
    def __init__(self, split="test", subset="wikitext-2-raw-v1"):
        print(f"加载 Wikitext ({subset} - {split}集) 数据")
        self.dataset = load_dataset("wikitext", subset, split=split)
        
        self.full_text = "\n\n".join([item["text"] for item in self.dataset if item["text"].strip()])
        print(f"Wikitext 数据集加载完毕，总字符数: {len(self.full_text)}")

    def fetch_chunk(self, tokenizer, max_tokens=1500, start_token_idx=0):
        tokens = tokenizer.encode(self.full_text)
        
        end_idx = min(start_token_idx + max_tokens, len(tokens))
        chunked_tokens = tokens[start_token_idx:end_idx]
        
        processed_text = tokenizer.decode(chunked_tokens)
        real_len = len(chunked_tokens)
        
        print(f"成功加载 Wikitext 测试块 | 起始索引: {start_token_idx} | 截取长度: {real_len} Tokens")
        return processed_text
    

def get_clean_model(model_id, device):
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device,
        output_attentions=True,
        attn_implementation="eager"
    )
    model.config.use_cache = True
    model.eval()
    return model


def summarize_runs(run_results):
    metric_names = [
        "TTFT (s)",
        "TPOT (s/token)",
        "Throughput (tokens/s)",
        "PPL",
    ]

    summary = {}
    for metric in metric_names:
        values = [run[metric] for run in run_results]
        summary[metric] = {
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        }

    summary["Context Length"] = run_results[0]["Context Length"]
    summary["Repeats"] = len(run_results)
    return summary


def run_single_benchmark(model_id, device, tokenizer, test_text, compressor_factory, gen_length):
    model = get_clean_model(model_id, device)
    compressor = compressor_factory()

    if compressor is not None:
        apply_compression_patch(model, compressor)

    evaluator = Evaluator(model, tokenizer)
    results = evaluator.run_all(test_text, gen_length=gen_length)

    del model
    del evaluator
    gc.collect()
    torch.cuda.empty_cache()

    return results


def fetch_test_text(loader, tokenizer, context_length):
    if isinstance(loader, PG19StreamLoader):
        return loader.fetch_chunk_by_index(
            index=0,
            tokenizer=tokenizer,
            max_tokens=context_length,
        )

    return loader.fetch_chunk(
        tokenizer=tokenizer,
        max_tokens=context_length,
        start_token_idx=0,
    )


def main():
    model_id = "EleutherAI/pythia-70m"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    benchmark_repeats = 3
    gen_length = 50
    context_lengths = [1024, 2048, 3072, 4096]
    compression_ratio = 0.5
    sink_size = 4
    window_size = 32
    kernel_size = 5
    
    print(f"正在加载模型环境: {model_id} 到 {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if device == "cuda":
        print("\n")
        print("执行 CUDA 预热...")
        
        warmup_model = get_clean_model(model_id, device)
        
        dummy_text = "test " * 2048
        warmup_evaluator = Evaluator(warmup_model, tokenizer)
        
        with torch.no_grad():
            _ = warmup_evaluator.run_all(dummy_text, gen_length=1)
            torch.cuda.synchronize()
            
        del warmup_model
        del warmup_evaluator
        gc.collect()
        torch.cuda.empty_cache()
        print("预热完成。")


    # 1. 初始化数据集加载器字典
    data_loaders = {
        "PG-19 (Fiction)": PG19StreamLoader(split="test"),
        "Wikitext (Encyclopedia)": WikitextLoader(split="test", subset="wikitext-2-raw-v1")
    }

    super_results = {}

    # 2. 遍历不同数据集
    for dataset_name, loader in data_loaders.items():
        print("\n")
        print(f"开始在数据集 [{dataset_name}] 上进行基准测试")

        dataset_results = {}

        for context_length in context_lengths:
            print("\n")
            print(f"当前上下文长度: {context_length} tokens")

            test_text = fetch_test_text(loader, tokenizer, context_length)
            context_results = {}
            current_capacity = max(
                int(context_length * compression_ratio),
                window_size + sink_size,
            )

            algorithms = {
                "1. Baseline (Dense)": lambda: None,
                f"2. StreamingLLM (Ratio={compression_ratio}, Cap={current_capacity})": (
                    lambda: StreamingLLMCompressor(
                        window_size=current_capacity - sink_size,
                        sink_size=sink_size,
                    )
                ),
                f"3. H2O (Ratio={compression_ratio}, Cap={current_capacity})": (
                    lambda: H2OCompressor(
                        max_capacity=current_capacity,
                        window_size=window_size,
                        sink_size=sink_size,
                    )
                ),
                f"4. SnapKV (Ratio={compression_ratio}, Cap={current_capacity})": (
                    lambda: SnapKVCompressor(
                        max_capacity=current_capacity,
                        window_size=window_size,
                        kernel_size=kernel_size,
                        sink_size=sink_size,
                    )
                ),
            }

            # 3. 遍历并测试不同压缩算法
            for algo_name, compressor_factory in algorithms.items():
                print("\n")
                print(f"当前评估算法: {algo_name}")
                run_results = []

                _ = run_single_benchmark(
                    model_id=model_id,
                    device=device,
                    tokenizer=tokenizer,
                    test_text=test_text,
                    compressor_factory=compressor_factory,
                    gen_length=gen_length,
                )

                for repeat_idx in range(benchmark_repeats):
                    print(f"重复测试 {repeat_idx + 1}/{benchmark_repeats}")

                    results = run_single_benchmark(
                        model_id=model_id,
                        device=device,
                        tokenizer=tokenizer,
                        test_text=test_text,
                        compressor_factory=compressor_factory,
                        gen_length=gen_length,
                    )
                    run_results.append(results)

                    print(
                        "  "
                        f"TTFT={results['TTFT (s)']:.4f}s | "
                        f"TPOT={results['TPOT (s/token)']:.4f}s | "
                        f"Throughput={results['Throughput (tokens/s)']:.2f} tk/s | "
                        f"PPL={results['PPL']:.4f}"
                    )

                summary = summarize_runs(run_results)
                context_results[algo_name] = summary

                print("汇总统计:")
                print(
                    "  "
                    f"TTFT={summary['TTFT (s)']['mean']:.4f} ± {summary['TTFT (s)']['std']:.4f} s"
                )
                print(
                    "  "
                    f"TPOT={summary['TPOT (s/token)']['mean']:.4f} ± {summary['TPOT (s/token)']['std']:.4f} s/token"
                )
                print(
                    "  "
                    f"Throughput={summary['Throughput (tokens/s)']['mean']:.2f} ± "
                    f"{summary['Throughput (tokens/s)']['std']:.2f} tk/s"
                )
                print(
                    "  "
                    f"PPL={summary['PPL']['mean']:.4f} ± {summary['PPL']['std']:.4f}"
                )

            dataset_results[context_length] = context_results
            
        super_results[dataset_name] = dataset_results


    # 4. 生成报告表格
    print("\n\n")
    print(" "*35 + "最终 Benchmark 汇总报告")
    
    for dataset_name, dataset_results in super_results.items():
        print(f"\n数据集: {dataset_name}")
        for context_length, context_results in dataset_results.items():
            print(f"\nContext Length: {context_length}")
            print("-" * 132)
            print(
                f"{'Algorithm':<25} | {'TTFT mean±std (s)':<22} | "
                f"{'TPOT mean±std (s/token)':<27} | {'Throughput mean±std (tk/s)':<29} | {'PPL mean±std':<18}"
            )
            print("-" * 132)
            
            for name, res in context_results.items():
                ttft = f"{res['TTFT (s)']['mean']:.4f} ± {res['TTFT (s)']['std']:.4f}"
                tpot = f"{res['TPOT (s/token)']['mean']:.4f} ± {res['TPOT (s/token)']['std']:.4f}"
                tput = (
                    f"{res['Throughput (tokens/s)']['mean']:.2f} ± "
                    f"{res['Throughput (tokens/s)']['std']:.2f}"
                )
                ppl = f"{res['PPL']['mean']:.4f} ± {res['PPL']['std']:.4f}"
                
                print(f"{name:<25} | {ttft:<22} | {tpot:<27} | {tput:<29} | {ppl:<18}")
            
    print("="*132)

if __name__ == "__main__":
    main()
