import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# 导入自定义压缩模块与评估模块
from core.compressor import (
    StreamingLLMCompressor, 
    SnapKVCompressor, 
    H2OCompressor, 
    PyramidKVCompressor
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
        """
        加载 Wikitext 数据集。
        默认使用 wikitext-2-raw-v1 (未经过分词处理的原始文本)。
        """
        print(f"加载 Wikitext ({subset} - {split}集) 数据")
        # 静态加载 Wikitext 数据集
        self.dataset = load_dataset("wikitext", subset, split=split)
        
        self.full_text = "\n\n".join([item["text"] for item in self.dataset if item["text"].strip()])
        print(f"Wikitext 数据集加载完毕，总字符数: {len(self.full_text)}")

    def fetch_chunk(self, tokenizer, max_tokens=1500, start_token_idx=0):
        """
        按 Token 数量截取连续文本用于测试。
        """
        tokens = tokenizer.encode(self.full_text)
        
        # 截取指定范围的 Token
        end_idx = min(start_token_idx + max_tokens, len(tokens))
        chunked_tokens = tokens[start_token_idx:end_idx]
        
        processed_text = tokenizer.decode(chunked_tokens)
        real_len = len(chunked_tokens)
        
        print(f"成功加载 Wikitext 测试块 | 起始索引: {start_token_idx} | 截取长度: {real_len} Tokens")
        return processed_text
    

def get_clean_model(model_id, device):
    """每次测试前加载未被修改的初始模型"""
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


def main():
    model_id = "EleutherAI/pythia-70m"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"正在加载模型环境: {model_id} 到 {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if device == "cuda":
        print("\n" + "="*70)
        print("执行 CUDA 预热...")
        print("="*70)
        
        warmup_model = get_clean_model(model_id, device)
        
        # 构造长度为 1600 的无意义文本
        dummy_text = "test " * 1600 
        warmup_evaluator = Evaluator(warmup_model, tokenizer)
        
        with torch.no_grad():
            # 调用底层完整的评估管道。
            # 设置 gen_length=1 即可触发完整的 Prefill 阶段与单步 Decoding 阶段，
            # 从而遍历所有内存分配、算子寻优与上层对象初始化逻辑。
            _ = warmup_evaluator.run_all(dummy_text, gen_length=1)
            
            # 强制 CPU 等待 GPU 完成所有异步算子执行
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

    # 用于存储所有数据集的最终结果
    super_results = {}

    # 2. 外层循环：遍历不同数据集
    for dataset_name, loader in data_loaders.items():
        print("\n" + "="*70)
        print(f"开始在数据集 [{dataset_name}] 上进行基准测试")
        print("="*70)

        # 获取测试文本
        if isinstance(loader, PG19StreamLoader):
            test_text = loader.fetch_chunk_by_index(index=0, tokenizer=tokenizer, max_tokens=1500)
        else:
            test_text = loader.fetch_chunk(tokenizer=tokenizer, max_tokens=1500, start_token_idx=0)

        # 每次数据集测试前重新实例化压缩器，确保状态重置
        algorithms = {
            "1. Baseline (Dense)": None,
            "2. StreamingLLM (Cap=512)": StreamingLLMCompressor(window_size=508, sink_size=4),
            "3. H2O (Cap=512)": H2OCompressor(max_capacity=512, window_size=32, sink_size=4),
            "4. SnapKV (Ratio=0.5)": SnapKVCompressor(compression_ratio=0.5, window_size=32, kernel_size=5, sink_size=4),
        }

        dataset_results = {}

        # 3. 内层循环：遍历并测试不同压缩算法
        for algo_name, compressor in algorithms.items():
            print("\n" + "-"*50)
            print(f"当前评估算法: {algo_name}")
            print("-"*50)
            
            # 获取初始模型
            model = get_clean_model(model_id, device)
            
            # 应用对应的压缩补丁
            if compressor is not None:
                apply_compression_patch(model, compressor)
                
            # 执行评估
            evaluator = Evaluator(model, tokenizer)
            try:
                # 统一生成 50 个 Token 用于测试吞吐量
                results = evaluator.run_all(test_text, gen_length=50)
                dataset_results[algo_name] = results
                
                # 输出当前算法测试结果
                for key, value in results.items():
                    print(f"{key:25}: {value}")
                    
            except Exception as e:
                print(f"算法 {algo_name} 测试异常: {e}")
                
            # 清理显存以防止内存溢出 (OOM)
            del model
            del evaluator
            gc.collect()
            torch.cuda.empty_cache()
            
        # 记录当前数据集测试结果
        super_results[dataset_name] = dataset_results


    # 4. 打印汇总对比表格
    print("\n\n" + "="*90)
    print(" "*35 + "最终 Benchmark 汇总报告")
    print("="*90)
    
    # 遍历打印每个数据集的测试数据
    for dataset_name, dataset_results in super_results.items():
        print(f"\n数据集: {dataset_name}")
        print("-" * 90)
        print(f"{'Algorithm':<25} | {'TTFT (s)':<10} | {'TPOT (s/token)':<15} | {'Throughput (tk/s)':<18} ")
        print("-" * 90)
        
        for name, res in dataset_results.items():
            ttft = f"{res['TTFT (s)']:.4f}"
            tpot = f"{res['TPOT (s/token)']:.4f}"
            tput = f"{res['Throughput (tokens/s)']:.2f}"
            
            print(f"{name:<25} | {ttft:<10} | {tpot:<15} | {tput:<18} ")
            
    print("="*90)

if __name__ == "__main__":
    main()