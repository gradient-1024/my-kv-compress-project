import torch
from datasets import load_dataset
from transformers import AutoTokenizer

class DataLoader:
    """
    负责加载和预处理大作业所需的数据集（pg-19 和 wikitext）。
    """
    def __init__(self, model_id="EleutherAI/pythia-70m", device="cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        # Pythia 系列通常没有设置 pad_token，需手动指定
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.device = device

    def get_pg19_sample(self, split="test", index=0):
        """
        加载 pg-19 数据集的一个单一样本，用于长文本外推性能测试 。
        使用 streaming=True 以避免下载整个数个 GB 的数据集。
        """
        print(f"正在流式加载 pg-19 ({split}) 的第 {index} 个样本...")
        dataset = load_dataset("deepmind/pg19", split=split, streaming=True)
        
        # 获取指定索引的样本
        sample = next(iter(dataset.skip(index)))
        text = sample["text"]
        
        # 将文本转化为 tensor
        inputs = self.tokenizer(text, return_tensors="pt", truncation=False)
        return inputs.to(self.device)

    def get_wikitext_test(self, split="test"):
        """
        加载 wikitext-2 数据集，用于标准的 PPL（困惑度）测试 。
        """
        print(f"正在加载 wikitext-2 ({split})...")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        
        # 过滤掉短句或空行，拼接成较长的文本块进行 PPL 测试
        full_text = "\n\n".join(dataset["text"])
        inputs = self.tokenizer(full_text, return_tensors="pt", truncation=False)
        return inputs.to(self.device)

    def get_dataloader_iterator(self, input_ids, chunk_size=2048):
        """
        将长文本切分为固定长度的块，方便进行批量的 PPL 评估。
        """
        seq_len = input_ids.size(1)
        for i in range(0, seq_len, chunk_size):
            yield input_ids[:, i : i + chunk_size]




# --- 简单测试逻辑 ---
if __name__ == "__main__":
    loader = DataLoader()
    
    # 测试 pg-19 加载
    pg_sample = loader.get_pg19_sample()
    print(f"pg-19 样本 Token 长度: {pg_sample.input_ids.shape[1]}")
    
    # 测试 wikitext 加载
    wiki_data = loader.get_wikitext_test()
    print(f"wikitext 总 Token 长度: {wiki_data.input_ids.shape[1]}")