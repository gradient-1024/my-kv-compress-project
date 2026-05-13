import torch
from datasets import load_dataset
from transformers import AutoTokenizer

class DataLoader:
    def __init__(self, model_id="EleutherAI/pythia-70m", device="cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.device = device

    def get_pg19_sample(self, split="test", index=0):
        print(f"加载 pg-19 ({split}) 的第 {index} 个样本...")
        dataset = load_dataset("deepmind/pg19", split=split, streaming=True)

        sample = next(iter(dataset.skip(index)))
        text = sample["text"]

        inputs = self.tokenizer(text, return_tensors="pt", truncation=False)
        return inputs.to(self.device)

    def get_wikitext_test(self, split="test"):
        print(f"正在加载 wikitext-2 ({split})...")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        
        # 过滤短句或空行，拼接为较长文本块
        full_text = "\n\n".join(dataset["text"])
        inputs = self.tokenizer(full_text, return_tensors="pt", truncation=False)
        return inputs.to(self.device)

    def get_dataloader_iterator(self, input_ids, chunk_size=2048):
        seq_len = input_ids.size(1)
        for i in range(0, seq_len, chunk_size):
            yield input_ids[:, i : i + chunk_size]


if __name__ == "__main__":
    loader = DataLoader()

    pg_sample = loader.get_pg19_sample()
    print(f"pg-19 样本 Token 长度: {pg_sample.input_ids.shape[1]}")

    wiki_data = loader.get_wikitext_test()
    print(f"wikitext 总 Token 长度: {wiki_data.input_ids.shape[1]}")