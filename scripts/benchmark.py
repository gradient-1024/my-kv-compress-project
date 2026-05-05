import torch
import math
import time
from tqdm import tqdm

class Evaluator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device

    @torch.no_grad()
    def calculate_ppl(self, input_ids):
        # 计算 PPL (Perplexity)
        outputs = self.model(input_ids, labels=input_ids)
        loss = outputs.loss
        return math.exp(loss.item())

    def run_all(self, text, gen_length=50):
        input_ids = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
        seq_len = input_ids.shape[1]
        results = {}

        # 记录 TTFT (Time To First Token)
        start_ttft = time.time()
        outputs = self.model(input_ids, use_cache=True)
        next_token_logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1)
        ttft = time.time() - start_ttft

        past_key_values = outputs.past_key_values   # outputs.past_key_values已包含生成的新token的信息
        current_input_ids = next_token.unsqueeze(0)

        # 记录 TPOT (Time Per Output Token)
        latencies = []
        for _ in range(gen_length):
            start_step = time.time()
            
            outputs = self.model(
                current_input_ids, 
                past_key_values=past_key_values, 
                use_cache=True
            )
            
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1)
            past_key_values = outputs.past_key_values
            current_input_ids = next_token.unsqueeze(0)
            
            latencies.append(time.time() - start_step)

        tpot = sum(latencies) / len(latencies)
        
        # 计算 Throughput (1 / TPOT)
        throughput = 1 / tpot

        # 计算 PPL (Perplexity)
        # ppl = self.calculate_ppl(input_ids)
        
        results = {
            "TTFT (s)": round(ttft, 4),
            "TPOT (s/token)": round(tpot, 4),
            "Throughput (tokens/s)": round(throughput, 2),
            "Context Length": seq_len
        }
        
        return results