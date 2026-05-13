import torch
import math
import time
import torch.nn.functional as F

class Evaluator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device

    def _sync_if_needed(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    @torch.no_grad()
    def calculate_ppl(self, input_ids, step_size=1):
        # 按自回归解码路径计算 PPL
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        total_tokens = input_ids.size(1) - 1

        total_nll = 0.0
        past_key_values = None

        for start in range(0, total_tokens, step_size):
            end = min(start + step_size, total_tokens)
            input_chunk = input_ids[:, start:end]
            target_chunk = input_ids[:, start + 1 : end + 1]

            outputs = self.model(
                input_chunk,
                past_key_values=past_key_values,
                use_cache=True,
            )

            logits = outputs.logits
            token_nll = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target_chunk.reshape(-1),
                reduction="sum",
            )

            total_nll += token_nll.item()
            past_key_values = outputs.past_key_values

        return math.exp(total_nll / total_tokens)

    def run_all(self, text, gen_length=50):
        input_ids = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
        seq_len = input_ids.shape[1]
        results = {}

        # 记录 TTFT (Time To First Token)
        self._sync_if_needed()
        start_ttft = time.perf_counter()
        outputs = self.model(input_ids, use_cache=True)
        self._sync_if_needed()
        next_token_logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1)
        ttft = time.perf_counter() - start_ttft

        past_key_values = outputs.past_key_values
        current_input_ids = next_token.unsqueeze(0)

        # 记录 TPOT (Time Per Output Token)
        latencies = []
        for _ in range(gen_length):
            self._sync_if_needed()
            start_step = time.perf_counter()
            
            outputs = self.model(
                current_input_ids, 
                past_key_values=past_key_values, 
                use_cache=True
            )
            self._sync_if_needed()
            
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1)
            past_key_values = outputs.past_key_values
            current_input_ids = next_token.unsqueeze(0)
            
            latencies.append(time.perf_counter() - start_step)

        tpot = sum(latencies) / len(latencies)
        
        # 计算 Throughput (1 / TPOT)
        throughput = 1 / tpot

        # 计算 PPL (Perplexity)
        ppl = self.calculate_ppl(input_ids, step_size=1)
        
        results = {
            "TTFT (s)": round(ttft, 4),
            "TPOT (s/token)": round(tpot, 4),
            "Throughput (tokens/s)": round(throughput, 2),
            "PPL": round(ppl, 4),
            "Context Length": seq_len
        }
        
        return results
