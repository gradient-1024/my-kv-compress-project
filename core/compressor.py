import torch

class StreamingLLMCompressor:
    def __init__(self, window_size=512, sink_size=4):
        self.window_size = window_size
        self.sink_size = sink_size

    def compress(self, key, value):
        # Tensor 维度: [batch, num_heads, seq_len, head_dim]
        seq_len = key.size(-2)
        
        if seq_len <= self.window_size + self.sink_size:
            return key, value

        k_sink = key[:, :, :self.sink_size, :]
        v_sink = value[:, :, :self.sink_size, :]

        k_window = key[:, :, -self.window_size:, :]
        v_window = value[:, :, -self.window_size:, :]
        
        compressed_k = torch.concat([k_sink, k_window], dim=-2)
        compressed_v = torch.concat([v_sink, v_window], dim=-2)

        return compressed_k, compressed_v
    
class SnapKVCompressor:
    def __init__(self, compression_ratio=0.5, max_capacity=None, window_size=32, kernel_size=5, sink_size=4):
        self.compression_ratio = compression_ratio
        self.max_capacity = max_capacity
        self.window_size = window_size
        self.kernel_size = kernel_size
        self.sink_size = sink_size
        self.computed_max_capacity = None
            
    def compress(self, key, value):
        # key&value 维度: [batch, num_heads, seq_len, head_dim]
        batch, num_heads, seq_len, head_dim = key.shape

        if self.max_capacity is not None:
            self.computed_max_capacity = max(
                self.max_capacity,
                self.window_size + self.sink_size
            )
        elif self.computed_max_capacity is None or seq_len > self.computed_max_capacity + 1:
            self.computed_max_capacity = max(
                int(seq_len * self.compression_ratio),
                self.window_size + self.sink_size
            )

        if (seq_len <= self.computed_max_capacity) or (self.computed_max_capacity <= self.window_size + self.sink_size):
            return key, value
        
        k_count = self.computed_max_capacity - self.sink_size - self.window_size

        with torch.no_grad():
            mid_key = key[:, :, self.sink_size : -self.window_size, :]
            scores = torch.norm(mid_key, p=2, dim=-1)

            # [batch, num_heads, mid_len] -> [batch * num_heads, 1, mid_len]
            # AvgPool1d要求输入格式：[Batch, Channels, Length]
            # 保证池化过程中各head间独立性
            scores=scores.view(-1, 1, scores.shape[-1])
            smoothed_scores = torch.nn.functional.avg_pool1d(scores, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size // 2).view(batch, num_heads, -1)
            _, topk_indices = torch.topk(smoothed_scores, k=k_count, dim=-1)
            topk_indices = topk_indices + self.sink_size
            topk_indices, _ = torch.sort(topk_indices, dim=-1)

        k_sink = key[:, :, :self.sink_size, :]
        v_sink = value[:, :, :self.sink_size, :]

        gather_indices = topk_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        k_mid = torch.gather(key, dim=2, index=gather_indices)
        v_mid = torch.gather(value, dim=2, index=gather_indices)

        k_window = key[:, :, -self.window_size:, :]
        v_window = value[:, :, -self.window_size:, :]

        # sequence维度进行拼接，dim=2
        k_concat = torch.concat([k_sink, k_mid, k_window], dim=2)
        v_concat = torch.concat([v_sink, v_mid, v_window], dim=2)

        return k_concat, v_concat


class H2OCompressor:
    def __init__(self, max_capacity=512, window_size=32, sink_size=4):
        self.max_capacity=max_capacity
        self.window_size = window_size
        self.sink_size = sink_size
        self.scores = {}

    def compress(self, key, value, attn_weights, layer_idx):
        batch, num_heads, seq_len, head_dim = key.shape
        
        # attn_weight 维度：Prefill阶段：[batch, num_heads, seq_len, seq_len]；Decoding阶段：[batch, num_heads, 1, seq_len]
        current_scores = attn_weights.sum(dim=-2) 
        if layer_idx not in self.scores:
            self.scores[layer_idx] = current_scores
        else:
            # 核验seq_len是否对齐
            old_scores = self.scores[layer_idx]
            if old_scores.size(-1) < current_scores.size(-1):
                pad_len = current_scores.size(-1) - old_scores.size(-1)
                pad = torch.zeros(
                    *old_scores.shape[:-1],
                    pad_len,
                    device=old_scores.device,
                    dtype=old_scores.dtype,
                )
                old_scores = torch.cat([old_scores, pad], dim=-1)
            elif old_scores.size(-1) > current_scores.size(-1):
                old_scores = old_scores[..., :current_scores.size(-1)]

            self.scores[layer_idx] = old_scores + current_scores

        if seq_len <= self.max_capacity:
            return key, value

        mask = torch.ones(seq_len, device=key.device, dtype=torch.bool)
        mask[:self.sink_size] = False
        mask[-self.window_size:] = False
        fill_value = torch.finfo(self.scores[layer_idx].dtype).min
        candidate_scores = self.scores[layer_idx].masked_fill(~mask, fill_value)

        k_rest = self.max_capacity - self.sink_size - self.window_size
        _, topk_indices = torch.topk(candidate_scores, k=k_rest, dim=-1)

        sink_indices = torch.arange(self.sink_size, device=key.device).expand(batch, num_heads, -1)
        window_indices = torch.arange(seq_len - self.window_size, seq_len, device=key.device).expand(batch, num_heads, -1)

        final_indices = torch.cat([sink_indices, topk_indices, window_indices], dim=-1)
        final_indices, _ = torch.sort(final_indices, dim=-1)
        gather_indices = final_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)

        k_hh = torch.gather(key, dim=2, index=gather_indices)
        v_hh = torch.gather(value, dim=2, index=gather_indices)
        
        self.scores[layer_idx] = torch.gather(self.scores[layer_idx], dim=-1, index=final_indices)

        return k_hh, v_hh


