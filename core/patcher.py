import torch
import types
from core.compressor import StreamingLLMCompressor, SnapKVCompressor, H2OCompressor, PyramidKVCompressor
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXAttention

def patched_forward(self, *args, **kwargs):
    kwargs["output_attentions"] = True
    kwargs["use_cache"] = True
    outputs=self.original_forward(*args, **kwargs)
    if outputs[1] is not None:
        key, value = outputs[1]
        attn_w = outputs[2] if len(outputs) > 2 else None
        if isinstance(self.compressor, H2OCompressor):
            k_opt, v_opt = self.compressor.compress(key, value, attn_w, self.layer_idx)
        elif isinstance(self.compressor, PyramidKVCompressor):
            k_opt, v_opt = self.compressor.compress(key, value, self.layer_idx)
        else:
            k_opt, v_opt = self.compressor.compress(key, value)
        outputs = (outputs[0], (k_opt, v_opt)) + outputs[2:]
    return outputs


def apply_compression_patch(model, compressor):
    total_layers = model.config.num_hidden_layers
    layer_count = 0
    for name, module in model.named_modules():
        if isinstance(module, GPTNeoXAttention):
            module.original_forward = module.forward
            module.compressor = compressor
            module.forward = types.MethodType(patched_forward, module)
            module.layer_idx = layer_count
            layer_count+=1

