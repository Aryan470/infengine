import math
import torch
from transformers import LlamaForCausalLM
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

def save_rmsnorm_data(layer):
    torch.manual_seed(42)
    x = torch.randn(1, 128, 4096, dtype=torch.float16)

    # Save input
    x.detach().numpy().tofile("test_data/rmsnorm_input.bin")

    # Save weights
    layer.input_layernorm.weight.data.numpy().tofile("test_data/rmsnorm_weight.bin")

    # Save expected output
    normed = layer.input_layernorm(x)
    normed.detach().numpy().tofile("test_data/rmsnorm_output.bin")

def save_rope_data(model):
    seq_len = 128
    q = torch.randn(1, 32, seq_len, 128, dtype=torch.float16)
    k = torch.randn(1, 8, seq_len, 128, dtype=torch.float16)

    # Compute cos/sin in float32 directly from the model's inv_freq
    inv_freq = model.model.rotary_emb.inv_freq.float()
    positions = torch.arange(seq_len).float()
    # [seq_len, head_dim/2]
    freqs = torch.outer(positions, inv_freq)
    # [1, seq_len, head_dim/2]
    cos_f32 = freqs.cos().unsqueeze(0).contiguous()
    sin_f32 = freqs.sin().unsqueeze(0).contiguous()

    # Build full-dim cos/sin for apply_rotary_pos_emb (it expects [1, seq_len, head_dim])
    # [1, 1, seq_len, head_dim]
    cos_full = torch.cat([cos_f32, cos_f32], dim=-1).unsqueeze(1)
    sin_full = torch.cat([sin_f32, sin_f32], dim=-1).unsqueeze(1)

    q_rot, k_rot = apply_rotary_pos_emb(q.float(), k.float(), cos_full, sin_full)

    q.detach().numpy().tofile("test_data/rope_q_input.bin")
    k.detach().numpy().tofile("test_data/rope_k_input.bin")
    cos_f32.detach().numpy().tofile("test_data/rope_cos.bin")
    sin_f32.detach().numpy().tofile("test_data/rope_sin.bin")
    q_rot.to(torch.float16).detach().numpy().tofile("test_data/rope_q_output.bin")
    k_rot.to(torch.float16).detach().numpy().tofile("test_data/rope_k_output.bin")

def save_scale_causal_softmax_data():
    seq_len = 128
    q_heads = 32
    # [q_heads, seq_len, seq_len]
    inp = torch.randn(q_heads, seq_len, seq_len, dtype=torch.float16)

    scale = 1 / math.sqrt(128)
    # Compute reference in float32 to avoid precision issues in the reference itself
    scores = inp.float() * scale

    # Causal mask: positions where col > row get -inf
    mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)
    scores = scores + mask

    out = torch.nn.functional.softmax(scores, dim=-1)

    inp.to(torch.float16).detach().numpy().tofile("test_data/scalesoftmax_input.bin")
    out.to(torch.float16).detach().numpy().tofile("test_data/scalesoftmax_output.bin")

if __name__ == "__main__":
    model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B", dtype=torch.float16)
    layer = model.model.layers[0]

    save_rmsnorm_data(layer)
    save_rope_data(model)
    save_scale_causal_softmax_data()