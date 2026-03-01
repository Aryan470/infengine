import math
import torch
from transformers import LlamaForCausalLM
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

N_Q_HEADS = 32
N_KV_HEADS = 8
HIDDEN_DIM = 4096
HEAD_DIM = 128
FFN_DIM = 14336

def save_rmsnorm_data():
    torch.manual_seed(42)
    x = torch.randn(1, 128, 4096, dtype=torch.float16)
    layer = model.model.layers[0]

    # Save input
    x.detach().numpy().tofile("test_data/rmsnorm_input.bin")

    # Save weights
    layer.input_layernorm.weight.data.numpy().tofile("test_data/rmsnorm_weight.bin")

    # Save expected output
    normed = layer.input_layernorm(x)
    normed.detach().numpy().tofile("test_data/rmsnorm_output.bin")

def save_rope_data(model):
    torch.manual_seed(42)
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
    torch.manual_seed(42)
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

def generate_attn_inputs():
    torch.manual_seed(42)
    # generate inputs of form [seq_len, hidden_dim]
    seq_len = 128
    hidden_dim = 4096
    return torch.randn(1, seq_len, hidden_dim, dtype=torch.float16)


def save_e2e_attn_data(model):
    inp = generate_attn_inputs()
    attn = model.model.layers[0].self_attn
    seq_len = inp.shape[1]
    position_ids = torch.arange(seq_len).unsqueeze(0)

    attn.q_proj.weight.data.to(torch.float16).cpu().numpy().tofile("test_data/attn_qproj.bin")
    attn.k_proj.weight.data.to(torch.float16).cpu().numpy().tofile("test_data/attn_kproj.bin")
    attn.v_proj.weight.data.to(torch.float16).cpu().numpy().tofile("test_data/attn_vproj.bin")
    attn.o_proj.weight.data.to(torch.float16).cpu().numpy().tofile("test_data/attn_oproj.bin")

    # store intermediate

    # Compute RoPE embeddings (cos, sin) from the model's rotary embedding
    rotary_emb = model.model.rotary_emb
    position_embeddings = rotary_emb(inp, position_ids)  # returns (cos, sin)

    output, attn_weights = attn(
        hidden_states=inp,
        position_embeddings=position_embeddings,
        attention_mask=None,
    )

    # save output to a bin
    inp.to(torch.float16).detach().numpy().tofile("test_data/attn_e2e_input.bin")
    output.to(torch.float16).detach().numpy().tofile("test_data/attn_e2e_output.bin")

def save_swiglu_data():
    seq_len = 128
    # generate two inputs: gate and up both [seq_len, FFN_DIM]
    torch.manual_seed(42)
    gate = torch.randn(seq_len, FFN_DIM, dtype=torch.float16)
    up = torch.randn(seq_len, FFN_DIM, dtype=torch.float16)

    # promote to float for intermediate comp
    out = torch.nn.functional.silu(gate.float()) * up.float()
    gate.to(torch.float16).detach().numpy().tofile("test_data/swiglu_gate.bin")
    up.to(torch.float16).detach().numpy().tofile("test_data/swiglu_up.bin")
    out.to(torch.float16).detach().numpy().tofile("test_data/swiglu_out.bin")

def save_ffn_data(model):
    torch.manual_seed(42)

    seq_len = 128
    mlp = model.model.layers[0].mlp

    # (14336, 4096)
    gate = mlp.gate_proj.weight
    # (14336, 4096)
    up = mlp.up_proj.weight
    # (4096, 14336)
    down = mlp.down_proj.weight
    # save weights (these are fp16)
    gate.detach().contiguous().numpy().tofile("test_data/ffn_wgate.bin")
    up.detach().contiguous().numpy().tofile("test_data/ffn_wup.bin")
    down.detach().contiguous().numpy().tofile("test_data/ffn_wdown.bin")

    # ffn takes [seq_len, hidden_dim] -> [seq_len, hidden_dim]
    x = torch.randn(seq_len, HIDDEN_DIM, dtype=torch.float16, device="cuda")
    mlp.to(device="cuda")
    out = mlp(x.to(torch.float16))
    x.to(torch.float16).detach().cpu().numpy().tofile("test_data/ffn_in.bin")
    out.to(torch.float16).detach().cpu().numpy().tofile("test_data/ffn_out.bin")


if __name__ == "__main__":
    model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B", torch_dtype=torch.float16)

    save_rmsnorm_data()
    save_rope_data(model)
    save_scale_causal_softmax_data()
    save_e2e_attn_data(model)
    save_swiglu_data()
    save_ffn_data(model)