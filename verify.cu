#include "verify.cuh"
#include "kernels/emblookup.cuh"
#include "kernels/rmsnorm.cuh"
#include "kernels/attn.cuh"
#include "kernels/ffn.cuh"
#include "kernels/residual_add.cuh"
#include "kernels/multiply_by_weight.cuh"
#include "config.h"

void verify_step(cublasHandle_t handle, const ModelWeights& weights,
                 int seq_len, int N_draft,
                 int* d_draft_tokens, int* d_positions, int8_t* d_tree_mask,
                 half* data_buffer, half* aux_buffer, half* attn_buffer,
                 half* output_logits,
                 void* workspace, void* kv_cache) {
    // Embed all draft tokens
    emblookup(N_draft, d_draft_tokens, weights.emb_lookup, data_buffer);

    for (int layer_idx = 0; layer_idx < InfEngineConfig::NUM_LAYERS; layer_idx++) {
        rmsnorm(N_draft, data_buffer, weights.layers[layer_idx].input_layernorm, aux_buffer);

        verify_attn(handle, weights.rope.cos, weights.rope.sin,
                    seq_len, N_draft,
                    d_positions, d_tree_mask,
                    aux_buffer,
                    weights.layers[layer_idx].transformer.w_qkv,
                    weights.layers[layer_idx].transformer.w_o,
                    attn_buffer, workspace, kv_cache, layer_idx);

        residual_add(N_draft, InfEngineConfig::HIDDEN_SIZE, data_buffer, attn_buffer);

        rmsnorm(N_draft, data_buffer, weights.layers[layer_idx].post_attention_layernorm, aux_buffer);

        ffn(handle, N_draft, aux_buffer, attn_buffer,
            weights.layers[layer_idx].ffn_block.w_up,
            weights.layers[layer_idx].ffn_block.w_down,
            weights.layers[layer_idx].ffn_block.w_gate, workspace);

        residual_add(N_draft, InfEngineConfig::HIDDEN_SIZE, data_buffer, attn_buffer);
    }

    // Final norm
    rmsnorm(N_draft, data_buffer, weights.final_norm, data_buffer);

    // LM head: project all N_draft tokens to get logits [N_draft, VOCAB_SIZE]
    multiply_by_weight(handle, N_draft, InfEngineConfig::HIDDEN_SIZE, InfEngineConfig::VOCAB_SIZE,
                       data_buffer, weights.lm_head, output_logits);
}
