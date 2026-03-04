#include "decode.cuh"
#include "kernels/emblookup.cuh"
#include "kernels/rmsnorm.cuh"
#include "kernels/attn.cuh"
#include "kernels/ffn.cuh"
#include "kernels/residual_add.cuh"
#include "kernels/lm_head.cuh"
#include "kernels/sampling.cuh"
#include "config.h"

int decode_step(cublasHandle_t handle, const ModelWeights& weights, int seq_len,
                int* tokenid_buff, half* data_buffer, half* aux_buffer, half* attn_buffer,
                half* output_distr, int* output_token,
                void* workspace, void* kv_cache) {
    emblookup(1, tokenid_buff + seq_len - 1, weights.emb_lookup, data_buffer);

    for (int layer_idx = 0; layer_idx < InfEngineConfig::NUM_LAYERS; layer_idx++) {
        // for most things we can call with seq len 1 to just use the first part of the data/aux buffers
        // apply rmsnorm from data buffer to aux buffer
        rmsnorm(1, data_buffer, weights.layers[layer_idx].input_layernorm, aux_buffer);
        // apply decode_attn, need a special codepath
        decode_attn(handle, weights.rope.cos, weights.rope.sin, seq_len, aux_buffer,
            weights.layers[layer_idx].transformer.w_q,
            weights.layers[layer_idx].transformer.w_k,
            weights.layers[layer_idx].transformer.w_v,
            weights.layers[layer_idx].transformer.w_o,
            attn_buffer, workspace, kv_cache, layer_idx);
        // add into x, copy x back to aux
        residual_add(1, InfEngineConfig::HIDDEN_SIZE, data_buffer, attn_buffer);
        // apply rmsnorm to aux buffer inplace
        rmsnorm(1, data_buffer, weights.layers[layer_idx].post_attention_layernorm, aux_buffer);
        // apply ffn (can use the same codepath)
        ffn(handle, 1, aux_buffer, attn_buffer,
            weights.layers[layer_idx].ffn_block.w_up,
            weights.layers[layer_idx].ffn_block.w_down,
            weights.layers[layer_idx].ffn_block.w_gate, workspace);
        // add back to x
        residual_add(1, InfEngineConfig::HIDDEN_SIZE, data_buffer, attn_buffer);
    }

    // apply final norm
    rmsnorm(1, data_buffer, weights.final_norm, data_buffer);
    lm_head(handle, 1, data_buffer, weights.lm_head, output_distr);
    sample_token_amax(output_distr, output_token);

    int token;
    cudaMemcpy(&token, output_token, sizeof(int), cudaMemcpyDeviceToHost);
    return token;
}
