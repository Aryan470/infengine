#include "../config.h"
#include "attn_permute.cuh"
#include <cuda_fp16.h>

const int K = InfEngineConfig::HEAD_DIM;

// permute a buffer of [n_heads, seq_len, head_dim] -> [seq_len, n_heads, head_dim]
void attn_permute(int seq_len, __half* d_input, __half* d_output) {
    // out[i][j][k] = in[j][i][k]
    // 1 block per (i = seq_len, j = n_head)
    // threads responsible for k = head_dim axis
    attn_permute_kern<<<seq_len * InfEngineConfig::NUM_Q_HEADS, K>>>(seq_len, d_input, d_output);
}

__global__ void attn_permute_kern(int seq_len, __half* d_input, __half* d_output) {
    const int block_id = blockIdx.x;
    const int num_threads = blockDim.x;
    const int thread_id = threadIdx.x;

    const int seq_id = block_id / InfEngineConfig::NUM_Q_HEADS;
    const int head_id = block_id % InfEngineConfig::NUM_Q_HEADS;

    // my block is responsible for out[seq i][head j]
    // this means head_dim copies
    // input[head_id][seq_id][dim_id]
    const int input_base = head_id * seq_len * InfEngineConfig::HEAD_DIM + seq_id * InfEngineConfig::HEAD_DIM;
    // input[seq_id][head_id][dim_id]
    const int output_base = seq_id * InfEngineConfig::NUM_Q_HEADS * InfEngineConfig::HEAD_DIM + head_id * InfEngineConfig::HEAD_DIM;
    for (int dim_id = thread_id; dim_id < InfEngineConfig::HEAD_DIM; dim_id += num_threads) {
        d_output[output_base + dim_id] = d_input[input_base + dim_id];
    }
}