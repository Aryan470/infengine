#include "../config.h"
#include "swiglu.cuh"
#include <cuda_fp16.h>

const int K = 1024;

// apply swiglu on [seq_len, FFN_DIM] -> [seq_len, FFN_DIM]
void swiglu(int seq_len, half* d_gate, half* d_up, half* d_output) {
    // out[i] = silu(gate[i]) * up[i]
    // silu(x) = x * sigmoid(x) = x * (1 / (1 + exp(-x)))
    // out[i] = gate[i] * (1 / (1 + e^(-gate[i]))) * up[i]
    // let's have one block per token in seq_len
    swiglu_kern<<<seq_len, K>>>(seq_len, d_gate, d_up, d_output);
}

__global__ void swiglu_kern(int seq_len, half* d_gate, half* d_up, half* d_output) {
    const int block_id = blockIdx.x;
    const int num_threads = blockDim.x;
    const int thread_id = threadIdx.x;
    // my block is responsible for FFN_DIM computations
    half* block_output = d_output + (block_id * InfEngineConfig::FFN_DIM);
    half* block_gate = d_gate + (block_id * InfEngineConfig::FFN_DIM);
    half* block_up = d_up + (block_id * InfEngineConfig::FFN_DIM);
    for (int i = thread_id; i < InfEngineConfig::FFN_DIM; i+= num_threads) {
        // out[i] = gate[i] * (1 / (1 + e^(-gate[i]))) * up[i]
        float gate = __half2float(block_gate[i]);
        float up = __half2float(block_up[i]);
        float result = gate * (1.0f / (1.0f + exp(-gate))) * up;
        block_output[i] = __float2half(result);
    }
}