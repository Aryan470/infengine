#include "../config.h"
#include "emblookup.cuh"

const int K = 1024;

// take an array of tokens (ints), and populate a [seq_len, hidden_dim] projection
void emblookup(const int seq_len, int* d_input, half* d_weights, half* d_output) {
    // simple kernel launch, use 1 block per seq_len
    emblookup_kern<<<seq_len, K>>>(seq_len, d_input, d_weights, d_output);
}

__global__ void emblookup_kern(int seq_len, int* d_input, half* d_weights, half* d_output) {
    // one block (i) per seq_len
    // my and my thread friends need to fill d_output
    const int block_id = blockIdx.x;
    const int thread_id = threadIdx.x;
    const int num_threads = blockDim.x;

    half* block_output = d_output + (block_id * InfEngineConfig::HIDDEN_SIZE);
    // we are all going to read from the same spot: d_weights[d_input[block_id]]
    half* block_input = d_weights + (d_input[block_id] * InfEngineConfig::HIDDEN_SIZE);
    for (int j = thread_id; j < InfEngineConfig::HIDDEN_SIZE; j += num_threads) {
        block_output[j] = block_input[j];
    }
}