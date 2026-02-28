#include "rmsnorm.cuh"
#include "../config.h"
#include <cuda_fp16.h>

const float EPSILON = 1e-5;

__host__ void rmsnorm(int seq_len, __half* d_input, __half* d_weight, __half* d_output) {
    rmsnorm_kern<<<seq_len, 1024>>>(seq_len, d_input, d_weight, d_output);
}

/*
    Input: [seq_len, hidden_dim]
    Output: [seq_len, hidden_dim]
*/
__global__ void rmsnorm_kern(int seq_len, half* d_input, half* d_weight, half* d_output) {
    // shared accumulator
    __shared__ float shared_buf[1];

    // design: 1 block per token (seq len)
    // K threads per block (hidden dim / max_threads)

    // input is [seq_len, hidden dim]
    // i care about [block_id, hidden_dim], and within that 4096: [tid * hidden_dim/threads_per_block, (tid+1) * hidden_dim/threads_per_block)
    const int block_id = blockIdx.x;
    const int threads_per_block = blockDim.x;
    const int data_per_thread = InfEngineConfig::HIDDEN_SIZE / threads_per_block;
    const int thread_id = threadIdx.x;
    half* my_input = d_input + block_id * InfEngineConfig::HIDDEN_SIZE + thread_id * data_per_thread;
    // in this case, output gets the same transform
    half* my_output = d_output + block_id * InfEngineConfig::HIDDEN_SIZE + thread_id * data_per_thread;
    // weight does not depend on seq len (block_di)
    half* my_weight = d_weight + thread_id * data_per_thread;

    float my_sum = 0.0;

    if (thread_id == 0) {shared_buf[0] = 0.0;}
    __syncthreads();

    for (int i = 0; i < data_per_thread; i++) {
        float data = __half2float(my_input[i]);
        my_sum += data * data;
    }
    atomicAdd(&shared_buf[0], my_sum);
    __syncthreads();
    if (thread_id == 0) {
        shared_buf[0] = rsqrt((shared_buf[0] / InfEngineConfig::HIDDEN_SIZE) + EPSILON);}
    __syncthreads();

    // = 1/sqrt{mean square + eps}
    const float scaling_denom = shared_buf[0];

    for (int i = 0; i < data_per_thread; i++) {
        my_output[i] = __hmul(__float2half(__half2float(my_input[i]) * scaling_denom), my_weight[i]);
    }
}