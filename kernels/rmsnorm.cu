#include "rmsnorm.cuh"
#include "../config.h"
#include <cuda_fp16.h>

const float EPSILON = 1e-5;

__host__ void rmsnorm(int seq_len, __half* d_input, __half* d_weight, __half* d_output) {
    const int num_threads = 1024;
    rmsnorm_kern<<<seq_len, num_threads, num_threads * sizeof(float)>>>(seq_len, d_input, d_weight, d_output);
}

__device__ float reduce_sum_rms(float my_val, float* reduce_space) {
    const int thread_id = threadIdx.x;
    const int num_threads = blockDim.x;
    reduce_space[thread_id] = my_val;

    __syncthreads();
    for (int delta = num_threads / 2; delta > 0; delta >>= 1) {
        if (thread_id < delta) {
            reduce_space[thread_id] += reduce_space[thread_id + delta];
        }
        __syncthreads();
    }
    return reduce_space[0];
}

/*
    Input: [seq_len, hidden_dim]
    Output: [seq_len, hidden_dim]
*/
__global__ void rmsnorm_kern(int seq_len, half* d_input, half* d_weight, half* d_output) {
    // shared accumulator
    extern __shared__ float shmem[];

    // design: 1 block per token (seq len)
    // K threads per block (hidden dim / max_threads)

    // input is [seq_len, hidden dim]
    // i care about [block_id, hidden_dim], and within that 4096: [tid * hidden_dim/threads_per_block, (tid+1) * hidden_dim/threads_per_block)
    const int block_id = blockIdx.x;
    const int threads_per_block = blockDim.x;
    const int thread_id = threadIdx.x;
    half* my_input = d_input + block_id * InfEngineConfig::HIDDEN_SIZE;
    // in this case, output gets the same transform
    half* my_output = d_output + block_id * InfEngineConfig::HIDDEN_SIZE;
    // weight does not depend on seq len (block_di)
    half* my_weight = d_weight;

    float my_sum = 0.0;

    for (int i = thread_id; i < InfEngineConfig::HIDDEN_SIZE; i += threads_per_block) {
        float data = __half2float(my_input[i]);
        my_sum += data * data;
    }

    float global_sum = reduce_sum_rms(my_sum, shmem);
    // = 1/sqrt{mean square + eps}
    const float scaling_denom = rsqrt((global_sum / InfEngineConfig::HIDDEN_SIZE) + EPSILON);

    for (int i = thread_id; i < InfEngineConfig::HIDDEN_SIZE; i += threads_per_block) {
        my_output[i] = __hmul(__float2half(__half2float(my_input[i]) * scaling_denom), my_weight[i]);
    }
}