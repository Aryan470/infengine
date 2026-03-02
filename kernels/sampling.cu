#include "../config.h"
#include "sampling.cuh"
#include <cublas_v2.h>
#include <driver_types.h>

const int K = 1024;

__device__ int reduce_amax(int my_amax, half my_max, int* amax_space, half* max_space) {
    const int thread_id = threadIdx.x;
    const int num_threads = blockDim.x;

    amax_space[thread_id] = my_amax;
    max_space[thread_id] = my_max;

    __syncthreads();
    for (int delta = num_threads / 2; delta > 0; delta >>= 1) {
        if (thread_id < delta) {
            if (max_space[thread_id] < max_space[thread_id + delta]) {
                max_space[thread_id] = max_space[thread_id + delta];
                amax_space[thread_id] = amax_space[thread_id + delta];
            }
        }
        __syncthreads();
    }
    return amax_space[0];
}

__global__ void amax_kern(const half* d_logits, int* d_out) {
    extern __shared__ char shmem[];
    const int num_threads = blockDim.x;
    const int thread_id = threadIdx.x;
    // simple reduction over d_logits
    int* amax_space = (int*) shmem;
    half* max_space = (half*) (shmem + sizeof(int) * num_threads);
    // first each thread will compute their local amax
    int local_amax = thread_id;
    half local_max = d_logits[thread_id];

    for (int i = thread_id + num_threads; i < InfEngineConfig::VOCAB_SIZE; i += num_threads) {
        half this_val = d_logits[i];
        if (this_val > local_max) {
            local_amax = i;
            local_max = this_val;
        }
    }

    int result = reduce_amax(local_amax, local_max, amax_space, max_space);
    if (thread_id == 0) {
        *d_out = result;
    }
}


void sample_token_amax(const half* d_logits, int* d_output) {
    // just one block, threads will work together to take amax over vocab_size, alloc num_threads * (sizeof(int) + sizeof(half)) shmem
    const int shmem_size = (sizeof(half) + sizeof(int)) * K;
    amax_kern<<<1, K, shmem_size>>>(d_logits, d_output);
}
