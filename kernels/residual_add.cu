#include "residual_add.cuh"

const int K = 1024;

void residual_add(const int rows, const int cols, half* d_x, half* d_delta) {
    // launch 1 block per row
    residual_add_kern<<<rows, K>>>(rows, cols, d_x, d_delta);
}

__global__ void residual_add_kern(const int rows, const int cols, half* d_x, half* d_delta) {
    const int block_id = blockIdx.x;
    const int num_threads = blockDim.x;
    const int thread_id = threadIdx.x;

    half* block_x = d_x + (cols * block_id);
    half* block_delta = d_delta + (cols * block_id);
    for (int i = thread_id; i < cols; i += num_threads) {
        block_x[i] += block_delta[i];
    }
}