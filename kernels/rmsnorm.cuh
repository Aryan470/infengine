#pragma once
#include <cuda_fp16.h>

__host__ void rmsnorm(int seq_len, __half* d_input, __half* d_weight, __half* d_output);
__global__ void rmsnorm_kern(int seq_len, __half* d_input, __half* d_weight, __half* d_output);