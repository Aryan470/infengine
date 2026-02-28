#pragma once
#include <cuda_fp16.h>

void apply_rope(int num_heads, int seq_len, const float* d_rope_cos, const float* d_rope_sin, __half* d_input, __half* d_output);

#ifdef __CUDACC__
__global__ void apply_rope_kern(int seq_len, const float* d_rope_cos, const float* d_rope_sin, half* d_input, half* d_output);
#endif

// [max_seq_len, hidden_dim]
void init_rope_buffer(float** cu_rope_cos, float** cu_rope_sin);
void cleanup_rope_buffer(float* cu_rope_cos, float* cu_rope_sin);
