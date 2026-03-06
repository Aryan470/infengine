#pragma once
#include <cuda_fp16.h>

void apply_rope(const int start_pos, const int num_heads, const int seq_len, const float* d_rope_cos, const float* d_rope_sin, __half* d_input, __half* d_output, bool is_kv_cache);

void apply_rope_positions(const int* d_positions, int num_heads, int seq_len,
                          const float* d_rope_cos, const float* d_rope_sin,
                          half* d_input, half* d_output, bool is_kv_cache);

void apply_rope_positions_strided(const int* d_positions, int num_heads, int seq_len,
                                  const float* d_rope_cos, const float* d_rope_sin,
                                  half* d_input, half* d_output,
                                  int head_stride, int token_stride);

#ifdef __CUDACC__
__global__ void apply_rope_kern(const int start_pos, const int seq_len, const float* d_rope_cos, const float* d_rope_sin, half* d_input, half* d_output, bool is_kv_cache);
__global__ void apply_rope_positions_kern(const int* d_positions, const int seq_len, const float* d_rope_cos, const float* d_rope_sin, half* d_input, half* d_output, bool is_kv_cache);
__global__ void apply_rope_positions_strided_kern(const int* d_positions, const int seq_len,
                                                   const float* d_rope_cos, const float* d_rope_sin,
                                                   half* d_input, half* d_output,
                                                   int head_stride, int token_stride);
#endif

// [max_seq_len, hidden_dim]
void init_rope_buffer(float** cu_rope_cos, float** cu_rope_sin);
void cleanup_rope_buffer(float* cu_rope_cos, float* cu_rope_sin);
