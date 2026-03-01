#pragma once
#include <cuda_fp16.h>

// permute a buffer of [n_heads, seq_len, head_dim] -> [seq_len, n_heads, head_dim]
void attn_permute(int seq_len, __half* d_input, __half* d_output);
__global__ void attn_permute_kern(int seq_len, __half* d_input, __half* d_output);