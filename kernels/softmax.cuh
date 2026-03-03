#pragma once
#include <cuda_fp16.h>

void scale_causal_softmax(int seq_len, __half* d_input, __half* d_output, const bool is_row);
__global__ void scale_causal_softmax_kern(int seq_len, __half* d_input, __half* d_output);
__global__ void scale_row_softmax_kern(int seq_len, __half* d_input, __half* d_output);