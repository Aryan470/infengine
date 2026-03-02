#pragma once
#include <cuda_fp16.h>

// take an array of tokens (ints), and populate a [seq_len, hidden_dim] projection
void emblookup(int seq_len, int* d_input, half* d_weights, half* d_output);
__global__ void emblookup_kern(int seq_len, int* d_input, half* d_weights, half* d_output);