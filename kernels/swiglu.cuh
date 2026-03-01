#pragma once
#include <cuda_fp16.h>

// apply swiglu on [seq_len, FFN_DIM] -> [seq_len, FFN_DIM]
void swiglu(int seq_len, half* d_gate, half* d_up, half* d_output);
__global__ void swiglu_kern(int seq_len, half* d_gate, half* d_up, half* d_output);