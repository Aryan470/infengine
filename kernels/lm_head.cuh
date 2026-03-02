#pragma once
#include <cuda_fp16.h>

void lm_head(const int seq_len, half* d_input, half* d_weight, half* d_output);