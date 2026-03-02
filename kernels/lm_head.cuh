#pragma once
#include <cuda_fp16.h>

void lm_head(half* d_input, half* d_weight, half* d_output);