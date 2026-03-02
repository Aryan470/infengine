#pragma once
#include <cuda_fp16.h>

void sample_token_amax(const half* d_logits, int* d_output);