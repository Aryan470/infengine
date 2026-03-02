#pragma once
#include <cuda_fp16.h>
#include <cublas_v2.h>
void lm_head(cublasHandle_t handle, const int seq_len, half* d_input, half* d_weight, half* d_output);