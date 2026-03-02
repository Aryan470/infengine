#pragma once
#include <cuda_fp16.h>
#include <cublas_v2.h>

size_t attn_workspace_size(int max_seq_len);
void attn(cublasHandle_t handle, const float* d_rope_cos, const float* d_rope_sin, const int seq_len, half* d_input, half* d_qproj, half* d_kproj, half* d_vproj, half* d_oproj, half* d_output, void* workspace);