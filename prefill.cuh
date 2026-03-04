#pragma once
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include "kernels/model_weights.cuh"

int prefill(cublasHandle_t handle, const ModelWeights& weights, int seq_len,
            int* tokenid_buff, half* data_buffer, half* aux_buffer, half* attn_buffer,
            half* output_distr, int* output_token,
            void* workspace, void* kv_cache);
