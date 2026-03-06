#pragma once
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cstdint>
#include "kernels/model_weights.cuh"

// Process N_draft tokens through the full model with tree attention
// Writes draft K,V into kv_cache at positions seq_len..seq_len+N_draft-1
// Returns logits for each draft token in output_logits [N_draft, VOCAB_SIZE]
void verify_step(cublasHandle_t handle, const ModelWeights& weights,
                 int seq_len, int N_draft,
                 int* d_draft_tokens, int* d_positions, int8_t* d_tree_mask,
                 half* data_buffer, half* aux_buffer, half* attn_buffer,
                 half* output_logits,
                 void* workspace, void* kv_cache);
