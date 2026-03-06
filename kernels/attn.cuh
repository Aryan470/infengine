#pragma once
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cstdint>

size_t attn_workspace_size(int max_seq_len);
size_t verify_attn_workspace_size(int N_draft, int total_seq_len);
size_t kv_cache_size();
void attn(cublasHandle_t handle, const float* d_rope_cos, const float* d_rope_sin, const int seq_len, half* d_input, half* d_qproj, half* d_kproj, half* d_vproj, half* d_oproj, half* d_output, void* workspace, void* kv_cache, const int layer_idx);
void decode_attn(cublasHandle_t handle, const float* d_rope_cos, const float* d_rope_sin, const int seq_len, half* d_input, half* d_qproj, half* d_kproj, half* d_vproj, half* d_oproj, half* d_output, void* workspace, void* kv_cache, const int layer_idx);

// Verification attention for speculative decoding
// Processes N_draft tokens, writes K,V at positions seq_len..seq_len+N_draft-1 in kv_cache
// Uses tree_mask for attention masking and per-token positions for RoPE
void verify_attn(cublasHandle_t handle, const float* d_rope_cos, const float* d_rope_sin,
                 int seq_len, int N_draft,
                 const int* d_positions, const int8_t* d_tree_mask,
                 half* d_input, half* d_qkv_proj, half* d_oproj,
                 half* d_output, void* workspace, void* kv_cache, int layer_idx);