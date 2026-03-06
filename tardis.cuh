#pragma once
#include <cuda_fp16.h>
#include <cublas_v2.h>

// TARDIS GPU kernels

// Process a single token: update embedding and context vectors
// Reads/writes half from global embedding table, computes in float
__global__ void tardis_process_token_kern(float* d_embedding, float* d_context,
                                           half* d_token_emb, int dim,
                                           float sqrt_1ma, float sqrt_a,
                                           float sqrt_1mb, float sqrt_b,
                                           float cos_omega, float sin_omega,
                                           unsigned long long seed, int step);

// Compute direction vectors for a batch of parents (float in/out)
// Scaled antisymmetric: dir[2j] = -2*sin_omega*emb[2j+1], dir[2j+1] = 2*sin_omega*emb[2j]
__global__ void tardis_compute_dir_vecs_kern(const float* d_parent_embs, float* d_dir_vecs,
                                              int num_parents, int dim, float sin_omega);

// Filter and find top-B candidates per parent based on sim and dir thresholds
__global__ void tardis_filter_topB_kern(const float* d_sim, const float* d_dir,
                                         int* d_topB_out, float* d_topB_sims,
                                         int vocab_size, int num_parents, int B,
                                         float min_sim, float min_dir);

// Compute local embedding for a child token (reads half global table, writes float)
__global__ void tardis_compute_local_emb_kern(const half* d_global_embs, const float* d_branch_contexts,
                                               float* d_local_embs, const int* d_parent_tokens,
                                               int num_parents, int dim,
                                               float sqrt_1ma, float sqrt_a);

// Combine sim and dir scores with optional sim_threshold masking
// dir_scale cancels the 2*sin_omega factor from dir_vecs kernel
__global__ void tardis_combine_scores_kern(const float* sim, const float* dir,
                                            float* out, int total,
                                            float cos_phi, float sin_phi,
                                            float dir_scale,
                                            float sim_threshold, bool use_filter);

// Convert float buffer to half
__global__ void float_to_half_kern(const float* src, half* dst, int n);

// Convert half buffer to float
__global__ void half_to_float_kern(const half* src, float* dst, int n);

// Gather dim-dimensional FP16 vectors by token index and convert to FP32
__global__ void gather_half_to_float_kern(const half* table, const int* indices,
                                           float* out, int num, int dim);

// ============================================================================
// Option B: Single Packed cuBLAS GEMM + Fused Combine+TopK+Gather
// ============================================================================

// Pack GEMM input: compute dir vecs + convert parent embs to FP16 column-major.
__global__ void tardis_pack_gemm_input_kern(
    const float* __restrict__ d_parent_embs,
    half* d_packed_h,
    int num_parents, int dim, float sin_omega, bool use_combined);

// Chunked top-B: splits vocab across many blocks for GPU saturation.
// Grid: dim3(num_parents, num_chunks).
__global__ void tardis_chunked_topk_kern(
    const float* __restrict__ d_gemm_out,
    int vocab_size, int num_parents, int B,
    int sim_col_offset,
    int* d_block_winners_ids, float* d_block_winners_vals,
    int chunk_size,
    float cos_phi, float sin_phi, float dir_scale,
    float sim_threshold_val,
    bool use_combined, bool use_sim_filter);

// Cross-block reduction of chunk winners + gather embeddings for next depth.
// Grid: num_parents blocks, 256 threads each.
__global__ void tardis_reduce_topk_gather_kern(
    const int* d_block_winners_ids, const float* d_block_winners_vals,
    int num_chunks, int num_parents, int B,
    int* d_winner_tokens,
    const half* __restrict__ d_embeddings,
    float* d_parent_embs,
    int dim, bool skip_gather);
