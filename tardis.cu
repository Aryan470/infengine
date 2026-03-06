#include "tardis.cuh"
#include <curand_kernel.h>
#include <cmath>

__global__ void tardis_process_token_kern(float* d_embedding, float* d_context,
                                           half* d_token_emb, int dim,
                                           float sqrt_1ma, float sqrt_a,
                                           float sqrt_1mb, float sqrt_b,
                                           float cos_omega, float sin_omega,
                                           unsigned long long seed, int step) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dim) return;

    // Update global embedding for this token: e[tok] = sqrt(1-α)*e[tok] + sqrt(α)*context
    float old_e = __half2float(d_token_emb[i]);
    float ctx = d_context[i];
    float new_e = sqrt_1ma * old_e + sqrt_a * ctx;
    d_token_emb[i] = __float2half(new_e);
    d_embedding[i] = new_e;

    // Rotate context by omega (pairwise 2D rotation)
    float rotated;
    if (i % 2 == 0) {
        float ctx_next = (i + 1 < dim) ? d_context[i + 1] : 0.0f;
        rotated = cos_omega * ctx - sin_omega * ctx_next;
    } else {
        float ctx_prev = d_context[i - 1];
        rotated = sin_omega * ctx_prev + cos_omega * ctx;
    }

    // Generate noise
    curandState state;
    curand_init(seed, i + step * dim, 0, &state);
    float noise = curand_normal(&state);

    // Update context
    d_context[i] = sqrt_1mb * rotated + sqrt_b * noise;
}

// Dir vec = scaled antisymmetric rearrangement of embedding:
//   dir[2j]   = -2*sin_omega*emb[2j+1]
//   dir[2j+1] = +2*sin_omega*emb[2j]
// GEMM result = 2*sin_omega * (q_even @ all_odd.T - q_odd @ all_even.T)
// The 2*sin_omega factor improves FP16 numerical stability and is
// cancelled out in the combine kernel when blending with sim via phi.
__global__ void tardis_compute_dir_vecs_kern(const float* d_parent_embs, float* d_dir_vecs,
                                              int num_parents, int dim, float sin_omega) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_parents * dim;
    if (idx >= total) return;

    int p = idx / dim;
    int i = idx % dim;

    const float* emb = d_parent_embs + p * dim;
    float val;
    if (i % 2 == 0) {
        val = -2.0f * sin_omega * emb[i + 1];
    } else {
        val = 2.0f * sin_omega * emb[i - 1];
    }
    d_dir_vecs[idx] = val;
}

// Top-B selection with parallel tree reduction.
// Phase 1: each thread scans vocab_size/num_threads elements, maintains local top-B.
// Phase 2: log2(num_threads) parallel merge steps instead of serial.
// Shared memory: num_threads * B * (sizeof(int) + sizeof(float))
__global__ void tardis_filter_topB_kern(const float* d_sim, const float* d_dir,
                                         int* d_topB_out, float* d_topB_sims,
                                         int vocab_size, int num_parents, int B,
                                         float min_sim, float min_dir) {
    int parent_id = blockIdx.x;
    if (parent_id >= num_parents) return;

    extern __shared__ char shmem[];
    int num_threads = blockDim.x;
    int* s_ids = (int*)shmem;                              // [num_threads * B]
    float* s_vals = (float*)(s_ids + num_threads * B);     // [num_threads * B]

    int tid = threadIdx.x;
    int base = tid * B;

    // Init this thread's slots
    for (int i = 0; i < B; i++) {
        s_ids[base + i] = -1;
        s_vals[base + i] = -1e30f;
    }

    const float* my_scores = d_sim + (size_t)parent_id * vocab_size;

    // Phase 1: parallel scan — each thread finds its local top-B
    int count = 0;
    for (int v = tid; v < vocab_size; v += num_threads) {
        float val = my_scores[v];
        if (count < B) {
            s_ids[base + count] = v;
            s_vals[base + count] = val;
            count++;
        } else {
            // Find min in local top-B
            int min_idx = 0;
            for (int j = 1; j < B; j++) {
                if (s_vals[base + j] < s_vals[base + min_idx]) min_idx = j;
            }
            if (val > s_vals[base + min_idx]) {
                s_ids[base + min_idx] = v;
                s_vals[base + min_idx] = val;
            }
        }
    }
    __syncthreads();

    // Phase 2: parallel tree reduction — log2(num_threads) steps
    for (int stride = num_threads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            int base_a = tid * B;
            int base_b = (tid + stride) * B;
            // Merge B entries from b into a, keeping top-B
            for (int i = 0; i < B; i++) {
                if (s_vals[base_b + i] <= -1e30f) continue;
                int min_idx = 0;
                for (int j = 1; j < B; j++) {
                    if (s_vals[base_a + j] < s_vals[base_a + min_idx]) min_idx = j;
                }
                if (s_vals[base_b + i] > s_vals[base_a + min_idx]) {
                    s_ids[base_a + min_idx] = s_ids[base_b + i];
                    s_vals[base_a + min_idx] = s_vals[base_b + i];
                }
            }
        }
        __syncthreads();
    }

    // Thread 0 writes final result
    if (tid == 0) {
        for (int i = 0; i < B; i++) {
            d_topB_out[parent_id * B + i] = s_ids[i];
            d_topB_sims[parent_id * B + i] = s_vals[i];
        }
    }
}

__global__ void tardis_compute_local_emb_kern(const half* d_global_embs, const float* d_branch_contexts,
                                               float* d_local_embs, const int* d_parent_tokens,
                                               int num_parents, int dim,
                                               float sqrt_1ma, float sqrt_a) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_parents * dim;
    if (idx >= total) return;

    int p = idx / dim;
    int i = idx % dim;

    int tok = d_parent_tokens[p];
    float global_e = __half2float(d_global_embs[(size_t)tok * dim + i]);
    float ctx = d_branch_contexts[p * dim + i];

    d_local_embs[p * dim + i] = sqrt_1ma * global_e + sqrt_a * ctx;
}

__global__ void float_to_half_kern(const float* src, half* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half(src[i]);
}

__global__ void half_to_float_kern(const half* src, float* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __half2float(src[i]);
}

// Combine sim and dir scores: out = cos_phi * sim + sin_phi * (dir * dir_scale)
// dir_scale = 1/(2*sin_omega) to cancel the scaling in dir_vecs kernel
// Optionally mask where sim < sim_threshold to -1e30
__global__ void tardis_combine_scores_kern(const float* sim, const float* dir,
                                            float* out, int total,
                                            float cos_phi, float sin_phi,
                                            float dir_scale,
                                            float sim_threshold, bool use_filter) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    float s = sim[idx];
    float d = dir[idx] * dir_scale;
    float score = cos_phi * s + sin_phi * d;
    if (use_filter && s < sim_threshold) score = -1e30f;
    out[idx] = score;
}

// Gather dim-dimensional FP16 vectors by token index and convert to FP32
__global__ void gather_half_to_float_kern(const half* table, const int* indices,
                                           float* out, int num, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num * dim;
    if (idx >= total) return;
    int p = idx / dim;
    int i = idx % dim;
    int tok = indices[p];
    out[p * dim + i] = __half2float(table[(size_t)tok * dim + i]);
}

// ============================================================================
// Option B: Single Packed cuBLAS GEMM + Fused Combine+TopK+Gather
// ============================================================================

// Pack GEMM input: compute dir vecs + convert parent embs to FP16.
// Output is column-major (dim × N_cols) for cuBLAS:
//   Columns 0..num_parents-1:                dir vecs (FP16)
//   Columns num_parents..2*num_parents-1:    parent embs (FP16, only if use_combined)
//
// Dir vec formula: dir[2j] = -2*sin_omega*emb[2j+1], dir[2j+1] = 2*sin_omega*emb[2j]
// Grid/Block: simple 1D, total = num_parents * dim threads
__global__ void tardis_pack_gemm_input_kern(
    const float* __restrict__ d_parent_embs,    // [num_parents, dim] row-major
    half* d_packed_h,                            // [dim, N_cols] column-major output
    int num_parents, int dim, float sin_omega, bool use_combined)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_parents * dim;
    if (idx >= total) return;

    int p = idx / dim;
    int i = idx % dim;

    const float* emb = d_parent_embs + p * dim;

    // Dir vec column (column p, offset = p * dim + i)
    float dir_val;
    if (i % 2 == 0) {
        dir_val = -2.0f * sin_omega * emb[i + 1];
    } else {
        dir_val = 2.0f * sin_omega * emb[i - 1];
    }
    d_packed_h[p * dim + i] = __float2half(dir_val);

    // Sim column (column p + num_parents) if combined
    if (use_combined) {
        d_packed_h[(p + num_parents) * dim + i] = __float2half(emb[i]);
    }
}

// Chunked top-B: splits vocab across many blocks for GPU saturation.
// Each block scans a chunk of the vocab, computes combined scores inline,
// and finds its local top-B via parallel tree reduction.
//
// Grid: dim3(num_parents, num_chunks)
// Block: 256 threads
// Shared mem: 256 * B * (sizeof(int) + sizeof(float))
__global__ void tardis_chunked_topk_kern(
    const float* __restrict__ d_gemm_out,   // [VOCAB_SIZE, N_cols] column-major
    int vocab_size, int num_parents, int B,
    int sim_col_offset,
    int* d_block_winners_ids,                // [num_chunks * num_parents * B]
    float* d_block_winners_vals,
    int chunk_size,
    float cos_phi, float sin_phi, float dir_scale,
    float sim_threshold_val,
    bool use_combined, bool use_sim_filter)
{
    int parent_id = blockIdx.x;
    int chunk_id = blockIdx.y;
    int chunk_start = chunk_id * chunk_size;
    if (parent_id >= num_parents) return;

    extern __shared__ char shmem[];
    const int num_threads = blockDim.x;
    int* s_ids = (int*)shmem;
    float* s_vals = (float*)(s_ids + num_threads * B);

    int tid = threadIdx.x;
    int base = tid * B;

    for (int i = 0; i < B; i++) {
        s_ids[base + i] = -1;
        s_vals[base + i] = -1e30f;
    }

    const float* dir_scores = d_gemm_out + (size_t)parent_id * vocab_size;
    const float* sim_scores = use_combined ?
        d_gemm_out + (size_t)(parent_id + sim_col_offset) * vocab_size : nullptr;

    int chunk_end = min(chunk_start + chunk_size, vocab_size);
    int count = 0;
    for (int v = chunk_start + tid; v < chunk_end; v += num_threads) {
        float score;
        if (use_combined) {
            float s = sim_scores[v];
            float d = dir_scores[v] * dir_scale;
            score = cos_phi * s + sin_phi * d;
            if (use_sim_filter && s < sim_threshold_val) score = -1e30f;
        } else {
            score = dir_scores[v];
        }

        if (count < B) {
            s_ids[base + count] = v;
            s_vals[base + count] = score;
            count++;
        } else {
            int min_idx = 0;
            for (int j = 1; j < B; j++) {
                if (s_vals[base + j] < s_vals[base + min_idx]) min_idx = j;
            }
            if (score > s_vals[base + min_idx]) {
                s_ids[base + min_idx] = v;
                s_vals[base + min_idx] = score;
            }
        }
    }
    __syncthreads();

    for (int stride = num_threads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            int base_a = tid * B;
            int base_b = (tid + stride) * B;
            for (int i = 0; i < B; i++) {
                if (s_vals[base_b + i] <= -1e30f) continue;
                int min_idx = 0;
                for (int j = 1; j < B; j++) {
                    if (s_vals[base_a + j] < s_vals[base_a + min_idx]) min_idx = j;
                }
                if (s_vals[base_b + i] > s_vals[base_a + min_idx]) {
                    s_ids[base_a + min_idx] = s_ids[base_b + i];
                    s_vals[base_a + min_idx] = s_vals[base_b + i];
                }
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        int out_base = (chunk_id * num_parents + parent_id) * B;
        for (int i = 0; i < B; i++) {
            d_block_winners_ids[out_base + i] = s_ids[i];
            d_block_winners_vals[out_base + i] = s_vals[i];
        }
    }
}

// Cross-block reduction of chunk-local winners + gather winning embeddings.
// Merges per-chunk top-B into global top-B, then cooperatively gathers
// FP16 embeddings for the next depth.
//
// Grid: num_parents blocks
// Block: 256 threads
// Shared mem: 256 * B * (sizeof(int) + sizeof(float))
__global__ void tardis_reduce_topk_gather_kern(
    const int* d_block_winners_ids,
    const float* d_block_winners_vals,
    int num_chunks, int num_parents, int B,
    int* d_winner_tokens,
    const half* __restrict__ d_embeddings,
    float* d_parent_embs,
    int dim, bool skip_gather)
{
    int parent_id = blockIdx.x;
    if (parent_id >= num_parents) return;

    extern __shared__ char shmem[];
    const int num_threads = blockDim.x;
    int* s_ids = (int*)shmem;
    float* s_vals = (float*)(s_ids + num_threads * B);

    int tid = threadIdx.x;
    int base = tid * B;

    for (int i = 0; i < B; i++) {
        s_ids[base + i] = -1;
        s_vals[base + i] = -1e30f;
    }

    int count = 0;
    for (int c = tid; c < num_chunks; c += num_threads) {
        int chunk_base = (c * num_parents + parent_id) * B;
        for (int i = 0; i < B; i++) {
            int id = d_block_winners_ids[chunk_base + i];
            float val = d_block_winners_vals[chunk_base + i];
            if (id < 0) continue;

            if (count < B) {
                s_ids[base + count] = id;
                s_vals[base + count] = val;
                count++;
            } else {
                int min_idx = 0;
                for (int j = 1; j < B; j++) {
                    if (s_vals[base + j] < s_vals[base + min_idx]) min_idx = j;
                }
                if (val > s_vals[base + min_idx]) {
                    s_ids[base + min_idx] = id;
                    s_vals[base + min_idx] = val;
                }
            }
        }
    }
    __syncthreads();

    for (int stride = num_threads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            int base_a = tid * B;
            int base_b = (tid + stride) * B;
            for (int i = 0; i < B; i++) {
                if (s_vals[base_b + i] <= -1e30f) continue;
                int min_idx = 0;
                for (int j = 1; j < B; j++) {
                    if (s_vals[base_a + j] < s_vals[base_a + min_idx]) min_idx = j;
                }
                if (s_vals[base_b + i] > s_vals[base_a + min_idx]) {
                    s_ids[base_a + min_idx] = s_ids[base_b + i];
                    s_vals[base_a + min_idx] = s_vals[base_b + i];
                }
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        for (int i = 0; i < B; i++) {
            d_winner_tokens[parent_id * B + i] = s_ids[i];
        }
    }
    __syncthreads();

    if (!skip_gather) {
        for (int i = 0; i < B; i++) {
            int tok = s_ids[i];
            if (tok < 0) continue;
            for (int d = tid; d < dim; d += num_threads) {
                d_parent_embs[(parent_id * B + i) * dim + d] =
                    __half2float(d_embeddings[(size_t)tok * dim + d]);
            }
        }
    }
}
