#include "drafters.h"
#include "tardis.cuh"
#include <random>

void TardisDrafter::init(const SpecDecodeConfig& cfg, cublasHandle_t handle) {
    dim = cfg.tardis_dim;
    alpha = cfg.alpha;
    beta = cfg.beta;
    cos_omega = cosf(cfg.omega);
    sin_omega = sinf(cfg.omega);
    sqrt_1ma = sqrtf(1.0f - alpha);
    sqrt_a = sqrtf(alpha);
    sqrt_1mb = sqrtf(1.0f - beta);
    sqrt_b = sqrtf(beta);
    cublas_handle = handle;
    rng_seed = 42;

    // Combined scoring setup
    use_combined = !std::isnan(cfg.phi);
    use_sim_filter = !std::isnan(cfg.sim_threshold);
    if (use_combined) {
        cos_phi = cosf(cfg.phi);
        sin_phi = sinf(cfg.phi);
    }
    if (use_sim_filter) {
        sim_threshold_val = cfg.sim_threshold;
    }

    int num_topk_chunks = (InfEngineConfig::VOCAB_SIZE + TOPK_CHUNK_SIZE - 1) / TOPK_CHUNK_SIZE;

    // FP16 global embedding table
    cudaMalloc(&d_embeddings, (size_t)InfEngineConfig::VOCAB_SIZE * dim * sizeof(half));
    cudaMalloc(&d_context, dim * sizeof(float));
    cudaMalloc(&d_embedding, dim * sizeof(float));

    // Parent embeddings: single buffer (no double-buffering needed)
    cudaMalloc(&d_parent_embs, (size_t)InfEngineConfig::MAX_DRAFT_NODES * dim * sizeof(float));

    // Packed FP16 GEMM input: dim × 2*MAX_BRANCH columns
    cudaMalloc(&d_packed_h, (size_t)dim * 2 * MAX_BRANCH * sizeof(half));

    // GEMM output: VOCAB_SIZE × 2*MAX_BRANCH columns
    cudaMalloc(&d_gemm_out, (size_t)InfEngineConfig::VOCAB_SIZE * 2 * MAX_BRANCH * sizeof(float));

    // Block winners for chunked topk: num_chunks × MAX_DRAFT_NODES entries
    size_t max_block_entries = (size_t)num_topk_chunks * InfEngineConfig::MAX_DRAFT_NODES;
    cudaMalloc(&d_block_winners_ids,  max_block_entries * sizeof(int));
    cudaMalloc(&d_block_winners_vals, max_block_entries * sizeof(float));

    // Winner tokens across all depths (at most MAX_DRAFT_NODES total)
    cudaMalloc(&d_winner_tokens, InfEngineConfig::MAX_DRAFT_NODES * sizeof(int));

    // Parent tokens buffer for root setup
    cudaMalloc(&d_parent_tokens, sizeof(int));

    // Initialize embeddings
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f / sqrtf((float)dim));
    size_t emb_count = (size_t)InfEngineConfig::VOCAB_SIZE * dim;
    std::vector<half> emb_init_h(emb_count);
    for (size_t i = 0; i < emb_count; i++) {
        emb_init_h[i] = __float2half(dist(rng));
    }
    cudaMemcpy(d_embeddings, emb_init_h.data(), emb_count * sizeof(half), cudaMemcpyHostToDevice);

    cudaMemset(d_context, 0, dim * sizeof(float));
    cudaMemset(d_embedding, 0, dim * sizeof(float));
}

void TardisDrafter::process_token(int token_id) {
    half* d_token_emb = d_embeddings + (size_t)token_id * dim;

    int threads = min(dim, 256);
    int blocks = (dim + threads - 1) / threads;
    tardis_process_token_kern<<<blocks, threads>>>(d_embedding, d_context, d_token_emb, dim,
                                                    sqrt_1ma, sqrt_a, sqrt_1mb, sqrt_b,
                                                    cos_omega, sin_omega, rng_seed, step_counter);
    step_counter++;
}

void TardisDrafter::process_prompt(const std::vector<int>& prompt) {
    for (int tok : prompt) {
        process_token(tok);
    }
}

// Build draft tree using packed cuBLAS GEMM + multi-block chunked topk.
// Per-depth pipeline: 4 operations
//   1. tardis_pack_gemm_input_kern    — dir vecs + FP16 conversion
//   2. cublasGemmEx (single call)     — tensor core GEMM, reads embedding table once
//   3. tardis_chunked_topk_kern       — multi-block combine + topk (GPU-saturating)
//   4. tardis_reduce_topk_gather_kern — reduce chunk winners + gather embeddings
void TardisDrafter::build_tree(DraftTree& tree, int last_token, int B, int max_depth, int seq_len) {
    tree.clear();

    // Profiling events
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);

    #define PROF_START() cudaEventRecord(t0)
    #define PROF_END(acc) do { \
        cudaEventRecord(t1); cudaEventSynchronize(t1); \
        float _ms; cudaEventElapsedTime(&_ms, t0, t1); acc += _ms; \
    } while(0)

    // Root: read global FP16 embedding for last_token, convert to FP32
    PROF_START();
    {
        int threads = 256;
        int blocks = (dim + threads - 1) / threads;
        half_to_float_kern<<<blocks, threads>>>(d_embeddings + (size_t)last_token * dim, d_parent_embs, dim);
    }
    PROF_END(prof_root_ms);

    int num_parents = 1;
    int num_topk_chunks = (InfEngineConfig::VOCAB_SIZE + TOPK_CHUNK_SIZE - 1) / TOPK_CHUNK_SIZE;
    float dir_scale = 1.0f / (2.0f * sin_omega);

    // Track winner offsets per depth for tree building
    struct DepthInfo {
        int num_parents;
        int winner_offset;
    };
    std::vector<DepthInfo> depth_infos;
    int winner_offset = 0;

    for (int depth = 1; depth <= max_depth; depth++) {
        if (num_parents == 0) break;

        // Cap parents so total winners don't exceed MAX_DRAFT_NODES
        // and parents fit in GEMM buffers (2*MAX_BRANCH columns)
        int remaining = InfEngineConfig::MAX_DRAFT_NODES - winner_offset;
        int effective_parents = std::min(num_parents, remaining / B);
        effective_parents = std::min(effective_parents, MAX_BRANCH);
        if (effective_parents <= 0) break;

        bool is_last = (depth == max_depth) ||
                       (effective_parents * B + winner_offset >= InfEngineConfig::MAX_DRAFT_NODES);

        int gemm_cols = use_combined ? 2 * effective_parents : effective_parents;

        // 1. Pack dir_vecs + parent_embs → FP16 column-major
        PROF_START();
        {
            int total = effective_parents * dim;
            int threads = 256;
            int blocks = (total + threads - 1) / threads;
            tardis_pack_gemm_input_kern<<<blocks, threads>>>(
                d_parent_embs, d_packed_h, effective_parents, dim, sin_omega, use_combined);
        }
        PROF_END(prof_pack_ms);

        // 2. Single cuBLAS GEMM: embeddings^T × packed_h → gemm_out
        PROF_START();
        {
            float alpha_gemm = 1.0f, beta_gemm = 0.0f;
            cublasGemmEx(cublas_handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                InfEngineConfig::VOCAB_SIZE, gemm_cols, dim,
                &alpha_gemm,
                d_embeddings, CUDA_R_16F, dim,
                d_packed_h, CUDA_R_16F, dim,
                &beta_gemm,
                d_gemm_out, CUDA_R_32F, InfEngineConfig::VOCAB_SIZE,
                CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        }
        PROF_END(prof_gemm_ms);

        // 3. Chunked topk: many blocks per parent for GPU saturation
        PROF_START();
        {
            size_t shmem = 256 * B * (sizeof(int) + sizeof(float));
            dim3 grid(effective_parents, num_topk_chunks);
            tardis_chunked_topk_kern<<<grid, 256, shmem>>>(
                d_gemm_out, InfEngineConfig::VOCAB_SIZE, effective_parents, B,
                effective_parents,  // sim_col_offset
                d_block_winners_ids, d_block_winners_vals,
                TOPK_CHUNK_SIZE,
                cos_phi, sin_phi, dir_scale,
                sim_threshold_val, use_combined, use_sim_filter);
        }
        PROF_END(prof_topk_ms);

        // 4. Reduce chunk winners + gather embeddings for next depth
        PROF_START();
        {
            size_t shmem = 256 * B * (sizeof(int) + sizeof(float));
            tardis_reduce_topk_gather_kern<<<effective_parents, 256, shmem>>>(
                d_block_winners_ids, d_block_winners_vals,
                num_topk_chunks, effective_parents, B,
                d_winner_tokens + winner_offset,
                d_embeddings, d_parent_embs, dim, is_last);
        }
        PROF_END(prof_reduce_ms);

        depth_infos.push_back({effective_parents, winner_offset});
        winner_offset += effective_parents * B;
        num_parents = effective_parents * B;

        if (is_last) break;
    }

    // Single D→H transfer of all winner tokens
    PROF_START();
    std::vector<int> h_all_winners(winner_offset);
    if (winner_offset > 0) {
        cudaMemcpy(h_all_winners.data(), d_winner_tokens,
                   winner_offset * sizeof(int), cudaMemcpyDeviceToHost);
    }
    PROF_END(prof_memcpy_ms);

    // Build tree nodes from winner data
    std::vector<int> parent_node_indices = {-1};  // root's "parent" in tree

    for (int d = 0; d < (int)depth_infos.size(); d++) {
        int depth = d + 1;
        int np = depth_infos[d].num_parents;
        int off = depth_infos[d].winner_offset;

        std::vector<int> next_parent_indices;

        for (int p = 0; p < np; p++) {
            for (int b = 0; b < B; b++) {
                int tok = h_all_winners[off + p * B + b];
                if (tok < 0) continue;
                if (tree.num_nodes >= InfEngineConfig::MAX_DRAFT_NODES) break;

                int idx = tree.num_nodes++;
                tree.nodes.push_back({tok, parent_node_indices[p], depth});
                tree.h_token_ids[idx] = tok;
                tree.h_positions[idx] = seq_len + depth - 1;

                next_parent_indices.push_back(idx);
            }
            if (tree.num_nodes >= InfEngineConfig::MAX_DRAFT_NODES) break;
        }

        parent_node_indices = next_parent_indices;
    }

    // Build tree mask
    memset(tree.h_tree_mask, 0, tree.num_nodes * tree.num_nodes * sizeof(int8_t));
    for (int i = 0; i < tree.num_nodes; i++) {
        tree.h_tree_mask[i * tree.num_nodes + i] = 1;
        int cur = tree.nodes[i].parent_idx;
        while (cur >= 0) {
            tree.h_tree_mask[i * tree.num_nodes + cur] = 1;
            cur = tree.nodes[cur].parent_idx;
        }
    }

    prof_calls++;
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    #undef PROF_START
    #undef PROF_END
}

void TardisDrafter::cleanup() {
    if (d_embeddings) cudaFree(d_embeddings);
    if (d_context) cudaFree(d_context);
    if (d_embedding) cudaFree(d_embedding);
    if (d_parent_embs) cudaFree(d_parent_embs);
    if (d_packed_h) cudaFree(d_packed_h);
    if (d_gemm_out) cudaFree(d_gemm_out);
    if (d_block_winners_ids) cudaFree(d_block_winners_ids);
    if (d_block_winners_vals) cudaFree(d_block_winners_vals);
    if (d_winner_tokens) cudaFree(d_winner_tokens);
    if (d_parent_tokens) cudaFree(d_parent_tokens);
    d_embeddings = nullptr;
}
