#include "spec_decode.cuh"
#include "verify.cuh"
#include "decode.cuh"
#include "drafters.h"
#include "kernels/sampling.cuh"
#include "kernels/emblookup.cuh"
#include "config.h"
#include <cstdio>
#include <cstring>
#include <vector>
#include <iostream>

// KV cache compaction kernel
// Moves accepted KV entries to contiguous positions
// Grid: k * NUM_LAYERS * 2 (K and V) * NUM_KV_HEADS blocks, HEAD_DIM threads
__global__ void compact_kv_cache_kern(half* kv_cache, const int* accepted_indices,
                                       int seq_len, int num_accepted) {
    // block decomposition: block_id = accepted_slot * (NUM_LAYERS * 2 * NUM_KV_HEADS) + layer * (2 * NUM_KV_HEADS) + kv * NUM_KV_HEADS + head
    const int block_id = blockIdx.x;
    const int thread_id = threadIdx.x;

    const int total_per_accepted = InfEngineConfig::NUM_LAYERS * 2 * InfEngineConfig::NUM_KV_HEADS;
    const int accepted_slot = block_id / total_per_accepted;
    const int remainder = block_id % total_per_accepted;
    const int layer = remainder / (2 * InfEngineConfig::NUM_KV_HEADS);
    const int kv_and_head = remainder % (2 * InfEngineConfig::NUM_KV_HEADS);
    const int kv = kv_and_head / InfEngineConfig::NUM_KV_HEADS;
    const int head = kv_and_head % InfEngineConfig::NUM_KV_HEADS;

    if (accepted_slot >= num_accepted) return;

    int src_tree_idx = accepted_indices[accepted_slot];
    int src_pos = seq_len + src_tree_idx;
    int dst_pos = seq_len + accepted_slot;

    if (src_pos == dst_pos) return;  // already in place

    // kv_cache layout: [NUM_LAYERS, 2, NUM_KV_HEADS, MAX_CONTEXT_LENGTH, HEAD_DIM]
    size_t base = (size_t)layer * 2 * InfEngineConfig::NUM_KV_HEADS * InfEngineConfig::MAX_CONTEXT_LENGTH * InfEngineConfig::HEAD_DIM
                + (size_t)kv * InfEngineConfig::NUM_KV_HEADS * InfEngineConfig::MAX_CONTEXT_LENGTH * InfEngineConfig::HEAD_DIM
                + (size_t)head * InfEngineConfig::MAX_CONTEXT_LENGTH * InfEngineConfig::HEAD_DIM;

    half* cache = kv_cache + base;

    // Copy HEAD_DIM elements from src_pos to dst_pos
    // Each thread handles one or more elements
    for (int i = thread_id; i < InfEngineConfig::HEAD_DIM; i += blockDim.x) {
        cache[dst_pos * InfEngineConfig::HEAD_DIM + i] = cache[src_pos * InfEngineConfig::HEAD_DIM + i];
    }
}

void compact_kv_cache(half* kv_cache, int* d_accepted_indices, int seq_len, int num_accepted) {
    if (num_accepted == 0) return;
    int total_blocks = num_accepted * InfEngineConfig::NUM_LAYERS * 2 * InfEngineConfig::NUM_KV_HEADS;
    compact_kv_cache_kern<<<total_blocks, InfEngineConfig::HEAD_DIM>>>((half*)kv_cache, d_accepted_indices, seq_len, num_accepted);
}

// Greedy acceptance: walk each root-to-leaf path, find longest prefix match
struct AcceptResult {
    std::vector<int> accepted_indices;  // tree indices of accepted nodes
    int bonus_token;
    int num_accepted;
};

AcceptResult greedy_accept(const DraftTree& tree, int saved_argmax_token,
                           const int* verify_argmax, int N_draft) {
    AcceptResult best;
    best.num_accepted = 0;
    best.bonus_token = saved_argmax_token;

    if (N_draft == 0) return best;

    // Find all leaf nodes (nodes with no children)
    std::vector<bool> has_child(N_draft, false);
    for (int i = 0; i < N_draft; i++) {
        if (tree.nodes[i].parent_idx >= 0) {
            has_child[tree.nodes[i].parent_idx] = true;
        }
    }

    // For each leaf, trace path to root and check acceptance
    for (int leaf = 0; leaf < N_draft; leaf++) {
        if (has_child[leaf]) continue;  // not a leaf

        // Build root-to-leaf path
        std::vector<int> path;
        int cur = leaf;
        while (cur >= 0) {
            path.push_back(cur);
            cur = tree.nodes[cur].parent_idx;
        }
        std::reverse(path.begin(), path.end());

        // Check acceptance along path
        int count = 0;
        // First node: check against saved_logits argmax
        if (tree.nodes[path[0]].token_id == saved_argmax_token) {
            count = 1;
            for (int i = 1; i < (int)path.size(); i++) {
                // Check if verify_logits[path[i-1]] argmax matches tree token at path[i]
                if (verify_argmax[path[i - 1]] == tree.nodes[path[i]].token_id) {
                    count++;
                } else {
                    break;
                }
            }
        }

        if (count > best.num_accepted) {
            best.num_accepted = count;
            best.accepted_indices.assign(path.begin(), path.begin() + count);
            if (count == (int)path.size()) {
                // All matched — bonus is argmax of verify_logits at last accepted
                best.bonus_token = verify_argmax[path[count - 1]];
            } else if (count > 0) {
                best.bonus_token = verify_argmax[path[count - 1]];
            } else {
                best.bonus_token = saved_argmax_token;
            }
        }
    }

    return best;
}

// Build merged batch: [pending_token, draft_0, ..., draft_N-1]
// Tree mask layout (N_merged x N_merged):
//   pending attends to self only (history via KV cache)
//   each draft token attends to pending + its draft-tree ancestors + self
void build_merged_batch(int pending_token, int pending_pos,
                        const DraftTree& tree, int N_draft,
                        int* h_token_ids, int* h_positions, int8_t* h_tree_mask) {
    int N = 1 + N_draft;
    h_token_ids[0] = pending_token;
    h_positions[0] = pending_pos;
    for (int i = 0; i < N_draft; i++) {
        h_token_ids[i + 1] = tree.h_token_ids[i];
        h_positions[i + 1] = tree.h_positions[i];
    }
    memset(h_tree_mask, 0, N * N * sizeof(int8_t));
    h_tree_mask[0] = 1;  // pending attends to self
    for (int i = 0; i < N_draft; i++) {
        h_tree_mask[(i + 1) * N + 0] = 1;  // all drafts attend to pending
        for (int j = 0; j < N_draft; j++)
            h_tree_mask[(i + 1) * N + (j + 1)] = tree.h_tree_mask[i * N_draft + j];
    }
}

void spec_decode_loop(cublasHandle_t handle, const ModelWeights& weights,
                      SpecDecodeConfig& cfg, SpecDecodeMetrics& metrics,
                      std::vector<int>& tokens,
                      int* tokenid_buff,
                      half* data_buffer, half* aux_buffer, half* attn_buffer,
                      half* output_distr, int* output_token,
                      void* decode_workspace, void* kv_cache,
                      DraftTree& tree, SpecDecodeBuffers& bufs,
                      void* drafter_ptr) {

    NgramDrafter* ngram_drafter = nullptr;
    TardisDrafter* tardis_drafter = nullptr;
    if (cfg.mode == SpecDecodeConfig::NGRAM) {
        ngram_drafter = (NgramDrafter*)drafter_ptr;
    } else if (cfg.mode == SpecDecodeConfig::TARDIS) {
        tardis_drafter = (TardisDrafter*)drafter_ptr;
    }

    cudaEvent_t total_start, total_end;
    cudaEvent_t step_start, step_end;
    cudaEventCreate(&total_start);
    cudaEventCreate(&total_end);
    cudaEventCreate(&step_start);
    cudaEventCreate(&step_end);

    cudaEventRecord(total_start);

    // Host buffers for merged batch and verify results
    int h_merged_token_ids[InfEngineConfig::MAX_DRAFT_NODES + 1];
    int h_merged_positions[InfEngineConfig::MAX_DRAFT_NODES + 1];
    int8_t h_merged_tree_mask[(InfEngineConfig::MAX_DRAFT_NODES + 1) * (InfEngineConfig::MAX_DRAFT_NODES + 1)];
    int h_verify_argmax[InfEngineConfig::MAX_DRAFT_NODES];
    int h_saved_argmax;

    int tokens_generated = 0;

    // Pending token: last token in the sequence, KV not yet computed.
    // After prefill, tokens[0..seq_len-2] have KV in cache.
    // tokens[seq_len-1] was produced by prefill but its KV was not written
    // (prefill writes KV for the prompt tokens, then produces the next token).
    int pending_token = tokens.back();
    bool first_iter = true;

    while (tokens_generated < cfg.num_tokens) {
        int seq_len = (int)tokens.size();
        metrics.total_iterations++;

        // Step 1: Draft
        // TARDIS: score from stale global embedding THEN update (matches tardis_sim.py)
        // Ngram: update n-gram table then build tree
        cudaEventRecord(step_start);
        int N_draft = 0;
        if (cfg.mode == SpecDecodeConfig::NGRAM && ngram_drafter) {
            if (!first_iter) {
                ngram_drafter->update(tokens.data(), seq_len - 1, pending_token);
            }
            ngram_drafter->build_tree(tree, tokens.data(), seq_len,
                                      cfg.branch_factor, cfg.max_depth, seq_len);
        } else if (cfg.mode == SpecDecodeConfig::TARDIS && tardis_drafter) {
            // Score from stale embedding first, then update
            tardis_drafter->build_tree(tree, pending_token,
                                        cfg.branch_factor, cfg.max_depth, seq_len);
            tardis_drafter->process_token(pending_token);
        }
        N_draft = tree.num_nodes;
        if (first_iter) first_iter = false;
        cudaEventRecord(step_end);
        cudaEventSynchronize(step_end);
        float draft_ms;
        cudaEventElapsedTime(&draft_ms, step_start, step_end);
        metrics.draft_time_ms += draft_ms;
        metrics.total_draft_tokens += N_draft;

        // Step 2: Build merged batch [pending, draft_0, ..., draft_N-1]
        int N_merged = 1 + N_draft;

        // pending_pos = seq_len - 1 (its KV slot)
        int pending_pos = seq_len - 1;

        if (N_draft > 0) {
            build_merged_batch(pending_token, pending_pos, tree, N_draft,
                               h_merged_token_ids, h_merged_positions, h_merged_tree_mask);
        } else {
            // No draft tokens — just the pending token
            h_merged_token_ids[0] = pending_token;
            h_merged_positions[0] = pending_pos;
            h_merged_tree_mask[0] = 1;
        }

        // Upload merged batch to GPU
        cudaMemcpy(bufs.merged_token_ids, h_merged_token_ids, N_merged * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(bufs.merged_positions, h_merged_positions, N_merged * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(bufs.merged_tree_mask, h_merged_tree_mask, N_merged * N_merged * sizeof(int8_t), cudaMemcpyHostToDevice);

        // Step 3: Single verify pass on merged batch
        // seq_len - 1 = history length (KV filled for positions 0..seq_len-2)
        cudaEventRecord(step_start);
        verify_step(handle, weights, seq_len - 1, N_merged,
                    bufs.merged_token_ids, bufs.merged_positions, bufs.merged_tree_mask,
                    data_buffer, aux_buffer, attn_buffer,
                    bufs.draft_logits,
                    bufs.verify_workspace, kv_cache);

        // Extract saved_argmax from logits[0] (pending token's prediction)
        sample_token_amax(bufs.draft_logits, bufs.saved_argmax);
        cudaMemcpy(&h_saved_argmax, bufs.saved_argmax, sizeof(int), cudaMemcpyDeviceToHost);

        // Extract verify_argmax from logits[1..N_merged-1] (draft tokens' predictions)
        if (N_draft > 0) {
            sample_tokens_amax(bufs.draft_logits + (size_t)InfEngineConfig::VOCAB_SIZE, bufs.draft_argmax, N_draft);
            cudaMemcpy(h_verify_argmax, bufs.draft_argmax, N_draft * sizeof(int), cudaMemcpyDeviceToHost);
        }
        cudaEventRecord(step_end);
        cudaEventSynchronize(step_end);
        float verify_ms;
        cudaEventElapsedTime(&verify_ms, step_start, step_end);
        metrics.verify_time_ms += verify_ms;

        // Step 4: Accept
        AcceptResult result = greedy_accept(tree, h_saved_argmax, h_verify_argmax, N_draft);

        // Step 5: Compact KV cache — move accepted draft entries to contiguous positions
        // pending's KV is at seq_len-1 (untouched), draft KV starts at seq_len
        if (result.num_accepted > 0) {
            cudaMemcpy(bufs.d_accepted_indices, result.accepted_indices.data(),
                       result.num_accepted * sizeof(int), cudaMemcpyHostToDevice);
            compact_kv_cache((half*)kv_cache, bufs.d_accepted_indices, seq_len, result.num_accepted);
        }

        // Step 6: Output accepted tokens + bonus
        // The pending token was already in tokens[] (it was the last element).
        // Its model prediction (saved_argmax) either matches draft[0] (accepted) or becomes the bonus.

        if (N_draft == 0) {
            // No drafts: saved_argmax is our only output token
            tokens.push_back(h_saved_argmax);
            int h_tok = h_saved_argmax;
            cudaMemcpy(tokenid_buff + seq_len, &h_tok, sizeof(int), cudaMemcpyHostToDevice);
            tokens_generated++;
            pending_token = h_saved_argmax;
        } else {
            // Push accepted draft tokens and feed them to TARDIS
            for (int i = 0; i < result.num_accepted; i++) {
                int tok = tree.nodes[result.accepted_indices[i]].token_id;
                tokens.push_back(tok);
                int h_tok = tok;
                cudaMemcpy(tokenid_buff + seq_len + i, &h_tok, sizeof(int), cudaMemcpyHostToDevice);
                tokens_generated++;
                // Update TARDIS embeddings and context for accepted tokens
                if (tardis_drafter) {
                    tardis_drafter->process_token(tok);
                }
            }
            metrics.total_accepted += result.num_accepted;

            // Push bonus token
            if (tokens_generated < cfg.num_tokens) {
                tokens.push_back(result.bonus_token);
                int h_bonus = result.bonus_token;
                cudaMemcpy(tokenid_buff + (int)tokens.size() - 1, &h_bonus, sizeof(int), cudaMemcpyHostToDevice);
                tokens_generated++;
            }
            pending_token = result.bonus_token;
        }
    }

    cudaEventRecord(total_end);
    cudaEventSynchronize(total_end);
    cudaEventElapsedTime(&metrics.total_time_ms, total_start, total_end);
    metrics.total_tokens = tokens_generated;

    cudaEventDestroy(total_start);
    cudaEventDestroy(total_end);
    cudaEventDestroy(step_start);
    cudaEventDestroy(step_end);
}
