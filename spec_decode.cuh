#pragma once
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include "spec_decode.h"
#include "kernels/model_weights.cuh"

struct SpecDecodeBuffers {
    half* draft_logits;     // [MAX_DRAFT_NODES + 1, VOCAB_SIZE]
    int* draft_argmax;      // [MAX_DRAFT_NODES + 1]
    half* saved_logits;     // [VOCAB_SIZE]
    int* saved_argmax;      // [1]
    void* verify_workspace; // workspace for verify_attn
    int* d_accepted_indices; // [MAX_DRAFT_NODES] for KV compaction
    // Merged batch buffers (pending token + draft tokens)
    int* merged_token_ids;     // [MAX_DRAFT_NODES + 1]
    int* merged_positions;     // [MAX_DRAFT_NODES + 1]
    int8_t* merged_tree_mask;  // [(MAX_DRAFT_NODES + 1)^2]
};

void spec_decode_loop(cublasHandle_t handle, const ModelWeights& weights,
                      SpecDecodeConfig& cfg, SpecDecodeMetrics& metrics,
                      std::vector<int>& tokens,
                      int* tokenid_buff,
                      half* data_buffer, half* aux_buffer, half* attn_buffer,
                      half* output_distr, int* output_token,
                      void* decode_workspace, void* kv_cache,
                      DraftTree& tree, SpecDecodeBuffers& bufs,
                      void* drafter_ptr);
