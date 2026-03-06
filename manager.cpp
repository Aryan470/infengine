#include <driver_types.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <tokenizers_cpp.h>
#include "manager.h"
#include "kernels/model_weights.cuh"
#include "kernels/attn.cuh"
#include "kernels/ffn.cuh"
#include "config.h"
#include "prefill.cuh"
#include "decode.cuh"
#include "spec_decode.cuh"
#include "drafters.h"
#include <fstream>
#include <iostream>
#include <optional>

Manager::Manager() {
    std::ifstream file(InfEngineConfig::TOKENIZER_PATH);
    std::string json_blob((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    std::cout << "Loading tokenizer..." << std::endl;
    tokenizer = tokenizers::Tokenizer::FromBlobJSON(json_blob);
    std::cout << "Precomputing RoPE..." << std::endl;
    std::cout << "Loading model weights..." << std::endl;
    model_weights = ModelWeights::from_safetensors(InfEngineConfig::MODEL_FILES);
}

Manager::~Manager() {
    model_weights.free();
}

std::optional<std::string> Manager::handle_request(const std::string& request, SpecDecodeConfig cfg) {
    RequestContext context;
    context.tokens = tokenize(request);

    std::cout << request << std::flush;

    // alloc buffers
    int* tokenid_buff;
    cudaMalloc((void**)&tokenid_buff, InfEngineConfig::MAX_CONTEXT_LENGTH * sizeof(int));

    half* data_buffer;
    half* aux_buffer;
    half* attn_buffer;
    half* output_distr;
    int* output_token;
    cudaMalloc((void**)&data_buffer, InfEngineConfig::MAX_CONTEXT_LENGTH * InfEngineConfig::HIDDEN_SIZE * sizeof(half));
    cudaMalloc((void**)&aux_buffer, InfEngineConfig::MAX_CONTEXT_LENGTH * InfEngineConfig::HIDDEN_SIZE * sizeof(half));
    cudaMalloc((void**)&attn_buffer, InfEngineConfig::MAX_CONTEXT_LENGTH * InfEngineConfig::HIDDEN_SIZE * sizeof(half));
    cudaMalloc((void**)&output_distr, InfEngineConfig::VOCAB_SIZE * sizeof(half));
    cudaMalloc((void**)&output_token, sizeof(int));

    cudaEvent_t startEvent;
    cudaEventCreate(&startEvent);
    cudaEvent_t endEvent;
    cudaEventCreate(&endEvent);

    cudaEventRecord(startEvent, 0);
    cublasHandle_t handle;
    cublasCreate(&handle);
    cudaProfilerStart();

    int max_seq_len = context.tokens.size() + cfg.num_tokens + InfEngineConfig::MAX_DRAFT_NODES + 100;
    if (max_seq_len > InfEngineConfig::MAX_CONTEXT_LENGTH) max_seq_len = InfEngineConfig::MAX_CONTEXT_LENGTH;

    void* workspace;
    void* kv_cache;
    cudaMalloc(&workspace, std::max(attn_workspace_size(max_seq_len), ffn_workspace_size(max_seq_len)));
    cudaMalloc(&kv_cache, kv_cache_size());

    /* PREFILL */
    int seq_len = context.tokens.size();
    cudaMemcpy(tokenid_buff, context.tokens.data(), seq_len * sizeof(int), cudaMemcpyHostToDevice);
    int final_token = prefill(handle, model_weights, seq_len, tokenid_buff,
        data_buffer, aux_buffer, attn_buffer, output_distr, output_token, workspace, kv_cache);
    context.tokens.push_back(final_token);
    cudaMemcpy(tokenid_buff + context.tokens.size() - 1, output_token, sizeof(int), cudaMemcpyDeviceToDevice);
    std::cout << detokenize({final_token}) << std::flush;

    if (cfg.mode == SpecDecodeConfig::NONE) {
        /* DECODE — baseline */
        for (int num_tokens = 1; num_tokens < cfg.num_tokens; num_tokens++) {
            seq_len = context.tokens.size();
            final_token = decode_step(handle, model_weights, seq_len, tokenid_buff,
                data_buffer, aux_buffer, attn_buffer, output_distr, output_token, workspace, kv_cache);
            context.tokens.push_back(final_token);
            cudaMemcpy(tokenid_buff + context.tokens.size() - 1, output_token, sizeof(int), cudaMemcpyDeviceToDevice);
            std::cout << detokenize({final_token}) << std::flush;
        }
    } else {
        /* SPECULATIVE DECODE */
        // Allocate spec decode buffers
        SpecDecodeBuffers bufs;
        cudaMalloc((void**)&bufs.draft_logits, (size_t)(InfEngineConfig::MAX_DRAFT_NODES + 1) * InfEngineConfig::VOCAB_SIZE * sizeof(half));
        cudaMalloc((void**)&bufs.draft_argmax, (InfEngineConfig::MAX_DRAFT_NODES + 1) * sizeof(int));
        cudaMalloc((void**)&bufs.saved_logits, InfEngineConfig::VOCAB_SIZE * sizeof(half));
        cudaMalloc((void**)&bufs.saved_argmax, sizeof(int));
        cudaMalloc((void**)&bufs.d_accepted_indices, InfEngineConfig::MAX_DRAFT_NODES * sizeof(int));

        // Merged batch buffers (pending token + draft tokens)
        cudaMalloc((void**)&bufs.merged_token_ids, (InfEngineConfig::MAX_DRAFT_NODES + 1) * sizeof(int));
        cudaMalloc((void**)&bufs.merged_positions, (InfEngineConfig::MAX_DRAFT_NODES + 1) * sizeof(int));
        cudaMalloc((void**)&bufs.merged_tree_mask, (InfEngineConfig::MAX_DRAFT_NODES + 1) * (InfEngineConfig::MAX_DRAFT_NODES + 1) * sizeof(int8_t));

        // Verify workspace: needs to handle N_merged (1 + N_draft) tokens attending to seq_len + N_merged
        size_t verify_ws_size = std::max(
            verify_attn_workspace_size(InfEngineConfig::MAX_DRAFT_NODES + 1, max_seq_len),
            ffn_workspace_size(InfEngineConfig::MAX_DRAFT_NODES + 1)
        );
        cudaMalloc(&bufs.verify_workspace, verify_ws_size);

        DraftTree tree;
        tree.alloc_gpu();

        SpecDecodeMetrics metrics;

        void* drafter_ptr = nullptr;
        NgramDrafter* ngram_drafter = nullptr;
        TardisDrafter* tardis_drafter = nullptr;

        if (cfg.mode == SpecDecodeConfig::NGRAM) {
            ngram_drafter = new NgramDrafter(cfg.max_ngram_size);
            ngram_drafter->update_from_prompt(context.tokens);
            drafter_ptr = ngram_drafter;
        } else if (cfg.mode == SpecDecodeConfig::TARDIS) {
            tardis_drafter = new TardisDrafter();
            tardis_drafter->init(cfg, handle);
            // Process all prompt tokens EXCEPT the last (pending token).
            // The loop scores from stale embedding then updates, matching tardis_sim.py.
            std::vector<int> prompt_for_tardis(context.tokens.begin(), context.tokens.end() - 1);
            tardis_drafter->process_prompt(prompt_for_tardis);
            drafter_ptr = tardis_drafter;
        }

        // Run spec decode loop
        int prev_size = context.tokens.size();
        spec_decode_loop(handle, model_weights, cfg, metrics,
                         context.tokens, tokenid_buff,
                         data_buffer, aux_buffer, attn_buffer,
                         output_distr, output_token,
                         workspace, kv_cache,
                         tree, bufs, drafter_ptr);

        // Print newly generated tokens
        for (int i = prev_size; i < (int)context.tokens.size(); i++) {
            std::cout << detokenize({context.tokens[i]}) << std::flush;
        }

        metrics.print();

        // Cleanup spec decode buffers
        tree.free_gpu();
        cudaFree(bufs.draft_logits);
        cudaFree(bufs.draft_argmax);
        cudaFree(bufs.saved_logits);
        cudaFree(bufs.saved_argmax);
        cudaFree(bufs.d_accepted_indices);
        cudaFree(bufs.merged_token_ids);
        cudaFree(bufs.merged_positions);
        cudaFree(bufs.merged_tree_mask);
        cudaFree(bufs.verify_workspace);

        if (ngram_drafter) delete ngram_drafter;
        if (tardis_drafter) { tardis_drafter->print_profile(); tardis_drafter->cleanup(); delete tardis_drafter; }
    }

    std::cout << std::endl;

    cudaProfilerStop();
    cudaEventRecord(endEvent, 0);
    cudaEventSynchronize(endEvent);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, startEvent, endEvent);
    printf("Total time: %.2f ms, Time per token: %.2f ms/token\n", ms, ms / (float)(context.tokens.size() - seq_len));

    cudaEventDestroy(endEvent);
    cudaEventDestroy(startEvent);

    cudaFree(workspace);
    cudaFree(kv_cache);
    cudaFree(data_buffer);
    cudaFree(aux_buffer);
    cudaFree(attn_buffer);
    cudaFree(output_distr);
    cudaFree(output_token);
    cudaFree(tokenid_buff);
    cublasDestroy(handle);
    return detokenize(context.tokens);
}

std::vector<int> Manager::tokenize(const std::string& text) { return tokenizer->Encode(text); }
std::string Manager::detokenize(const std::vector<int>& tokens) { return tokenizer->Decode(tokens); }
