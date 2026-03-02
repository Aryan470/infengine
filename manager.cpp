#include <driver_types.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <tokenizers_cpp.h>
#include "manager.h"
#include "kernels/emblookup.cuh"
#include "kernels/lm_head.cuh"
#include "kernels/model_weights.cuh"
#include "kernels/residual_add.cuh"
#include "kernels/rmsnorm.cuh"
#include "kernels/attn.cuh"
#include "kernels/ffn.cuh"
#include "config.h"
#include "kernels/sampling.cuh"
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

std::optional<std::string> Manager::handle_request(const std::string& request) {
    RequestContext context;
    context.tokens = tokenize(request);

    // alloc 2 buffers for input
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

    int max_seq_len = context.tokens.size() + 100;
    size_t workspace_bytes = std::max(attn_workspace_size(max_seq_len), ffn_workspace_size(max_seq_len));
    void* workspace;
    cudaMalloc(&workspace, workspace_bytes);

    std::cout << request;
    for (int num_tokens = 0; num_tokens < 100; num_tokens++) {
        int seq_len = context.tokens.size();
        cudaMemcpy(tokenid_buff, context.tokens.data(), seq_len * sizeof(int), cudaMemcpyHostToDevice);
        emblookup(context.tokens.size(), tokenid_buff, model_weights.emb_lookup, data_buffer);


        for (int i = 0; i < InfEngineConfig::NUM_LAYERS; i++) {
            // apply rmsnorm from data buffer to aux buffer
            rmsnorm(seq_len, data_buffer, model_weights.layers[i].input_layernorm, aux_buffer);
            // apply attn
            attn(handle, model_weights.rope.cos, model_weights.rope.sin, seq_len, aux_buffer,
                model_weights.layers[i].transformer.w_q,
                model_weights.layers[i].transformer.w_k,
                model_weights.layers[i].transformer.w_v,
                model_weights.layers[i].transformer.w_o,
                attn_buffer, workspace);

            // add into x, copy x back to aux
            residual_add(seq_len, InfEngineConfig::HIDDEN_SIZE, data_buffer, attn_buffer);

            // apply rmsnorm to aux buffer inplace
            rmsnorm(seq_len, data_buffer, model_weights.layers[i].post_attention_layernorm, aux_buffer);
            // apply ffn
            ffn(handle, seq_len, aux_buffer, attn_buffer,
                model_weights.layers[i].ffn_block.w_up,
                model_weights.layers[i].ffn_block.w_down,
                model_weights.layers[i].ffn_block.w_gate, workspace);
            // add back to x
            residual_add(seq_len, InfEngineConfig::HIDDEN_SIZE, data_buffer, attn_buffer);
        }

        // apply final norm
        rmsnorm(seq_len, data_buffer, model_weights.final_norm, data_buffer);
        lm_head(handle, seq_len, data_buffer, model_weights.lm_head, output_distr);
        sample_token_amax(output_distr, output_token);

        int final_token;
        cudaMemcpy(&final_token, output_token, sizeof(int), cudaMemcpyDeviceToHost);
        context.tokens.push_back(final_token);
        std::string token_str = tokenizer->Decode({final_token});
        std::cout << token_str;
        fflush(stdout);
    }
    std::cout << std::endl;
    cudaProfilerStop();
    cudaEventRecord(endEvent, 0);
    cudaEventSynchronize(endEvent);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, startEvent, endEvent);
    printf("Total time: %.2f ms, Time per token: %.2f ms/token\n", ms, ms / 100.0f);

    cudaEventDestroy(endEvent);
    cudaEventDestroy(startEvent);
    
    cudaFree(workspace);
    cudaFree(data_buffer);
    cudaFree(aux_buffer);
    cudaFree(attn_buffer);
    cudaFree(output_distr);
    cudaFree(output_token);
    cudaFree(tokenid_buff);
    cublasDestroy(handle);
    return detokenize(context.tokens);
}

/*std::optional<std::string> Manager::handle_request_future(const std::string& request) {
    // tokenize the request
    RequestContext context;
    context.tokens = tokenize(request);

    // alloc kv cache and token buffers on gpu
    int init_result = initialize_request_context(&context);
    if (init_result != 0) {return {};}

    // prefill will build kv cache
    int prefill_result = initialize_request_context(&context);
    if (prefill_result != 0) {cleanup_request_context(&context); return {};}

    // call decode kernel, which will just work on kvcache + 
    int decode_result = decode(&context);
    if (decode_result != 0) {cleanup_request_context(&context); return {};}

    int cleanup_result = cleanup_request_context(&context);
    if (cleanup_result != 0) {return {};}
    return detokenize(context.tokens);
}*/

std::vector<int> Manager::tokenize(const std::string& text) { return tokenizer->Encode(text); }
std::string Manager::detokenize(const std::vector<int>& tokens) { return tokenizer->Decode(tokens); }