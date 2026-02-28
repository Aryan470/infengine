#include <tokenizers_cpp.h>
#include "manager.h"
#include "kernel.cuh"
#include "kernels/rope.cuh"
#include "config.h"
#include <fstream>
#include <iostream>
#include <optional>

Manager::Manager() {
    std::ifstream file(InfEngineConfig::TOKENIZER_PATH);
    std::string json_blob((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    std::cout << "Loading tokenizer..." << std::endl;
    tokenizer = tokenizers::Tokenizer::FromBlobJSON(json_blob);
    std::cout << "Precomputing RoPE..." << std::endl;
    init_rope_buffer(&cu_rope_cos, &cu_rope_sin);
}

Manager::~Manager() {
    cleanup_rope_buffer(cu_rope_cos, cu_rope_sin);
}


std::optional<std::string> Manager::handle_request(const std::string& request) {
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
}

std::vector<int> Manager::tokenize(const std::string& text) { return tokenizer->Encode(text); }
std::string Manager::detokenize(const std::vector<int>& tokens) { return tokenizer->Decode(tokens); }