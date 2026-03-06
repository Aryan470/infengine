#pragma once
#include <tokenizers_cpp.h>
#include <optional>
#include "kernels/model_weights.cuh"
#include "spec_decode.h"

class Manager {
    public:
        Manager();
        ~Manager();
        std::optional<std::string> handle_request(const std::string& request, SpecDecodeConfig cfg = SpecDecodeConfig());
    private:
        std::vector<int> tokenize(const std::string& text);
        std::string detokenize(const std::vector<int>& tokens);
        std::unique_ptr<tokenizers::Tokenizer> tokenizer;
        ModelWeights model_weights;
};

struct RequestContext {
    std::vector<int> tokens;
    int* cu_kv_cache_buffer;
    int* cu_token_buffer;
    int num_output_tokens;
};
