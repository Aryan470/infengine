#pragma once
#include <tokenizers_cpp.h>
#include <optional>

class Manager {
    public:
        Manager();
        // take a prompt, maybe return the response
        std::optional<std::string> handle_request(const std::string& request);
    private:
        std::vector<int> tokenize(const std::string& text);
        std::string detokenize(const std::vector<int>& tokens);
        std::unique_ptr<tokenizers::Tokenizer> tokenizer;
        int* cu_model_weights;
        int num_output_tokens;
};

struct RequestContext {
    std::vector<int> tokens;
    int* cu_kv_cache_buffer;
    int* cu_token_buffer;
};