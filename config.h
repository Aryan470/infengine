#pragma once
#include <string>
#include <vector>

namespace InfEngineConfig {
    static const std::string TOKENIZER_PATH = "models/llama_3.1_8b/tokenizer.json";
    static const int MAX_CONTEXT_LENGTH = 8192;
    static const int VOCAB_SIZE = 128256;

    static const int NUM_LAYERS = 32;
    static const int HIDDEN_SIZE = 4096;

    static const int NUM_Q_HEADS = 32;
    static const int NUM_KV_HEADS = 8;

    static const int HEAD_DIM = 128;
    static const int FFN_DIM = 14336;

    static const int HALF_SIZE = 2;

    static const int MAX_DRAFT_NODES = 256;

    static const std::vector<std::string> MODEL_FILES = {
        "models/llama_3.1_8b/model-00001-of-00004.safetensors",
        "models/llama_3.1_8b/model-00002-of-00004.safetensors",
        "models/llama_3.1_8b/model-00003-of-00004.safetensors",
        "models/llama_3.1_8b/model-00004-of-00004.safetensors"
    };
}