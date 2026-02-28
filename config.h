#pragma once
#include <string>

namespace InfEngineConfig {
    static const std::string TOKENIZER_PATH = "../models/llama_3.1_8b/tokenizer.json";
    static const int MAX_CONTEXT_LENGTH = 32768;
    static const int VOCAB_SIZE = 128256;

    static const int NUM_LAYERS = 32;
    static const int HIDDEN_SIZE = 4096;

    static const int NUM_Q_HEADS = 32;
    static const int NUM_KV_HEADS = 8;

    static const int HEAD_DIM = 128;
    static const int FFN_DIM = 14336;

    static const int HALF_SIZE = 2;
}