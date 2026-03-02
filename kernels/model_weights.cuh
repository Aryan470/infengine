#pragma once
#include <vector>
#include <string>
#include <cuda_fp16.h>
#include "../config.h"

struct TransformerBlockWeights { half* w_k; half* w_q; half* w_v; half* w_o; };
struct FFNBlockWeights { half* w_up; half* w_gate; half* w_down; };
struct RoPEWeights { float* cos; float* sin; };

struct Layer {
    half* input_layernorm;
    // transformer: wk, wq, wv, wo
    TransformerBlockWeights transformer;

    half* post_attention_layernorm;
    // ffn: wup, wgate, wdown
    FFNBlockWeights ffn_block;
};

struct ModelWeights {
    // initial embedding lookup
    half* emb_lookup;
    // per layer weights
    Layer layers[InfEngineConfig::NUM_LAYERS];
    // final lmhead
    half* final_norm;
    half* lm_head;

    RoPEWeights rope;

    static ModelWeights from_safetensors(std::vector<std::string> safetensors_paths);
    void free();
};