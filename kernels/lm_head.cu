#include "../config.h"
#include "lm_head.cuh"
#include "multiply_by_weight.cuh"

void lm_head(cublasHandle_t handle, const int seq_len, half* d_input, half* d_weight, half* d_output) {
    // offset the input by seq_len for the user
    d_input += (seq_len-1) * InfEngineConfig::HIDDEN_SIZE;

    // simple matmul, we can use multiply_by_weight
    // input is [1, hidden_dim], weight is [vocab_size, hidden_dim]
    // m = input rows = 1, k = hidden_dim, n = vocab_size
    // want to find [1, vocab_size]
    multiply_by_weight(handle,
        1,
        InfEngineConfig::HIDDEN_SIZE,
        InfEngineConfig::VOCAB_SIZE,
        d_input,
        d_weight,
        d_output);

}