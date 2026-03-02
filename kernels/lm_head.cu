#include "../config.h"
#include "lm_head.cuh"
#include "multiply_by_weight.cuh"

void lm_head(half* d_input, half* d_weight, half* d_output) {
    cublasHandle_t handle;
    cublasCreate(&handle);

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

    cublasDestroy(handle);
}