#include <driver_types.h>
#include <gtest/gtest.h>
#include "test_utils.h"
#include "../config.h"
#include "../kernels/emblookup.cuh"

TEST(EmbLookup, MatchesPyTorch) {
    std::vector<int> token_ids = load_int_tensor("test_data/emblookup_tokens.bin");
    std::vector<half> emb_weights = load_tensor("test_data/emblookup_weights.bin");
    std::vector<half> expected = load_tensor("test_data/emblookup_output.bin");
    std::vector<half> actual(expected.size());

    const int seq_len = token_ids.size();

    int* d_input;
    half* d_weights;
    half* d_out;

    cudaMalloc(&d_input, sizeof(int) * token_ids.size());
    cudaMalloc(&d_weights, sizeof(half) * emb_weights.size());
    cudaMalloc(&d_out, sizeof(half) * expected.size());

    cudaMemcpy(d_input, token_ids.data(), sizeof(int) * token_ids.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, emb_weights.data(), sizeof(half) * emb_weights.size(), cudaMemcpyHostToDevice);

    emblookup(seq_len, d_input, d_weights, d_out);
    cudaMemcpy(actual.data(), d_out, expected.size() * sizeof(half), cudaMemcpyDeviceToHost);

    CompareResult result = compare_tensors(actual.data(), expected.data(), expected.size(), 0.0);

    EXPECT_LE(result.max_abs_err, 0.0f);
    EXPECT_GE(result.pct_within_tol, 100.0);

    cudaFree(d_input);
    cudaFree(d_weights);
    cudaFree(d_out);
}