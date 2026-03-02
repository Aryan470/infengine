#include <driver_types.h>
#include <gtest/gtest.h>
#include "test_utils.h"
#include "../kernels/sampling.cuh"

TEST(SamplingAmax, MatchesPyTorch) {
    std::vector<half> input = load_tensor("test_data/sampling_amax_input.bin");
    std::vector<int> expected = load_int_tensor("test_data/sampling_amax_output.bin");

    half* d_input;
    int* d_output;
    int actual;
    cudaMalloc(&d_input, sizeof(half) * input.size());
    cudaMalloc(&d_output, sizeof(int));
    cudaMemcpy(d_input, input.data(), sizeof(half) * input.size(), cudaMemcpyHostToDevice);

    sample_token_amax(d_input, d_output);
    cudaMemcpy(&actual, d_output, sizeof(int), cudaMemcpyDeviceToHost);

    EXPECT_EQ(actual, expected[0]);

    cudaFree(d_input);
    cudaFree(d_output);
}