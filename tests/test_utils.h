#pragma once
#include <vector>
#include <string>
#include <cuda_fp16.h>

std::vector<__half> load_tensor(const std::string& path);
std::vector<float> load_float_tensor(const std::string& path);

struct CompareResult {
    float max_abs_err;
    float mean_abs_err;
    float pct_within_tol;
};

CompareResult compare_tensors(const __half* actual, const __half* expected, size_t n, float atol = 1e-5);
CompareResult compare_float_tensors(const float* actual, const float* expected, size_t n, float atol = 1e-5);