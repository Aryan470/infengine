#include "test_utils.h"
#include <fstream>

std::vector<__half> load_tensor(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    f.seekg(0, std::ios::end);
    size_t size = f.tellg() / sizeof(__half);
    f.seekg(0);
    std::vector<__half> data(size);
    f.read(reinterpret_cast<char*>(data.data()), size * sizeof(__half));
    return data;
}

CompareResult compare_tensors(const __half* actual, const __half* expected, size_t n, float atol) {
    CompareResult r{0.f, 0.f, 0.f};
    size_t within = 0;
    for (size_t i = 0; i < n; i++) {
        float err = std::abs(actual[i] - expected[i]);
        r.max_abs_err = std::max(r.max_abs_err, err);
        r.mean_abs_err += err;
        if (err < atol) within++;
    }
    r.mean_abs_err /= n;
    r.pct_within_tol = 100.f * within / n;
    return r;
}