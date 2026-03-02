#include <driver_types.h>
#include <gtest/gtest.h>
#include <vector>
#include "../kernels/model_weights.cuh"
#include "test_utils.h"

TEST(ModelWeights, LoadCorrectly) {
    std::vector<std::string> filenames = {
        "models/llama_3.1_8b/model-00001-of-00004.safetensors",
        "models/llama_3.1_8b/model-00002-of-00004.safetensors",
        "models/llama_3.1_8b/model-00003-of-00004.safetensors",
        "models/llama_3.1_8b/model-00004-of-00004.safetensors"
    };
    ModelWeights m = ModelWeights::from_safetensors(filenames);

    // do a sample comparison against first layer's kproj matrix
    half* d_Wk0_loaded = m.layers[0].transformer.w_k;
    std::vector<half> Wk0_saved = load_tensor("test_data/attn_kproj.bin");
    std::vector<half> Wk0_loaded(Wk0_saved.size());

    cudaMemcpy(Wk0_loaded.data(), d_Wk0_loaded, sizeof(half) * Wk0_saved.size(), cudaMemcpyDeviceToHost);

    CompareResult res = compare_tensors(Wk0_loaded.data(), Wk0_saved.data(), Wk0_loaded.size(), 0.0);

    EXPECT_LE(res.max_abs_err, 0.0f);
    EXPECT_GE(res.pct_within_tol, 100.0);

    m.free();
}