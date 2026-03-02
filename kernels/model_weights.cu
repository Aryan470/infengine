#include <cassert>
#include <cuda_runtime_api.h>
#include <iostream>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <fstream>
#include <nlohmann/json.hpp>
#include "model_weights.cuh"
#include "rope.cuh"

#include <sys/mman.h>
#include <fcntl.h>

bool check_all_present(ModelWeights& m) {
    if (m.emb_lookup == nullptr) { printf("ModelWeights: no pointer for emb_lookup\n"); return false; }
    if (m.final_norm == nullptr) { printf("ModelWeights: no pointer for final_norm\n"); return false; }
    if (m.lm_head == nullptr) { printf("ModelWeights: no pointer for lm_head\n"); return false; }
    for (int i = 0; i < InfEngineConfig::NUM_LAYERS; i++) {
        if (m.layers[i].input_layernorm == nullptr) { printf("ModelWeights: no pointer for layers[%d].input_layernorm\n", i); return false; }
        if (m.layers[i].transformer.w_k == nullptr) { printf("ModelWeights: no pointer for layers[%d].transformer.w_k\n", i); return false; }
        if (m.layers[i].transformer.w_q == nullptr) { printf("ModelWeights: no pointer for layers[%d].transformer.w_q\n", i); return false; }
        if (m.layers[i].transformer.w_v == nullptr) { printf("ModelWeights: no pointer for layers[%d].transformer.w_v\n", i); return false; }
        if (m.layers[i].transformer.w_o == nullptr) { printf("ModelWeights: no pointer for layers[%d].transformer.w_o\n", i); return false; }
        if (m.layers[i].post_attention_layernorm == nullptr) { printf("ModelWeights: no pointer for layers[%d].post_attention_layernorm\n", i); return false; }
        if (m.layers[i].ffn_block.w_up == nullptr) { printf("ModelWeights: no pointer for layers[%d].ffn_block.w_up\n", i); return false; }
        if (m.layers[i].ffn_block.w_gate == nullptr) { printf("ModelWeights: no pointer for layers[%d].ffn_block.w_gate\n", i); return false; }
        if (m.layers[i].ffn_block.w_down == nullptr) { printf("ModelWeights: no pointer for layers[%d].ffn_block.w_down\n", i); return false; }
    }
    return true;
}

half** get_model_weights_entry(ModelWeights& m, const std::string tensor_name) {
    if (tensor_name == "model.embed_tokens.weight") { return &m.emb_lookup; }
    else if (tensor_name == "model.norm.weight") { return &m.final_norm; }
    else if (tensor_name == "lm_head.weight") { return &m.lm_head; }
    else if (tensor_name.starts_with("model.layers.")) {
        // tensor_name is "model.layers.{layer_idx}.something"
        size_t start = std::string("model.layers.").size();
        size_t end = tensor_name.find('.', start);
        if (end == std::string::npos) {
            return nullptr;
        }
        int layer = std::stoi(tensor_name.substr(start, end - start));
        assert (0 <= layer && layer < InfEngineConfig::NUM_LAYERS);
        // now, get the rest of the name
        std::string subname = tensor_name.substr(end + 1);
        if (subname == "input_layernorm.weight") { return &m.layers[layer].input_layernorm; }
        else if (subname == "self_attn.q_proj.weight") { return &m.layers[layer].transformer.w_q; }
        else if (subname == "self_attn.k_proj.weight") { return &m.layers[layer].transformer.w_k; }
        else if (subname == "self_attn.v_proj.weight") { return &m.layers[layer].transformer.w_v; }
        else if (subname == "self_attn.o_proj.weight") { return &m.layers[layer].transformer.w_o; }
        else if (subname == "post_attention_layernorm.weight") { return &m.layers[layer].post_attention_layernorm; }
        else if (subname == "mlp.gate_proj.weight") { return &m.layers[layer].ffn_block.w_gate; }
        else if (subname == "mlp.up_proj.weight") { return &m.layers[layer].ffn_block.w_up; }
        else if (subname == "mlp.down_proj.weight") { return &m.layers[layer].ffn_block.w_down; }
    }
    
    throw std::runtime_error(tensor_name + " is an unknown tensor name");
}

__global__ void convert_bf16_fp16_kern(half* d_weight, const int num_elems) {
    const int block_id = blockIdx.x;
    const int thread_id = threadIdx.x;
    const int num_threads = blockDim.x;

    const int my_offset = num_threads * block_id + thread_id;
    if (my_offset < num_elems) {
        nv_bfloat16 orig_data = ((nv_bfloat16*) d_weight)[my_offset];
        d_weight[my_offset] = __float2half(__bfloat162float(orig_data));
    }
}

void convert_bf16_fp16(half* d_weight, const int num_elems) {
    const int num_threads = 512;
    const int num_blocks = (num_elems + (num_threads - 1)) / num_threads;

    convert_bf16_fp16_kern<<<num_blocks, num_threads>>>(d_weight, num_elems);
}

void populate_modelweights(ModelWeights& m, std::string filepath) {
    std::ifstream f(filepath, std::ios::binary | std::ios::ate);
    if (!f.is_open()) {
        throw std::runtime_error("Failed to open file: " + filepath);
    }
    size_t file_size = f.tellg();
    f.seekg(0);

    // get the length of the json blob
    uint64_t header_len;
    f.read(reinterpret_cast<char*>(&header_len), 8);

    // get the blob
    std::string header_json(header_len, '\0');
    f.read(header_json.data(), header_len);
    auto header = nlohmann::json::parse(header_json);

    size_t data_offset_in_file = 8 + header_len;

    // mmap the file
    int fd = open(filepath.c_str(), O_RDONLY);
    if (fd == -1) {
        throw std::runtime_error("Failed to open file for mmap: " + filepath);
    }
    void* mapped = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapped == MAP_FAILED) {
        close(fd);
        throw std::runtime_error("Failed to mmap file: " + filepath);
    }
    char* data_start = (char*) mapped;

    for (auto& [name, val] : header.items()) {
        if (name == "__metadata__") continue;
        // go from name to pointer within the struct
        half** model_weights_entry = get_model_weights_entry(m, name);
        // make sure it has not already been initialized
        assert (*model_weights_entry == nullptr);

        const std::string dtype = val["dtype"];
        assert(dtype == "BF16");

        const std::vector<int64_t> shape = val["shape"].get<std::vector<int64_t>>();
        size_t num_elems = 1;
        for (const int dim : shape) {num_elems *= dim;}
        const int weights_size_bytes = sizeof(half) * num_elems;
        const size_t offset_begin = val["data_offsets"][0].get<size_t>() + data_offset_in_file;
        const size_t offset_end = val["data_offsets"][1].get<size_t>() + data_offset_in_file;
        assert (offset_end - offset_begin == weights_size_bytes);

        half* d_weight;
        cudaMalloc(&d_weight, weights_size_bytes);
        cudaMemcpy(d_weight, data_start + offset_begin, weights_size_bytes, cudaMemcpyHostToDevice);


        // convert bf16 to fp16
        convert_bf16_fp16(d_weight, num_elems);

        *model_weights_entry = d_weight;
    }

}

ModelWeights ModelWeights::from_safetensors(std::vector<std::string> safetensors_paths) {
    ModelWeights m{0};
    // for each file, read the json header
    for (auto& filepath : safetensors_paths) {
        populate_modelweights(m, filepath);
    }
    // precompute rope buffer
    init_rope_buffer(&m.rope.cos, &m.rope.sin);

    // ensure that we have populated each pointer
    if (!check_all_present(m)) { throw std::runtime_error("ModelWeights: not all required pointers are present after loading from safetensors."); }
    return m;
}

void ModelWeights::free() {
    cudaFree(emb_lookup);
    for (int i = 0; i < InfEngineConfig::NUM_LAYERS; i++) {
        cudaFree(layers[i].input_layernorm);
        cudaFree(layers[i].transformer.w_k);
        cudaFree(layers[i].transformer.w_q);
        cudaFree(layers[i].transformer.w_v);
        cudaFree(layers[i].transformer.w_o);
        cudaFree(layers[i].post_attention_layernorm);
        cudaFree(layers[i].ffn_block.w_up);
        cudaFree(layers[i].ffn_block.w_gate);
        cudaFree(layers[i].ffn_block.w_down);
    }
    cudaFree(final_norm);
    cudaFree(lm_head);
    cudaFree(rope.cos);
    cudaFree(rope.sin);
}