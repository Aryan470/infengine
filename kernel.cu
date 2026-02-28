#include "kernel.cuh"
#include "manager.h"
#include "config.h"

int initialize_request_context(RequestContext* context) {
	// alloc token buffer, kv cache on gpu
	cudaMalloc(&context->cu_token_buffer, InfEngineConfig::MAX_CONTEXT_LENGTH * sizeof(int));
	const size_t kv_cache_size = (size_t) (InfEngineConfig::MAX_CONTEXT_LENGTH) * InfEngineConfig::NUM_LAYERS * InfEngineConfig::NUM_KV_HEADS * 2 * InfEngineConfig::HEAD_DIM * InfEngineConfig::FLOAT_SIZE;
	cudaMalloc(&context->cu_kv_cache_buffer, kv_cache_size);
	cudaMemcpy(context->cu_token_buffer, context->tokens.data(), context->tokens.size() * sizeof(int), cudaMemcpyHostToDevice);
	return 0;
}

int prefill(RequestContext* context) {
	// prefill: fill kv cache with input tokens, one pass through model weights
	return 0;
}

int decode(RequestContext* context) {
	// decode kernel: which work on kvcache + token buffer iteratively until EOS or max seq len

	// once we finish decode, copy output tokens to token buffer
	cudaMemcpy(context->tokens.data(), context->cu_token_buffer, context->tokens.size() * sizeof(int), cudaMemcpyDeviceToHost);
	return 0;
}

int cleanup_request_context(RequestContext* context) {
	cudaFree(context->cu_token_buffer);
	cudaFree(context->cu_kv_cache_buffer);
	return 0;
}