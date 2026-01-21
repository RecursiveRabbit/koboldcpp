#include "common.cuh"

#define CUDA_SOFT_MAX_BLOCK_SIZE 1024

void ggml_cuda_op_soft_max(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_soft_max_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// Attention tap infrastructure - side-channel for capturing attention during forward pass
void attention_tap_init(int max_layers, int max_heads, int max_ctx);
void attention_tap_free();
void attention_tap_reset();  // Call at start of each token generation
bool attention_tap_extract(float ** out_data, int * out_n_layers, int * out_n_heads, int * out_seq_len);
bool attention_tap_get_gpu_data(float ** out_buffer, int * out_n_layers, int * out_n_heads, int * out_seq_len, int * out_max_ctx);
void attention_tap_update_brightness(void * stream);  // Update brightness engine with GPU tap data (void* for ABI compat)
