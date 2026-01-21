#pragma once

#include <cuda_runtime.h>

// Brightness engine - GPU-side token importance scoring
// Computes per-token brightness based on attention patterns

void brightness_init(int max_ctx);
void brightness_free();
void brightness_reset();  // Reset for new conversation

// Update brightness based on attention data from tap buffer
// Called after each token generation
void brightness_update(
    const float * attention_data,   // Tap buffer [n_layers, n_heads, seq_len]
    int seq_len,
    int n_layers,
    int n_heads,
    int max_ctx,
    cudaStream_t stream
);

// Async readback to host memory (void* for ABI compatibility)
void brightness_request_readback(void * stream);

// Get brightness data after readback
bool brightness_get_data(const float ** out_data, int * out_len);

// Get raw GPU buffer pointer (for direct texture mapping)
bool brightness_get_gpu_buffer(float ** out_buffer, int * out_len);

// Get detected sink position
int brightness_get_sink_pos();
