// =============================================================================
// BRIGHTNESS ENGINE - GPU-side token importance scoring
//
// Computes "brightness" (importance) for each token based on attention patterns.
// Runs entirely on GPU using the attention tap buffer as input.
// Output is a persistent texture that accumulates across token generations.
//
// Algorithm (Magnitude-Weighted Voting):
// 1. Detect attention sink (highest attention position, detected once)
// 2. For each token generation:
//    - Aggregate attention across all layers/heads
//    - Calculate threshold: (1 - sink_attention) / (n_tokens - 1)
//    - If attention > threshold: brightness += int(attention / threshold)
//    - Else: brightness -= 1
// 3. New tokens initialize at 10000.0
// 4. Sink position always stays at max brightness
//
// Output: R32F texture, 1 x max_ctx, where each pixel = token brightness
// =============================================================================

#include "common.cuh"
#include <cstdio>
#include <mutex>
#include <cfloat>

// =============================================================================
// BRIGHTNESS STATE
// =============================================================================

struct brightness_state {
    float * gpu_buffer;         // [max_ctx] brightness per token
    float * host_buffer;        // Pinned host memory for async readback
    int max_ctx;                // Maximum context length
    int current_ctx;            // Current context length
    int sink_pos;               // Attention sink position (-1 = not detected)
    float sink_attention;       // Attention at sink position
    bool initialized;
    std::mutex mtx;
};

static brightness_state g_brightness = {};

// =============================================================================
// CUDA KERNELS
// =============================================================================

// Kernel to find attention sink (position with max aggregated attention)
// Run once on first token generation
__global__ void find_sink_kernel(
    const float * attention,    // [n_layers * n_heads, seq_len]
    float * max_attention,      // [seq_len] - aggregated attention per position
    int seq_len,
    int n_layers,
    int n_heads
) {
    int token_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (token_idx >= seq_len) return;

    // Aggregate attention across all layers and heads
    float sum = 0.0f;
    int total_heads = n_layers * n_heads;
    for (int i = 0; i < total_heads; i++) {
        sum += attention[i * seq_len + token_idx];
    }

    max_attention[token_idx] = sum / float(total_heads);
}

// Kernel to initialize new tokens to starting brightness
__global__ void init_new_tokens_kernel(
    float * brightness,
    int old_ctx,
    int new_ctx,
    float init_value
) {
    int token_idx = blockIdx.x * blockDim.x + threadIdx.x + old_ctx;
    if (token_idx >= new_ctx) return;

    brightness[token_idx] = init_value;
}

// Main brightness update kernel
// Runs after each token generation, updates brightness based on attention
__global__ void brightness_update_kernel(
    const float * attention,    // [n_layers * n_heads, seq_len] - aggregated attention
    float * brightness,         // [seq_len] - brightness per token (in/out)
    int seq_len,
    int n_layers,
    int n_heads,
    int sink_pos,
    float threshold             // Pre-computed: (1 - sink_attention) / (seq_len - 1)
) {
    int token_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (token_idx >= seq_len) return;

    // Sink position maintains max brightness (white = 0xFFFFFF)
    if (token_idx == sink_pos) {
        brightness[token_idx] = 16777215.0f;
        return;
    }

    // Aggregate attention across all layers and heads for this token
    float aggregated = 0.0f;
    int total_heads = n_layers * n_heads;
    for (int i = 0; i < total_heads; i++) {
        aggregated += attention[i * seq_len + token_idx];
    }
    aggregated /= float(total_heads);

    // Calculate update
    float ratio = aggregated / threshold;
    float update;
    if (aggregated > threshold) {
        // Strong reference: boost by ratio (clamped to reasonable range)
        update = fminf(ratio, 100.0f);  // Cap at +100 per token
    } else {
        // Weak/no reference: gentle decay
        update = -1.0f;
    }

    // Update brightness (clamp to [0, 0xFFFFFF])
    float new_val = brightness[token_idx] + update;
    brightness[token_idx] = fmaxf(0.0f, fminf(16777215.0f, new_val));
}

// =============================================================================
// PUBLIC API
// =============================================================================

void brightness_init(int max_ctx) {
    std::lock_guard<std::mutex> lock(g_brightness.mtx);

    if (g_brightness.initialized) {
        return;
    }

    size_t buffer_size = (size_t)max_ctx * sizeof(float);

    // Allocate GPU buffer
    cudaError_t err = cudaMalloc(&g_brightness.gpu_buffer, buffer_size);
    if (err != cudaSuccess) {
        fprintf(stderr, "[BRIGHTNESS] Failed to allocate GPU buffer: %s\n", cudaGetErrorString(err));
        return;
    }

    // Initialize to 0 (will be set to 10000 as tokens are added)
    cudaMemset(g_brightness.gpu_buffer, 0, buffer_size);

    // Allocate pinned host buffer for async readback
    err = cudaMallocHost(&g_brightness.host_buffer, buffer_size);
    if (err != cudaSuccess) {
        fprintf(stderr, "[BRIGHTNESS] Failed to allocate pinned host buffer: %s\n", cudaGetErrorString(err));
        cudaFree(g_brightness.gpu_buffer);
        g_brightness.gpu_buffer = nullptr;
        return;
    }

    g_brightness.max_ctx = max_ctx;
    g_brightness.current_ctx = 0;
    g_brightness.sink_pos = -1;
    g_brightness.sink_attention = 0.0f;
    g_brightness.initialized = true;

    fprintf(stderr, "[BRIGHTNESS] Initialized: max_ctx=%d (%.2f MB GPU)\n",
            max_ctx, buffer_size / (1024.0f * 1024.0f));
}

void brightness_free() {
    std::lock_guard<std::mutex> lock(g_brightness.mtx);

    if (g_brightness.gpu_buffer) {
        cudaFree(g_brightness.gpu_buffer);
        g_brightness.gpu_buffer = nullptr;
    }
    if (g_brightness.host_buffer) {
        cudaFreeHost(g_brightness.host_buffer);
        g_brightness.host_buffer = nullptr;
    }
    g_brightness.initialized = false;
}

void brightness_reset() {
    std::lock_guard<std::mutex> lock(g_brightness.mtx);

    if (!g_brightness.initialized) return;

    // Reset to initial state (new conversation)
    cudaMemset(g_brightness.gpu_buffer, 0, g_brightness.max_ctx * sizeof(float));
    g_brightness.current_ctx = 0;
    g_brightness.sink_pos = -1;
    g_brightness.sink_attention = 0.0f;
}

// Called after attention tap captures data for a token
// attention_data: pointer to tap buffer [n_layers, n_heads, seq_len]
// The tap buffer layout is [layer][head][ctx] with max dimensions
void brightness_update(
    const float * attention_data,
    int seq_len,
    int n_layers,
    int n_heads,
    int max_ctx,
    cudaStream_t stream
) {
    if (!g_brightness.initialized || attention_data == nullptr) {
        return;
    }

    // Initialize new tokens if context grew
    if (seq_len > g_brightness.current_ctx) {
        int old_ctx = g_brightness.current_ctx;
        int new_tokens = seq_len - old_ctx;

        dim3 block(256);
        dim3 grid((new_tokens + block.x - 1) / block.x);
        init_new_tokens_kernel<<<grid, block, 0, stream>>>(
            g_brightness.gpu_buffer, old_ctx, seq_len, 16777215.0f  // White = 0xFFFFFF
        );

        g_brightness.current_ctx = seq_len;
    }

    // Detect sink on first token (or if not yet detected)
    if (g_brightness.sink_pos < 0) {
        // Allocate temp buffer for aggregated attention
        float * d_aggregated;
        cudaMalloc(&d_aggregated, seq_len * sizeof(float));

        // Compute aggregated attention per position
        dim3 block(256);
        dim3 grid((seq_len + block.x - 1) / block.x);
        find_sink_kernel<<<grid, block, 0, stream>>>(
            attention_data, d_aggregated, seq_len, n_layers, n_heads
        );
        cudaStreamSynchronize(stream);

        // Copy to host and find max
        std::vector<float> h_aggregated(seq_len);
        cudaMemcpy(h_aggregated.data(), d_aggregated, seq_len * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_aggregated);

        // Find position with max attention
        int max_pos = 0;
        float max_val = h_aggregated[0];
        for (int i = 1; i < seq_len; i++) {
            if (h_aggregated[i] > max_val) {
                max_val = h_aggregated[i];
                max_pos = i;
            }
        }

        g_brightness.sink_pos = max_pos;
        g_brightness.sink_attention = max_val;

        fprintf(stderr, "[BRIGHTNESS] Detected sink at position %d (attention=%.4f)\n",
                max_pos, max_val);
    }

    // Calculate threshold
    float threshold = (1.0f - g_brightness.sink_attention) / float(seq_len - 1);
    if (threshold <= 0.0f) threshold = 1e-6f;  // Avoid division by zero

    // Update brightness for all tokens
    dim3 block(256);
    dim3 grid((seq_len + block.x - 1) / block.x);
    brightness_update_kernel<<<grid, block, 0, stream>>>(
        attention_data,
        g_brightness.gpu_buffer,
        seq_len,
        n_layers,
        n_heads,
        g_brightness.sink_pos,
        threshold
    );
}

// Get brightness data (async copy to host, returns immediately)
// Call this to trigger readback, then poll brightness_get_data() for results
// Takes void* for ABI compatibility with non-CUDA callers
void brightness_request_readback(void * stream_ptr) {
    if (!g_brightness.initialized || g_brightness.current_ctx == 0) {
        return;
    }

    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    size_t bytes = g_brightness.current_ctx * sizeof(float);
    cudaMemcpyAsync(g_brightness.host_buffer, g_brightness.gpu_buffer,
                    bytes, cudaMemcpyDeviceToHost, stream);
}

// Get brightness data (returns host buffer pointer)
// Call after brightness_request_readback() and stream sync
bool brightness_get_data(const float ** out_data, int * out_len) {
    if (!g_brightness.initialized || g_brightness.current_ctx == 0) {
        return false;
    }

    *out_data = g_brightness.host_buffer;
    *out_len = g_brightness.current_ctx;
    return true;
}

// Get raw GPU buffer pointer (for direct texture mapping)
bool brightness_get_gpu_buffer(float ** out_buffer, int * out_len) {
    if (!g_brightness.initialized) {
        return false;
    }

    *out_buffer = g_brightness.gpu_buffer;
    *out_len = g_brightness.current_ctx;
    return true;
}

// Get sink position (for external use)
int brightness_get_sink_pos() {
    return g_brightness.sink_pos;
}
