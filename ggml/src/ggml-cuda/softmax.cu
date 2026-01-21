#include "common.cuh"
#include "ggml.h"
#include "softmax.cuh"
#include "brightness.cuh"
#include <cstdint>
#include <utility>
#include <cstring>
#include <mutex>

// =============================================================================
// ATTENTION TAP INFRASTRUCTURE
// Side-channel buffer for capturing attention data during kernel execution.
// Invisible to ggml's allocator - lives outside the compute graph.
// =============================================================================

struct attention_tap_state {
    float * buffer;           // GPU buffer: [max_layers, max_heads, max_ctx]
    float * host_buffer;      // CPU staging buffer for extraction
    int max_layers;
    int max_heads;
    int max_ctx;
    int current_ctx;          // Actual context length for current token
    int n_layers_captured;    // Number of layers written this token
    bool enabled;
    std::mutex mtx;
};

static attention_tap_state g_attn_tap = {};

void attention_tap_init(int max_layers, int max_heads, int max_ctx) {
    std::lock_guard<std::mutex> lock(g_attn_tap.mtx);

    if (g_attn_tap.buffer != nullptr) {
        return;  // Already initialized
    }

    size_t buffer_size = (size_t)max_layers * max_heads * max_ctx * sizeof(float);

    cudaError_t err = cudaMalloc(&g_attn_tap.buffer, buffer_size);
    if (err != cudaSuccess) {
        fprintf(stderr, "[ATTN_TAP] Failed to allocate GPU buffer: %s\n", cudaGetErrorString(err));
        return;
    }

    g_attn_tap.host_buffer = (float *)malloc(buffer_size);
    if (g_attn_tap.host_buffer == nullptr) {
        cudaFree(g_attn_tap.buffer);
        g_attn_tap.buffer = nullptr;
        fprintf(stderr, "[ATTN_TAP] Failed to allocate host buffer\n");
        return;
    }

    g_attn_tap.max_layers = max_layers;
    g_attn_tap.max_heads = max_heads;
    g_attn_tap.max_ctx = max_ctx;
    g_attn_tap.current_ctx = 0;
    g_attn_tap.n_layers_captured = 0;
    g_attn_tap.enabled = true;

    fprintf(stderr, "[ATTN_TAP] Initialized: %d layers, %d heads, %d max_ctx (%.2f MB GPU)\n",
            max_layers, max_heads, max_ctx, buffer_size / (1024.0f * 1024.0f));
}

void attention_tap_free() {
    std::lock_guard<std::mutex> lock(g_attn_tap.mtx);

    if (g_attn_tap.buffer != nullptr) {
        cudaFree(g_attn_tap.buffer);
        g_attn_tap.buffer = nullptr;
    }
    if (g_attn_tap.host_buffer != nullptr) {
        free(g_attn_tap.host_buffer);
        g_attn_tap.host_buffer = nullptr;
    }
    g_attn_tap.enabled = false;
}

void attention_tap_reset() {
    // Called at start of each token generation
    g_attn_tap.n_layers_captured = 0;
    g_attn_tap.current_ctx = 0;
}

// Returns pointer into GPU buffer for a specific layer
static float * attention_tap_get_layer_ptr(int layer, int n_heads, int seq_len) {
    if (!g_attn_tap.enabled || g_attn_tap.buffer == nullptr) {
        return nullptr;
    }
    if (layer >= g_attn_tap.max_layers || n_heads > g_attn_tap.max_heads || seq_len > g_attn_tap.max_ctx) {
        return nullptr;
    }

    // Update context length (should be same for all layers)
    g_attn_tap.current_ctx = seq_len;

    // Layout: [layer, head, ctx]
    size_t offset = (size_t)layer * g_attn_tap.max_heads * g_attn_tap.max_ctx;
    return g_attn_tap.buffer + offset;
}

static void attention_tap_layer_complete(int layer) {
    if (layer >= g_attn_tap.n_layers_captured) {
        g_attn_tap.n_layers_captured = layer + 1;
    }
}

// Extract all captured attention to host memory
bool attention_tap_extract(float ** out_data, int * out_n_layers, int * out_n_heads, int * out_seq_len) {
    if (!g_attn_tap.enabled || g_attn_tap.n_layers_captured == 0) {
        return false;
    }

    int n_layers = g_attn_tap.n_layers_captured;
    int n_heads = g_attn_tap.max_heads;
    int seq_len = g_attn_tap.current_ctx;

    // Copy from GPU to host - only the data we actually captured
    // We copy full max_heads * max_ctx per layer for simplicity
    size_t bytes_per_layer = (size_t)g_attn_tap.max_heads * g_attn_tap.max_ctx * sizeof(float);

    cudaError_t err = cudaMemcpy(g_attn_tap.host_buffer, g_attn_tap.buffer,
                                  bytes_per_layer * n_layers, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "[ATTN_TAP] Extract failed: %s\n", cudaGetErrorString(err));
        return false;
    }

    *out_data = g_attn_tap.host_buffer;
    *out_n_layers = n_layers;
    *out_n_heads = n_heads;
    *out_seq_len = seq_len;

    return true;
}

// Get raw GPU buffer for brightness calculation (no CPU copy)
bool attention_tap_get_gpu_data(float ** out_buffer, int * out_n_layers, int * out_n_heads, int * out_seq_len, int * out_max_ctx) {
    if (!g_attn_tap.enabled || g_attn_tap.n_layers_captured == 0) {
        return false;
    }

    *out_buffer = g_attn_tap.buffer;
    *out_n_layers = g_attn_tap.n_layers_captured;
    *out_n_heads = g_attn_tap.max_heads;
    *out_seq_len = g_attn_tap.current_ctx;
    *out_max_ctx = g_attn_tap.max_ctx;

    return true;
}

// Update brightness using GPU tap buffer (call after all layers captured)
// Takes void* for ABI compatibility with non-CUDA callers (cast to cudaStream_t internally)
void attention_tap_update_brightness(void * stream_ptr) {
    float * buffer;
    int n_layers, n_heads, seq_len, max_ctx;

    if (!attention_tap_get_gpu_data(&buffer, &n_layers, &n_heads, &seq_len, &max_ctx)) {
        return;
    }

    // Cast void* to cudaStream_t (nullptr = default stream)
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);

    // Call brightness engine with GPU attention data
    brightness_update(buffer, seq_len, n_layers, n_heads, max_ctx, stream);
}

// =============================================================================
// END ATTENTION TAP INFRASTRUCTURE
// =============================================================================

template <typename T>
static __device__ __forceinline__ float t2f32(T val) {
    return (float) val;
}

template <>
__device__ float __forceinline__ t2f32<half>(half val) {
    return __half2float(val);
}

struct soft_max_params {

    int64_t nheads;
    uint32_t n_head_log2;
    int64_t ncols;
    int64_t nrows_x;
    int64_t nrows_y;
    int64_t ne00;
    int64_t ne01;
    int64_t ne02;
    int64_t ne03;
    int64_t nb11;
    int64_t nb12;
    int64_t nb13;

    int64_t ne12;
    int64_t ne13;
    float scale;
    float max_bias;
    float m0;
    float m1;
};

// When ncols_template == 0 the bounds for the loops in this function are not known and can't be unrolled.
// As we want to keep pragma unroll for all other cases we supress the clang transformation warning here.
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"
#endif // __clang__
template <bool use_shared, int ncols_template, int block_size_template, typename T>
static __global__ void soft_max_f32(
        const float * x, const T * mask, const float * sinks, float * dst, const soft_max_params p) {
    const int ncols = ncols_template == 0 ? p.ncols : ncols_template;

    const int tid  = threadIdx.x;

    const int64_t i03 = blockIdx.z;
    const int64_t i02 = blockIdx.y;
    const int64_t i01 = blockIdx.x;

    //TODO: noncontigous inputs/outputs
    const int rowx = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;

    const int64_t i11 = i01;
    const int64_t i12 = i02 % p.ne12;
    const int64_t i13 = i03 % p.ne13;

    x    += int64_t(rowx)*ncols;
    mask += (i11*p.nb11 + i12*p.nb12 + i13*p.nb13) / sizeof(T) * (mask != nullptr);
    dst  += int64_t(rowx)*ncols;

    const int block_size = block_size_template == 0 ? blockDim.x : block_size_template;

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;

    const float slope = get_alibi_slope(p.max_bias, i02, p.n_head_log2, p.m0, p.m1);

    extern __shared__ float data_soft_max_f32[];
    float * buf_iw = data_soft_max_f32; // shared memory buffer for inter-warp communication
    // shared memory buffer to cache values between iterations:
    float * vals = use_shared ? buf_iw + WARP_SIZE : dst;

    float max_val = sinks ? sinks[i02] : -INFINITY;

#pragma unroll
    for (int col0 = 0; col0 < ncols; col0 += block_size) {
        const int col = col0 + tid;

        if (ncols_template == 0 && col >= ncols) {
            break;
        }

        const float val = x[col]*p.scale + (mask ? slope*t2f32(mask[col]) : 0.0f);

        vals[col] = val;
        max_val = max(max_val, val);
    }

    // find the max value in the block
    max_val = warp_reduce_max(max_val);
    if (block_size > WARP_SIZE) {
        if (warp_id == 0) {
            buf_iw[lane_id] = -INFINITY;
        }
        __syncthreads();

        if (lane_id == 0) {
            buf_iw[warp_id] = max_val;
        }
        __syncthreads();

        max_val = buf_iw[lane_id];
        max_val = warp_reduce_max(max_val);
    }

    float tmp = 0.0f; // partial sum

#pragma unroll
    for (int col0 = 0; col0 < ncols; col0 += block_size) {
        const int col = col0 + tid;

        if (ncols_template == 0 && col >= ncols) {
            break;
        }

        const float val = expf(vals[col] - max_val);
        tmp += val;
        vals[col] = val;
    }

    // find the sum of exps in the block
    tmp = warp_reduce_sum(tmp);
    if (block_size > WARP_SIZE) {
        __syncthreads();
        if (warp_id == 0) {
            buf_iw[lane_id] = 0.0f;
        }
        __syncthreads();

        if (lane_id == 0) {
            buf_iw[warp_id] = tmp;
        }
        __syncthreads();

        tmp = buf_iw[lane_id];
        tmp = warp_reduce_sum(tmp);
    }

    if (sinks) {
        tmp += expf(sinks[i02] - max_val);
    }

    const float inv_sum = 1.0f / tmp;

#pragma unroll
    for (int col0 = 0; col0 < ncols; col0 += block_size) {
        const int col = col0 + tid;

        if (ncols_template == 0 && col >= ncols) {
            return;
        }

        dst[col] = vals[col] * inv_sum;
    }
}
#ifdef __clang__
#pragma clang diagnostic pop
#endif // __clang__

static __global__ void soft_max_back_f32(
        const float * grad, const float * dstf, float * dst, const int ncols, const float scale) {
    const int tid  = threadIdx.x;
    const int rowx = blockIdx.x;

    grad += int64_t(rowx)*ncols;
    dstf += int64_t(rowx)*ncols;
    dst  += int64_t(rowx)*ncols;

    float dgf_dot = 0.0f; // dot product of dst from forward pass and gradients

    for (int col = tid; col < ncols; col += WARP_SIZE) {
        dgf_dot += dstf[col]*grad[col];
    }

    dgf_dot = warp_reduce_sum(dgf_dot);

    for (int col = tid; col < ncols; col += WARP_SIZE) {
        dst[col] = scale * (grad[col] - dgf_dot) * dstf[col];
    }
}

template<int... Ns, typename T>
static void launch_soft_max_kernels(const float * x, const T * mask, const float * sinks, float * dst,
                             const soft_max_params & p, cudaStream_t stream, dim3 block_dims, dim3 block_nums, size_t nbytes_shared)
{
    const int id       = ggml_cuda_get_device();
    const size_t smpbo = ggml_cuda_info().devices[id].smpbo;

    auto launch_kernel = [=](auto I) -> bool {
        constexpr int ncols = decltype(I)::value;
        constexpr int block = (ncols > 1024 ? 1024 : ncols);

        if (p.ncols == ncols) {
            CUDA_SET_SHARED_MEMORY_LIMIT((soft_max_f32<true, ncols, block, T>), smpbo);
            soft_max_f32<true, ncols, block><<<block_nums, block_dims, nbytes_shared, stream>>>
                (x, mask, sinks, dst, p);
            return true;
        }
        return false;
    };

    // unary fold over launch_kernel
    if ((launch_kernel(std::integral_constant<int, Ns>{}) || ...)) {
        return;
    }

    //default case
    CUDA_SET_SHARED_MEMORY_LIMIT((soft_max_f32<true, 0, 0, T>), smpbo);
    soft_max_f32<true, 0, 0><<<block_nums, block_dims, nbytes_shared, stream>>>(x, mask, sinks, dst, p);
}


template<typename T>
static void soft_max_f32_cuda(const float * x, const T * mask, const float * sinks, float * dst, const soft_max_params & params, cudaStream_t stream) {
    int nth = WARP_SIZE;
    const int64_t ncols_x = params.ncols;

    while (nth < ncols_x && nth < CUDA_SOFT_MAX_BLOCK_SIZE) nth *= 2;
    const dim3 block_dims(nth,     1, 1);
    const dim3 block_nums(params.ne01, params.ne02, params.ne03);
    const size_t nbytes_shared = (GGML_PAD(ncols_x, WARP_SIZE) + WARP_SIZE)*sizeof(float);
    static_assert(CUDA_SOFT_MAX_BLOCK_SIZE == 1024, "These values need to be adjusted.");


    const int id       = ggml_cuda_get_device();
    const size_t smpbo = ggml_cuda_info().devices[id].smpbo;


    if (nbytes_shared <= smpbo) {
        launch_soft_max_kernels<32, 64, 128, 256, 512, 1024, 2048, 4096>(x, mask, sinks, dst, params, stream, block_dims, block_nums, nbytes_shared);
    } else {
        const size_t nbytes_shared_low = WARP_SIZE*sizeof(float);
        soft_max_f32<false, 0, 0><<<block_nums, block_dims, nbytes_shared_low, stream>>>(x, mask, sinks, dst, params);
    }
}

static void soft_max_back_f32_cuda(
        const float * grad, const float * dstf, float * dst,
        const int ncols, const int nrows, const float scale, cudaStream_t stream) {
    const dim3 block_dims(WARP_SIZE, 1, 1);
    const dim3 block_nums(nrows,     1, 1);

    soft_max_back_f32<<<block_nums, block_dims, 0, stream>>>(grad, dstf, dst, ncols, scale);
}

void ggml_cuda_op_soft_max(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * src2 = dst->src[2];

    const float * src0_d = (const float *) src0->data;
    const void  * src1_d = src1 ? (const void *) src1->data : nullptr;
    const void  * src2_d = src2 ? (const void *) src2->data : nullptr;
    float       *  dst_d = (float *) dst->data;

    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_ASSERT(!src1 || src1->type == GGML_TYPE_F16 || src1->type == GGML_TYPE_F32); // src1 contains mask and it is optional

    const int64_t nrows_x = ggml_nrows(src0);
    const int64_t nrows_y = src0->ne[1];

    const int64_t ne00 = src0->ne[0];

    float scale    = 1.0f;
    float max_bias = 0.0f;

    memcpy(&scale,    (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));

    const bool use_f16 = (src1 && src1->type == GGML_TYPE_F16);

    const int64_t nb11 = src1 ? src1->nb[1] : 1;
    const int64_t nb12 = src1 ? src1->nb[2] : 1;
    const int64_t nb13 = src1 ? src1->nb[3] : 1;

    const int64_t ne12 = src1 ? src1->ne[2] : 1;
    const int64_t ne13 = src1 ? src1->ne[3] : 1;

    const uint32_t n_head      = src0->ne[2];
    const uint32_t n_head_log2 = 1u << (uint32_t) floorf(log2f((float) n_head));

    const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);


    soft_max_params params = {};
    params.nheads = src0->ne[2];
    params.n_head_log2 = n_head_log2;
    params.ncols = ne00;
    params.nrows_x = nrows_x;
    params.nrows_y = nrows_y;
    params.ne00 = src0->ne[0];
    params.ne01 = src0->ne[1];
    params.ne02 = src0->ne[2];
    params.ne03 = src0->ne[3];
    params.nb11 = nb11;
    params.nb12 = nb12;
    params.nb13 = nb13;
    params.ne12 = ne12;
    params.ne13 = ne13;
    params.scale = scale;
    params.max_bias = max_bias;
    params.m0 = m0;
    params.m1 = m1;

    if (use_f16) {
        soft_max_f32_cuda(src0_d, (const half  *) src1_d, (const float *) src2_d, dst_d, params, stream);
    } else {
        soft_max_f32_cuda(src0_d, (const float *) src1_d, (const float *) src2_d, dst_d, params, stream);
    }

    // =========================================================================
    // ATTENTION TAP: Copy attention data to side-channel buffer
    // This happens immediately after kernel on same stream, before buffer reuse
    // =========================================================================
    if (dst->name && strncmp(dst->name, "kq_soft_max-", 12) == 0) {
        // Only tap single-token generation (ne01 == 1), not prompt processing
        if (params.ne01 == 1) {
            // Extract layer index from name "kq_soft_max-N"
            int layer = atoi(dst->name + 12);

            int n_heads = (int)params.ne02;
            int seq_len = (int)params.ne00;

            float * tap_ptr = attention_tap_get_layer_ptr(layer, n_heads, seq_len);
            if (tap_ptr != nullptr) {
                // Async copy on same stream - will complete before any subsequent kernel
                // that might reuse dst_d buffer
                size_t bytes = (size_t)n_heads * seq_len * sizeof(float);
                cudaMemcpyAsync(tap_ptr, dst_d, bytes, cudaMemcpyDeviceToDevice, stream);
                attention_tap_layer_complete(layer);
            }
        }
    }
}

void ggml_cuda_op_soft_max_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0]; // grad
    const ggml_tensor * src1 = dst->src[1]; // forward pass output

    const float * src0_d = (const float *) src0->data;
    const float * src1_d = (const float *) src1->data;
    float       * dst_d  = (float       *) dst->data;

    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    const int64_t ncols = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    float scale    = 1.0f;
    float max_bias = 0.0f;

    memcpy(&scale,    (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));

    GGML_ASSERT(max_bias == 0.0f);

    soft_max_back_f32_cuda(src0_d, src1_d, dst_d, ncols, nrows, scale, stream);
}
