# CLAUDE.md - KoboldCpp Attention Extraction & Brightness Engine

## Project Overview

Modified fork of KoboldCpp with two major additions:
1. **Attention Tap** - CUDA kernel side-channel capturing ALL transformer layers during forward pass
2. **Brightness Engine** - GPU-side token importance scoring for context culling

---

## The Problem

### Context Length Management via Brightness-Based Culling

**Goal**: Delete unimportant tokens from context while preserving important information.

Traditional approaches fail:
- **FIFO/Sliding window**: Deletes oldest tokens regardless of importance
- **Summarization**: Model rewords everything, losing direct quotes and precise data

**Solution**: Track attention patterns over time, assign "brightness" scores to tokens, cull lowest-brightness sentences when context exceeds limit.

### Technical Challenges

1. **ggml Buffer Aliasing**: All 48 `kq_soft_max` tensors share the same VRAM address due to lifetime-based allocation. By extraction time, only the last layer's data remains.

2. **Performance**: Can't block inference for CPU-side brightness calculations.

---

## Architecture: Attention Tap + Brightness Engine

### The Solution: CUDA Kernel Side-Channel

Instead of fighting ggml's allocator, we tap attention data **during kernel execution**, before buffers get reused.

```
Forward Pass (CUDA)                    Side Channel
       │
       ▼
   softmax kernel computes
   writes to dst buffer ─────────────► cudaMemcpyAsync to tap_buffer
       │                               (same stream, before reuse)
       ▼
   next layer overwrites dst           tap_buffer preserves data
       │                                      │
       ▼                                      │
   ... repeat 48 layers ...                   │
       │                                      │
       ▼                                      ▼
   graph_compute() returns             tap_buffer has ALL 48 layers
       │                                      │
       ▼                                      ▼
   GPU sync                            brightness_update_kernel()
       │                               (parallel, all tokens)
       ▼                                      │
   attention_tap_extract()                    ▼
   (GPU→CPU copy)                      brightness_texture persists
```

### Key Insight

The tap is **invisible to ggml**:
- Not a tensor in the graph
- Not tracked by the allocator
- Not subject to lifetime analysis

We copy data while it's still live, before ggml can reclaim the buffer.

---

## Implementation Details

### File Structure

```
ggml/src/ggml-cuda/
├── softmax.cu          # Attention tap infrastructure + softmax kernel
├── softmax.cuh         # Tap API declarations
├── brightness.cu       # Brightness engine (GPU kernels)
└── brightness.cuh      # Brightness API declarations

gpttype_adapter.cpp     # Integration, init, extraction, brightness wrappers
src/llama-context.cpp   # Hook points (tap reset, extraction call)
expose.h/cpp            # C API for brightness_outputs
koboldcpp.py            # REST endpoint /api/extra/brightness
```

### Attention Tap (softmax.cu)

```cpp
// Global state - invisible to ggml
struct attention_tap_state {
    float * buffer;           // GPU: [max_layers, max_heads, max_ctx]
    float * host_buffer;      // CPU staging for extraction
    int max_layers, max_heads, max_ctx;
    int current_ctx, n_layers_captured;
    bool enabled;
};

// Called during softmax kernel execution
void ggml_cuda_op_soft_max(...) {
    // ... normal softmax computation ...

    // TAP: Copy to side-channel BEFORE buffer reuse
    if (is_attention_tensor && is_single_token_generation) {
        int layer = extract_layer_from_name(dst->name);  // "kq_soft_max-N"
        float * tap_ptr = attention_tap_get_layer_ptr(layer, n_heads, seq_len);
        cudaMemcpyAsync(tap_ptr, dst_d, bytes, cudaMemcpyDeviceToDevice, stream);
    }
}
```

### Brightness Engine (brightness.cu)

#### 24-Bit Color Encoding

Brightness values ARE 24-bit RGB integers directly:
- **16777215** (0xFFFFFF) = white = max brightness (new tokens)
- **16777214** (0xFFFFFE) = one decay step
- **0** (0x000000) = black = fully decayed

No mapping layer - the brightness value can be decomposed directly into RGB:
```cpp
int brightness = 16777166;  // Example decayed token
int r = (brightness >> 16) & 0xFF;  // 255
int g = (brightness >> 8) & 0xFF;   // 255
int b = brightness & 0xFF;          // 206
// RGB(255, 255, 206) - slightly yellow-tinted white
```

#### Algorithm: Magnitude-Weighted Voting

```cpp
__global__ void brightness_update_kernel(
    const float * attention,    // [n_layers * n_heads, seq_len]
    float * brightness,         // [seq_len] - persistent
    int seq_len, int n_layers, int n_heads,
    int sink_pos, float threshold
) {
    int token_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Sink position always stays white
    if (token_idx == sink_pos) {
        brightness[token_idx] = 16777215.0f;  // 0xFFFFFF
        return;
    }

    // Aggregate attention across all layers/heads
    float aggregated = 0.0f;
    for (int i = 0; i < n_layers * n_heads; i++) {
        aggregated += attention[i * seq_len + token_idx];
    }
    aggregated /= float(n_layers * n_heads);

    // Update: boost if above threshold, decay otherwise
    float ratio = aggregated / threshold;
    float update = (aggregated > threshold) ? min(ratio, 100.0f) : -1.0f;

    brightness[token_idx] = clamp(brightness[token_idx] + update, 0.0f, 16777215.0f);
}
```

**Key behaviors:**
- New tokens initialize at 16777215 (0xFFFFFF = white)
- Above-threshold attention boosts brightness by ratio
- Below-threshold decays by -1 per generation
- Sink position (detected once) always stays at max
- Runs entirely on GPU, no CPU round-trip

### Sink Detection

The attention sink is the position that consistently receives high attention (often BOS, but model-dependent). Detected on first token generation:

```cpp
// Find position with max aggregated attention
// Store as constant for all future updates
g_brightness.sink_pos = argmax(aggregated_attention);
g_brightness.sink_attention = aggregated_attention[sink_pos];
```

---

## API

### REST Endpoint (koboldcpp.py)

```
GET /api/extra/brightness
```

Response:
```json
{
  "data": [16777215.0, 16777166.0, 16777166.0, ...],
  "ctx_len": 256,
  "sink_pos": 0,
  "valid": true
}
```

- `data`: Array of brightness values (24-bit integers as floats)
- `ctx_len`: Current context length
- `sink_pos`: Detected attention sink position
- `valid`: Whether brightness data is available

### C API (expose.cpp)

```c
// Attention data (all 48 layers)
attention_outputs get_token_attention(int token_idx);
typedef struct {
    const float * data;    // [n_layers, n_heads, seq_len]
    int n_layers;          // 48 (all layers now!)
    int n_heads;           // 32
    int seq_len;           // current context length
    bool valid;
} attention_outputs;

// Brightness texture
brightness_outputs get_brightness();
typedef struct {
    const float * data;    // [ctx_len] - brightness per token (24-bit values)
    int ctx_len;           // current context length
    int sink_pos;          // detected attention sink position
    bool valid;
} brightness_outputs;
```

### ODR Fix: Wrapper Functions

expose.cpp is compiled once without `-DGGML_USE_CUDA`, so it can't call brightness.cu functions directly. Solution: wrapper functions in gpttype_adapter.cpp that route through the CUDA-compiled translation unit.

```cpp
// gpttype_adapter.cpp - compiled as gpttype_adapter_cublas.o with GGML_USE_CUDA
bool brightness_wrapper_get_data(const float ** out_data, int * out_len) {
    return brightness_get_data(out_data, out_len);
}

int brightness_wrapper_get_sink_pos() {
    return brightness_get_sink_pos();
}

// expose.cpp calls these wrappers instead of brightness.cu functions directly
extern bool brightness_wrapper_get_data(const float ** out_data, int * out_len);
extern int brightness_wrapper_get_sink_pos();
```

### Initialization

```cpp
// Called during model load (gpttype_adapter.cpp)
attention_tap_init(n_layer, n_head, n_ctx_max);  // ~60MB GPU for 48×32×10k
brightness_init(n_ctx_max);                       // ~40KB GPU for 10k tokens
```

---

## Compilation

```bash
make LLAMA_CUBLAS=1 -j8
```

New files (`brightness.cu`) are picked up automatically by the Makefile's wildcard rule.

---

## Performance

The tap and brightness engine add minimal overhead:
- **cudaMemcpyAsync**: Device-to-device on same stream, overlaps with next layer's compute
- **brightness_update_kernel**: Parallel across all tokens, ~0.1ms for 10k context
- **No CPU blocking**: Brightness stays on GPU until explicitly requested

Target: **<1% inference overhead** (not yet benchmarked rigorously)

---

## Visualization

Test script `test_brightness_viz.py` generates a BMP visualization:

```bash
python3 test_brightness_viz.py
# Outputs: /tmp/plato_brightness.bmp (256x100)
```

The BMP encodes brightness directly as 24-bit RGB:
- Each pixel column = one token
- RGB value = brightness value decomposed into channels
- White (255,255,255) = max brightness
- Decay shows in blue channel first, then green, then red

---

## Integration with Halo Weave

Halo Weave consumes brightness data for context culling:

1. After each generation, brightness texture reflects cumulative importance
2. When `context_len > max_context`, find lowest-brightness sentence
3. Cull that sentence from context
4. Semantic resurrection can later restore culled content if re-referenced

See `../Halo_Weave/halo_weave/CLAUDE.md` for culling strategy details.

---

## Status

**Last Updated**: 2026-01-15

**Completed:**
- [x] Attention tap - captures all 48 layers during kernel execution
- [x] Brightness engine - GPU-side magnitude-weighted voting
- [x] 24-bit color encoding - brightness value IS the RGB int (16.7M levels)
- [x] Sink detection - finds attention sink on first token
- [x] C API - `get_brightness()` returns texture data
- [x] REST endpoint - `/api/extra/brightness`
- [x] ODR fix - wrapper functions route through CUDA-compiled TU
- [x] Visualization - `test_brightness_viz.py` outputs BMP
- [x] Initialization wiring - tap and brightness init during model load

**TODO:**
- [ ] Async readback with PBO for non-blocking CPU access
- [ ] Benchmark performance impact
- [ ] Integration tests with Halo Weave
- [ ] WebGL streaming endpoint for real-time visualization

---

## Key Files Quick Reference

| File | Purpose |
|------|---------|
| `ggml/src/ggml-cuda/softmax.cu` | Attention tap side-channel |
| `ggml/src/ggml-cuda/brightness.cu` | Brightness CUDA kernels (24-bit encoding) |
| `gpttype_adapter.cpp:60-90` | Extern declarations, init calls |
| `gpttype_adapter.cpp:485-500` | Brightness wrapper functions (ODR fix) |
| `gpttype_adapter.cpp:420-480` | `extract_pending_attention_data()` |
| `src/llama-context.cpp:791-807` | Tap reset + extraction hook |
| `expose.h:170-176` | `brightness_outputs` struct |
| `expose.cpp:23-26` | Wrapper extern declarations |
| `expose.cpp:293-314` | `get_brightness()` implementation |
| `koboldcpp.py:3780` | REST endpoint `/api/extra/brightness` |
| `test_brightness_viz.py` | BMP visualization script |

---

## Debugging

Enable debug output by checking server logs for:
```
[ATTN_TAP] Initialized: 48 layers, 32 heads, 4224 max_ctx (24.75 MB GPU)
[BRIGHTNESS] Initialized: max_ctx=4224 (0.02 MB GPU)
[BRIGHTNESS] Detected sink at position 0 (attention=0.0625)
[ATTN_TAP] Extraction #N: 48 layers, 32 heads, 256 ctx
[ATTN_TAP] Layer 0 data (first 5): [0.3932, 0.0498, 0.5571, ...]
[ATTN_TAP] Layer 47 data (first 5): [0.8245, 0.0424, 0.1331, ...]
```

Different data between Layer 0 and Layer 47 confirms the tap is working (no aliasing).

### Common Issues

**Brightness returns `valid=false, ctx_len=0`:**
- Check that generation has occurred (brightness populates on first token generation)
- Verify ODR fix: wrapper functions must route through `gpttype_adapter_cublas.o`

**All tokens same brightness:**
- Normal for first few generations - variance builds over time
- Check threshold calculation if no differentiation after many generations
