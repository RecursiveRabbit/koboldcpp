# CLAUDE.md - KoboldCpp Attention Extraction

## Project Overview

This is a fork of KoboldCpp modified to extract attention weights during text generation. The goal is to expose raw attention tensors through the API for use with Halo Weave (attention visualization tool).

**Fork**: https://github.com/RecursiveRabbit/koboldcpp
**Upstream**: https://github.com/LostRuins/koboldcpp

## Motivation

The Positronic Brain/Halo Weave project needs attention weights to visualize which tokens the model attends to during generation. Using HuggingFace Transformers with `output_attentions=True` works but:
- Uses ~14GB VRAM for 7B models (bfloat16)
- Incompatible with quantization (causes NaN/Inf in sampling)
- 14B+ models don't fit on 24GB GPU

**Solution**: Use koboldcpp with GGUF quantization + extract attention from llama.cpp internals.

## Implementation (2025-11-15)

### Core Design

**Attention capture happens at the callback level:**
```
User sets output_attentions=True
  ↓
g_attention.enabled = true
  ↓
llama_decode() runs
  ↓
For each layer: build_attn_mha() → ggml_soft_max_ext() → cb("kq_soft_max")
  ↓
attention_capture_callback() fires → copies GPU tensor to CPU buffer
  ↓
After generation: output.attention_weights points to static buffer
```

**Static memory architecture:**
- Pre-allocate 128 MB buffer at model load (once)
- Reuse buffer across all requests (overwrite each generation)
- Shape: `[n_layers, n_heads, seq_len]`
- Example: 28 layers × 28 heads × 4096 ctx = **~12 MB per generation**

### Files Modified

#### 1. `gpttype_adapter.cpp` (lines 152-266, 2697-2709, 3353-3354, 4767-4778)

**Added attention capture system:**
```cpp
struct AttentionCapture {
    float* buffer;              // Pre-allocated static buffer
    size_t buffer_capacity;     // Max floats we can store
    size_t buffer_used;         // Floats written this request
    int n_layers_captured;      // Layers captured so far
    int n_heads;                // Heads per layer
    int seq_len;                // Context length
    bool enabled;               // Capture flag for this request
};

static AttentionCapture g_attention;

void attention_capture_callback(const llama_ubatch & ubatch,
                                ggml_tensor * cur,
                                const char * name,
                                int il) {
    if (!g_attention.enabled || strcmp(name, "kq_soft_max") != 0) {
        return;
    }

    // Extract attention tensor from GPU to CPU
    // Transpose from [seq_len, n_heads] to [n_heads, seq_len]
    // Append to global buffer
}
```

**Initialization at model load:**
```cpp
// After model warmup (line 2697)
int n_layer = llama_n_layer(llamamodel);
int n_head = llama_n_head(llamamodel);
int n_ctx_max = llama_n_ctx(llama_ctx_v4);
g_attention.init(n_head, n_ctx_max, n_layer);
```

**Enable/disable per request:**
```cpp
// At generation start (line 3353)
g_attention.reset();
g_attention.enabled = inputs.output_attentions;
```

**Copy to output:**
```cpp
// Before return (line 4767)
if (g_attention.enabled && g_attention.buffer_used > 0) {
    output.attention_weights = g_attention.buffer;
    output.attention_n_layers = g_attention.n_layers_captured;
    output.attention_n_heads = g_attention.n_heads;
    output.attention_seq_len = g_attention.seq_len;
}
```

#### 2. `expose.h` (lines 134, 143-146)

**Added API fields:**
```cpp
struct generation_inputs {
    ...
    const bool output_attentions = false;  // NEW
};

struct generation_outputs {
    ...
    const float * attention_weights = nullptr;  // [n_layers, n_heads, seq_len]
    int attention_n_layers = 0;
    int attention_n_heads = 0;
    int attention_seq_len = 0;
};
```

#### 3. `src/llama-graph.cpp` (lines 16-20, 603-609)

**Hooked callback into graph execution:**
```cpp
// External declaration at top
extern void attention_capture_callback(const llama_ubatch & ubatch,
                                      ggml_tensor * cur,
                                      const char * name,
                                      int il);

// Modified cb() function (line 603)
void llm_graph_context::cb(ggml_tensor * cur, const char * name, int il) const {
    if (cb_func) {
        cb_func(ubatch, cur, name, il);
    }
    // Also call our global attention capture callback
    attention_capture_callback(ubatch, cur, name, il);
}
```

## How It Works

### Tensor Extraction

**The `kq_soft_max` tensor contains attention weights after softmax:**
- Shape during generation: `[seq_len_k, n_heads, 1, 1]`
- This is the new token's attention to all previous tokens in KV cache
- We transpose to `[n_heads, seq_len_k]` for easier processing

**GPU → CPU copy:**
```cpp
size_t tensor_bytes = seq_len_k * n_heads * sizeof(float);
float* temp = (float*)malloc(tensor_bytes);
ggml_backend_tensor_get(cur, temp, 0, tensor_bytes);  // GPU → CPU
```

**Memory layout:**
```
Buffer: [n_layers, n_heads, seq_len]

Layer 0: [head_0: [attn_0, attn_1, ..., attn_seq_len],
          head_1: [...],
          ...
          head_n: [...]]
Layer 1: [...]
...
Layer n: [...]
```

### API Usage (Future - koboldcpp.py integration)

```python
# Python API wrapper (NOT YET IMPLEMENTED)
from koboldcpp import generate

result = generate(
    prompt="Hello, world!",
    max_length=50,
    output_attentions=True  # Enable attention extraction
)

print(f"Generated text: {result.text}")
print(f"Attention shape: {result.attention_n_layers} × {result.attention_n_heads} × {result.attention_seq_len}")
print(f"Attention data: {result.attention_weights}")  # NumPy array via ctypes
```

## Compilation

**Requirements:**
- CUDA Toolkit (for CUDA build)
- GCC/G++ 13+
- Make

**Build command:**
```bash
cd /home/evans/Coding_Projects/koboldcpp
make LLAMA_CUBLAS=1 -j8  # CUDA build with 8 parallel jobs
```

**Output:**
- `koboldcpp_cublas.so` - Shared library with attention extraction

**Known warnings (harmless):**
- Format string warnings in `gpttype_adapter.cpp` (printf with size_t)
- These don't affect functionality

## Testing (TODO)

1. **Compile koboldcpp** with CUDA support
2. **Load a test model** (e.g., Qwen2.5-7B-Instruct Q4_K_M)
3. **Generate with attention** using `output_attentions=True`
4. **Verify output**:
   - `attention_weights` pointer is non-null
   - Shape matches `[n_layers, n_heads, seq_len]`
   - Values are in range `[0, 1]` (normalized attention)
5. **Update `koboldcpp.py`** to expose attention through REST/WebSocket API

## Integration with Halo Weave

**Next steps:**
1. Finish koboldcpp.py API exposure (REST + WebSocket)
2. Update Halo Weave backend to use koboldcpp instead of Transformers
3. Verify attention accumulation still works correctly
4. Test with 14B+ models (should fit with Q4_K_M quantization)

**Expected benefits:**
- 14B models fit on 24GB GPU (~7GB VRAM with Q4_K_M)
- Faster generation (optimized inference)
- Still get raw attention weights for visualization

## Known Limitations

1. **Flash attention disabled**: Cannot extract attention from fused flash attention kernels (requires fallback path)
2. **Single-token generation only**: Callback assumes `seq_len_q = 1` during generation
3. **Memory not pinned**: CPU buffer, no zero-copy from GPU
4. **No streaming attention**: Must wait until generation completes to get full attention data

## Future Improvements

- **Streaming attention**: Send attention data per-token via WebSocket
- **GPU buffer reuse**: Keep attention on GPU, copy only when requested
- **Multi-batch support**: Handle `seq_len_q > 1` for prompt processing
- **Attention compression**: Send only top-K attention values per head

---

**Last Updated**: 2025-11-15
**Status**: ✅ Code implemented, ⏳ Compilation in progress, ❌ Not yet tested
**Tested On**: (TBD)
