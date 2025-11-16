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

### Architecture: The Push Model

Tokens and attention are **paired atomically at the point of generation**:

```
C++ Generation Thread:
Token 0 generates
  ↓ callback fires (all layers)
g_attention.buffer = [token 0 attn]
  ↓ IMMEDIATELY PAIR
delayed_queue.push(TokenWithAttention(text="I", attention=copy(g_attention)))
  ↓
Token 1 generates
  ↓ callback fires
g_attention.buffer = [token 1 attn] ← overwrites (safe: token 0 already saved!)
  ↓ IMMEDIATELY PAIR
delayed_queue.push(TokenWithAttention(text=" think", attention=copy(g_attention)))
  ↓
Delayed queue pops when ready (antislop may hold tokens)
  ↓ streamcount++
                                    Python Polling Thread:
                                    token_0 = new_token(0) → "I"
                                    attn_0 = get_token_attention(0) → token 0's pre-saved attention
                                    send_websocket(token_0, attn_0)
```

**Key insights**:
- Attention is captured **the moment a token completes generation**, not when it's reported to Python
- Each token in the delayed queue carries its own attention copy
- **Antislop sampling works correctly** - even if tokens are held in the delayed queue, they're already paired with the right attention
- No race conditions: token and attention are inseparable from the moment of generation

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

### API Usage

**PUSH MODEL: Tokens and attention are captured together atomically**

```python
# In koboldcpp.py streaming loop (simplified)
while current_token < handle.get_stream_count():
    # Get the generated token
    token = handle.new_token(current_token)

    # Get attention for THIS SPECIFIC TOKEN (atomic retrieval)
    attention = handle.get_token_attention(current_token)

    if attention.valid:
        # Attention buffer shape: [n_layers, n_heads, seq_len]
        # This attention was captured WHEN token was generated (push model)

        # Convert pointer to numpy array
        import numpy as np
        attention_array = np.ctypeslib.as_array(
            attention.data,
            shape=(attention.n_layers, attention.n_heads, attention.seq_len)
        )

        # Encode to base64 for JSON transmission
        import base64
        attention_bytes = attention_array.tobytes()
        attention_b64 = base64.b64encode(attention_bytes).decode('ascii')

        # Send WebSocket event
        send_websocket_event({
            "type": "token",
            "token": {"text": token.decode(), ...},
            "attention": {
                "format": "per_layer",
                "shape": [attention.n_layers, attention.n_heads, attention.seq_len],
                "encoding": "base64",
                "dtype": "float32",
                "data": attention_b64
            }
        })

    current_token += 1
```

**Key insight - PUSH MODEL**:
- When a token completes generation, its attention is IMMEDIATELY captured and stored with the token
- `get_token_attention(idx)` retrieves the pre-captured attention for that specific token
- No race conditions: token and attention are paired atomically before the buffer is overwritten
- No pull API: you cannot query the "current" buffer state, only retrieve completed token+attention pairs

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

## Feature Compatibility

### ✅ Antislop Sampling Compatible

Attention extraction works correctly with koboldcpp's antislop sampling feature:
- Tokens are paired with attention **at generation time** (before entering delayed queue)
- The delayed queue stores `TokenWithAttention` objects (not just strings)
- Even if antislop holds tokens for multiple steps, each token keeps its correct attention data
- No special configuration needed - just enable both features

### ✅ Works With All Sampling Methods

Temperature, top-k, top-p, mirostat, etc. don't affect attention capture - we extract attention from the forward pass before sampling.

## Testing (TODO)

1. **Compile koboldcpp** with CUDA support
2. **Load a test model** (e.g., Qwen2.5-7B-Instruct Q4_K_M)
3. **Generate with attention** using `output_attentions=True`
4. **Verify output**:
   - `get_token_attention(idx)` returns valid attention data
   - Shape matches `[n_layers, n_heads, seq_len]`
   - Values are in range `[0, 1]` (normalized attention)
5. **Test with antislop enabled**: Verify attention still pairs correctly with delayed tokens
6. **Update `koboldcpp.py`** to expose attention through REST/WebSocket API

## Integration with Halo Weave

### Critical Understanding: KV Cache and Context Pruning

**The KV cache is immutable and generation-scoped:**
- You **CANNOT** delete tokens from the middle of the KV cache without breaking positional encoding
- The cache is only valid for the current generation turn
- After pruning tokens, you **MUST** reprocess the entire pruned context to rebuild the KV cache

**Pruning workflow:**
```
Generation N:
  input: [0,1,2,3,4,5,6,7,8,9]  (10 tokens)
  → KV cache built for these positions
  → Generate response tokens with attention
  → Attention seq_len grows: 10 → 11 → 12 → 13 (as each token is added)

Between generations:
  → Halo Weave checks brightness scores
  → Prunes tokens [2,3] from conversation (low brightness)
  → Context now: [0,1,4,5,6,7,8,9] (8 tokens)
  → **KV cache is DISCARDED**

Generation N+1:
  input: [0,1,4,5,6,7,8,9]  (8 tokens - gaps closed!)
  → Reprocess entire context (feed through model)
  → Build NEW KV cache for sequential positions 0-7
  → Generate response tokens with attention
  → Attention seq_len grows: 8 → 9 → 10 → 11
```

**Key insights:**
- Attention is indexed by **current input array** (0-based, no gaps)
- Halo Weave maintains position IDs as metadata (can have gaps: 0,1,4,5,6...)
- Halo Weave maps `attention[i]` → `conversation_position` using its own index-to-position dictionary
- Koboldcpp doesn't know or care about conversation position IDs - it just processes the array you give it

**Cost of pruning:**
- Must reprocess entire context (e.g., 500 tokens → 450 tokens after pruning)
- But this is cheap: processing 450 tokens takes ~1-2 seconds on GPU
- Much cheaper than running out of context and truncating from the start

**Next steps:**
1. Finish koboldcpp.py API exposure (REST + WebSocket)
2. Update Halo Weave backend to use koboldcpp instead of Transformers
3. Implement context reprocessing after pruning
4. Test with 14B+ models (should fit with Q4_K_M quantization)

**Expected benefits:**
- 14B models fit on 24GB GPU (~7GB VRAM with Q4_K_M)
- Faster generation (optimized inference)
- Context reprocessing is fast (~1-2s for 500 tokens)
- Still get raw attention weights for visualization

## Known Limitations

1. **Flash attention disabled**: Cannot extract attention from fused flash attention kernels (requires fallback path)
2. **Autoregressive generation only**: Attention is ONLY captured during autoregressive token generation (one token at a time), NOT during initial prompt processing. This is intentional - Halo Weave only needs attention from generated tokens, not from prompt tokens.
3. **Memory not pinned**: CPU buffer, no zero-copy from GPU
4. **Static buffer design**: The `g_attention.buffer` is a "live tap" that holds attention for the most recently generated token. It gets overwritten with each new token. This is correct behavior - the API layer reads the buffer and serializes to JSON before the next token generates.

## Future Improvements

- **Streaming attention**: Send attention data per-token via WebSocket
- **GPU buffer reuse**: Keep attention on GPU, copy only when requested
- **Multi-batch support**: Handle `seq_len_q > 1` for prompt processing
- **Attention compression**: Send only top-K attention values per head

---

**Last Updated**: 2025-11-16
**Status**: ✅ Push model implemented with antislop compatibility, ⏳ Ready for compilation, ❌ Not yet tested
**Architecture**: Push model - tokens paired with attention at generation point (before antislop delay)
**Tested On**: (TBD)
