# CLAUDE.md - KoboldCpp Async Attention Extraction

## Project Overview

This is a modified fork of KoboldCpp that extracts attention weights during text generation with minimal performance impact. The system uses an async VRAM→VRAM→CPU pipeline to avoid blocking the generation loop.

**Performance**: ~62 tokens/sec on RTX 4090 with Qwen2.5-VL-7B-Instruct-Q8_0 (768 context)
**Overhead**: <1ms per token (vs. 15-30ms with blocking copy)

---

## The Problem

### Context Length Management via Brightness-Based Culling

**Goal**: Delete unimportant tokens from context while preserving important information.

Traditional approaches fail:
- **FIFO/Sliding window**: Deletes oldest tokens regardless of importance (loses critical early context)
- **Summarization**: Model rewords everything, losing direct quotes and precise data

**Solution**: Brightness-based culling using attention scores.

The model already computes attention during forward pass. These scores reveal which tokens the model considers important. By tracking attention over time and giving each token a "brightness" score (based on peak attention received), we can intelligently prune low-brightness chunks while preserving what the model actually uses.

### Technical Challenge: Extracting Attention from llama.cpp

llama.cpp (the inference engine KoboldCpp uses) doesn't expose attention weights by default. We need to:
1. Intercept `kq_soft_max` tensors during graph execution (these contain attention weights)
2. Copy attention data from VRAM to CPU without blocking generation
3. Pair attention with generated tokens for downstream processing (Halo Weave)

**Previous approach (blocking)**: ~15-30ms overhead per token
- 28 GPU→CPU copies over PCIe (one per layer)
- 56 malloc/free operations per token
- Blocking generation until all copies complete

**New approach (async)**: <1ms overhead per token
- Single fast VRAM→VRAM copy (~0.5-1ms, blocking)
- Async VRAM→CPU copy (happens in parallel with next token)
- Background worker transposes and packages data

---

## Architecture: VRAM→VRAM→CPU Async Pipeline

### The Flow

```
Token N generates
  ↓
GPU compute completes (attention tensor in VRAM)
  ↓
ggml_backend_sched_synchronize() ← Block until GPU done (unavoidable)
  ↓
┌─────────────────────────────────────────────────────────────────┐
│ extract_pending_attention_data() [FAST PATH - Main Thread]     │
│                                                                 │
│ STEP 1: cudaMemcpy Device→Device (~0.5-1ms BLOCKING)          │
│         Copy from attention tensor → safe VRAM buffer           │
│         (Must be fast because main thread is blocked here)      │
│                                                                 │
│ STEP 2: cudaMemcpyAsync Device→Host (QUEUED, returns now!)    │
│         Start async copy: safe VRAM buffer → pinned CPU RAM    │
│         (DMA happens in background, ~10-20ms total)             │
│                                                                 │
│ STEP 3: Queue work for background thread                       │
│         Increment token counter, notify worker                  │
│                                                                 │
│ RETURN IMMEDIATELY (~1ms total blocking time)                  │
└─────────────────────────────────────────────────────────────────┘
  ↓
Token N+1 generation starts (NON-BLOCKING!)
  |
  |... meanwhile in background worker thread ...
  ↓
┌─────────────────────────────────────────────────────────────────┐
│ async_copy_worker() [SLOW PATH - Background Thread]            │
│                                                                 │
│ WAIT: cudaStreamSynchronize() ← Wait for async DMA (~10-20ms)  │
│                                                                 │
│ TRANSPOSE: [seq_len, n_heads] → [n_heads, seq_len]            │
│            Loop over heads and sequence positions               │
│                                                                 │
│ STORE: Copy to g_attention.buffer for API retrieval            │
│        Reset + append_layer() with transposed data             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Why VRAM→VRAM First?

**The Race Condition**: ggml's memory allocator reuses buffers. All 28 attention layers use the same VRAM buffer (aliased). By the time graph execution completes, only layer 27's data remains in the shared buffer (all previous layers were overwritten).

**The Fix**: Copy from the aliased buffer to a *safe* VRAM buffer that ggml will never touch. This copy must be:
- **Fast** (VRAM→VRAM is ~20x faster than VRAM→CPU)
- **Synchronous** (must complete before next token's graph execution starts)
- **Small** (only one layer: 28 heads × 768 seq_len × 4 bytes = ~84KB per token)

Once in the safe buffer, we can take our time copying to CPU asynchronously.

---

## Implementation Details

### Key Files Modified

**gpttype_adapter.cpp** (main implementation):
- CUDA forward declarations (search for `#ifdef GGML_USE_CUDA`, near top of file)
- `AsyncAttentionState` structure + globals (search for `struct AsyncAttentionState`)
- Init/cleanup functions (search for `init_async_attention` and `cleanup_async_attention`)
- Background worker thread (search for `async_copy_worker`)
- Async extraction (search for `extract_pending_attention_data`)
- Init call at model load (search for `init_async_attention(device_id`)
- Cleanup happens automatically during re-initialization inside `init_async_attention()`

**src/llama-context.cpp** (unchanged, but important context):
- GPU sync happens (search for `ggml_backend_sched_synchronize()`)
- Hook point (search for `extract_pending_attention_data()` call)

### Memory Layout

**VRAM Buffers**:
```
g_async_attn.vram_safe_buffer:
  Size: n_heads × max_seq_len × sizeof(float)
  Example: 28 × 30000 × 4 = 3.36 MB (negligible)
  Purpose: Safe storage between GPU sync and async copy
```

**CPU Buffers**:
```
g_async_attn.cpu_staging_buffer:
  Size: n_heads × max_seq_len × sizeof(float)
  Example: 28 × 30000 × 4 = 3.36 MB
  Type: Pinned memory (cudaMallocHost) for faster DMA
  Purpose: Async DMA destination

g_attention.buffer:
  Size: n_heads × max_seq_len × n_layers × sizeof(float)
  Example: 28 × 30000 × 28 × 4 = 94 MB
  Purpose: Final storage for API retrieval (pre-allocated, reused)
```

### Tensor Aliasing in llama.cpp

**The Problem**:
```
Graph Construction:
  Layer 0: kq_soft_max tensor created → callback fires, pointer saved
  Layer 1: kq_soft_max tensor created → callback fires, pointer saved
  ...
  Layer 27: kq_soft_max tensor created → callback fires, pointer saved

Memory Allocation:
  ggml_backend_sched_alloc_graph() assigns buffers
  Sees: Layer 0 needs N bytes → assign buffer[A]
  Sees: Layer 1 needs N bytes, Layer 0 done → REUSE buffer[A]!
  ...
  Result: All 28 pointers → SAME physical address

Graph Execution:
  Layer 0: compute attention → write to buffer[A]
  Layer 0: multiply with V → read from buffer[A] → done
  Layer 1: compute attention → write to buffer[A] (OVERWRITES Layer 0!)
  ...
  Layer 27: compute attention → write to buffer[A]
  Final state: buffer[A] contains ONLY Layer 27's data

Our Hook (after execution):
  Read all 28 pointers → all point to buffer[A] → get 28 copies of Layer 27
```

**Why We Only Extract Layer 27**:
- Due to aliasing, only the last layer's data remains
- Layer 27 IS actionable - it's the final attention pattern before output
- Extracting one layer is faster anyway (1/28th the work)

### Thread Safety

**Coordination**:
```cpp
g_async_attn.token_counter (atomic<int>):
  - Incremented atomically in extract_pending_attention_data()
  - Used to tag which token this attention belongs to
  - Prevents race conditions between generation and worker threads

g_async_attn.copy_queue (protected by mutex):
  - Main thread pushes: {stream, n_heads, seq_len, token_counter}
  - Worker thread pops and processes
  - Condition variable notifies worker of new work

g_attention.buffer (NO protection needed):
  - Only accessed by worker thread (single reader/writer)
  - API retrieves data after worker completes
```

---

## Performance Characteristics

### Timing Breakdown (per token)

**Main Thread (Blocking)**:
- GPU compute: ~10-20ms (unavoidable, model inference)
- GPU sync: <1ms (unavoidable, wait for kernels to finish)
- VRAM→VRAM copy: ~0.5-1ms (our only blocking overhead!)
- Queue work: <0.1ms
- **Total blocking: ~11-21ms** (most of which is model inference, not our code)

**Background Thread (Non-blocking)**:
- Wait for async DMA: ~10-20ms (happens in parallel with Token N+1 generation)
- Transpose: ~1-2ms
- Store: ~0.5ms
- **Total background: ~12-23ms** (doesn't block generation!)

**Net Impact**: <1ms added to generation time (just the VRAM→VRAM copy)

### Memory Overhead

- **VRAM**: ~3.36 MB (safe buffer) - negligible for modern GPUs
- **RAM**: ~3.36 MB (pinned staging) + ~94 MB (API buffer)
- **Total**: ~100 MB (0.4% of 24GB GPU memory)

### Throughput

**Measured**:
- **62 tokens/sec** on RTX 4090 with Qwen2.5-VL-7B-Instruct-Q8_0 (768 context)
- **~800KB per token** (base64-encoded attention in JSON)
- **~50 MB/sec** attention data throughput

**Scaling**:
- At 30,000 context length: ~3.2 MB per token → ~200 MB/sec
- Background worker keeps up because async DMA is parallel with generation

---

## API Usage

### Enabling Attention Extraction

Attention is extracted **unconditionally** for all tokens during generation. No opt-in required.

### Retrieving Attention Data

**C API** (expose.cpp):
```c
attention_outputs get_token_attention(int token_idx);

typedef struct {
    const float * data;        // [n_layers, n_heads, seq_len]
    int n_layers;              // Number of layers (1 for async extraction)
    int n_heads;               // Number of attention heads (e.g., 28)
    int seq_len;               // Context length at this token
    bool valid;                // True if data is available
} attention_outputs;
```

**Python** (koboldcpp.py):
```python
attn = handle.get_token_attention(token_idx)
if attn.valid:
    # Convert C pointer to numpy array
    import numpy as np
    attention_array = np.ctypeslib.as_array(
        attn.data,
        shape=(attn.n_layers, attn.n_heads, attn.seq_len)
    )

    # Shape: [1, 28, 768] for single-layer extraction
    # Data: Post-softmax attention probabilities (range: [0, 1], sums to ~1.0 per head)
```

**REST API** (koboldcpp.py, search for `output_attentions`):
```json
{
  "type": "token",
  "token": {
    "token_id": 4330,
    "text": "ato"
  },
  "request_id": "example-request-123",
  "attention": {
    "format": "per_layer",
    "shape": [1, 28, 768],
    "context_length": 768,
    "encoding": "base64",
    "dtype": "float32",
    "data": "sNCcP7KRvj..." // base64-encoded float array
  }
}
```

---

## Compilation

**CUDA Build** (required for async extraction):
```bash
make LLAMA_CUBLAS=1 -j8
```

**Output**: `koboldcpp_cublas.so` (~211 MB)

**CPU Build** (fallback to blocking extraction):
```bash
make -j8
```

**Output**: `koboldcpp_default.so` (~11 MB)

### Forward Declarations

We use CUDA forward declarations (lines 31-50 in gpttype_adapter.cpp) instead of including CUDA headers directly. This avoids compilation conflicts between C++ and CUDA code.

```cpp
#ifdef GGML_USE_CUDA
typedef struct cudaStream_st* cudaStream_t;
typedef int cudaError_t;
extern "C" {
    cudaError_t cudaMalloc(void **devPtr, size_t size);
    cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, cudaError_t kind);
    cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count, cudaError_t kind, cudaStream_t stream);
    // ... etc
}
#endif
```

If CUDA headers are not available at compile time, the CUDA-specific code is disabled and the system falls back to blocking extraction.

---

## Debugging

### Log Messages

**At Model Load**:
```
Initializing attention capture buffer (max: 28 heads, 30000 ctx, 28 layers)...
Attention buffer allocated: 94.0 MB
Initializing async attention extraction (device 0, 28 heads, 30000 max seq_len)...
  Allocated 3.4 MB VRAM + 3.4 MB pinned RAM
Async attention extraction initialized successfully
Async attention worker thread started
```

**During Generation** (per token):
```
DEBUG append_layer: heads=28, len=768, count=21504, buffer_used=0, capacity=23620352
DEBUG append_layer DONE: buffer_used=21504, n_layers_captured=1
DEBUG: Token ID=4330 'ato' paired with attention [1, 28, 768]
```

**At Cleanup**:
```
Cleaning up async attention extraction...
  Stats: 517 copies queued, 517 completed
Async attention cleanup complete
Async attention worker thread exiting
```

### Common Issues

**Issue**: `buffer_used=0` in logs
- **Status**: Normal! Buffer is reset before each append (search for `g_attention.reset()`)
- **Verify**: Check `buffer_used` in "DONE" message (should be 21504 for 28 heads × 768 seq_len)

**Issue**: Fallback to blocking extraction
- **Symptom**: Log message "DEBUG: Using fallback blocking extraction"
- **Cause**: Async system not initialized (CUDA not available or init failed)
- **Fix**: Check model load logs for initialization errors

**Issue**: Worker thread not keeping up
- **Symptom**: `copies_queued` >> `copies_completed` in cleanup stats
- **Cause**: Generation faster than background processing (rare)
- **Fix**: Acceptable - queue will drain after generation completes

---

## Integration with Downstream Clients

### Primary Consumer: Halo Weave

**Halo Weave** is a pure frontend application that uses the attention data for brightness-based context pruning with semantic resurrection. See `../Halo_Weave/halo_weave/CLAUDE.md` for complete details on how it uses the attention data.

### What KoboldCpp Provides

**Current Output Format (V2 - Server-Side Aggregation)**:

Via SSE (Server-Sent Events) streaming endpoint `/api/extra/generate/stream`:
```json
{
  "type": "token",
  "token": {
    "token_id": 4330,
    "text": "ato"
  },
  "attention": {
    "format": "aggregated",
    "shape": [768],                 // [seq_len] - pre-aggregated across heads
    "context_length": 768,
    "encoding": "base64",
    "dtype": "float32",
    "data": "sNCcP7..."             // ~3KB base64-encoded (28x smaller!)
  }
}
```

**Key characteristics**:
- **Single layer**: Only layer 27 extracted (due to tensor aliasing)
- **Pre-aggregated**: Mean across 28 attention heads computed on server
- **Post-softmax probabilities**: Range [0, 1], represents average attention weight
- **Indexed to input**: `attention[i]` corresponds to `input_ids[i]`, not original conversation positions
- **Per-token extraction**: Attention captured for each generated token
- **Bandwidth optimized**: 3KB per token vs 86KB (28x reduction)

**Performance impact** (measured on Qwen2.5-VL-7B-Instruct, 700 token generation):
- Old format: 33.2s wall clock (2.3x slower than generation due to TCP send buffer blocking)
- New format: 14.8s wall clock (1.06x generation time, effectively real-time)

### Attention Indexing

**Important**: Attention arrays are indexed by the **input token array**, not by conversation position IDs.

If client sends pruned context with gaps:
```
Client maintains: positions [0, 1, 4, 5, 8, ...]  (gaps from pruning)
Client sends: input_ids = [9707, 11, 5234, 1532, ...]  (sequential array)

KoboldCpp returns:
attention[0] → input_ids[0] → client position 0
attention[1] → input_ids[1] → client position 1
attention[2] → input_ids[2] → client position 4  (gap!)
attention[3] → input_ids[3] → client position 5
```

The client is responsible for maintaining the mapping between array indices and conversation positions.

### Historical: V1 Format (Deprecated)

The original format sent raw per-head attention tensors:

```json
{
  "attention": {
    "format": "per_layer",
    "shape": [1, 28, 768],
    "data": "..."  // 86KB base64-encoded
  }
}
```

This format caused severe TCP send buffer blocking (90-100ms stalls every ~50 tokens) because the client couldn't read fast enough to keep up with generation. Replaced with server-side aggregation in V2.

---

## Future Improvements

### Potential Optimizations

1. **Multi-layer extraction** - If tensor aliasing is fixed upstream, extract all 28 layers
2. **Compression** - Send only top-K attention values per head (sparse format)
3. **GPU-side transposition** - Use CUDA kernel instead of CPU loop
4. **Pooled temp buffers** - Reuse transposition buffer instead of malloc/free

### Upstream Changes Needed

**llama.cpp tensor aliasing**:
- Current: ggml allocator reuses buffers aggressively (all layers share one buffer)
- Needed: Flag attention tensors as non-aliasable OR expose cb_eval callback
- Impact: Would allow extracting all 28 layers instead of just layer 27

**Flash Attention compatibility**:
- Current: Flash attention fuses attention computation (no intermediate tensors)
- Needed: API to extract attention from fused kernels
- Impact: Would enable attention extraction with flash attention enabled

---

## Status

**Last Updated**: 2025-12-20 (Async pipeline implementation)

**Current State**: ✅ **PRODUCTION READY**
- Compilation: ✅ CUDA + CPU builds
- Extraction: ✅ Async VRAM→VRAM→CPU pipeline
- Performance: ✅ 62 tokens/sec on RTX 4090 (<1ms overhead)
- Data Format: ✅ Post-softmax attention probabilities, normalized and interpretable
- API: ✅ C/Python/REST APIs functional
- Integration: ✅ Ready for Halo Weave

**Testing**:
- ✅ Qwen2.5-VL-7B-Instruct-Q8_0 @ 768 context
- ✅ Single-layer extraction (layer 27)
- ✅ Streaming generation
- ✅ Token-attention pairing
- [ ] Antislop sampling compatibility (assumed OK, not explicitly tested)
- [ ] Multi-thousand context lengths
- [ ] 14B+ models

---

## Contact

For questions about this implementation:
- Original brightness-based culling concept: Evans
- Async pipeline implementation: Claude (Anthropic)
- KoboldCpp upstream: https://github.com/LostRuins/koboldcpp
