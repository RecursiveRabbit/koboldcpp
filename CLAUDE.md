# CLAUDE.md - KoboldCpp Attention Extraction

## Project Overview

This is a modified fork of KoboldCpp that extracts attention weights during text generation. The system uses the same synchronous GPU→CPU transfer path as logits, ensuring reliable per-token attention data.

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
1. Intercept `kq_soft_max` tensors during graph construction (these contain attention weights)
2. Copy attention data from VRAM to CPU after graph execution
3. Pair attention with generated tokens for downstream processing (Halo Weave)

---

## Architecture: Synchronous Extraction via ggml Backend

### The Flow

```
Token N generates
  ↓
GPU compute completes (attention tensor in VRAM)
  ↓
ggml_backend_sched_synchronize() ← Block until GPU done (unavoidable)
  ↓
extract_pending_attention_data()
  │
  ├─ ggml_backend_tensor_get() ← Synchronous GPU→CPU copy (same as logits!)
  │
  ├─ Transpose [seq_len, n_heads] → [n_heads, seq_len]
  │
  └─ Store in g_attention buffer
  ↓
Token paired with attention data
  ↓
Next token generation starts
```

### Why Synchronous?

Previous async attempts failed because:
1. `cudaMemcpyAsync` on the default CUDA stream serializes with compute
2. Single shared buffers got overwritten before the background worker could process them
3. Result: all tokens received the same stale attention snapshot

The solution: use `ggml_backend_tensor_get()` - the same synchronous mechanism that reliably transfers logits. This ensures each token gets its own correct attention data.

---

## Implementation Details

### Key Files

**gpttype_adapter.cpp** (main implementation):
- `attention_capture_callback()` - Captures tensor pointers during graph construction
- `extract_pending_attention_data()` - Synchronous extraction after graph execution
- `g_attention` (AttentionCapture) - Stores current token's attention data
- `g_pending_attentions` - Vector of tensor pointers captured during graph build

**src/llama-context.cpp**:
- Hook point after GPU sync (calls `extract_pending_attention_data()`)

### Tensor Aliasing in llama.cpp

**The Problem**:
```
Graph Construction:
  Layer 0-27: kq_soft_max tensors created, pointers saved

Memory Allocation:
  ggml allocator reuses buffers aggressively
  Result: All 28 pointers → SAME physical VRAM address

Graph Execution:
  Each layer overwrites the shared buffer
  Final state: buffer contains ONLY Layer 27's data
```

**Why We Only Extract Layer 27**:
- Due to aliasing, only the last layer's data remains
- Layer 27 IS actionable - it's the final attention pattern before output
- Extracting one layer reduces overhead

### Code Structure

```cpp
// Callback during graph construction - saves tensor pointers
void attention_capture_callback(const llama_ubatch & ubatch,
                                ggml_tensor * cur,
                                const char * name,
                                int il) {
    if (strcmp(name, "kq_soft_max") != 0) return;
    if (cur->ne[1] != 1 || cur->ne[3] != 1) return;  // Single-token only
    g_pending_attentions.push_back({cur, il});
}

// Called after graph execution - extracts data synchronously
void extract_pending_attention_data() {
    if (g_pending_attentions.empty()) return;

    const auto & pending = g_pending_attentions.back();  // Layer 27
    ggml_tensor * cur = pending.tensor;

    // Synchronous GPU→CPU copy (same path as logits)
    ggml_backend_tensor_get(cur, tensor_data.data(), 0, tensor_bytes);

    // Transpose and store
    g_attention.append_layer(transposed.data(), n_heads, seq_len_k);
    g_pending_attentions.clear();
}
```

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
    int n_layers;              // Number of layers (1 - only layer 27)
    int n_heads;               // Number of attention heads (e.g., 28)
    int seq_len;               // Context length at this token
    bool valid;                // True if data is available
} attention_outputs;
```

**REST API** (SSE streaming):
```json
{
  "type": "token",
  "token": {
    "token_id": 4330,
    "text": "ato"
  },
  "attention": {
    "format": "aggregated",
    "shape": [768],
    "context_length": 768,
    "encoding": "base64",
    "dtype": "float32",
    "data": "sNCcP7..."
  }
}
```

**Key characteristics**:
- **Single layer**: Only layer 27 extracted (due to tensor aliasing)
- **Pre-aggregated**: Mean across attention heads computed on server
- **Post-softmax probabilities**: Range [0, 1]
- **Per-token extraction**: Attention captured for each generated token

---

## Compilation

```bash
make LLAMA_CUBLAS=1 -j8
```

No special CUDA forward declarations needed - extraction uses ggml's backend system.

---

## Integration with Downstream Clients

### Primary Consumer: Halo Weave

**Halo Weave** uses the attention data for brightness-based context pruning with semantic resurrection. See `../Halo_Weave/halo_weave/CLAUDE.md` for details.

### Attention Indexing

Attention arrays are indexed by the **input token array**, not by conversation position IDs.

```
Client sends: input_ids = [9707, 11, 5234, 1532, ...]

KoboldCpp returns:
attention[0] → input_ids[0]
attention[1] → input_ids[1]
...
```

The client maintains mapping between array indices and conversation positions.

---

## Future Improvements

1. **Multi-layer extraction** - If tensor aliasing is fixed upstream, extract all 28 layers
2. **Flash Attention compatibility** - Currently disabled; fused kernels don't expose intermediate tensors

---

## Status

**Last Updated**: 2026-01-02 (Simplified to synchronous extraction)

**Current State**:
- Extraction: Synchronous via ggml_backend_tensor_get
- Reliability: Each token gets correct, unique attention data
- API: C/Python/REST APIs functional

---

## Contact

For questions about this implementation:
- Original brightness-based culling concept: Evans
- KoboldCpp upstream: https://github.com/LostRuins/koboldcpp
