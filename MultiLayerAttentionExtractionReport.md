# Multi-Layer Attention Extraction Debug Report

## Goal
Extract attention data from ALL transformer layers (not just layer 27) for brightness-based context culling.

## Problem Statement
Currently, only layer 27's attention data is accessible after graph execution. All other layers show identical data due to ggml's aggressive buffer aliasing - the allocator reuses the same VRAM buffer for all 48 `kq_soft_max` tensors since their lifetimes don't overlap in the compute graph.

## Attempts Made

### Attempt 1: `ggml_set_output()` in Callback (Failed)
**Location**: `gpttype_adapter.cpp:attention_capture_callback()`

**Approach**: Call `ggml_set_output(cur)` during the graph construction callback to mark attention tensors as outputs.

**Result**: Flag was set (`flags=0x2`), but allocation had already been planned during `graph_reserve()`. Setting the flag after allocation planning has no effect.

**Evidence**:
```
[ATTN_DEBUG] Layer 0: tensor=0x307ce440, flags=0x2, OUTPUT_FLAG=1
```
Flag is set, but buffer aliasing persists.

---

### Attempt 2: `ggml_set_output()` During Tensor Creation (Failed)
**Location**: `src/llama-graph.cpp:1444`

**Approach**: Call `ggml_set_output(kq)` immediately after `ggml_soft_max_ext()` creates the tensor, BEFORE any allocation planning.

**Code Added**:
```cpp
kq = ggml_soft_max_ext(ctx0, kq, kq_mask, kq_scale, hparams.f_max_alibi_bias);
ggml_soft_max_add_sinks(kq, sinks);

// ATTENTION EXTRACTION: Mark selected layers as output
if (il == 0 || il == n_layer/2 || il == n_layer - 1) {
    ggml_set_output(kq);
}
```

**Result**: When marking ALL layers, VRAM explodes (20+ GB allocation failure). When marking only 3 layers (0, 24, 47), allocation succeeds but:
- Layers 0 and 24 STILL share the same buffer address
- Data is `[1.0, 0.0, 0.0, ...]` - uninitialized memory

**Evidence**:
```
[ATTN_DEBUG] Extract layer 0: buffer=0x468d20c0
[ATTN_DEBUG] Extract layer 24: buffer=0x468d20c0  <-- SAME BUFFER
[ATTN_DEBUG] Extract layer 47: buffer=0x468d2310  <-- Different
```

**Analysis**: `ggml_set_output()` prevents OTHER tensors from reusing this buffer, but does NOT prevent two OUTPUT tensors from sharing the same address if their lifetimes don't overlap in the compute graph.

---

### Attempt 3: Graph Lookup by Name (Failed)
**Location**: `gpttype_adapter.cpp:extract_pending_attention_data()`

**Approach**: Instead of storing tensor pointers during graph construction (which may come from `gf_res_reserve` - wrong graph), look up tensors by name from the EXECUTED graph using `ggml_graph_get_tensor(gf, "kq_soft_max-N")`.

**Result**: Same issue - correct tensors are found, but data is still `[1.0, 0.0, 0.0, ...]`.

**Evidence**:
```
[ATTN_DEBUG] Extract layer 0: tensor=0x44bb5400, name='kq_soft_max-0', buffer=0x468d20c0
[ATTN_DEBUG] Layer 0 data (first 5): [1.0000, 0.0000, 0.0000, 0.0000, 0.0000]
```

---

## Key Findings

### 1. Buffer Aliasing is Fundamental
ggml's allocator determines buffer reuse based on tensor LIFETIMES in the compute graph, not on OUTPUT flags. Two OUTPUT tensors can share a buffer if their lifetimes don't overlap:
- Layer 0's attention is consumed before layer 24 computes
- Allocator sees no conflict, reuses buffer
- By extraction time, only the last writer's data remains

### 2. The `[1.0, 0.0, 0.0, ...]` Pattern
This is `softmax([0, 0, 0, ...])` - the result of applying softmax to uninitialized/zeroed memory. This confirms we're reading stale or never-computed data.

### 3. Graph Reserve vs Inference Graphs
- `graph_reserve()` builds planning graphs with `gf_res_reserve`
- Actual inference uses `gf_res_prev`
- Tensor pointers captured during reserve point to wrong graph objects
- This was fixed by graph name lookup, but didn't solve the core aliasing issue

### 4. VRAM Impact of Preventing Aliasing
Marking all 48 layers as OUTPUT requires ~20GB additional VRAM because:
- Each attention tensor can no longer share buffers with ANY other tensor
- This cascades through the entire graph allocation

---

## What Would Actually Work

### Option A: Copy During Execution (Invasive)
Modify the softmax CUDA kernel to copy attention data to a separate buffer DURING computation, before the buffer gets overwritten by the next layer.

**Pros**: Gets actual computed data
**Cons**: Requires modifying ggml CUDA kernels, significant performance impact

### Option B: Disable All Buffer Reuse for Attention Tensors
Modify `ggml-alloc.c` to treat OUTPUT tensors as having infinite lifetime (never reusable), not just "protected from others".

**Pros**: Would work with existing tensor marking
**Cons**: Massive VRAM increase, may break other ggml assumptions

### Option C: Sequential Layer-by-Layer Extraction
After each layer computes (but before the next layer), extract that layer's attention. Would require hooks into the per-layer execution flow.

**Pros**: Each layer's data is captured before overwrite
**Cons**: ggml executes the entire graph as a unit - no per-layer hooks exist

### Option D: Use Flash Attention's Debug Mode (If Available)
Some Flash Attention implementations have debug modes that preserve intermediate attention matrices.

**Pros**: Designed for this use case
**Cons**: May not be available in ggml's flash attention implementation

### Option E: Aggregate at the Model Level
Instead of extracting raw attention, use the model's own representations (e.g., hidden states) which ARE preserved per-layer, and derive importance signals from those.

**Pros**: Hidden states are already extractable (working in this codebase)
**Cons**: Less direct signal than raw attention scores

---

## Current Code State

### Files Modified
1. `src/llama-graph.cpp:1442-1451` - `ggml_set_output()` on selected layers
2. `src/llama-context.cpp:805-807` - Pass executed graph to extraction
3. `gpttype_adapter.cpp:405-475` - Graph name lookup extraction

### Debug Output Available
- Tensor addresses, names, flags, shapes
- Buffer addresses (to detect aliasing)
- Raw data samples
- Graph build vs extraction counts

---

## Questions for Engineering

1. **Is there a way to prevent two OUTPUT tensors from sharing the same buffer allocation?** The current behavior allows OUTPUT tensors to alias if lifetimes don't overlap.

2. **Can ggml be modified to support "persistent" tensors that survive for the entire graph execution?** Different from OUTPUT which just prevents reuse BY others.

3. **Is there a hook point during graph execution (between layers) where we could extract data?** Currently `graph_compute()` is atomic.

4. **Does Flash Attention have any mechanism to preserve intermediate attention matrices?** The fused kernel might have debug/analysis modes.

5. **Would it be feasible to add a custom CUDA kernel that extracts attention during the softmax operation itself?**

---

## Reproduction Steps

```bash
# Build
make LLAMA_CUBLAS=1 -j8

# Run koboldcpp with a model
python koboldcpp.py --model /path/to/model.gguf --port 5001

# Generate with attention output
curl -X POST http://localhost:5001/api/extra/generate/stream \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "max_length": 1, "output_attentions": true}'

# Check debug output in terminal and /tmp/attention_dump_token_0.txt
```

---

## Date
2026-01-15

## Author
Evans + Claude (debugging session)
