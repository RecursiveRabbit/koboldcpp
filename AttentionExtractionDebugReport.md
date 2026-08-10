# Attention Extraction Debug Report

 

## Executive Summary (For Humans)

 

### The Problem

 

The attention extraction system is capturing 28 copies of layer 27's data instead of unique data from all 28 layers. This is a **tensor aliasing bug** caused by a timing mismatch between when we capture tensor pointers and when we read their data.

 

### Root Cause (One-Sentence)

 

We're saving *pointers* during graph construction, but reading *data* after graph execution — by which point ggml's memory allocator has reused the same buffer for all 28 layers (overwriting each layer's attention before we read it).

 

### The Fix (Conceptual)

 

Instead of capturing pointers during graph *construction* and reading later, we need to extract data *during* graph execution using the `cb_eval` callback, which fires immediately after each layer computes (before the buffer is reused).

 

### Performance Impact

 

The current implementation adds **~10-30ms overhead per token** from:

- 28 GPU→CPU memory copies

- 56 malloc/free operations

- Unnecessary transposition loops

 

This is acceptable for real-time visualization but could be optimized by ~90% with buffer pooling and async copies.

 

---

 

## Technical Documentation (For AI Collaborators)

 

### Architecture Overview

 

```

Graph Construction Phase (wrong time to copy):

┌─────────────────────────────────────────────────────────────────┐

│  for layer 0..27:                                               │

│    kq = ggml_mul_mat(k, q)                                      │

│    kq = ggml_soft_max_ext(kq, ...)                              │

│    cb(kq, "kq_soft_max", layer_idx)  ← CALLBACK FIRES HERE      │

│         ↓                                                       │

│    attention_capture_callback() stores {tensor_ptr, layer_idx}  │

│         ↓                                                       │

│    g_pending_attentions.push_back({tensor_ptr, layer_idx})      │

│                                                                 │

│  PROBLEM: tensor_ptr has no buffer yet - just a graph node!    │

└─────────────────────────────────────────────────────────────────┘

                              ↓

                    ggml_backend_sched_alloc_graph()

                              ↓

┌─────────────────────────────────────────────────────────────────┐

│  Memory Allocator assigns buffers:                              │

│    - Sees layer 0's kq_soft_max: needs N bytes, assign buf[A]   │

│    - Layer 0 used and discarded before layer 1 computes        │

│    - Sees layer 1's kq_soft_max: can reuse buf[A]!              │

│    - ... all 28 layers share buf[A] ...                         │

│                                                                 │

│  RESULT: All tensor pointers in g_pending_attentions point to   │

│          the SAME physical buffer address                       │

└─────────────────────────────────────────────────────────────────┘

                              ↓

                        graph_compute()

                              ↓

┌─────────────────────────────────────────────────────────────────┐

│  GPU executes layers sequentially:                              │

│    Layer 0: compute softmax → write to buf[A]                   │

│    Layer 0: multiply with V → read from buf[A] → done           │

│    Layer 1: compute softmax → write to buf[A] (OVERWRITES!)     │

│    ...                                                          │

│    Layer 27: compute softmax → write to buf[A]                  │

│    Layer 27: multiply with V → read from buf[A]                 │

│                                                                 │

│  Final state: buf[A] contains ONLY layer 27's attention data    │

└─────────────────────────────────────────────────────────────────┘

                              ↓

                    extract_pending_attention_data()

                              ↓

┌─────────────────────────────────────────────────────────────────┐

│  for each {tensor_ptr, layer_idx} in g_pending_attentions:      │

│    ggml_backend_tensor_get(tensor_ptr, ...)  ← ALL READ buf[A]! │

│                                                                 │

│  RESULT: 28 copies of layer 27's data                           │

└─────────────────────────────────────────────────────────────────┘

```

 

### Data Flow: Current (Broken) Implementation

 

| Stage | Location | What Happens |

|-------|----------|--------------|

| 1. Graph construction | `src/llama-graph.cpp:608` | `cb()` calls `attention_capture_callback()` |

| 2. Pointer capture | `gpttype_adapter.cpp:260` | Store `{tensor*, layer_idx}` in `g_pending_attentions` |

| 3. Memory allocation | `src/llama-context.cpp:775` | `ggml_backend_sched_alloc_graph()` assigns aliased buffers |

| 4. GPU execution | `src/llama-context.cpp:791` | `graph_compute()` - each layer overwrites shared buffer |

| 5. GPU sync | `src/llama-context.cpp:801` | `ggml_backend_sched_synchronize()` - wait for GPU |

| 6. Extraction | `src/llama-context.cpp:806` | `extract_pending_attention_data()` reads same data 28x |

| 7. Copy to token | `gpttype_adapter.cpp:101-116` | `TokenWithAttention` copies corrupted data |

 

### Data Flow: Correct Implementation (Proposed)

 

| Stage | Location | What Happens |

|-------|----------|--------------|

| 1. Configure eval callback | Context initialization | Set `cb_eval` to attention capture function |

| 2. Graph execution | During `graph_compute()` | Eval callback fires AFTER each op completes |

| 3. Immediate extraction | Inside eval callback | Copy layer N's attention BEFORE layer N+1 overwrites it |

| 4. Buffer to token | Same as before | `TokenWithAttention` copies correct data |

 

### Key Files and Line Numbers

 

#### Modified Files (Our Changes)

 

| File | Lines | Purpose |

|------|-------|---------|

| `gpttype_adapter.cpp` | 60-95 | `AttentionCapture` struct methods (`init`, `reset`, `append_layer`) |

| `gpttype_adapter.cpp` | 97-116 | `TokenWithAttention` constructors |

| `gpttype_adapter.cpp` | 227-235 | `g_attention` and `g_pending_attentions` globals |

| `gpttype_adapter.cpp` | 240-261 | `attention_capture_callback()` - stores tensor pointers (BUG) |

| `gpttype_adapter.cpp` | 266-335 | `extract_pending_attention_data()` - extracts after execution (BUG) |

| `src/llama-graph.cpp` | 16-20 | `extern` declaration for callback |

| `src/llama-graph.cpp` | 603-609 | Hook callback into `cb()` function |

| `src/llama-context.cpp` | 798-806 | GPU sync + extraction hook after `graph_compute()` |

| `expose.h` | 145-148 | `generation_outputs` attention fields |

| `expose.h` | 150-192 | `attention_outputs`, `AttentionCapture`, `TokenWithAttention` structs |

 

#### llama.cpp Internals (Unmodified, For Reference)

 

| File | Lines | Purpose |

|------|-------|---------|

| `src/llama-graph.cpp` | 1340-1460 | `build_attn_mha()` - where `kq_soft_max` tensor is created |

| `src/llama-graph.cpp` | 1439-1441 | `ggml_soft_max_ext()` + `cb(kq, "kq_soft_max", il)` |

| `src/llama-context.cpp` | 775 | `ggml_backend_sched_alloc_graph()` - memory aliasing happens here |

| `src/llama-context.cpp` | 791 | `graph_compute()` - GPU execution |

| `src/llama-cparams.h` | 39-40 | `cb_eval` - the correct callback mechanism |

 

### Performance Bottlenecks

 

#### Current Overhead Per Token

 

| Operation | Count | Time (est.) | Notes |

|-----------|-------|-------------|-------|

| `ggml_backend_tensor_get()` | 28 | 10-20ms | GPU→CPU DMA per layer |

| `malloc()` + `free()` | 56 | 1-2ms | 2 temp buffers per layer |

| Transposition loop | 28 | 1-2ms | O(n_heads × seq_len) |

| `memcpy()` in `append_layer()` | 28 | 0.5-1ms | O(n_heads × seq_len) |

| `fprintf()` debug | 1 | 0.1ms | Syscall overhead |

| **Total** | | **~15-30ms** | |

 

#### Optimization Opportunities

 

1. **Buffer pooling**: Pre-allocate transposition buffers, reuse across layers

2. **Batch GPU copy**: Single `ggml_backend_tensor_get()` for all layers if possible

3. **Remove debug prints**: `fprintf()` in hot path

4. **Async extraction**: Overlap GPU compute with CPU extraction

5. **Pinned memory**: Use CUDA pinned memory for faster DMA

 

### Tensor Shape Reference

 

```

kq_soft_max tensor during generation:

  ne[0] = seq_len_k    (context length, e.g., 256)

  ne[1] = seq_len_q    (query length, 1 during generation)

  ne[2] = n_heads      (attention heads, e.g., 28)

  ne[3] = batch        (batch size, 1)

 

Memory layout: [seq_len_k, seq_len_q, n_heads, batch]

With seq_len_q=1, batch=1: effectively [seq_len_k, n_heads]

 

We transpose to: [n_heads, seq_len_k]

Then append 28 layers to get: [n_layers, n_heads, seq_len_k]

```

 

### How to Fix the Bug

 

#### Option 1: Use `cb_eval` Callback (Recommended)

 

The `cb_eval` callback fires *during* graph execution, after each operation completes. This allows extracting attention data before the buffer is reused.

 

```cpp

// In llama_context_params initialization:

params.cb_eval = attention_eval_callback;

params.cb_eval_user_data = &g_attention;

 

// Callback signature:

bool attention_eval_callback(struct ggml_tensor * t, bool ask, void * user_data) {

    if (ask) return true;  // "Should I evaluate this?" → Yes, always

 

    // Only capture kq_soft_max tensors

    if (strstr(ggml_get_name(t), "kq_soft_max") == nullptr) return true;

 

    // Extract layer index from tensor name (e.g., "kq_soft_max-0")

    int layer_idx = extract_layer_from_name(ggml_get_name(t));

 

    // Copy data NOW, before next layer overwrites it

    AttentionCapture* attn = (AttentionCapture*)user_data;

    // ... copy tensor data to attn->buffer ...

 

    return true;

}

```

 

#### Option 2: Mark Tensors as Non-Aliasable (Invasive)

 

Modify ggml to prevent the allocator from aliasing `kq_soft_max` tensors. This would require upstream changes and increase memory usage by 28× for attention buffers.

 

#### Option 3: Instrument `ggml_soft_max_ext` (Invasive)

 

Modify the softmax kernel to copy output to a side buffer. Very invasive, breaks abstraction.

 

### Testing the Fix

 

```python

# After implementing the fix, verify layer uniqueness:

import numpy as np

 

# Extract attention for one token

attn = get_token_attention(0)  # Shape: [28, 28, seq_len]

 

# Check that layers are different

for i in range(27):

    layer_i = attn[i]

    layer_i_plus_1 = attn[i + 1]

    correlation = np.corrcoef(layer_i.flatten(), layer_i_plus_1.flatten())[0, 1]

    print(f"Layer {i} vs {i+1}: correlation = {correlation:.4f}")

    # Should be < 0.9 for different layers, currently ~1.0 (identical)

```

 

### Summary

 

| Aspect | Current State | Fix |

|--------|--------------|-----|

| Callback type | `cb` (graph construction) | `cb_eval` (graph execution) |

| When data copied | After all layers complete | After each layer completes |

| Buffer aliasing | All layers share one buffer | Each layer copied before reuse |

| Layer uniqueness | All identical (layer 27) | All unique |

| Performance | 28 serial GPU→CPU copies | Same, but could optimize |