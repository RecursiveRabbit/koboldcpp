# 🐛 Critical Bug: Attention Extraction Race Condition

**Date Discovered:** 2025-11-22
**Severity:** HIGH - Causes ~50% data corruption in attention extraction
**Affects:** All CUDA/GPU backends when extracting attention data
**Status:** CONFIRMED

---

## 📋 Summary

The attention extraction system has a **race condition** between GPU computation and CPU data reading. The extraction function reads tensor data **before the GPU has finished writing it**, resulting in corrupted attention values that oscillate between correct normalized probabilities and stale raw logits.

---

## 🔍 Technical Details

### The Problem

**File:** `src/llama-context.cpp:791-801`

```cpp
const auto status = graph_compute(res->get_gf(), ubatch.n_tokens > 1);
if (status != GGML_STATUS_SUCCESS) {
    LLAMA_LOG_ERROR("%s: failed to compute graph, compute status: %d\n", __func__, status);
    ret = status;
    return nullptr;
}

// UNCONDITIONAL ATTENTION EXTRACTION HOOK
// Called after every graph compute - cannot be bypassed
extern void extract_pending_attention_data();
extract_pending_attention_data();  // ← BUG: No synchronization!
```

**File:** `src/llama-context.cpp:1472`

```cpp
auto status = ggml_backend_sched_graph_compute_async(sched.get(), gf);
```

### The Race Condition

1. **Graph construction phase:**
   - `attention_capture_callback()` stores pointers to "kq_soft_max" tensors
   - At this point, tensors contain **uninitialized or stale data**

2. **Graph execution phase:**
   - `ggml_backend_sched_graph_compute_async()` is called
   - **ASYNC** - GPU kernels are scheduled but may not execute immediately
   - Function returns to caller **before GPU completes**

3. **Data extraction phase:**
   - `extract_pending_attention_data()` is called **immediately**
   - `ggml_backend_tensor_get()` reads from GPU memory
   - **PROBLEM:** GPU may still be writing the softmax results

### Timeline Diagram

```
Time    CPU Thread              GPU
─────   ─────────────           ───────────────────
T0      graph_construct()       [idle]
        └─ save tensor ptrs

T1      graph_compute()         [idle]
        └─ schedule kernels

T2      return from compute     [executing layer 0]

T3      extract_attention()     [executing layer 5]
        └─ read tensor data     ↑
                                ├─ RACE: reading while GPU writing
                                ↓
T4      [done]                  [executing layer 15]

T5                              [done]
```

**Result:** Sometimes we read before GPU writes (get stale logits), sometimes after (get correct softmax).

---

## 🧪 Observed Symptoms

### Experimental Evidence

**Test Setup:**
- Model: Qwen2.5-VL-7B-Instruct-Q8_0
- Context: 512 tokens
- Flash Attention: DISABLED (all tokens use same code path)
- Generated: 387 tokens

**Results:**
- **182 transitions** between normalized and raw logits
- **Frequency:** ~every 2 tokens
- **Pattern:** Semi-random (timing-dependent)

**Example Data:**

| Token | Type | Min Value | Max Value | Sum |
|-------|------|-----------|-----------|-----|
| 1 | ✅ NORMALIZED | 0.00 | 0.99 | 784 |
| 6 | ❌ RAW LOGITS | -126.99 | 41.76 | -255,087 |
| 7 | ✅ NORMALIZED | 0.00 | 0.94 | 784 |
| 8 | ❌ RAW LOGITS | -130.37 | 27.77 | -261,543 |
| 9 | ✅ NORMALIZED | 0.00 | 0.99 | 784 |
| 11 | ❌ RAW LOGITS | -131.58 | 10.94 | -264,196 |

**Key observation:** Sum of 784 = 28 layers × 28 heads = perfect for normalized probabilities that should sum to 1.0 per head.

### Why The Pattern Is Semi-Random

The race condition is **timing-dependent**, affected by:
- GPU scheduling variability
- CPU/GPU clock speed fluctuations
- System load
- Memory bandwidth contention
- CUDA stream scheduling

This explains why there's no periodic pattern - it's determined by microsecond-level timing variance.

---

## 🛠️ Root Cause Analysis

### Code Path Investigation

**gpttype_adapter.cpp:245-269** - Callback captures tensor pointers:
```cpp
void attention_capture_callback(const llama_ubatch & ubatch,
                                ggml_tensor * cur,
                                const char * name,
                                int il) {
    if (strcmp(name, "kq_soft_max") != 0) {
        return;
    }
    // Store the tensor pointer for later extraction (after graph execution)
    g_pending_attentions.push_back({cur, il});
}
```

**gpttype_adapter.cpp:310-311** - Extraction reads tensor data:
```cpp
// Copy tensor from backend (GPU/CPU) to our CPU buffer
ggml_backend_tensor_get(cur, tensor_data, 0, tensor_bytes);
```

**The Missing Link:**

Between graph_compute() returning and extract_pending_attention_data() reading, there **MUST** be a synchronization barrier:

```cpp
// NOT PRESENT in current code:
ggml_backend_synchronize(backend);  // Wait for GPU to finish
```

---

## ✅ Proposed Solution

### Option 1: Add Synchronization (RECOMMENDED)

**File:** `src/llama-context.cpp`

**Current Code (lines 791-801):**
```cpp
const auto status = graph_compute(res->get_gf(), ubatch.n_tokens > 1);
if (status != GGML_STATUS_SUCCESS) {
    LLAMA_LOG_ERROR("%s: failed to compute graph, compute status: %d\n", __func__, status);
    ret = status;
    return nullptr;
}

extern void extract_pending_attention_data();
extract_pending_attention_data();
```

**Fixed Code:**
```cpp
const auto status = graph_compute(res->get_gf(), ubatch.n_tokens > 1);
if (status != GGML_STATUS_SUCCESS) {
    LLAMA_LOG_ERROR("%s: failed to compute graph, compute status: %d\n", __func__, status);
    ret = status;
    return nullptr;
}

// Synchronize ALL backends before reading attention data
// CRITICAL: GPU kernels must finish before we read tensor buffers
ggml_backend_sched_synchronize(sched.get());

extern void extract_pending_attention_data();
extract_pending_attention_data();
```

**Trade-offs:**
- ✅ **Pros:** Simple, guaranteed correct, minimal code change
- ⚠️ **Cons:** Adds ~0.1-0.5ms latency per token (GPU idle wait time)

### Option 2: Use Async Copy with Callback

Add completion callback after GPU work finishes:

```cpp
// In llama-context.cpp
struct AttentionExtractionContext {
    llama_context * ctx;
    std::vector<PendingAttentionTensor> pending;
};

void attention_extraction_callback(void * user_data) {
    auto * ctx = (AttentionExtractionContext*)user_data;
    extract_pending_attention_data_from_context(ctx);
}

// After graph_compute_async:
auto extract_ctx = new AttentionExtractionContext{this, g_pending_attentions};
ggml_backend_sched_set_callback(sched.get(), attention_extraction_callback, extract_ctx);
```

**Trade-offs:**
- ✅ **Pros:** Zero added latency
- ❌ **Cons:** Complex, requires refactoring, harder to debug

### Option 3: Double-Buffer with Validation

Keep current code but validate and retry:

```cpp
void extract_pending_attention_data() {
    // Try extraction with validation
    for (int retry = 0; retry < 3; retry++) {
        // Extract data
        ggml_backend_tensor_get(cur, tensor_data, 0, tensor_bytes);

        // Validate: check if data looks normalized
        float min_val = *std::min_element(tensor_data, tensor_data + count);
        float max_val = *std::max_element(tensor_data, tensor_data + count);

        if (min_val >= -0.01 && max_val <= 1.01) {
            break;  // Data looks good
        }

        // Wait and retry
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
}
```

**Trade-offs:**
- ✅ **Pros:** Automatic retry/recovery
- ❌ **Cons:** Hacky, adds unpredictable latency, may still fail

---

## 🎯 Recommended Fix

**Use Option 1** - Add explicit synchronization.

**Patch:**

```diff
diff --git a/src/llama-context.cpp b/src/llama-context.cpp
index abc123..def456 100644
--- a/src/llama-context.cpp
+++ b/src/llama-context.cpp
@@ -796,6 +796,10 @@ llama_graph_plan * llama_context::graph_schedule(
         return nullptr;
     }

+    // Synchronize backend scheduler to ensure all GPU kernels complete
+    // before reading attention tensor data
+    ggml_backend_sched_synchronize(sched.get());
+
     // UNCONDITIONAL ATTENTION EXTRACTION HOOK
     // Called after every graph compute - cannot be bypassed
     extern void extract_pending_attention_data();
```

**Testing:**
1. Apply patch
2. Rebuild KoboldCPP
3. Capture 500+ tokens with attention
4. Verify ALL tokens have normalized attention (min ≥ 0, max ≤ 1, sum ≈ 784)
5. Verify no transitions occur in `find_transitions.py` output

---

## 📊 Performance Impact

**Expected overhead:** ~0.1-0.5ms per token

**Calculation:**
- Current: ~100ms per token (model inference)
- Sync wait: ~0.1-0.5ms (GPU already 99% done when we call sync)
- Overhead: **0.1-0.5%** slowdown

**Worth it?** Absolutely. Data corruption is unacceptable.

---

## 🧪 Validation Script

**Test for race condition:**

```bash
cd /home/evans/Coding_Projects/Halo_Weave/halo_weave
python3 find_transitions.py Capture_Data/capture_<timestamp>/
```

**Expected BEFORE fix:**
```
Total transitions: 182
```

**Expected AFTER fix:**
```
Total transitions: 0
```

---

## 📝 Related Files

**Primary:**
- `src/llama-context.cpp:791-801` - Where sync is needed
- `gpttype_adapter.cpp:274-345` - Extraction implementation

**Secondary:**
- `src/llama-graph.cpp:607-608` - Attention callback registration
- `src/llama-graph.cpp:1439-1441` - Softmax operation

---

## 🔗 References

**GGML Backend API:**
- `ggml_backend_sched_graph_compute_async()` - Async graph execution
- `ggml_backend_sched_synchronize()` - Wait for completion
- `ggml_backend_tensor_get()` - Copy tensor from device to host

**KoboldCPP Attention System:**
- `KOBOLD_API_SPEC.md` - Original attention extraction spec
- `SESSION_7_SUMMARY.md` - Implementation notes

---

## ✅ Action Items

- [ ] Apply synchronization patch to `src/llama-context.cpp`
- [ ] Rebuild KoboldCPP
- [ ] Run validation test with 500+ token capture
- [ ] Verify zero transitions in output
- [ ] Benchmark performance impact (should be < 1%)
- [ ] Update documentation with findings
- [ ] Consider upstreaming fix to llama.cpp if they use similar pattern

---

**Status:** Ready for implementation
**Priority:** HIGH - Blocks Halo Weave production use
**Estimated Fix Time:** 5 minutes
**Estimated Test Time:** 15 minutes
