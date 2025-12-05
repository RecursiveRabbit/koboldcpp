# Session 7: API Documentation & Reality Check (2025-11-18)

## What We Did ✅

### 1. Tested Live API
- Started koboldcpp server with Qwen2.5-VL-7B-Instruct-Q8_0
- Ran real API calls and captured responses
- Verified attention extraction works (20 tokens with ~1MB attention data each)

### 2. Rewrote API Spec
- **Before**: 952 lines of aspirational design document
- **After**: 294 lines of tested, working API calls
- Every example is from actual server responses
- Backed up original as `KOBOLD_API_SPEC_ASPIRATIONAL.md`

### 3. Identified Critical Blockers

**Output side**: ✅ **WORKING PERFECTLY**
- Attention extraction: Raw pre-softmax logits
- Shape: `[n_layers, n_heads, seq_len]`
- Streaming via SSE
- Request tracking
- Base64 encoding

**Input side**: 🔴 **NEEDS WORK**

## Critical Findings

### 🔴 BLOCKING: No Token-Level Input Control

**Discovery**: API accepts `prompt` (text), not `input_ids` (token array)

**Why this blocks Halo Weave**:
- Can't send pruned context (gaps in token sequence)
- Can't guarantee deterministic tokenization
- Have to retokenize everything on each request
- Lose precise token-level control

**Example of the problem**:
```python
# What Halo Weave wants to send after pruning:
context_tokens = [151644, 1587, 198, 2610, 4521, 7890]  # Tokens 3-4 pruned!

# What koboldcpp API accepts:
{"prompt": "Hello world"}  # Must reconstruct text and hope it tokenizes the same
```

**Next step**: Find where koboldcpp tokenizes prompts, add `input_ids` parameter

---

### 🔴 BLOCKING: No Tokenization Endpoints

**Discovery**: No `/api/v1/tokenize` or `/api/v1/detokenize` endpoints

**Why this blocks Halo Weave**:
- Can't tokenize user input deterministically
- Can't build token dictionary with metadata
- Can't reconstruct text from token IDs after pruning

**Next step**: Add tokenization endpoints to koboldcpp.py

---

### 🟡 NICE TO HAVE: Limited Model Info

**Discovery**: `/api/v1/model` returns only model name

**Impact**: Not blocking, but Halo Weave needs layer/head counts to validate attention shapes

**Next step**: Extract GGUF metadata and expose via API

---

## Key Insight

> **"We know the system eats tokens at some point, we just have to find it."**

The good news: We're getting the **output** we need (attention works!). We just need to fix the **input** side (token control).

---

## Files Changed

- `KOBOLD_API_SPEC.md` - Rewritten with real tested responses (294 lines)
- `KOBOLD_API_SPEC_ASPIRATIONAL.md` - Backup of original spec (952 lines)
- `CLAUDE.md` - Added TODO section with blockers
- `test_streaming_attention.py` - Working test script
- `capture_full_example.py` - Response capture script

---

## Next Steps (Priority Order)

1. **Add `input_ids` parameter to generation API** (CRITICAL)
   - Find tokenization point in koboldcpp.py
   - Add bypass for direct token array input
   - Verify attention indices match

2. **Add tokenization endpoints** (CRITICAL)
   - `/api/v1/tokenize` - text → tokens
   - `/api/v1/detokenize` - tokens → text
   - Return both IDs and text

3. **Enhance model info endpoint** (nice to have)
   - Extract GGUF metadata
   - Return layer/head counts, vocab size, etc.

---

## Test Results

**Model**: Qwen2.5-VL-7B-Instruct-Q8_0 (28 layers, 28 heads)
**Prompt**: "The capital of France is"
**Generated**: 20 tokens
**Attention data**: 19 tokens with attention (first token had none)
**Attention size**: 1,070,424 chars base64 (~802KB decoded)
**Shape**: [28, 28, 256] per token

✅ Test PASSED - Attention extraction works perfectly!

---

**Status**: Output side complete, input side needs work. Ready to dig into source code for `input_ids` support.
