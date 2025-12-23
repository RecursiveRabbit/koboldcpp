# Session 8: Tokenization API Implementation (2025-11-19)

## What We Built ✅

### Problem
Halo Weave needs tokenization/detokenization endpoints to:
- Tokenize user messages ONCE when added to conversation
- Store tokens with metadata (position, turn_id, sentence_id)
- Never retokenize existing context (deterministic tokenization)
- Reconstruct text from token IDs after pruning

### Solution: Added `/api/v1/tokenize` and `/api/v1/detokenize` Endpoints

## Implementation

### 1. C++ Layer: Token-to-Text Conversion

**Added `gpttype_token_to_str()` in `gpttype_adapter.cpp:3234-3249`**:
```cpp
std::string gpttype_token_to_str(int token_id, bool render_special)
{
    if(kcpp_data==nullptr) {
        printf("\nWarning: KCPP text generation not initialized!\n");
        return "";
    }
    if(token_id<0 || token_id>=n_vocab) {
        return "";
    }
    return FileFormatTokenizeID(token_id, file_format, render_special);
}
```

**Added declaration in `model_adapter.h:106`**:
```cpp
std::string gpttype_token_to_str(int token_id, bool render_special);
```

**Exposed via C API in `expose.cpp:399-404`**:
```cpp
static std::string token_str = "";
const char * token_to_str(int token_id)
{
    token_str = gpttype_token_to_str(token_id, false);
    return token_str.c_str();
}
```

### 2. Python Bindings

**Added ctypes binding in `koboldcpp.py:625-626`**:
```python
handle.token_to_str.argtypes = [ctypes.c_int]
handle.token_to_str.restype = ctypes.c_char_p
```

### 3. REST API Endpoints

**Added `/api/v1/tokenize` in `koboldcpp.py:3839-3868`**:
- Accepts: `{"text": "...", "add_special_tokens": false}`
- Returns: Token IDs AND text for each token
- Format matches Halo Weave requirements

**Added `/api/v1/detokenize` in `koboldcpp.py:3870-3881`**:
- Accepts: `{"token_ids": [123, 456, ...]}`
- Returns: `{"text": "..."}`

### 4. Test Suite

**Created `test_tokenization.py`**:
- Tests tokenization with text output
- Tests detokenization
- Tests round-trip (tokenize → detokenize)
- All tests pass ✅

## Test Results

```bash
$ python3 test_tokenization.py

=== Testing /api/v1/tokenize ===
Request: Hello, how are you?
✅ Success! Got 6 tokens:
  ID   9707 → "Hello"
  ID     11 → ","
  ID   1246 → " how"
  ID    525 → " are"
  ID    498 → " you"
  ID     30 → "?"

=== Testing /api/v1/detokenize ===
Request: [9707, 11, 1246, 525, 498, 30]
✅ Success! Detokenized text: "Hello, how are you?"

=== Testing Round-trip ===
Original text: "The capital of France is Paris."
Token IDs: [785, 6722, 315, 9625, 374, 12095, 13]
Reconstructed text: "The capital of France is Paris."
✅ Round-trip test PASSED!
```

## API Examples

### Tokenize Endpoint

**Request**:
```bash
curl -X POST http://localhost:5001/api/v1/tokenize \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, how are you?", "add_special_tokens": false}'
```

**Response**:
```json
{
  "tokens": [
    {"token_id": 9707, "text": "Hello"},
    {"token_id": 11, "text": ","},
    {"token_id": 1246, "text": " how"},
    {"token_id": 525, "text": " are"},
    {"token_id": 498, "text": " you"},
    {"token_id": 30, "text": "?"}
  ],
  "token_ids": [9707, 11, 1246, 525, 498, 30],
  "token_count": 6
}
```

### Detokenize Endpoint

**Request**:
```bash
curl -X POST http://localhost:5001/api/v1/detokenize \
  -H "Content-Type: application/json" \
  -d '{"token_ids": [9707, 11, 1246, 525, 498, 30]}'
```

**Response**:
```json
{
  "text": "Hello, how are you?"
}
```

## Benefits for Halo Weave

1. **Deterministic Tokenization**: Same text → same tokens, always
2. **Token Metadata**: Can attach position, turn_id, sentence_id to each token
3. **No Retokenization**: Tokenize once, store forever
4. **Pruning Support**: Can reconstruct text from pruned token arrays
5. **Token-Level Control**: Foundation for `input_ids` parameter (next step)

## Files Modified

- `gpttype_adapter.cpp` - Added `gpttype_token_to_str()` function
- `model_adapter.h` - Added function declaration
- `expose.cpp` - Exposed function via C API
- `koboldcpp.py` - Added Python binding + REST endpoints
- `test_tokenization.py` - Test suite (NEW)

## Compilation

```bash
make LLAMA_CUBLAS=1 -j8
# Output: koboldcpp_cublas.so (211 MB)
# Warnings: format strings (harmless, pre-existing)
```

## Next Steps (From CLAUDE.md TODO)

### 🔴 CRITICAL: Input Token Control

**Problem**: API still accepts `prompt` (text), not `input_ids` (token array).

**Next step**: Modify generation endpoints to accept `input_ids` parameter:
```json
{
  "input_ids": [151644, 1587, 198, 2610, ...],
  "max_new_tokens": 50
}
```

**Action**: Find where koboldcpp.py parses generation requests and add bypass for `input_ids`.

### 🟡 NICE TO HAVE: Enhanced Model Info

**Problem**: `/api/v1/model` returns only model name, not architecture details.

**Desired**:
```json
{
  "model_name": "Qwen2.5-VL-7B-Instruct-Q8_0",
  "num_layers": 28,
  "num_attention_heads": 28,
  "vocab_size": 151936,
  "max_context_length": 32768
}
```

**Action**: Extract GGUF metadata and expose via API.

## Status

**Session 8 Complete**: ✅ Tokenization API implemented and tested
**Overall Progress**:
- ✅ Phase 1: C++ attention extraction (Session 5)
- ✅ Phase 2: REST API with SSE streaming (Session 6)
- ✅ **Phase 3: Tokenization endpoints (Session 8)**
- ⏳ Phase 4: Input token control (next)

**Ready for**: Halo Weave integration with deterministic tokenization!

---

**Last Updated**: 2025-11-19
**Model Tested**: Qwen2.5-VL-7B-Instruct-Q8_0 (28L, 28H)
**Server**: koboldcpp with CUDA support
