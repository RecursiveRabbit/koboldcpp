# KoboldCPP Attention Extraction API - TESTED DOCUMENTATION
## Real API Responses and Working Endpoints

**Version**: 3.1 (Model Info Edition)
**Date**: 2025-11-19
**Status**: ✅ **VERIFIED WORKING** - All endpoints tested with live server
**Model Tested**: Qwen2.5-VL-7B-Instruct-Q8_0 (28 layers, 28 heads)

---

## Implementation Status

### ✅ Phase 1: C++ Extraction Layer (COMPLETE)
- Unconditional attention tensor extraction at core `process_ubatch()`
- GPU→CPU copy after graph execution
- Shape: `[n_layers, n_heads, seq_len]` per token
- Raw pre-softmax logits
- Verified with 20+ token generation

### ✅ Phase 2: REST API (COMPLETE)
- Streaming endpoint `/api/extra/generate/stream` working
- Token IDs exposed
- Base64-encoded attention data (~1MB per token)
- Request ID tracking functional
- SSE (Server-Sent Events) format

### ✅ Phase 3: Tokenization API (COMPLETE - Session 8)
- `/api/v1/tokenize` endpoint with token text
- `/api/v1/detokenize` endpoint
- Round-trip tokenize/detokenize verified
- Deterministic tokenization for Halo Weave

### ✅ Phase 4: Model Information API (COMPLETE - Session 9)
- Enhanced `/api/v1/model` endpoint with full architecture metadata
- 12 fields exposed: layers, heads, vocab size, context limits, special tokens, RoPE params
- Essential for attention tensor shape validation
- Enables model-agnostic client implementations

### ⚠️ What Doesn't Exist
- No non-streaming `/api/v1/generate` endpoint with attention
- No `input_ids` parameter support (generation still requires text prompt)

**This document only shows TESTED, WORKING API calls.**

---

## What Works (Tested)

### Attention Extraction
- **Format**: Raw pre-softmax logits (NOT normalized probabilities)
- **Shape**: `[n_layers, n_heads, seq_len]` per generated token
- **Size**: ~1.07MB per token (base64-encoded) for 28L/28H model
- **Encoding**: base64 string in JSON
- **Note**: First generated token may not have attention data

### Streaming Generation
- **Protocol**: Server-Sent Events (SSE)
- **Format**: `event: message\ndata: {json}\n\n`
- **Request tracking**: Works with `request_id` field
- **Token delivery**: Real-time as generated

---

## Working API Endpoints

Only endpoints verified by actual testing are documented below.

### 1. Tokenization (NEW - Session 8)

#### Tokenize Text to Token IDs + Text
**`POST /api/v1/tokenize`**

Convert text to tokens, returning both token IDs and text for each token. Essential for deterministic tokenization in Halo Weave.

**Request**:
```bash
curl -X POST http://localhost:5001/api/v1/tokenize \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, how are you?",
    "add_special_tokens": false
  }'
```

**Request Parameters**:
- `text` (string): Text to tokenize
- `add_special_tokens` (bool): Include BOS/EOS tokens (default: false)
- `with_pieces` (bool): Return token text alongside IDs (default: true)

**ACTUAL Response** (from Qwen2.5-VL-7B-Instruct-Q8_0):
```json
{
  "tokens": [
    {
      "token_id": 9707,
      "text": "Hello"
    },
    {
      "token_id": 11,
      "text": ","
    },
    {
      "token_id": 1246,
      "text": " how"
    },
    {
      "token_id": 525,
      "text": " are"
    },
    {
      "token_id": 498,
      "text": " you"
    },
    {
      "token_id": 30,
      "text": "?"
    }
  ],
  "token_ids": [9707, 11, 1246, 525, 498, 30],
  "token_count": 6
}
```

**Notes**:
- Each token includes both `token_id` and `text` fields
- Useful for building token dictionaries with metadata
- Tokenization is deterministic (same text → same tokens)
- Compatible with context pruning workflows

---

#### Detokenize Token IDs to Text
**`POST /api/v1/detokenize`**

Convert token IDs back to text. Useful for reconstructing text after context pruning.

**Request**:
```bash
curl -X POST http://localhost:5001/api/v1/detokenize \
  -H "Content-Type: application/json" \
  -d '{
    "token_ids": [9707, 11, 1246, 525, 498, 30]
  }'
```

**Request Parameters**:
- `token_ids` (array): List of token IDs to convert back to text

**ACTUAL Response**:
```json
{
  "text": "Hello, how are you?"
}
```

**Notes**:
- Faithful reconstruction of original text
- Round-trip tokenize/detokenize preserves text exactly
- Works with any token ID array (useful after pruning)

---

### 2. Model Information
**`GET /api/v1/model`**

Get comprehensive model architecture details and metadata.

**Request**:
```bash
curl http://localhost:5001/api/v1/model
```

**ACTUAL Response** (from Qwen2.5-VL-7B-Instruct-Q8_0):
```json
{
  "result": "koboldcpp/Qwen2.5-VL-7B-Instruct-Q8_0",
  "model_name": "koboldcpp/Qwen2.5-VL-7B-Instruct-Q8_0",
  "vocab_size": 151936,
  "num_layers": 28,
  "num_attention_heads": 28,
  "num_key_value_heads": 4,
  "embedding_size": 3584,
  "max_context_length": 512,
  "max_trained_context": 32768,
  "bos_token_id": 151643,
  "eos_token_id": 151645,
  "eot_token_id": 151644,
  "rope_freq_base": 10000.0,
  "rope_freq_scale": 1.0
}
```

**Response Fields**:
- `result` (string): Model name (for backwards compatibility)
- `model_name` (string): Full model name
- `vocab_size` (int): Total vocabulary size - validate token IDs are in `[0, vocab_size)`
- `num_layers` (int): Number of transformer layers - validates attention shape `[num_layers, ...]`
- `num_attention_heads` (int): Attention heads per layer - validates attention shape `[..., num_heads, ...]`
- `num_key_value_heads` (int): KV cache heads (for GQA models like Qwen2.5)
- `embedding_size` (int): Hidden dimension size
- `max_context_length` (int): Current context window (set via --contextsize)
- `max_trained_context` (int): Maximum context the model was trained on
- `bos_token_id` (int): Beginning-of-sentence token ID
- `eos_token_id` (int): End-of-sentence token ID
- `eot_token_id` (int): End-of-turn token ID (or -1 if not available)
- `rope_freq_base` (float): RoPE frequency base (default: 10000.0)
- `rope_freq_scale` (float): RoPE frequency scaling factor

**Notes**:
- **NEW**: Now returns full architecture details (previously only returned model name)
- Essential for validating attention tensor shapes: `[num_layers, num_attention_heads, seq_len]`
- Use `vocab_size` to validate token IDs before generation
- Use `max_context_length` to prevent context overflow
- `max_trained_context` shows model's training limit (useful for context extension)
- Special token IDs needed for proper tokenization and generation control
- If model info retrieval fails, falls back to basic `{"result": "model_name"}` format

---

### 2. Streaming Text Generation with Attention
**`POST /api/extra/generate/stream`**

Generate text token-by-token with real-time attention extraction.

**Protocol**: HTTP POST with Server-Sent Events (SSE) response

**Request**:
```bash
curl -X POST http://localhost:5001/api/extra/generate/stream \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  --no-buffer \
  -d '{
    "prompt": "The capital of France is",
    "max_length": 20,
    "temperature": 0.7,
    "output_attentions": true,
    "request_id": "test-123"
  }'
```

**Request Parameters**:
- `prompt` (string): Text prompt (NOT token IDs)
- `max_length` (int): Maximum tokens to generate
- `temperature` (float): Sampling temperature
- `output_attentions` (bool): Enable attention extraction
- `request_id` (string): Optional request tracking ID

**Response Format**: Server-Sent Events (SSE)

Each event follows SSE format:
```
event: message
data: {json}

```

**Token Event** (ACTUAL tested response, attention data truncated for display):
```json
{
  "type": "token",
  "token": {
    "token_id": 13,
    "text": "."
  },
  "request_id": "test-123",
  "attention": {
    "format": "per_layer",
    "shape": [28, 28, 256],
    "context_length": 256,
    "encoding": "base64",
    "dtype": "float32",
    "data": "dTNEwgpPRkGR/PrBtRrZwaRn78Gm...[1,070,424 chars total]"
  }
}
```

**Attention Data**:
- **Shape**: `[num_layers, num_heads, context_length]`
- **Format**: RAW PRE-SOFTMAX LOGITS (not normalized)
- **Size**: ~1.07MB base64 for 28 layers × 28 heads × 256 context
- **Values**: Typically range from -100 to +100
- **Not normalized**: Sum does NOT equal 1.0

**Done Event** (ACTUAL tested response):
```json
{
  "type": "done",
  "finish_reason": "length",
  "total_tokens": 20,
  "request_id": "test-123"
}
```

**Important Notes**:
- First generated token often has `"attention": null`
- Subsequent tokens include attention data
- Attention is indexed by prompt position, not conversation position IDs
- Client must handle base64 decoding and array reshaping

---

## Decoding Attention Data (Python Example)

**Server encodes** (already implemented in koboldcpp.py):
```python
import numpy as np
import base64

# Get attention from C++ layer (float32 array)
attention_np = ...  # shape: [n_layers, n_heads, seq_len]

# Encode to base64
attention_bytes = attention_np.tobytes()
attention_b64 = base64.b64encode(attention_bytes).decode('ascii')
```

**Client decodes**:
```python
import numpy as np
import base64

# Parse JSON response
token_event = json.loads(event_data)
attention_info = token_event["attention"]

# Decode base64 to bytes
attention_bytes = base64.b64decode(attention_info["data"])

# Convert to numpy array
attention = np.frombuffer(attention_bytes, dtype=np.float32)
attention = attention.reshape(attention_info["shape"])

# Result: numpy array with shape [28, 28, 256]
# attention[layer, head, position] = raw logit value
```

**Memory**: ~1MB per token with attention data

---

## Complete Working Example (Tested)

**Start the server**:
```bash
python3 koboldcpp.py \
  --model /path/to/model.gguf \
  --port 5001 \
  --usecublas 0 \
  --gpulayers 999 \
  --contextsize 512
```

**Test the API** (Python):
```python
import requests
import json
import base64
import numpy as np

# Stream generation with attention
response = requests.post(
    "http://localhost:5001/api/extra/generate/stream",
    json={
        "prompt": "The capital of France is",
        "max_length": 20,
        "temperature": 0.7,
        "output_attentions": True,
        "request_id": "test-123"
    },
    stream=True,
    headers={"Accept": "text/event-stream"}
)

# Process stream
for line in response.iter_lines():
    if not line:
        continue
    
    line = line.decode('utf-8')
    if line.startswith('data: '):
        data = json.loads(line[6:])
        
        if data["type"] == "token":
            token_text = data["token"]["text"]
            print(f"Token: {token_text}")
            
            if data.get("attention"):
                # Decode attention
                attn_b64 = data["attention"]["data"]
                attn_bytes = base64.b64decode(attn_b64)
                attn = np.frombuffer(attn_bytes, dtype=np.float32)
                attn = attn.reshape(data["attention"]["shape"])
                print(f"  Attention shape: {attn.shape}")
        
        elif data["type"] == "done":
            print(f"Done: {data['total_tokens']} tokens")
            break
```

---

## Summary

### What Works ✅
1. **GET /api/v1/model** - Returns comprehensive model architecture details (NEW: full metadata!)
2. **POST /api/v1/tokenize** - Tokenize text to token IDs + text
3. **POST /api/v1/detokenize** - Convert token IDs back to text
4. **POST /api/extra/generate/stream** - Streaming generation with attention
5. Attention extraction: Raw pre-softmax logits, shape `[layers, heads, context]`
6. Base64 encoding for JSON transmission
7. Request ID tracking
8. SSE protocol for real-time streaming

### What Doesn't Exist ❌
- No non-streaming generation with attention
- No WebSocket support (uses HTTP + SSE instead)
- No `input_ids` parameter for generation (still requires text prompt)

### For Halo Weave Integration
- ✅ Model metadata available via `/api/v1/model` (architecture validation)
- ✅ Tokenization available via `/api/v1/tokenize` (deterministic)
- ✅ Detokenization available via `/api/v1/detokenize` (for reconstruction)
- ✅ Special token IDs exposed (BOS, EOS, EOT) for proper handling
- ✅ Attention shape validation: use `num_layers` and `num_attention_heads` from model info
- Client must map attention indices to conversation positions
- Attention is indexed by input prompt array position
- KV cache cannot survive pruning - must reprocess context after pruning
- ⏳ Next step: Add `input_ids` parameter to generation endpoints

**Backup of Original Spec**: See `KOBOLD_API_SPEC_ASPIRATIONAL.md` for the original design document with planned features.

---

**Last Updated**: 2025-11-19 (Session 9 - Model Information API enhanced)
**Tested With**: Qwen2.5-VL-7B-Instruct-Q8_0 (28L, 28H, Q8_0 quantization)
**Server**: koboldcpp v1.101.1 with custom attention extraction + tokenization patches
**New in Session 9**: Enhanced `/api/v1/model` endpoint with full architecture metadata (12 fields)
