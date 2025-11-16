# KoboldCPP Dream API Specification
## Attention Extraction for Halo Weave Integration

**Version**: 1.0
**Date**: 2025-11-16
**Purpose**: Define the ideal KoboldCPP API for extracting attention patterns during text generation

---

## Executive Summary

This specification defines REST and WebSocket APIs that expose transformer attention patterns during text generation. The goal is to enable **brightness-based context pruning** in Halo Weave by providing real-time attention scores for every token in the conversation.

**Critical requirement**: The API must return **raw per-layer, per-head attention tensors**, not pre-aggregated values, to support flexible aggregation strategies (mean, max, weighted layers, etc.) and advanced features like distance weighting.

---

## Design Principles

### 1. Raw Data Over Convenience
- Return full attention tensors (all layers, all heads) even though it's more data
- Client decides how to aggregate; server doesn't make assumptions
- Enables experimentation with different aggregation modes

### 2. Deterministic Tokenization
- Expose exact tokenizer output so clients can pre-tokenize
- Token positions must be stable and predictable
- Client maintains conversation state (token dictionary)

### 3. Position-Based Mapping (CLIENT RESPONSIBILITY)
- **Position IDs are CLIENT metadata, not exposed in API**
- Kobold returns attention indexed by input array: `attention[i]` corresponds to `input_ids[i]`
- Client maintains mapping: `array_index → position_id`
- This allows client to handle non-sequential position IDs after pruning (e.g., 0,1,2,15,16,65,66...)
- **Kobold doesn't know or care about position IDs** - it just processes the array you give it

**CRITICAL: KV cache cannot survive pruning**
- The KV cache is **generation-scoped** and assumes sequential positions
- You CANNOT delete tokens from the middle of the cache
- After pruning, you MUST reprocess the entire pruned context to rebuild the KV cache
- This is expected behavior and relatively cheap (~1-2s for 500 tokens)

### 4. Streaming-First
- Token-by-token generation with real-time attention
- Client renders tokens as they arrive
- WebSocket for low latency

### 5. Memory Efficiency
- Server doesn't store conversation state
- Client sends full context on each request
- Server is stateless (easier to scale, restart, swap models)

---

## API Endpoints

### 1. Model Information
**`GET /api/v1/model/info`**

Get model architecture details needed for attention processing.

**Response**:
```json
{
  "model_name": "Qwen/Qwen2.5-7B-Instruct",
  "architecture": "Qwen2ForCausalLM",
  "vocab_size": 151936,
  "num_layers": 28,
  "num_attention_heads": 28,
  "num_key_value_heads": 4,
  "hidden_size": 3584,
  "max_position_embeddings": 32768,
  "rope_theta": 1000000.0,
  "bos_token_id": 151643,
  "eos_token_id": 151645,
  "special_tokens": {
    "bos_token": "<|endoftext|>",
    "eos_token": "<|im_end|>",
    "pad_token": "<|endoftext|>",
    "im_start_id": 151644,
    "im_end_id": 151645
  },
  "chat_template": "{% for message in messages %}...",
  "torch_dtype": "bfloat16",
  "context_length": 32768
}
```

**Purpose**:
- Client needs `num_layers` and `num_attention_heads` to validate attention tensor shapes
- Special token IDs for banning during sampling (prevent `<|im_start|>` generation)
- Chat template for formatting conversation

---

### 2. Tokenization
**`POST /api/v1/tokenize`**

Convert text to tokens using the model's tokenizer. Enables client-side token dictionary management.

**Request**:
```json
{
  "text": "Hello, how are you?",
  "add_special_tokens": false
}
```

**Response**:
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
      "token_id": 1234,
      "text": " how"
    },
    {
      "token_id": 527,
      "text": " are"
    },
    {
      "token_id": 499,
      "text": " you"
    },
    {
      "token_id": 30,
      "text": "?"
    }
  ],
  "token_ids": [9707, 11, 1234, 527, 499, 30],
  "token_count": 6
}
```

**Purpose**:
- Client tokenizes messages ONCE when they enter the conversation
- Stores tokens in token dictionary with metadata (position, turn_id, sentence_id)
- Never retokenizes existing context (deterministic, no position drift)
- **Once tokenized, original text can be discarded** - token list with text fields is the source of truth

**Parameters**:
- `add_special_tokens`: Whether to add BOS/EOS tokens (usually false for manual conversation assembly)

**Why no character offsets?**
- We rebuild text from token.text fields, not from source text
- After pruning, character offsets would be meaningless anyway
- Simpler API, less data to transmit

---

### 3. Detokenization
**`POST /api/v1/detokenize`**

Convert token IDs back to text. Useful for reconstructing context from token dictionary.

**Request**:
```json
{
  "token_ids": [9707, 11, 1234, 527, 499, 30]
}
```

**Response**:
```json
{
  "text": "Hello, how are you?"
}
```

---

### 4. Text Generation (Non-Streaming)
**`POST /api/v1/generate`**

Generate text with attention extraction (non-streaming, returns complete response).

**Request**:
```json
{
  "input_ids": [151644, 1587, 198, 2610, ...],
  "max_new_tokens": 50,
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 40,
  "repetition_penalty": 1.0,
  "stop_tokens": [151645],
  "banned_tokens": [151644],
  "return_attention": true,
  "attention_format": "per_layer"
}
```

**Response**:
```json
{
  "generated_tokens": [
    {
      "token_id": 40,
      "text": "I"
    },
    {
      "token_id": 2846,
      "text": "'m"
    },
    {
      "token_id": 3291,
      "text": " doing"
    }
  ],
  "generated_text": "I'm doing well, thank you for asking!",
  "finish_reason": "stop_token",
  "attention_data": [
    {
      "token_id": 40,
      "text": "I",
      "attention": {
        "format": "per_layer",
        "shape": [28, 28, 267],
        "data": "base64encodeddata..."
      }
    }
  ]
}
```

**Purpose**: Simple API for one-shot generation with full attention data.

**Note on positions**: Kobold doesn't track or return position IDs. Client assigns positions when adding tokens to ConversationState.

---

### 5. Text Generation (Streaming WebSocket)
**`WS /api/v1/generate/stream`**

Generate text token-by-token with real-time attention extraction. **This is the primary endpoint for Halo Weave.**

#### Connection
```javascript
const ws = new WebSocket('ws://localhost:5001/api/v1/generate/stream');
```

#### Request Message (Client → Server)
```json
{
  "type": "generate",
  "request_id": "uuid-1234-5678",
  "input_ids": [151644, 1587, 198, 2610, ...],
  "max_new_tokens": 50,
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 40,
  "repetition_penalty": 1.0,
  "stop_tokens": [151645],
  "banned_tokens": [151644],
  "return_attention": true,
  "attention_format": "per_layer"
}
```

**Parameters**:
- `input_ids`: Full conversation context as token IDs (NOT text)
- `max_new_tokens`: Maximum tokens to generate
- `stop_tokens`: Stop generation if these tokens are sampled (e.g., `<|im_end|>`)
- `banned_tokens`: Force logit to `-inf` for these tokens (e.g., `<|im_start|>` to prevent role breaking)
- `return_attention`: Whether to include attention tensors in response
- `attention_format`: `"per_layer"` (default) or `"aggregated"` (not recommended)

#### Response Messages (Server → Client)

**Token Event**:
```json
{
  "type": "token",
  "request_id": "uuid-1234-5678",
  "token": {
    "token_id": 40,
    "text": "I",
    "logprob": -0.234,
    "top_logprobs": [
      {"token_id": 40, "text": "I", "logprob": -0.234},
      {"token_id": 791, "text": "The", "logprob": -1.567}
    ]
  },
  "attention": {
    "format": "per_layer",
    "shape": [28, 28, 267],
    "context_length": 267,
    "encoding": "base64",
    "dtype": "float32",
    "data": "AAAA... (base64 encoded)"
  }
}
```

**Attention Data Structure** (after base64 decode):
- **Shape**: `[num_layers, num_heads, context_length]`
- **Interpretation**: Attention FROM the newly generated token TO all previous tokens
- **Example**: For Qwen 7B (28 layers, 28 heads) with 267 tokens in context:
  - Shape: `[28, 28, 267]`
  - `attention[layer][head][i]` = attention weight from new token to `input_ids[i]`
  - Client maintains mapping: `input_ids[i]` → conversation position ID
  - Sum over last dimension (context) = 1.0 (attention is normalized)

**Why this shape?**
- Attention indexed by input array position, NOT conversation position IDs
- Client maps `input_ids[i]` to position ID using its own metadata
- Full layer/head breakdown allows flexible aggregation
- Matches PyTorch `outputs.attentions[layer][batch, head, query, key]` format
- We extract `[:, :, -1, :]` (attention from last token to all tokens)

**Memory management**:
- **Attention tensors must be in system RAM, not VRAM**
- After forward pass, immediately copy attention from GPU to CPU
- Only model weights and active computation should consume VRAM
- With 128GB system RAM, storing 75MB per generation is trivial
- This prevents CUDA OOM errors during long conversations

**Done Event**:
```json
{
  "type": "done",
  "request_id": "uuid-1234-5678",
  "finish_reason": "stop_token",
  "total_tokens": 23,
  "generation_time_ms": 1234
}
```

**Error Event**:
```json
{
  "type": "error",
  "request_id": "uuid-1234-5678",
  "error": "CUDA out of memory",
  "error_code": "OOM"
}
```

---

## Data Format Details

### Attention Tensor Encoding

**Recommended format**: Base64-encoded NumPy array (float32)

**Why?**
- JSON-serializable (can send over WebSocket)
- Compact (much smaller than JSON array of floats)
- Fast to decode client-side (`base64.b64decode()` + `np.frombuffer()`)

**Alternative formats** (for optimization):
- **MessagePack**: Binary protocol, more efficient than JSON
- **FlatBuffers**: Zero-copy deserialization
- **Raw binary WebSocket**: Maximum efficiency (not JSON)

**Reference implementation** (Python server side):
```python
import numpy as np
import base64

# Extract attention from model output
# attention shape: (num_layers, num_heads, seq_len, seq_len)
attention_to_all_tokens = attention[:, :, -1, :]  # From last token to all tokens
# Shape: (num_layers, num_heads, seq_len)

# Convert to float32 (bfloat16 not widely supported)
attention_np = attention_to_all_tokens.cpu().numpy().astype(np.float32)

# Encode as base64
attention_bytes = attention_np.tobytes()
attention_base64 = base64.b64encode(attention_bytes).decode('ascii')

# Send in JSON
response = {
    "attention": {
        "format": "per_layer",
        "shape": list(attention_np.shape),
        "encoding": "base64",
        "dtype": "float32",
        "data": attention_base64
    }
}
```

**Client-side decode** (Python):
```python
import numpy as np
import base64

# Receive JSON
attention_info = response["attention"]

# Decode base64
attention_bytes = base64.b64decode(attention_info["data"])

# Reconstruct NumPy array
attention = np.frombuffer(attention_bytes, dtype=np.float32)
attention = attention.reshape(attention_info["shape"])

# Shape: (num_layers, num_heads, context_length)
# Now pass to AttentionTracker.update_attention()
```

**Memory cost**:
- Example: 28 layers × 28 heads × 500 tokens × 4 bytes (float32) = **1.5 MB per token**
- For 50-token generation: **75 MB total attention data**
- This is acceptable for local inference (no network egress costs)

**Optimization**: Only send attention if `return_attention=true` (for normal generation, skip it)

---

## Integration with Halo Weave

### Requirements for Halo Weave Changes

**This spec places requirements on attention_heatmap, not just KoboldCPP**. The following changes would need to be made to attention_heatmap to work with this API:

#### 1. Dynamic Model Configuration (BREAKING CHANGE)

**Current behavior**: Special tokens and chat templates are hardcoded in `text_generator.py`

**Required change**: Query `/api/v1/model/info` on startup and use returned values
- `special_tokens.im_start_id`, `special_tokens.im_end_id` for banning
- `num_layers`, `num_attention_heads` for validation
- `chat_template` for formatting (if implementing client-side chat formatting)

**Implementation location**: `backend/services/text_generator.py` or new `backend/services/kobold_client.py`

**Why this matters**: Makes attention_heatmap model-agnostic. Can swap models in KoboldCPP without modifying attention_heatmap code.

#### 2. Adapter Layer

**Create new file**: `backend/services/kobold_client.py`
- Wraps KoboldCPP API calls
- Handles tokenization via `/api/v1/tokenize`
- Manages WebSocket connection to `/api/v1/generate/stream`
- Decodes base64 attention tensors
- Converts NumPy arrays back to PyTorch tuple format for AttentionTracker

**Why separate?**: Isolates KoboldCPP-specific logic. If API changes, only this file needs updating.

#### 3. Array Index to Position ID Mapping

**Current behavior**: AttentionTracker expects attention indexed by conversation position

**Required change**: Add mapping layer in attention flow

**The problem**: After pruning, position IDs become non-sequential
```python
# Before pruning: positions are sequential
conversation_state.tokens = [
    TokenAttention(position=0, token="Hello"),
    TokenAttention(position=1, token=","),
    TokenAttention(position=2, token=" world"),
]
input_ids = [9707, 11, 1234]  # len=3

# After pruning: position 1 is deleted, but positions don't renumber
conversation_state.tokens = [
    TokenAttention(position=0, token="Hello", deleted=False),
    TokenAttention(position=1, token=",", deleted=True),  # DELETED
    TokenAttention(position=2, token=" world", deleted=False),
]
input_ids = [9707, 1234]  # len=2 (skips deleted token)

# Kobold returns attention shape: [layers, heads, 2]
# attention[:, :, 0] → input_ids[0] → position 0 ✓
# attention[:, :, 1] → input_ids[1] → position 2 (NOT 1!) ✓
```

**Solution**: Build index-to-position mapping before generation
```python
active_tokens = [t for t in conversation_state.tokens if not t.deleted]
index_to_position = {i: token.position for i, token in enumerate(active_tokens)}
# {0: 0, 1: 2}

# When processing attention:
for i in range(len(attention[0, 0, :])):
    position = index_to_position[i]
    score = attention[:, :, i].mean()  # or other aggregation
    conversation_state.tokens[position].attention_score = score
```

**Why needed**: Kobold returns attention indexed by input array, not by position ID. After pruning, these diverge.

#### 4. Remove PyTorch Model Loading

**Files to modify/delete**:
- Remove `AutoModelForCausalLM.from_pretrained()` code from `text_generator.py`
- Remove `transformers` dependency (or keep only for tokenizer if not using Kobold's)
- Remove CUDA device management code

**Why**: KoboldCPP handles model loading. Attention_heatmap becomes a pure client.

#### 5. Base64 Attention Decoding

**Add to** `backend/services/kobold_client.py`:
```python
def _decode_attention(self, attention_info):
    """Decode base64 attention to tuple of PyTorch tensors"""
    import base64
    import numpy as np
    import torch

    # Decode base64
    attention_bytes = base64.b64decode(attention_info["data"])
    attention_np = np.frombuffer(attention_bytes, dtype=np.float32)
    attention_np = attention_np.reshape(attention_info["shape"])

    # Convert to tuple of tensors (AttentionTracker expects this)
    num_layers = attention_np.shape[0]
    attention_tensors = tuple(
        torch.from_numpy(attention_np[i:i+1, :, :])
        for i in range(num_layers)
    )

    return attention_tensors
```

**Why**: AttentionTracker expects `Tuple[torch.Tensor, ...]`, not NumPy array

#### 6. Documentation Updates

**Update** `CLAUDE.md`:
- Document KoboldCPP as the inference backend
- Remove PyTorch-specific setup instructions
- Add KoboldCPP API URL configuration
- Document model info querying requirement

**Why**: Future Claude needs to know the architecture has changed

---

### Current Halo Weave Architecture

**ConversationState** (`text_generator.py`):
- Maintains master token list with metadata (position, turn_id, sentence_id, attention_score)
- Tokenizes messages ONCE when added to conversation
- Provides `get_input_ids()` to convert token dictionary → input_ids for model

**AttentionTracker** (`attention_tracker.py`):
- Receives raw attention tensors from model
- Aggregates across layers/heads (mean/max/weighted/last_layer)
- Applies distance weighting (filter local attention)
- Accumulates scores over time: `score = old_score + attention - decay_rate`
- Syncs updated scores back to ConversationState

### Integration Flow

1. **User sends message**:
   - Halo Weave calls `/api/v1/tokenize` with message text
   - Stores tokens in ConversationState with metadata
   - Increments turn counter, sentence counter

2. **Generate response**:
   - Halo Weave calls `ConversationState.get_input_ids()` to get full context
   - Opens WebSocket to `/api/v1/generate/stream`
   - Sends `input_ids` (NOT text) with generation parameters

3. **Receive tokens**:
   - For each token event:
     - Decode attention data from base64
     - Reshape to `(num_layers, num_heads, context_length)`
     - Convert to PyTorch tensor or keep as NumPy
     - Pass to `AttentionTracker.update_attention(attentions, token_obj, step, conversation_tokens)`
     - Tracker aggregates, applies distance weighting, updates scores
     - Sync scores back to ConversationState
     - Display token in UI with color-coded brightness

4. **End of generation**:
   - Receive `{"type": "done"}` event
   - Check if context exceeds threshold
   - If yes, prune low-brightness sentences
   - Send pruning events to frontend for animation

### Adapter Layer (Recommended)

Create `kobold_client.py` in Halo Weave to handle KoboldCPP-specific logic:

```python
class KoboldClient:
    def __init__(self, base_url="http://localhost:5001"):
        self.base_url = base_url
        self.ws = None

    async def get_model_info(self):
        """Get model architecture details"""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/api/v1/model/info") as resp:
                return await resp.json()

    def tokenize(self, text: str) -> List[TokenAttention]:
        """Tokenize text and return token objects ready for ConversationState"""
        response = requests.post(f"{self.base_url}/api/v1/tokenize", json={
            "text": text,
            "add_special_tokens": False
        })
        data = response.json()

        tokens = []
        for idx, token_data in enumerate(data["tokens"]):
            tokens.append(TokenAttention(
                token=token_data["text"],
                token_id=token_data["token_id"],
                position=None,  # ConversationState assigns this
                attention_score=1.0,  # Fail bright
                turn_id=None,  # ConversationState assigns this
                message_role=None,
                sentence_id=None,
                line_id=None,
                deleted=False
            ))
        return tokens

    async def generate_stream(self, input_ids, config, on_token_callback):
        """Stream generation with real-time attention"""
        self.ws = await websockets.connect(f"{self.base_url}/api/v1/generate/stream")

        # Send generation request
        await self.ws.send(json.dumps({
            "type": "generate",
            "request_id": str(uuid.uuid4()),
            "input_ids": input_ids,
            "max_new_tokens": config.max_length,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "return_attention": True,
            "attention_format": "per_layer",
            "banned_tokens": [self.tokenizer.im_start_id] if hasattr(self.tokenizer, 'im_start_id') else []
        }))

        # Receive token stream
        async for message in self.ws:
            data = json.loads(message)

            if data["type"] == "token":
                # Decode attention
                attention = self._decode_attention(data["attention"])

                # Call user's callback
                await on_token_callback(
                    token_id=data["token"]["token_id"],
                    token_text=data["token"]["text"],
                    attention=attention
                )

            elif data["type"] == "done":
                break

            elif data["type"] == "error":
                raise Exception(data["error"])

        await self.ws.close()

    def _decode_attention(self, attention_info):
        """Decode base64 attention data to NumPy array or PyTorch tensor"""
        attention_bytes = base64.b64decode(attention_info["data"])
        attention_np = np.frombuffer(attention_bytes, dtype=np.float32)
        attention_np = attention_np.reshape(attention_info["shape"])

        # Convert to PyTorch tensor (AttentionTracker expects this)
        # Convert back to tuple of tensors (one per layer)
        num_layers = attention_np.shape[0]
        attention_tensors = tuple(
            torch.from_numpy(attention_np[i:i+1, :, :]) for i in range(num_layers)
        )

        return attention_tensors
```

---

## Alternative: Aggregated Attention (NOT RECOMMENDED)

If bandwidth or latency is a concern, KoboldCPP could offer pre-aggregated attention:

**Request**:
```json
{
  "attention_format": "aggregated",
  "aggregation_mode": "mean"
}
```

**Response** (much smaller):
```json
{
  "type": "token",
  "token": {...},
  "attention": {
    "format": "aggregated",
    "shape": [267],
    "data": [0.0023, 0.0045, 0.0012, ...]
  }
}
```

**Attention shape**: `[context_length]` - one score per token

**Why NOT recommended**:
- Loses flexibility (can't change aggregation mode without re-generating)
- Halo Weave has 5 aggregation modes (mean, max, weighted_layers, last_layer, etc.)
- Client-side aggregation is cheap (few milliseconds)
- Server shouldn't make UX decisions for the client

**When to use**: Production deployment with bandwidth constraints or mobile clients

---

## Error Handling

### Common Errors

**OOM (Out of Memory)**:
```json
{
  "type": "error",
  "error": "CUDA out of memory",
  "error_code": "OOM",
  "context_length": 8192,
  "vram_used_mb": 23456,
  "vram_total_mb": 24576
}
```

**Invalid Input**:
```json
{
  "type": "error",
  "error": "Token ID 999999 not in vocabulary (vocab_size=151936)",
  "error_code": "INVALID_TOKEN"
}
```

**Model Not Loaded**:
```json
{
  "type": "error",
  "error": "No model loaded",
  "error_code": "NO_MODEL"
}
```

---

## Performance Considerations

### Attention Tensor Size

**Calculation**:
```
Size = num_layers × num_heads × context_length × 4 bytes (float32)

Examples:
- Qwen 7B (28 layers, 28 heads, 500 tokens): 1.5 MB per token
- Llama 7B (32 layers, 32 heads, 500 tokens): 2.0 MB per token
- Qwen 14B (40 layers, 40 heads, 500 tokens): 3.1 MB per token
```

**Mitigation strategies**:
1. **Only send when needed**: `return_attention=false` for normal generation
2. **Compression**: gzip the base64 data (often 10x reduction for attention patterns)
3. **Sparse attention**: Send only top-K attention weights per token
4. **Half precision**: Use float16 instead of float32 (2x smaller, minimal accuracy loss)

### Latency

**Bottlenecks**:
1. **Model forward pass**: ~50-200ms per token (GPU-bound)
2. **Attention extraction**: ~1-5ms (copy from GPU to CPU)
3. **Base64 encoding**: ~5-20ms (CPU-bound, scales with context length)
4. **WebSocket transmission**: ~1-10ms (network-bound)

**Total latency per token**: ~60-250ms (dominated by model inference, not attention encoding)

**Optimization**: Run encoding/transmission in parallel with next forward pass

---

## Security Considerations

### Input Validation

- **Token IDs**: Validate all token IDs are in `[0, vocab_size)`
- **Context length**: Reject if `len(input_ids) > max_position_embeddings`
- **Banned tokens**: Validate banned token IDs exist in vocabulary

### Resource Limits

- **Max tokens per request**: Prevent infinite generation
- **Max context length**: Prevent OOM attacks
- **Rate limiting**: Prevent API abuse
- **Timeout**: Kill generation after N seconds

### WebSocket Security

- **Authentication**: Token-based auth for production
- **Origin validation**: Check `Origin` header
- **Message size limits**: Prevent memory exhaustion

---

## Testing & Validation

### Unit Tests

1. **Tokenization consistency**: Same text → same tokens every time
2. **Attention shape validation**: Verify `(num_layers, num_heads, seq_len)`
3. **Base64 encoding**: Round-trip test (encode → decode → compare)
4. **Stop token handling**: Generation stops when stop token sampled

### Integration Tests

1. **Halo Weave end-to-end**: Full conversation with attention tracking
2. **Memory leak test**: Generate 1000 tokens, check memory usage
3. **Concurrent requests**: Multiple WebSocket connections
4. **Model switching**: Unload model A, load model B, verify attention shape changes

### Validation Checklist

- [ ] Attention tensor shape matches `(num_layers, num_heads, context_length)`
- [ ] Sum of attention across context dimension ≈ 1.0 (normalized)
- [ ] All attention values in `[0, 1]`
- [ ] Token positions map correctly to attention indices
- [ ] Special tokens (BOS, EOS, im_start, im_end) handled correctly
- [ ] Stop tokens terminate generation
- [ ] Banned tokens never appear in output
- [ ] OOM errors handled gracefully
- [ ] WebSocket reconnection works after disconnect

---

## Future Extensions

### 1. Sparse Attention
Only send top-K attention weights per token:
```json
{
  "attention": {
    "format": "sparse",
    "top_k": 50,
    "data": [
      {"position": 245, "layers": [0.023, 0.045, ...]},
      {"position": 198, "layers": [0.019, 0.038, ...]}
    ]
  }
}
```

### 2. Attention Visualization Data
Pre-compute attention aggregations for visualization:
```json
{
  "attention": {
    "format": "viz",
    "mean_by_layer": [0.002, 0.003, ...],
    "max_by_layer": [0.045, 0.067, ...],
    "entropy": 4.23
  }
}
```

### 3. Multi-Turn Pruning
Server maintains short-term history and suggests pruning points:
```json
{
  "type": "pruning_suggestion",
  "prunable_sentences": [
    {"turn_id": 1, "sentence_id": 3, "max_attention": 0.05}
  ]
}
```

### 4. Batch Generation
Generate multiple responses in parallel (for sampling, beam search):
```json
{
  "input_ids": [...],
  "num_samples": 4,
  "return_attention": true
}
```

---

## Appendix A: Complete Example Flow

### Scenario: User asks "What is the capital of France?"

**Step 1: Tokenize user message**
```bash
POST /api/v1/tokenize
{
  "text": "What is the capital of France?",
  "add_special_tokens": false
}

Response:
{
  "tokens": [
    {"token_id": 3555, "text": "What"},
    {"token_id": 374, "text": " is"},
    {"token_id": 279, "text": " the"},
    {"token_id": 6864, "text": " capital"},
    {"token_id": 315, "text": " of"},
    {"token_id": 9822, "text": " France"},
    {"token_id": 30, "text": "?"}
  ]
}
```

**Step 2: Halo Weave stores tokens**
```python
conversation_state.add_message(
    role="user",
    tokens=[
        TokenAttention(token="What", token_id=3555, position=0, ...),
        TokenAttention(token=" is", token_id=374, position=1, ...),
        ...
    ]
)
```

**Step 3: Build input_ids for generation**
```python
input_ids = conversation_state.get_input_ids()
# [151644, 1587, 198, ..., 3555, 374, 279, 6864, 315, 9822, 30]
# <|im_start|>system\nYou are...<|im_end|><|im_start|>user\nWhat is...
```

**Step 4: Open WebSocket and send generation request**
```javascript
ws.send(JSON.stringify({
  "type": "generate",
  "input_ids": input_ids,
  "max_new_tokens": 50,
  "temperature": 0.7,
  "return_attention": true
}))
```

**Step 5: Receive token stream**

Token 1: "The"
```json
{
  "type": "token",
  "token": {"token_id": 791, "text": "The"},
  "attention": {
    "shape": [28, 28, 267],
    "data": "AAAA..."
  }
}
```

Halo Weave:
1. Decodes attention: `(28 layers, 28 heads, 267 tokens)`
2. Passes to AttentionTracker
3. Tracker aggregates: `(267,)` - one score per token
4. Applies distance weighting
5. Updates scores: `score = old_score + weighted_attention - decay_rate`
6. Syncs to ConversationState
7. Displays "The" in UI, updates heatmap colors

Token 2: " capital"
```json
{
  "type": "token",
  "token": {"token_id": 6864, "text": " capital"},
  "attention": {
    "shape": [28, 28, 268],
    "data": "BBBB..."
  }
}
```

Note: context_length incremented (267 → 268) because "The" was added

... (repeat for 10 more tokens) ...

**Step 6: Generation complete**
```json
{
  "type": "done",
  "finish_reason": "stop_token",
  "total_tokens": 12
}
```

Halo Weave:
1. Checks context length: 279 tokens
2. If > max_context_tokens (500), prune low-brightness sentences
3. Updates UI with final message
4. Closes WebSocket

---

## Appendix B: Attention Extraction Implementation Notes

### For Future Claude Implementing This in KoboldCPP

**Key files to modify**:
1. `llama.cpp` or Python inference wrapper - Extract attention during forward pass
2. API handler - Add WebSocket endpoint, encode attention data
3. Model config - Expose num_layers, num_heads to API

**PyTorch attention extraction** (reference):
```python
# Load model with attention output enabled
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0"
)

# Forward pass with attention
outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    output_attentions=True  # KEY FLAG
)

# outputs.attentions is a tuple of tensors, one per layer
# Each tensor: (batch_size, num_heads, seq_len, seq_len)
attention_tensors = outputs.attentions

# Extract attention FROM new token (last position) TO all tokens
attention_from_new_token = torch.stack([
    attn[0, :, -1, :]  # [batch=0, all_heads, from_last_token, to_all_tokens]
    for attn in attention_tensors
])
# Shape: (num_layers, num_heads, seq_len)

# Convert to numpy for JSON serialization
attention_np = attention_from_new_token.cpu().numpy().astype(np.float32)

# Encode as base64
attention_bytes = attention_np.tobytes()
attention_base64 = base64.b64encode(attention_bytes).decode('ascii')

# Send via WebSocket
await websocket.send(json.dumps({
    "type": "token",
    "token": {"token_id": next_token.item(), "text": tokenizer.decode(next_token)},
    "attention": {
        "format": "per_layer",
        "shape": list(attention_np.shape),
        "encoding": "base64",
        "dtype": "float32",
        "data": attention_base64
    }
}))
```

**llama.cpp attention extraction** (pseudocode):
```cpp
// During forward pass, store attention weights
struct llama_context {
    // ...
    float * attention_weights;  // Allocated: n_layers * n_heads * n_ctx
    bool output_attentions;
};

// After softmax(Q @ K.T / sqrt(d_k))
if (ctx->output_attentions) {
    // Copy attention weights from new token to buffer
    memcpy(
        ctx->attention_weights + layer_idx * n_heads * n_ctx,
        attention_probs,
        n_heads * n_ctx * sizeof(float)
    );
}

// In API handler, read attention_weights buffer and encode for transmission
```

**Performance tip**: Attention extraction adds ~5-10% overhead to generation (copying from GPU to CPU). Only enable when client requests it.

---

## Conclusion

This API specification prioritizes **flexibility, correctness, and debuggability** over convenience. By exposing raw attention tensors, we enable Halo Weave to experiment with different aggregation strategies, distance weighting modes, and pruning thresholds without requiring server-side changes.

The stateless design keeps KoboldCPP simple (no conversation state management) while giving Halo Weave full control over the token dictionary and attention accumulation logic.

**Next step**: Implement this API in KoboldCPP and verify attention tensor shapes match expectations using the validation checklist in the Testing section.
