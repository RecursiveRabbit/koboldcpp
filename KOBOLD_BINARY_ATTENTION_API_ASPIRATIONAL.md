# KoboldCPP Binary Attention Streaming API

## The Problem

Current attention data delivery adds **64 seconds** of overhead to a 9-second generation:

| Stage | Time | Per Token | Issue |
|-------|------|-----------|-------|
| Buffer ops | 37.6s | 73.5ms | String concat + split on 9MB chunks |
| JSON parse | 5.5s | 10.9ms | Parsing 9MB JSON per token |
| Base64 decode | 21.0s | 41.0ms | Character-by-character conversion |
| **Total overhead** | **64.1s** | **125.4ms** | 7x slower than inference |

The root cause: attention tensors are base64-encoded inside JSON inside SSE text streams. Each token sends ~9MB of base64 text that must be accumulated, parsed, and decoded.

## The Solution

WebSocket with binary frames. Zero encoding. Zero parsing. Zero copying.

## Protocol Design

### Connection
```
ws://localhost:5001/api/extra/generate/stream/ws
```

### Client → Server (JSON text frame)
```json
{
  "input_ids": [1, 2, 3, ...],
  "max_length": 512,
  "temperature": 0.7,
  "top_p": 0.9,
  "output_attentions": true
}
```

### Server → Client (alternating frames)

**Frame 1: Text (JSON) - Token metadata**
```json
{
  "type": "token",
  "token_id": 1234,
  "text": "Hello"
}
```

**Frame 2: Binary - Raw attention tensor**
```
[float32 × num_layers × num_heads × context_length]
```

No shape metadata needed. Client knows:
- `num_layers`, `num_heads` from `/api/v1/model` at startup
- `context_length` from its own token count

**Final Frame: Text (JSON) - Completion**
```json
{
  "type": "done",
  "finish_reason": "length",
  "total_tokens": 512
}
```

### Client Implementation

```javascript
const ws = new WebSocket('ws://localhost:5001/api/extra/generate/stream/ws');
let pendingToken = null;

ws.onmessage = (event) => {
    if (typeof event.data === 'string') {
        const data = JSON.parse(event.data);
        
        if (data.type === 'token') {
            pendingToken = data;
        } else if (data.type === 'done') {
            onComplete(data);
        }
    } else {
        // Binary frame - attention for pendingToken
        const attention = {
            data: new Float32Array(event.data),  // Zero-copy!
            shape: [numLayers, numHeads, contextLength]
        };
        onToken(pendingToken, attention);
        pendingToken = null;
    }
};

ws.send(JSON.stringify(config));
```

### Server Implementation (Pseudocode)

```cpp
// In generation loop, after sampling token:

// 1. Send token JSON (existing code, just remove attention field)
json token_msg = {
    {"type", "token"},
    {"token_id", token_id},
    {"text", token_text}
};
ws_send_text(token_msg.dump());

// 2. Send attention binary (new - one line)
ws_send_binary(
    attention_data.data(),
    num_layers * num_heads * context_len * sizeof(float)
);
```

## Performance Target

| Stage | Current | Target | Savings |
|-------|---------|--------|---------|
| Buffer ops | 37.6s | 0s | Eliminated - no accumulation |
| JSON parse | 5.5s | 0.5s | 10x smaller JSON (no base64) |
| Base64 decode | 21.0s | 0s | Eliminated - raw binary |
| **Total** | **64.1s** | **0.5s** | **99% reduction** |

Expected wall clock: **80s → 13s** for 512 tokens.

## Backward Compatibility

- Keep existing SSE endpoint for clients that don't need attention
- New WebSocket endpoint is opt-in
- `/api/v1/model` already exposes layer/head counts

## Data Flow Visualization

```
CURRENT (9MB per token):
┌─────────────────────────────────────────────────────────────┐
│ SSE Text Stream                                             │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ data: {"type":"token","attention":{"data":"SGVsbG8g... │ │
│ │ ...8.7MB of base64...","shape":[28,28,2000]}}          │ │
│ └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
        ↓ accumulate string (37.6s)
        ↓ JSON.parse (5.5s)  
        ↓ atob + charCodeAt loop (21.0s)
        ↓
    Float32Array

TARGET (6.9MB per token):
┌──────────────────────┐  ┌──────────────────────────────────┐
│ Text Frame (50 bytes)│  │ Binary Frame (6.9MB raw floats)  │
│ {"type":"token",...} │  │ [f32][f32][f32][f32][f32]...     │
└──────────────────────┘  └──────────────────────────────────┘
        ↓ JSON.parse (0.1ms)      ↓ new Float32Array (0ms)
        ↓                         ↓
    Token metadata            Float32Array (zero-copy)
```

## Why This Matters

Halo Weave uses attention data to score token importance in real-time. Every token generation triggers:
1. Receive attention tensor
2. Aggregate across layers/heads  
3. Update brightness scores
4. Re-render visualization

At 512 tokens, we're processing **3.5GB** of attention data. The current encoding adds 64 seconds of pure overhead. With binary streaming, attention processing becomes invisible - the math takes 3 seconds, hidden entirely behind model inference time.

The UI already runs smoothly. The algorithm works. The only thing standing between "research prototype" and "usable tool" is this data format change.

## Files That Will Change

**KoboldCPP (your side):**
- Add WebSocket endpoint handler
- Send binary frame after each token
- Remove base64 encoding path for WS clients

**Halo Weave (my side):**
- `kobold_client.js` - WebSocket client, binary frame handling
- Already optimized and waiting

## Testing

Generate 512 tokens with attention. Compare:
- Current: ~80s wall clock
- Target: ~13s wall clock

The model runs at 50 tokens/sec. We should see that speed in the UI.
