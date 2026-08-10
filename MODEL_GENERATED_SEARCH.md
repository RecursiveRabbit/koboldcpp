# Design Document: Model-Generated Semantic Search Queries

## Overview

We want the LLM to generate a "search preview" by appending the user's new message to the KV cache from the previous turn's generation, generating 50 tokens, and using that as a semantic search query to resurrect relevant context.

**The key insight:** KoboldCPP already has the full context cached from the previous assistant response. We just need to append the new user message and generate a preview WITHOUT committing it to cache.

## Current Flow (Without Preview)

```
Turn N (Assistant response completes):
  Context: [System, User1, Asst1, User2, Asst2, ..., UserN, AsstN]
  KoboldCPP: Generates AsstN → Returns tokens + attention
  KoboldCPP: [Saves KV cache for entire context]

Turn N+1 (User types new message):
  Halo Weave: User types "Tell me more about brightness"
  Halo Weave: Tokenize user message
  Halo Weave: Do semantic resurrection (search using user message text)
  Halo Weave: Build context: [System, ..., AsstN, UserN+1] + resurrected chunks
  Halo Weave: Cull/prune low brightness chunks
  Halo Weave: Tokenize full context (10-30k tokens)
  Halo Weave → KoboldCPP: Full context
  KoboldCPP: [Discards old cache, rebuilds from scratch]
  KoboldCPP: Process 10-30k tokens → Generate AsstN+1
```

**Problem:** Semantic search uses user's raw message "Tell me more about brightness" which might not match the answer well.

## Proposed Flow (With Preview)

```
Turn N (Assistant response completes):
  Context: [System, User1, Asst1, User2, Asst2, ..., UserN, AsstN]
  KoboldCPP: Generates AsstN → Returns tokens + attention
  KoboldCPP: [Saves KV cache for entire context] ← CACHE PERSISTS

Turn N+1 (User types new message):
  Halo Weave: User types "Tell me more about brightness"
  Halo Weave: Tokenize user message → [token1, token2, token3]

  ↓

  Halo Weave → KoboldCPP.preview: Append [token1, token2, token3] to cached context
  KoboldCPP: [Loads cache from Turn N]
  KoboldCPP: Append UserN+1 tokens to cache
  KoboldCPP: Generate 50 tokens: "Brightness scoring uses magnitude voting..."
  KoboldCPP: [DISCARD preview, restore cache to Turn N state]
  KoboldCPP → Halo Weave: "Brightness scoring uses magnitude voting..."

  ↓

  Halo Weave: Search semantic index using preview text
  Halo Weave: Resurrect relevant chunks (2.5k tokens)
  Halo Weave: Build context: [System, ..., AsstN, UserN+1] + resurrected chunks
  Halo Weave: Cull/prune low brightness chunks
  Halo Weave: Tokenize full context (10-30k tokens)

  ↓

  Halo Weave → KoboldCPP.generate: Full context
  KoboldCPP: [Discards Turn N cache, rebuilds from scratch]
  KoboldCPP: Process 10-30k tokens → Generate AsstN+1
  KoboldCPP: [Saves new KV cache for Turn N+1]
```

**Key insight:** Preview reuses Turn N cache (fast), then we discard it and rebuild for Turn N+1 generation anyway (because resurrection changed context).

## The Value of Cache Reuse

**Without preview (current system):**
- Full generation: Process 20,000 tokens (typical active context)

**With preview:**
- Preview: Append 10 tokens to cached 20k context = ~10 token processing + 50 token generation
- Full generation: Process 22,500 tokens (20k + 2.5k resurrected)

**Overhead:**
- Preview: ~60 tokens worth of work (10 tokens KV + 50 tokens generation)
- NOT 20,000 tokens (because we reuse cache!)

**Time:**
- Preview: ~60 tokens * 27ms/token = ~1.6 seconds for 50 tokens = ~32ms/token
- Actually more like ~1-2 seconds total (50 tokens at typical generation speed)

This is MUCH better than regenerating 20k tokens!

## API Design

### Preview Endpoint

**Endpoint:** `POST /api/v1/generate/preview`

**Request:**
```javascript
{
  "append_tokens": [
    {"token_id": 1234, "text": "Tell"},
    {"token_id": 5678, "text": " me"},
    {"token_id": 9012, "text": " more"}
  ],
  "max_tokens": 50,
  "temperature": 0.7,
  "top_p": 0.9,
  "use_cached_context": true  // Use KV cache from last generation
}
```

**What Halo Weave is asking:**
> "You have a KV cache from the last generation. Append these user tokens and generate 50 tokens."

**Response:**
```javascript
{
  "text": "Brightness scoring uses magnitude voting across all active tokens...",
  "token_count": 47,
  "stopped_reason": "stop_token",
  "cache_hit": true  // Confirm cache was used
}
```

**What KoboldCPP needs to do:**
1. Load KV cache from last generation
2. Append user tokens to cache 
3. Generate up to max_tokens
4. Return generated text

### Full Generation Endpoint (Existing)

**Endpoint:** `POST /api/extra/generate/stream` (no changes needed)

**Request:**
```javascript
{
  "prompt": [/* full context with resurrected chunks */],
  "max_length": 200,
  "temperature": 0.7,
  "top_p": 0.9,
  "stream": true
}
```

**What KoboldCPP does:**
- Discard old cache (context changed due to resurrection/pruning)
- Process full prompt from scratch
- Generate response with streaming
- Save new KV cache

## KoboldCPP Implementation

### Cache Management

```python
class CacheManager:
    def __init__(self):
        self.current_cache = None  # KVCache from last generation

    def save(self, kv_cache):
        """Save KV cache after generation"""
        self.current_cache = kv_cache

    def get(self):
        """Get current cached KV"""
        return self.current_cache

    def clear(self):
        """Clear cache (happens on full generation with new context)"""
        self.current_cache = None

# Global instance
cache_manager = CacheManager()
```

### Preview Endpoint

```python
@app.route('/api/v1/generate/preview', methods=['POST'])
def generate_preview():
    """Generate preview by appending to cached context"""
    data = request.json

    append_tokens = data['append_tokens']
    max_tokens = data.get('max_tokens', 50)
    temperature = data.get('temperature', 0.7)
    top_p = data.get('top_p', 0.9)

    # Get cached KV from last generation
    cached_kv = cache_manager.get()

    if cached_kv is None:
        return jsonify({'error': 'No cached context available'}), 400

    # Make a temporary copy of the cache
    temp_kv = copy.deepcopy(cached_kv)

    # Append user tokens to temporary cache
    # (This is model-specific - adjust for your inference engine)
    # For transformers: past_key_values parameter

    start_time = time.time()

    generated_tokens = model.generate(
        input_ids=append_tokens,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        past_key_values=temp_kv,  # Use cached KV
        return_dict_in_generate=True
    )

    # Extract text from generated tokens
    generated_text = tokenizer.decode(generated_tokens.sequences[0])

    # DON'T save temp_kv - we're discarding this preview
    # cache_manager still holds the original cached_kv

    return jsonify({
        'text': generated_text,
        'token_count': len(generated_tokens.sequences[0]),
        'stopped_reason': 'max_tokens',  # or check for stop token
        'cache_hit': True,
        'generation_time_ms': int((time.time() - start_time) * 1000)
    })
```

### Generation Endpoint (Modified)

```python
@app.route('/api/extra/generate/stream', methods=['POST'])
def generate_stream():
    """Full generation with streaming"""
    data = request.json
    prompt_tokens = data['prompt']

    # Clear old cache - context has changed due to resurrection/pruning
    cache_manager.clear()

    # Process full context from scratch
    new_kv = None

    for token_data in model.generate_stream(
        input_ids=prompt_tokens,
        max_new_tokens=data.get('max_length', 200),
        temperature=data.get('temperature', 0.7),
        top_p=data.get('top_p', 0.9),
        past_key_values=new_kv  # Start fresh
    ):
        # Extract attention data
        attention = extract_attention_from_model()

        yield sse_format({
            'type': 'token',
            'token': token_data,
            'attention': {
                'format': 'aggregated',
                'encoding': 'base64',
                'data': encode_attention(attention)
            }
        })

        # Update KV cache as we generate
        new_kv = model.get_past_key_values()

    # Save final KV cache for next turn's preview
    cache_manager.save(new_kv)

    yield sse_format({'type': 'done'})
```

## Halo Weave Implementation

### kobold_client.js

```javascript
class KoboldClient {
    async generatePreview(userTokens, maxTokens = 50) {
        const response = await fetch(`${this.baseUrl}/api/v1/generate/preview`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                append_tokens: userTokens.map(t => ({
                    token_id: t.token_id,
                    text: t.text
                })),
                max_tokens: maxTokens,
                temperature: this.config.temperature,
                top_p: this.config.top_p,
                use_cached_context: true
            })
        });

        if (!response.ok) {
            throw new Error(`Preview generation failed: ${response.statusText}`);
        }

        const result = await response.json();

        if (!result.cache_hit) {
            console.warn('Preview did not use cached context');
        }

        return result.text;
    }
}
```

### app.js

```javascript
async _handleUserMessage(text) {
    // 1. Tokenize user message
    const userTokens = await this.client.tokenize(text);

    // 2. Add to conversation (in memory, will persist later)
    await this._addUserMessage(userTokens);

    // 3. Generate preview using cached context from last turn
    let previewText = text;  // Fallback

    if (this.settings.previewEnabled) {
        try {
            previewText = await this.client.generatePreview(
                userTokens,
                this.settings.previewMaxTokens || 50
            );

            console.log('Generated preview:', previewText.substring(0, 100));
        } catch (error) {
            console.warn('Preview generation failed, using user message:', error);
        }
    }

    // 4. Semantic resurrection using preview
    if (this.semanticIndex) {
        await this._performSemanticResurrection(previewText);
    }

    // 5. Prune low brightness chunks
    await this._checkPruning();

    // 6. Full generation with resurrected + pruned context
    await this._startGeneration();
}
```

## Performance Analysis

**Turn N+1 with preview:**

1. **Preview generation:**
   - Cached context: 20,000 tokens (already in KV cache)
   - Append user tokens: 10 tokens (~10ms for KV calculation)
   - Generate preview: 50 tokens (~50 * 25ms = 1,250ms typical)
   - **Total: ~1,260ms**

2. **Semantic search:**
   - Embed preview: ~40-60ms
   - Search index: ~4ms
   - Resurrect chunks: ~30ms
   - **Total: ~94ms**

3. **Full generation:**
   - Process context: 22,500 tokens (20k + 2.5k resurrected)
   - KV rebuild time: ~22,500 * 1ms = 22,500ms (depends on model/hardware)
   - Generate: 200 tokens (~200 * 25ms = 5,000ms typical)
   - **Total: ~27,500ms**

**Total time: ~28,854ms (~29 seconds)**

**Turn N+1 without preview:**
- Full generation: ~27,500ms

**Overhead: ~1,354ms (~1.3 seconds)**

This is MUCH better than regenerating 20k tokens (would be ~20 seconds overhead).

## Why This Works

**The magic:** KoboldCPP has already processed the entire Turn N context when it generated the assistant's response. That KV cache is sitting there, unused.

**What we're asking:** "Hey, you have that cache from last turn. Just append these 10 new tokens and generate 50 more. Don't save it, I'm going to ask you to regenerate anyway, but I need to know what you'd say first."

**Cost:** Processing 10 tokens + generating 50 tokens = ~1.3 seconds

**Value:** Semantic search gets a context-aware, answer-shaped query instead of the raw user message.

## Edge Cases

### 1. No Cached Context Available

**Scenario:** First turn, or KoboldCPP restarted

**Handling:**
```javascript
try {
    previewText = await client.generatePreview(userTokens, 50);
} catch (error) {
    if (error.message.includes('No cached context')) {
        console.warn('No cache available, using user message for search');
        previewText = text;
    }
}
```

### 2. Cache Mismatch

**Scenario:** We pruned/resurrected between turns, cache is for different context

**Handling:** This is actually fine! The preview will be based on the OLD context (before resurrection), which is what we want - we're asking "given what you know now, what would you say?" to determine what to resurrect.

### 3. Preview Generation Fails

**Fallback:** Always use user message text as search query (current behavior)

## UI Settings

```html
<div class="setting-group">
    <h3>Preview Generation</h3>

    <label>
        <input type="checkbox" id="preview-enabled" checked>
        Enable model-generated search queries
        <span class="help">Use model's response preview for semantic search (~1s added latency)</span>
    </label>

    <label>
        Preview Max Tokens:
        <input type="number" id="preview-max-tokens" value="50" min="10" max="200">
        <span class="help">Tokens to generate for search query (more tokens = better query, higher latency)</span>
    </label>
</div>
```

## Summary: What KoboldCPP Needs to Implement

**One new endpoint that:**
1. Loads KV cache from previous generation
2. Appends new tokens to cached KV
3. Generates up to N tokens
4. Returns generated text

**One modification to existing endpoint:**
- Save KV cache after each generation for next preview

**That's it.**

## Success Metrics

**This feature succeeds if:**
1. ✅ Preview reuses cached KV (not regenerating 20k tokens)
2. ✅ Preview adds ~1-2s latency (acceptable)
3. ✅ Semantic search quality improves (answer-shaped queries)
4. ✅ Graceful fallback when no cache available
5. ✅ Optional (user can disable if they want)

## Open Questions

1. **How long should preview be?** 50 tokens seems reasonable, but might need tuning
    a. The absolute max is 256 tokens, that's as large as the embedding model will take as input. We are currently generating at 109.6T/s and that includes pre-processing which we won't do. Leave it as an open variable on the endpoint and we can see if we feel like extending it.
2. **Should preview use different temperature?** Higher temp for diverse queries?
    a. Another dial to turn, we expose temperature for normal generation, I don't know why we wouldn't here. Make it one of those variables with a sensible default in case we decide we don't care. 
3. **Cache lifetime?** How long should KoboldCPP keep the cache? (Probably just until next generation)
    a. This flows into 4. Typical KoboldCPP behavior is to persist cache per conversation, logic to do this must already exist. 
4. **Multi-user support?** If KoboldCPP serves multiple clients, need per-session cache management
    a. Again, Kobold already handles this for normal users, if possible we should use existing code paths. 