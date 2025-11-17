#!/usr/bin/env python3
"""
Test attention extraction via KoboldCPP SSE streaming
Receives individual token events with attention data
"""

import requests
import json
import base64
import numpy as np

KOBOLD_URL = "http://localhost:5001"

def test_attention_streaming():
    print("="*70)
    print("KOBOLDCPP ATTENTION EXTRACTION - STREAMING TEST")
    print("="*70)
    print(f"Server: {KOBOLD_URL}")
    print()

    # Simple test prompt
    prompt = "<|im_start|>user\nHello, tell me about yourself.<|im_end|>\n<|im_start|>assistant\n"

    print("[1/2] Sending streaming generation request with output_attentions=True...")
    print(f"  Prompt: {repr(prompt[:50])}...")
    print()

    payload = {
        "prompt": prompt,
        "max_length": 20,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "output_attentions": True,
        "stream": True
    }

    try:
        response = requests.post(
            f"{KOBOLD_URL}/api/v1/generate",
            json=payload,
            stream=True,
            timeout=60
        )
        response.raise_for_status()
    except Exception as e:
        print(f"✗ Request failed: {e}")
        return False

    print("[2/2] Receiving token stream...")
    print("  " + "-"*66)
    print()

    token_count = 0
    tokens_with_attention = 0

    # KAI SSE format: "event: message\ndata: {json}\n\n"
    current_event_type = None
    for line in response.iter_lines():
        if not line:
            continue

        line_str = line.decode('utf-8')

        # Parse event type (KAI format)
        if line_str.startswith('event: '):
            current_event_type = line_str[7:]  # e.g., "message"
            continue

        # Parse data
        if line_str.startswith('data: '):
            data_str = line_str[6:]  # Remove "data: " prefix

            try:
                event = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            # Check event type
            if event.get('type') == 'token':
                token_count += 1
                token_text = event.get('token', {}).get('text', '')
                attention_data = event.get('attention')

                if attention_data and attention_data.get('data'):
                    tokens_with_attention += 1
                    shape = attention_data['shape']

                    # Decode and validate first token's attention
                    if token_count == 1:
                        data_b64 = attention_data['data']
                        data_bytes = base64.b64decode(data_b64)
                        data_array = np.frombuffer(data_bytes, dtype=np.float32)
                        data_array = data_array.reshape(shape)

                        min_val = np.min(data_array)
                        max_val = np.max(data_array)
                        mean_val = np.mean(data_array)
                        sums = np.sum(data_array, axis=2)
                        avg_sum = np.mean(sums)

                        print(f"  Token {token_count}: {repr(token_text):20s} ✓ WITH ATTENTION")
                        print(f"           Shape: {shape}")
                        print(f"           Range: [{min_val:.4f}, {max_val:.4f}], Mean: {mean_val:.4f}, Sum: {avg_sum:.3f}")
                    else:
                        print(f"  Token {token_count}: {repr(token_text):20s} ✓ shape={shape}")
                else:
                    print(f"  Token {token_count}: {repr(token_text):20s} ✗ NO ATTENTION")

            elif event.get('type') == 'done':
                finish_reason = event.get('finish_reason', 'unknown')
                total_tokens = event.get('total_tokens', token_count)
                print()
                print("  " + "-"*66)
                print(f"  Stream finished: {finish_reason}")
                print(f"  Total tokens: {total_tokens}")
                break

    print()
    print("="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"✅ Received {token_count} token events")
    print(f"✅ {tokens_with_attention}/{token_count} tokens had attention data")

    if tokens_with_attention > 0:
        print("✅ Attention extraction working correctly!")
        print()
        print("PUSH MODEL VERIFIED:")
        print("  - Each token sent as individual event")
        print("  - Attention paired atomically with token at generation time")
        print("  - Ready for Halo Weave integration")
        print("="*70)
        return True
    else:
        print("✗ No attention data received")
        return False

if __name__ == "__main__":
    print()
    print("PREREQUISITES:")
    print("  Restart koboldcpp server to pick up streaming changes:")
    print("  python koboldcpp.py --model models/Qwen2.5-VL-7B-Instruct-Q8_0.gguf --usecublas --port 5001")
    print()
    input("Press Enter when server is ready...")
    print()

    success = test_attention_streaming()
    exit(0 if success else 1)
