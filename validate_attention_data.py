#!/usr/bin/env python3
"""
Validate that attention data contains post-softmax probabilities with correct range.
Captures one token event and decodes the attention tensor.
"""

import requests
import json
import base64
import numpy as np
import sys

KOBOLD_URL = "http://localhost:5001"
API_ENDPOINT = f"{KOBOLD_URL}/api/extra/generate/stream"

def validate_attention():
    """Capture and validate one token's attention data"""
    print("Validating Attention Data")
    print("="*80)

    payload = {
        "prompt": "Hello world",
        "max_length": 5,
        "temperature": 0.7,
        "output_attentions": True
    }

    print("Sending request...\n")

    response = requests.post(
        API_ENDPOINT,
        json=payload,
        stream=True,
        headers={"Accept": "text/event-stream"}
    )

    if response.status_code != 200:
        print(f"✗ Error: HTTP {response.status_code}")
        return False

    # Read until we get a token with attention data
    for line in response.iter_lines():
        if not line:
            continue

        line = line.decode('utf-8')

        if line.startswith('data: '):
            data_str = line[6:]
            event = json.loads(data_str)

            if event.get("type") == "token":
                attention_info = event.get("attention")

                if attention_info and attention_info.get("data"):
                    token_text = event["token"]["text"]
                    print(f"Token: '{token_text}'")
                    print(f"Shape: {attention_info['shape']}")

                    # Decode base64
                    attention_b64 = attention_info["data"]
                    attention_bytes = base64.b64decode(attention_b64)

                    # Reconstruct numpy array
                    attention_array = np.frombuffer(attention_bytes, dtype=np.float32)
                    n_layers, n_heads, seq_len = attention_info["shape"]
                    attention_array = attention_array.reshape((n_layers, n_heads, seq_len))

                    print(f"\nDecoded attention tensor:")
                    print(f"  Shape: {attention_array.shape}")
                    print(f"  Data type: {attention_array.dtype}")
                    print(f"  Min value: {attention_array.min():.2f}")
                    print(f"  Max value: {attention_array.max():.2f}")
                    print(f"  Mean value: {attention_array.mean():.2f}")
                    print(f"  Std dev: {attention_array.std():.2f}")

                    # Check if these are post-softmax probabilities (normalized)
                    sum_per_head = attention_array[0, 0, :].sum()
                    print(f"\nSum of attention for first head: {sum_per_head:.2f}")

                    if abs(sum_per_head - 1.0) < 0.01:
                        print("  ✓ Confirmed: Post-softmax probabilities (sum ≈ 1.0)")
                    else:
                        print("  ⚠ Unexpected: Sum does not equal 1.0")
                        print("  Expected: Post-softmax probabilities (sum ≈ 1.0)")
                        return False

                    # Check value range (probabilities should be [0, 1])
                    if attention_array.min() >= 0 and attention_array.max() <= 1.0:
                        print(f"  ✓ Value range consistent with probabilities [0, 1]")
                        print(f"  ✓ Dynamic range: {attention_array.max() - attention_array.min():.6f}")
                        return True
                    else:
                        print(f"  ⚠ Values outside [0, 1] range")
                        print(f"  Expected: 0.0 to 1.0 range")
                        print(f"  Got: {attention_array.min():.6f} to {attention_array.max():.6f}")
                        return False

    print("✗ No attention data received")
    return False

if __name__ == "__main__":
    success = validate_attention()
    sys.exit(0 if success else 1)
