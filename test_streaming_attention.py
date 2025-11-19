#!/usr/bin/env python3
"""
Test script for KoboldCPP streaming attention extraction API.
Tests the /api/extra/generate/stream endpoint with output_attentions=True.
"""

import requests
import json
import sys
import time
import uuid

KOBOLD_URL = "http://localhost:5001"
API_ENDPOINT = f"{KOBOLD_URL}/api/extra/generate/stream"

def wait_for_server(timeout=30):
    """Wait for koboldcpp server to be ready"""
    print("Waiting for koboldcpp server...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{KOBOLD_URL}/api/v1/model")
            if response.status_code == 200:
                model_info = response.json()
                print(f"✓ Server ready! Model: {model_info.get('result', 'unknown')}")
                return True
        except requests.exceptions.ConnectionError:
            time.sleep(1)
    print("✗ Timeout waiting for server")
    return False

def test_streaming_attention():
    """Test streaming generation with attention extraction"""
    print("\n" + "="*80)
    print("Testing Streaming Attention Extraction")
    print("="*80)

    # Generate a unique request ID
    request_id = str(uuid.uuid4())

    # Prepare generation request
    payload = {
        "prompt": "The capital of France is",
        "max_length": 20,
        "temperature": 0.7,
        "output_attentions": True,
        "request_id": request_id
    }

    print(f"\nRequest ID: {request_id}")
    print(f"Prompt: {payload['prompt']}")
    print(f"Max tokens: {payload['max_length']}")
    print("\nSending request...\n")

    try:
        # Send streaming request
        response = requests.post(
            API_ENDPOINT,
            json=payload,
            stream=True,
            headers={"Accept": "text/event-stream"}
        )

        if response.status_code != 200:
            print(f"✗ Error: HTTP {response.status_code}")
            print(response.text)
            return False

        print("✓ Connected to stream\n")
        print("-"*80)

        token_count = 0
        has_attention = False

        # Read SSE stream
        for line in response.iter_lines():
            if not line:
                continue

            line = line.decode('utf-8')

            # Parse SSE format: "data: {json}"
            if line.startswith('data: '):
                data_str = line[6:]  # Remove "data: " prefix

                try:
                    event = json.loads(data_str)

                    if event.get("type") == "token":
                        token_count += 1
                        token_info = event.get("token", {})
                        attention_info = event.get("attention")

                        print(f"\n[Token {token_count}]")
                        print(f"  Text: '{token_info.get('text', 'N/A')}'")
                        print(f"  Token ID: {token_info.get('token_id', 'N/A')}")
                        print(f"  Request ID: {event.get('request_id', 'N/A')}")

                        if attention_info:
                            has_attention = True
                            shape = attention_info.get('shape', [])
                            data_len = len(attention_info.get('data', ''))
                            print(f"  Attention:")
                            print(f"    Format: {attention_info.get('format', 'N/A')}")
                            print(f"    Shape: {shape}")
                            print(f"    Context length: {attention_info.get('context_length', 'N/A')}")
                            print(f"    Encoding: {attention_info.get('encoding', 'N/A')}")
                            print(f"    Data type: {attention_info.get('dtype', 'N/A')}")
                            print(f"    Data size: {data_len} chars (base64)")

                            # Calculate expected size
                            if len(shape) == 3:
                                n_layers, n_heads, seq_len = shape
                                expected_floats = n_layers * n_heads * seq_len
                                expected_bytes = expected_floats * 4  # float32
                                expected_b64 = (expected_bytes * 4) // 3 + 4  # base64 overhead
                                print(f"    Expected: {expected_floats} floats = {expected_bytes} bytes ≈ {expected_b64} b64 chars")
                        else:
                            print(f"  Attention: None")

                        print("-"*80)

                    elif event.get("type") == "done":
                        print(f"\n[Generation Complete]")
                        print(f"  Finish reason: {event.get('finish_reason', 'N/A')}")
                        print(f"  Total tokens: {event.get('total_tokens', 'N/A')}")
                        print(f"  Request ID: {event.get('request_id', 'N/A')}")
                        break

                except json.JSONDecodeError as e:
                    print(f"✗ JSON parse error: {e}")
                    print(f"  Data: {data_str[:100]}...")

        print("="*80)
        print(f"\nSummary:")
        print(f"  Tokens received: {token_count}")
        print(f"  Attention data present: {'✓ YES' if has_attention else '✗ NO'}")

        return has_attention

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("KoboldCPP Streaming Attention Test")
    print("="*80)

    # Wait for server
    if not wait_for_server():
        print("\n✗ Server not available. Start koboldcpp first:")
        print("  python3 koboldcpp.py --model <model.gguf> --port 5001 --usecublas")
        sys.exit(1)

    # Run test
    success = test_streaming_attention()

    if success:
        print("\n✓ Test PASSED - Attention data received successfully!")
        sys.exit(0)
    else:
        print("\n✗ Test FAILED - No attention data received")
        sys.exit(1)

if __name__ == "__main__":
    main()
