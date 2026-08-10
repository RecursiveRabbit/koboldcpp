#!/usr/bin/env python3
"""
Test input_ids parameter for direct token input
"""

import requests
import json

BASE_URL = "http://localhost:5001"

def test_tokenize_roundtrip():
    """Test: tokenize → input_ids → generate (SSE)"""
    print("=" * 80)
    print("Test 1: Tokenize → input_ids → SSE Generation")
    print("=" * 80)
    
    # Step 1: Tokenize text
    text = "The capital of France is"
    print(f"\n1. Tokenizing: '{text}'")
    
    response = requests.post(
        f"{BASE_URL}/api/v1/tokenize",
        json={"text": text, "add_special_tokens": False, "with_pieces": True},
        timeout=5
    )
    
    if response.status_code != 200:
        print(f"❌ Tokenize failed: {response.status_code}")
        return False
    
    token_data = response.json()
    token_ids = token_data["token_ids"]
    print(f"   Token IDs: {token_ids}")
    print(f"   Token count: {len(token_ids)}")
    
    # Step 2: Generate using input_ids (SSE)
    print(f"\n2. Generating with input_ids (SSE)...")
    
    response = requests.post(
        f"{BASE_URL}/api/extra/generate/stream",
        json={
            "input_ids": token_ids,
            "max_length": 3,
            "temperature": 0.7,
            "sampler_seed": 12345
        },
        stream=True,
        timeout=30
    )
    
    if response.status_code != 200:
        print(f"❌ Generation failed: {response.status_code}")
        return False
    
    print("   Generated tokens:")
    generated_text = ""
    for line in response.iter_lines():
        if line:
            line = line.decode('utf-8')
            if line.startswith('data: '):
                data = json.loads(line[6:])
                token = data.get('token', '')
                if token:
                    generated_text += token
                    print(f"     '{token}'")
                finish_reason = data.get('finish_reason')
                if finish_reason:
                    print(f"   ✅ Finished: {finish_reason}")
    
    print(f"\n   Full output: '{text}{generated_text}'")
    return True


def test_input_ids_with_attention():
    """Test: input_ids + attention capture"""
    print("\n" + "=" * 80)
    print("Test 2: input_ids + Attention Capture")
    print("=" * 80)
    
    # First tokenize a prompt to get valid token IDs
    print("\n1. Tokenizing prompt first...")
    response = requests.post(
        f"{BASE_URL}/api/v1/tokenize",
        json={"text": "Once upon a time there was a", "add_special_tokens": False},
        timeout=5
    )
    token_ids = response.json()["token_ids"]
    print(f"   Token IDs: {token_ids}")
    
    response = requests.post(
        f"{BASE_URL}/api/extra/generate/stream",
        json={
            "input_ids": token_ids,
            "max_length": 2,
            "output_attentions": True,
            "temperature": 0.7
        },
        stream=True,
        timeout=30
    )
    
    if response.status_code != 200:
        print(f"❌ Generation failed: {response.status_code}")
        return False
    
    print("\n2. Streaming with attention:")
    token_count = 0
    attention_count = 0
    
    for line in response.iter_lines():
        if line:
            line = line.decode('utf-8')
            if line.startswith('data: '):
                data = json.loads(line[6:])
                token = data.get('token', '')
                if token:
                    token_count += 1
                    print(f"   Token {token_count}: '{token}'")
                    
                    attention = data.get('attention')
                    if attention and attention.get('data'):
                        attention_count += 1
                        attn_len = len(attention['data'])
                        attn_sum = sum(attention['data'])
                        print(f"     ✅ Attention: {attn_len} floats, sum={attn_sum:.4f}")
                    else:
                        print(f"     ⚠️  No attention data")
                
                finish_reason = data.get('finish_reason')
                if finish_reason:
                    print(f"\n   Finished: {finish_reason}")
    
    print(f"\n   Tokens: {token_count}, Attention frames: {attention_count}")
    return attention_count > 0


def test_websocket_input_ids():
    """Test: input_ids via WebSocket"""
    print("\n" + "=" * 80)
    print("Test 3: input_ids + WebSocket Binary Streaming")
    print("=" * 80)
    
    try:
        import websocket
    except ImportError:
        print("⚠️  websocket-client not installed, skipping WebSocket test")
        return True
    
    token_ids = [785, 6722, 315, 9625, 374]  # "The cat sat on the"
    print(f"\n1. Using token IDs: {token_ids}")
    
    ws_url = "ws://localhost:5001/api/extra/generate/stream/ws"
    
    received_tokens = 0
    received_attention = 0
    received_token_ids = []
    
    def on_message(ws, message):
        nonlocal received_tokens, received_attention, received_token_ids
        if isinstance(message, bytes):
            # Binary frame - attention data
            num_floats = len(message) // 4
            received_attention += 1
            print(f"   📊 Binary attention: {num_floats} floats")
        else:
            # Text frame
            data = json.loads(message)
            if data.get('type') == 'token':
                received_tokens += 1
                token_id = data.get('token_id', None)
                received_token_ids.append(token_id)
                print(f"   🔤 Token {received_tokens}: id={token_id} '{data['text']}'")
            elif data.get('type') == 'done':
                print(f"   ✅ Done: {data['finish_reason']}")
                ws.close()
    
    def on_open(ws):
        print("\n2. WebSocket connected, sending request...")
        ws.send(json.dumps({
            "input_ids": token_ids,
            "max_length": 5,  # Request 5 tokens to verify per-token streaming
            "output_attentions": True
        }))
    
    def on_error(ws, error):
        print(f"   ❌ Error: {error}")
    
    ws = websocket.WebSocketApp(
        ws_url,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error
    )
    
    ws.run_forever()
    
    print(f"\n   Tokens: {received_tokens}, Attention frames: {received_attention}")
    
    # Verify token_ids are present (not None)
    has_token_ids = all(tid is not None and tid >= 0 for tid in received_token_ids)
    if received_token_ids:
        print(f"   Token IDs received: {received_token_ids}")
        if has_token_ids:
            print("   ✅ All tokens have valid token_id")
        else:
            print("   ❌ Some tokens missing token_id!")
    
    return received_tokens > 0 and has_token_ids


if __name__ == "__main__":
    print("\n🧪 Testing input_ids Parameter Support\n")
    
    results = []
    
    # Test 1: Basic roundtrip
    try:
        results.append(("Tokenize → input_ids → SSE", test_tokenize_roundtrip()))
    except Exception as e:
        print(f"❌ Test 1 failed: {e}")
        results.append(("Tokenize → input_ids → SSE", False))
    
    # Test 2: With attention
    try:
        results.append(("input_ids + Attention", test_input_ids_with_attention()))
    except Exception as e:
        print(f"❌ Test 2 failed: {e}")
        results.append(("input_ids + Attention", False))
    
    # Test 3: WebSocket
    try:
        results.append(("input_ids + WebSocket", test_websocket_input_ids()))
    except Exception as e:
        print(f"❌ Test 3 failed: {e}")
        results.append(("input_ids + WebSocket", False))
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(result[1] for result in results)
    print("\n" + ("🎉 All tests passed!" if all_passed else "⚠️  Some tests failed"))
