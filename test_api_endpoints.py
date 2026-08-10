#!/usr/bin/env python3
"""
Test all KoboldCPP API endpoints and document actual responses.
Run against a live server to verify the API spec.
"""

import requests
import json

BASE_URL = "http://localhost:5001"

def test_model_info():
    """Test GET /api/v1/model"""
    print("\n" + "="*80)
    print("TEST: GET /api/v1/model")
    print("="*80)
    
    response = requests.get(f"{BASE_URL}/api/v1/model")
    print(f"Status: {response.status_code}")
    print(f"Response:\n{json.dumps(response.json(), indent=2)}")
    return response.json()

def test_tokenize():
    """Test POST /api/v1/tokenize"""
    print("\n" + "="*80)
    print("TEST: POST /api/v1/tokenize")
    print("="*80)
    
    payload = {
        "text": "Hello, how are you?",
        "add_special_tokens": False
    }
    print(f"Request:\n{json.dumps(payload, indent=2)}")
    
    response = requests.post(f"{BASE_URL}/api/v1/tokenize", json=payload)
    print(f"Status: {response.status_code}")
    if response.status_code == 404:
        print("ENDPOINT NOT FOUND")
        return None
    result = response.json()
    print(f"Response:\n{json.dumps(result, indent=2)}")
    return result

def test_detokenize(token_ids):
    """Test POST /api/v1/detokenize"""
    print("\n" + "="*80)
    print("TEST: POST /api/v1/detokenize")
    print("="*80)
    
    payload = {"token_ids": token_ids}
    print(f"Request:\n{json.dumps(payload, indent=2)}")
    
    response = requests.post(f"{BASE_URL}/api/v1/detokenize", json=payload)
    print(f"Status: {response.status_code}")
    if response.status_code == 404:
        print("ENDPOINT NOT FOUND")
        return None
    result = response.json()
    print(f"Response:\n{json.dumps(result, indent=2)}")
    return result

def test_generate_stream_with_prompt():
    """Test POST /api/extra/generate/stream with text prompt"""
    print("\n" + "="*80)
    print("TEST: POST /api/extra/generate/stream (with prompt)")
    print("="*80)
    
    payload = {
        "prompt": "The capital of France is",
        "max_length": 3,
        "temperature": 0.7,
        "sampler_seed": 12345
    }
    print(f"Request:\n{json.dumps(payload, indent=2)}")
    
    response = requests.post(
        f"{BASE_URL}/api/extra/generate/stream",
        json=payload,
        stream=True,
        headers={"Accept": "text/event-stream"}
    )
    print(f"Status: {response.status_code}")
    print("Response (SSE stream):")
    
    for line in response.iter_lines():
        if line:
            print(line.decode('utf-8'))

def test_generate_stream_with_input_ids(token_ids):
    """Test POST /api/extra/generate/stream with input_ids"""
    print("\n" + "="*80)
    print("TEST: POST /api/extra/generate/stream (with input_ids)")
    print("="*80)
    
    payload = {
        "input_ids": token_ids,
        "max_length": 3,
        "temperature": 0.7,
        "sampler_seed": 12345
    }
    print(f"Request:\n{json.dumps(payload, indent=2)}")
    
    response = requests.post(
        f"{BASE_URL}/api/extra/generate/stream",
        json=payload,
        stream=True,
        headers={"Accept": "text/event-stream"}
    )
    print(f"Status: {response.status_code}")
    print("Response (SSE stream):")
    
    for line in response.iter_lines():
        if line:
            print(line.decode('utf-8'))

def test_generate_stream_with_attention():
    """Test POST /api/extra/generate/stream with output_attentions"""
    print("\n" + "="*80)
    print("TEST: POST /api/extra/generate/stream (with attention)")
    print("="*80)
    
    payload = {
        "prompt": "The cat sat on the",
        "max_length": 3,
        "output_attentions": True
    }
    print(f"Request:\n{json.dumps(payload, indent=2)}")
    
    response = requests.post(
        f"{BASE_URL}/api/extra/generate/stream",
        json=payload,
        stream=True,
        headers={"Accept": "text/event-stream"}
    )
    print(f"Status: {response.status_code}")
    print("Response (SSE stream, attention data truncated):")
    
    for line in response.iter_lines():
        if not line:
            continue
        line_str = line.decode('utf-8')
        if line_str.startswith('data: '):
            data = json.loads(line_str[6:])
            if data.get('attention'):
                # Truncate attention data for display
                attn = data['attention']
                data['attention'] = {
                    **attn,
                    'data': attn['data'][:5] + ['...'] + attn['data'][-2:]
                }
            print(f"data: {json.dumps(data)}")
        else:
            print(line_str)

if __name__ == "__main__":
    print("KoboldCPP API Endpoint Tests")
    print("Testing against:", BASE_URL)
    
    # Test 1: Model info
    model_info = test_model_info()
    
    # Test 2: Tokenization
    tokenize_result = test_tokenize()
    token_ids = tokenize_result.get('token_ids', []) if tokenize_result else []
    
    # Test 3: Detokenization
    if token_ids:
        test_detokenize(token_ids)
    else:
        print("\nSkipping detokenize test (no token_ids from tokenize)")
    
    # Test 4: Generate with prompt
    test_generate_stream_with_prompt()
    
    # Test 5: Generate with input_ids
    if token_ids:
        test_generate_stream_with_input_ids(token_ids)
    
    # Test 6: Generate with attention
    test_generate_stream_with_attention()
    
    print("\n" + "="*80)
    print("All tests complete!")
    print("="*80)
