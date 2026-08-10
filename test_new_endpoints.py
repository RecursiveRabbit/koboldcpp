#!/usr/bin/env python3
"""
Test newly implemented tokenize and detokenize endpoints.
"""

import requests
import json

BASE_URL = "http://localhost:5001"

def test_tokenize():
    """Test POST /api/v1/tokenize"""
    print("\n" + "="*80)
    print("TEST: POST /api/v1/tokenize")
    print("="*80)
    
    payload = {
        "text": "Hello, how are you?",
        "add_special_tokens": False,
        "with_pieces": True
    }
    print(f"Request:\n{json.dumps(payload, indent=2)}")
    
    response = requests.post(f"{BASE_URL}/api/v1/tokenize", json=payload)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Response:\n{json.dumps(result, indent=2)}")
        return result
    else:
        print(f"Error: {response.text}")
        return None

def test_detokenize(token_ids):
    """Test POST /api/v1/detokenize"""
    print("\n" + "="*80)
    print("TEST: POST /api/v1/detokenize")
    print("="*80)
    
    payload = {"token_ids": token_ids}
    print(f"Request:\n{json.dumps(payload, indent=2)}")
    
    response = requests.post(f"{BASE_URL}/api/v1/detokenize", json=payload)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Response:\n{json.dumps(result, indent=2)}")
        return result
    else:
        print(f"Error: {response.text}")
        return None

def test_roundtrip():
    """Test tokenize -> detokenize roundtrip"""
    print("\n" + "="*80)
    print("TEST: Roundtrip (tokenize -> detokenize)")
    print("="*80)
    
    original_text = "The cat sat on the mat."
    print(f"Original text: {original_text}")
    
    # Tokenize
    tok_result = requests.post(f"{BASE_URL}/api/v1/tokenize", json={
        "text": original_text,
        "add_special_tokens": False
    }).json()
    
    token_ids = tok_result.get('token_ids', [])
    print(f"Token IDs: {token_ids}")
    
    # Detokenize
    detok_result = requests.post(f"{BASE_URL}/api/v1/detokenize", json={
        "token_ids": token_ids
    }).json()
    
    reconstructed_text = detok_result.get('text', '')
    print(f"Reconstructed text: {reconstructed_text}")
    
    if original_text == reconstructed_text:
        print("✅ Roundtrip successful - texts match!")
    else:
        print("⚠️  Roundtrip mismatch")

if __name__ == "__main__":
    print("Testing Tokenize/Detokenize Endpoints")
    print("Testing against:", BASE_URL)
    
    # Test tokenize
    tokenize_result = test_tokenize()
    
    # Test detokenize
    if tokenize_result and 'token_ids' in tokenize_result:
        test_detokenize(tokenize_result['token_ids'])
    
    # Test roundtrip
    test_roundtrip()
    
    print("\n" + "="*80)
    print("All tests complete!")
    print("="*80)
