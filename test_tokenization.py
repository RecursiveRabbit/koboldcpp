#!/usr/bin/env python3
"""
Test script for new /api/v1/tokenize and /api/v1/detokenize endpoints.
"""
import requests
import json

BASE_URL = "http://localhost:5001"

def test_tokenize():
    """Test the /api/v1/tokenize endpoint"""
    print("\n=== Testing /api/v1/tokenize ===\n")

    test_text = "Hello, how are you?"

    response = requests.post(
        f"{BASE_URL}/api/v1/tokenize",
        json={
            "text": test_text,
            "add_special_tokens": False
        }
    )

    print(f"Request: {test_text}")
    print(f"Status Code: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"\nResponse:")
        print(json.dumps(data, indent=2))

        print(f"\n✅ Success! Got {data['token_count']} tokens:")
        for token in data['tokens']:
            print(f"  ID {token['token_id']:6d} → \"{token['text']}\"")

        return data
    else:
        print(f"❌ Error: {response.text}")
        return None

def test_detokenize(token_ids):
    """Test the /api/v1/detokenize endpoint"""
    print("\n\n=== Testing /api/v1/detokenize ===\n")

    response = requests.post(
        f"{BASE_URL}/api/v1/detokenize",
        json={
            "token_ids": token_ids
        }
    )

    print(f"Request: {token_ids}")
    print(f"Status Code: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"\nResponse:")
        print(json.dumps(data, indent=2))

        print(f"\n✅ Success! Detokenized text:")
        print(f"  \"{data['text']}\"")

        return data
    else:
        print(f"❌ Error: {response.text}")
        return None

def test_roundtrip():
    """Test that tokenize → detokenize gives back original text"""
    print("\n\n=== Testing Round-trip (tokenize → detokenize) ===\n")

    original_text = "The capital of France is Paris."

    # Tokenize
    print(f"Original text: \"{original_text}\"")
    tokenize_response = requests.post(
        f"{BASE_URL}/api/v1/tokenize",
        json={"text": original_text, "add_special_tokens": False}
    )

    if tokenize_response.status_code != 200:
        print(f"❌ Tokenization failed: {tokenize_response.text}")
        return

    token_data = tokenize_response.json()
    token_ids = token_data['token_ids']

    print(f"Token IDs: {token_ids}")

    # Detokenize
    detokenize_response = requests.post(
        f"{BASE_URL}/api/v1/detokenize",
        json={"token_ids": token_ids}
    )

    if detokenize_response.status_code != 200:
        print(f"❌ Detokenization failed: {detokenize_response.text}")
        return

    detok_data = detokenize_response.json()
    reconstructed_text = detok_data['text']

    print(f"Reconstructed text: \"{reconstructed_text}\"")

    if original_text.strip() == reconstructed_text.strip():
        print("\n✅ Round-trip test PASSED! Text matches.")
    else:
        print(f"\n⚠️  Round-trip test FAILED!")
        print(f"  Expected: \"{original_text}\"")
        print(f"  Got:      \"{reconstructed_text}\"")

def main():
    print("="*70)
    print("KoboldCPP Tokenization API Test")
    print("="*70)

    try:
        # Test 1: Tokenize
        tokenize_result = test_tokenize()

        if tokenize_result:
            # Test 2: Detokenize using the token IDs from test 1
            token_ids = tokenize_result['token_ids']
            test_detokenize(token_ids)

        # Test 3: Round-trip
        test_roundtrip()

        print("\n" + "="*70)
        print("All tests completed!")
        print("="*70)

    except requests.exceptions.ConnectionError:
        print(f"\n❌ Error: Could not connect to server at {BASE_URL}")
        print("Make sure KoboldCPP is running with:")
        print("  python3 koboldcpp.py --model <model.gguf> --port 5001")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
