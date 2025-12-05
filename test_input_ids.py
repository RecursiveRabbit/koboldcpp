#!/usr/bin/env python3
"""
Test input_ids parameter for direct token input to generation endpoints.
"""

import requests
import json

KOBOLD_URL = "http://localhost:5001"

def test_tokenization():
    """First, tokenize a prompt to get token IDs"""
    print("=== Step 1: Tokenize text ===")
    prompt = "The capital of France is"

    response = requests.post(
        f"{KOBOLD_URL}/api/v1/tokenize",
        json={"text": prompt, "add_special_tokens": False}
    )

    tokenize_result = response.json()
    print(f"Prompt: {prompt}")
    print(f"Token IDs: {tokenize_result['token_ids']}")
    print(f"Token count: {tokenize_result['token_count']}")
    print()

    return tokenize_result['token_ids']

def test_generation_with_prompt():
    """Generate with text prompt (normal path)"""
    print("=== Step 2: Generate with text prompt (baseline) ===")
    response = requests.post(
        f"{KOBOLD_URL}/api/extra/generate/stream",
        json={
            "prompt": "The capital of France is",
            "max_length": 5,
            "temperature": 0.7,
            "output_attentions": False
        },
        headers={"Accept": "text/event-stream"},
        stream=True
    )

    generated_text = ""
    for line in response.iter_lines():
        if not line:
            continue
        line = line.decode('utf-8')
        if line.startswith('data: '):
            data = json.loads(line[6:])
            if data["type"] == "token":
                generated_text += data["token"]["text"]
            elif data["type"] == "done":
                break

    print(f"Generated: The capital of France is{generated_text}")
    print()
    return generated_text

def test_generation_with_input_ids(token_ids):
    """Generate with input_ids (new feature)"""
    print("=== Step 3: Generate with input_ids (new feature) ===")
    response = requests.post(
        f"{KOBOLD_URL}/api/extra/generate/stream",
        json={
            "input_ids": token_ids,
            "max_length": 5,
            "temperature": 0.7,
            "output_attentions": False
        },
        headers={"Accept": "text/event-stream"},
        stream=True
    )

    generated_text = ""
    for line in response.iter_lines():
        if not line:
            continue
        line = line.decode('utf-8')
        if line.startswith('data: '):
            data = json.loads(line[6:])
            if data["type"] == "token":
                generated_text += data["token"]["text"]
            elif data["type"] == "done":
                break

    print(f"Generated with input_ids: {generated_text}")
    print()
    return generated_text

def test_pruning_workflow():
    """Simulate Halo Weave's pruning workflow"""
    print("=== Step 4: Simulate Context Pruning Workflow ===")

    # Original prompt
    original_prompt = "The quick brown fox jumps over the lazy dog"
    print(f"Original prompt: {original_prompt}")

    # Tokenize
    response = requests.post(
        f"{KOBOLD_URL}/api/v1/tokenize",
        json={"text": original_prompt, "add_special_tokens": False}
    )
    tokens = response.json()
    print(f"Original tokens: {tokens['token_ids']}")
    print(f"Token count: {len(tokens['token_ids'])}")

    # Simulate pruning: remove tokens at indices 2, 3 (simulate removing "brown fox" based on low brightness)
    pruned_token_ids = tokens['token_ids'][:2] + tokens['token_ids'][4:]
    print(f"Pruned tokens (removed indices 2, 3): {pruned_token_ids}")

    # Detokenize to see what we're left with
    response = requests.post(
        f"{KOBOLD_URL}/api/v1/detokenize",
        json={"token_ids": pruned_token_ids}
    )
    pruned_text = response.json()['text']
    print(f"Pruned text: {pruned_text}")

    # Generate with pruned input_ids
    print("\nGenerating with pruned context...")
    response = requests.post(
        f"{KOBOLD_URL}/api/extra/generate/stream",
        json={
            "input_ids": pruned_token_ids,
            "max_length": 10,
            "temperature": 0.7,
            "output_attentions": False
        },
        headers={"Accept": "text/event-stream"},
        stream=True
    )

    generated_text = ""
    for line in response.iter_lines():
        if not line:
            continue
        line = line.decode('utf-8')
        if line.startswith('data: '):
            data = json.loads(line[6:])
            if data["type"] == "token":
                generated_text += data["token"]["text"]
            elif data["type"] == "done":
                break

    print(f"Generated: {pruned_text}{generated_text}")
    print()

if __name__ == "__main__":
    print("Testing input_ids parameter for brightness-based context pruning\n")
    print("="*60)

    try:
        # Test tokenization
        token_ids = test_tokenization()

        # Test normal generation (baseline)
        baseline_text = test_generation_with_prompt()

        # Test input_ids generation (should be deterministic)
        input_ids_text = test_generation_with_input_ids(token_ids)

        # Test pruning workflow
        test_pruning_workflow()

        print("="*60)
        print("✅ ALL TESTS COMPLETED")
        print("="*60)
        print("\nKey Observations:")
        print("1. Tokenization: Text → Token IDs")
        print("2. Normal generation: Text input → Tokenization → Generation")
        print("3. input_ids generation: Token IDs → Generation (bypasses tokenization)")
        print("4. Pruning workflow: Tokenize → Prune → Generate with pruned input_ids")
        print("\nThis enables Halo Weave to:")
        print("- Tokenize conversation once")
        print("- Track brightness scores per token")
        print("- Prune low-brightness tokens")
        print("- Feed pruned token array back to model without retokenization")

    except requests.exceptions.ConnectionError:
        print("❌ ERROR: Could not connect to koboldcpp server")
        print("Start the server first: python3 koboldcpp.py --model <path> --port 5001")
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
