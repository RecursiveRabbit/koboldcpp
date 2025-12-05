#!/usr/bin/env python3
import requests
import json

SEED = 12345
URL = "http://localhost:5001/api/extra/generate/stream"

# Test 1: With input_ids
print("=== Test 1: With input_ids ===")
response1 = requests.post(URL, json={
    "input_ids": [785, 6722, 315, 9625, 374],
    "max_length": 5,
    "temperature": 0.7,
    "sampler_seed": SEED
}, stream=True, headers={"Accept": "text/event-stream"})

text1 = ""
for line in response1.iter_lines():
    if line and line.startswith(b'data: '):
        data = json.loads(line[6:])
        if "token" in data:
            text1 += data["token"]

print(f"Output: {text1}")

# Test 2: With text prompt
print("\n=== Test 2: With text prompt ===")
response2 = requests.post(URL, json={
    "prompt": "The capital of France is",
    "max_length": 5,
    "temperature": 0.7,
    "sampler_seed": SEED
}, stream=True, headers={"Accept": "text/event-stream"})

text2 = ""
for line in response2.iter_lines():
    if line and line.startswith(b'data: '):
        data = json.loads(line[6:])
        if "token" in data:
            text2 += data["token"]

print(f"Output: {text2}")

print("\n=== Comparison ===")
if text1 == text2:
    print("✅ IDENTICAL - input_ids works correctly!")
else:
    print(f"⚠️  Different outputs:")
    print(f"  input_ids: {text1}")
    print(f"  prompt:    {text2}")
