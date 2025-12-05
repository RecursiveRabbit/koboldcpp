#!/usr/bin/env python3
"""
Capture raw API responses for documentation.
"""
import requests
import json
import sys

KOBOLD_URL = "http://localhost:5001"

def capture_model_info():
    """Capture /api/v1/model response"""
    print("=" * 80)
    print("GET /api/v1/model")
    print("=" * 80)
    try:
        response = requests.get(f"{KOBOLD_URL}/api/v1/model")
        print(f"Status: {response.status_code}")
        print(f"Response:")
        print(json.dumps(response.json(), indent=2))
        return response.json()
    except Exception as e:
        print(f"Error: {e}")
        return None

def capture_streaming_generation():
    """Capture /api/extra/generate/stream responses"""
    print("\n" + "=" * 80)
    print("POST /api/extra/generate/stream")
    print("=" * 80)

    payload = {
        "prompt": "ABC",
        "max_length": 3,
        "temperature": 0.7,
        "output_attentions": True,
        "request_id": "test-123"
    }

    print("\nRequest payload:")
    print(json.dumps(payload, indent=2))
    print("\nResponse (Server-Sent Events):")
    print("-" * 80)

    try:
        response = requests.post(
            f"{KOBOLD_URL}/api/extra/generate/stream",
            json=payload,
            stream=True,
            headers={"Accept": "text/event-stream"}
        )

        token_count = 0
        for line in response.iter_lines():
            if not line:
                continue

            line = line.decode('utf-8')
            print(line)  # Print raw SSE line

            if line.startswith('data: '):
                data_str = line[6:]
                try:
                    event = json.loads(data_str)

                    # Truncate attention data for readability
                    if event.get("type") == "token" and event.get("attention"):
                        attn = event["attention"]
                        if "data" in attn and len(attn["data"]) > 100:
                            attn["data"] = attn["data"][:100] + "..." + attn["data"][-20:]

                    token_count += 1
                    if token_count > 2:  # Only show first 2 tokens
                        print("\n[... remaining tokens omitted for brevity ...]")
                        break

                except json.JSONDecodeError:
                    pass
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    capture_model_info()
    capture_streaming_generation()
