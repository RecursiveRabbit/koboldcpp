#!/usr/bin/env python3
"""
Capture ONE complete token event with attention for documentation.
"""
import requests
import json

KOBOLD_URL = "http://localhost:5001"

def capture_complete_token_event():
    """Capture complete token event"""
    payload = {
        "prompt": "The capital of France is",
        "max_length": 5,
        "temperature": 0.7,
        "output_attentions": True,
        "request_id": "doc-example-123"
    }

    response = requests.post(
        f"{KOBOLD_URL}/api/extra/generate/stream",
        json=payload,
        stream=True,
        headers={"Accept": "text/event-stream"}
    )

    print("Capturing token events...")
    token_count = 0

    for line in response.iter_lines():
        if not line:
            continue

        line = line.decode('utf-8')

        if line.startswith('data: '):
            data_str = line[6:]
            try:
                event = json.loads(data_str)

                if event.get("type") == "token":
                    token_count += 1
                    attn = event.get("attention")

                    # Only capture ONE token with attention
                    if attn and token_count >= 2:
                        # Truncate attention data for display
                        original_len = len(attn["data"])
                        attn_display = attn.copy()
                        attn_display["data"] = attn["data"][:80] + "...[truncated]..." + attn["data"][-20:]

                        print("=" * 80)
                        print("ACTUAL TOKEN EVENT (with attention truncated for display)")
                        print("=" * 80)
                        print(json.dumps(event, indent=2))
                        print(f"\nNote: Full attention data is {original_len} chars")
                        print(f"      (802,816 bytes = 200,704 floats in base64)")
                        return

                elif event.get("type") == "done":
                    print("\nGeneration complete, no suitable token captured")
                    return

            except json.JSONDecodeError:
                pass

if __name__ == "__main__":
    capture_complete_token_event()
