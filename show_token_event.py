#!/usr/bin/env python3
"""
Display a complete Token Event JSON in pretty-printed format.
This is what Halo Weave will receive from the API.
"""

import requests
import json

KOBOLD_URL = "http://localhost:5001"
API_ENDPOINT = f"{KOBOLD_URL}/api/extra/generate/stream"

def show_token_event():
    """Capture and display one complete token event"""
    payload = {
        "prompt": "Hello",
        "max_length": 3,
        "temperature": 0.7,
        "output_attentions": True,
        "request_id": "example-request-123"
    }

    print("="*80)
    print("COMPLETE TOKEN EVENT JSON")
    print("="*80)
    print("\nSending request with output_attentions=True...\n")

    response = requests.post(
        API_ENDPOINT,
        json=payload,
        stream=True,
        headers={"Accept": "text/event-stream"}
    )

    token_count = 0
    for line in response.iter_lines():
        if not line:
            continue

        line = line.decode('utf-8')

        if line.startswith('data: '):
            data_str = line[6:]
            event = json.loads(data_str)

            if event.get("type") == "token":
                token_count += 1
                attention_info = event.get("attention")

                if attention_info and attention_info.get("data"):
                    # Truncate the base64 data for display
                    data_truncated = attention_info["data"][:100] + "... (" + str(len(attention_info["data"])) + " chars total)"
                    event_display = event.copy()
                    event_display["attention"]["data"] = data_truncated

                    print(f"Token {token_count}:")
                    print(json.dumps(event_display, indent=2))
                    print("\n" + "-"*80 + "\n")

                    # Stop after first token with attention
                    break

            elif event.get("type") == "done":
                print("Done event:")
                print(json.dumps(event, indent=2))
                break

if __name__ == "__main__":
    show_token_event()
