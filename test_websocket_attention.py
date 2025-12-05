#!/usr/bin/env python3
"""
Test script for WebSocket binary attention streaming endpoint.

Usage:
    python test_websocket_attention.py [--host HOST] [--port PORT]

Requires: websocket-client (pip install websocket-client)
"""

import argparse
import json
import time
import numpy as np

try:
    import websocket
except ImportError:
    print("Please install websocket-client: pip install websocket-client")
    exit(1)


def test_websocket_attention(host="localhost", port=5001):
    url = f"ws://{host}:{port}/api/extra/generate/stream/ws"
    print(f"Connecting to {url}...")
    
    ws = websocket.create_connection(url)
    print("Connected!")
    
    # Send generation config
    config = {
        "prompt": "The capital of France is",
        "max_length": 10,
        "temperature": 0.7,
        "output_attentions": True,
        "request_id": "ws-test-001"
    }
    print(f"Sending config: {json.dumps(config, indent=2)}")
    ws.send(json.dumps(config))
    
    tokens_received = 0
    attention_bytes_total = 0
    start_time = time.time()
    
    try:
        while True:
            opcode, data = ws.recv_data()
            
            if opcode == websocket.ABNF.OPCODE_TEXT:
                # Text frame - token metadata or done event
                msg = json.loads(data.decode('utf-8'))
                
                if msg.get("type") == "token":
                    tokens_received += 1
                    print(f"Token {tokens_received}: id={msg.get('token_id')} text={repr(msg.get('text'))}")
                    
                elif msg.get("type") == "done":
                    elapsed = time.time() - start_time
                    print(f"\n--- Generation Complete ---")
                    print(f"Finish reason: {msg.get('finish_reason')}")
                    print(f"Total tokens: {msg.get('total_tokens')}")
                    print(f"Tokens received: {tokens_received}")
                    print(f"Total attention data: {attention_bytes_total / 1024 / 1024:.2f} MB")
                    print(f"Elapsed time: {elapsed:.2f}s")
                    print(f"Tokens/sec: {tokens_received / elapsed:.1f}")
                    break
                    
            elif opcode == websocket.ABNF.OPCODE_BINARY:
                # Binary frame - raw attention data
                attention_bytes_total += len(data)
                
                if len(data) > 0:
                    # Decode as float32 array
                    attention = np.frombuffer(data, dtype=np.float32)
                    print(f"  Attention: {len(data)} bytes, {len(attention)} floats, "
                          f"range=[{attention.min():.2f}, {attention.max():.2f}]")
                else:
                    print(f"  Attention: (none)")
                    
            elif opcode == websocket.ABNF.OPCODE_CLOSE:
                print("Connection closed by server")
                break
                
    except Exception as e:
        print(f"Error: {e}")
    finally:
        ws.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test WebSocket attention streaming")
    parser.add_argument("--host", default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=5001, help="Server port")
    args = parser.parse_args()
    
    test_websocket_attention(args.host, args.port)
