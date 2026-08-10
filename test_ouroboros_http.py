#!/usr/bin/env python3
"""
Ouroboros HTTP Test Client

This script tests the ouroboros functionality via the koboldcpp HTTP API.
Run koboldcpp server first with a model, then use this to test.

Usage:
    python test_ouroboros_http.py --url http://localhost:5001
"""

import requests
import numpy as np
import base64
import json
import argparse
import time
from typing import Optional, Dict, Any, List, Tuple


class OuroborosClient:
    """Client for testing ouroboros functionality via HTTP API."""

    def __init__(self, base_url: str = "http://localhost:5001"):
        self.base_url = base_url.rstrip('/')
        self.stored_hidden_states: List[np.ndarray] = []
        self.stored_token_ids: List[int] = []
        self.n_embd: int = 0

    def check_server(self) -> bool:
        """Check if the server is running."""
        try:
            resp = requests.get(f"{self.base_url}/api/v1/model", timeout=5)
            return resp.status_code == 200
        except:
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information including embedding dimension."""
        resp = requests.get(f"{self.base_url}/api/extra/true/model")
        if resp.status_code == 200:
            return resp.json()
        return {}

    def tokenize(self, text: str) -> List[int]:
        """Tokenize text and return token IDs."""
        resp = requests.post(
            f"{self.base_url}/api/extra/tokencount",
            json={"prompt": text}
        )
        if resp.status_code == 200:
            data = resp.json()
            # Response format: {"count": N, "ids": [...]}
            return data.get("ids", [])
        return []

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in text."""
        return len(self.tokenize(text))

    def generate_with_hidden_states(self,
                                    prompt: str,
                                    max_length: int = 50,
                                    temperature: float = 0.7,
                                    **kwargs) -> Tuple[str, List[Dict]]:
        """
        Generate text and capture hidden states for each token.

        Returns:
            (generated_text, list of {token_id, hidden_state} dicts)
        """
        payload = {
            "prompt": prompt,
            "max_length": max_length,
            "temperature": temperature,
            "output_hidden_states": True,  # Enable hidden state extraction
            **kwargs
        }

        # Use SSE streaming endpoint to get per-token hidden states
        resp = requests.post(
            f"{self.base_url}/api/extra/generate/stream",
            json=payload,
            stream=True
        )

        generated_text = ""
        hidden_states = []

        for line in resp.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data = json.loads(line[6:])

                    if data.get('token'):
                        token_info = data['token']
                        generated_text += token_info.get('text', '')

                        # Check for hidden state
                        # Format: {"shape": [n_embd], "encoding": "base64", "dtype": "float32", "data": "..."}
                        hs = data.get('hidden_state')
                        if hs is not None and 'data' in hs:
                            # Decode base64 hidden state
                            hs_bytes = base64.b64decode(hs['data'])
                            hs_array = np.frombuffer(hs_bytes, dtype=np.float32)
                            n_embd = hs.get('shape', [len(hs_array)])[0]
                            hidden_states.append({
                                'token_id': token_info.get('token_id', -1),
                                'text': token_info.get('text', ''),
                                'hidden_state': hs_array,
                                'n_embd': n_embd
                            })

                    if data.get('finish'):
                        break

        return generated_text, hidden_states

    def store_hidden_states(self, hidden_states: List[Dict]):
        """Store hidden states for later injection."""
        self.stored_hidden_states = []
        self.stored_token_ids = []

        for hs in hidden_states:
            if hs.get('hidden_state') is not None:
                self.stored_hidden_states.append(hs['hidden_state'])
                self.stored_token_ids.append(hs['token_id'])
                if self.n_embd == 0:
                    self.n_embd = hs.get('n_embd', len(hs['hidden_state']))

        print(f"Stored {len(self.stored_hidden_states)} hidden states, n_embd={self.n_embd}")

    def generate_with_ouroboros(self,
                                prompt: str,
                                injection_start_position: int,
                                max_length: int = 50,
                                temperature: float = 0.7,
                                **kwargs) -> str:
        """
        Generate text while injecting stored hidden states at specified positions.

        Args:
            prompt: The prompt text
            injection_start_position: Position in input where stored embeddings start
            max_length: Maximum tokens to generate
            temperature: Sampling temperature
        """
        if not self.stored_hidden_states:
            raise ValueError("No hidden states stored. Call store_hidden_states first.")

        # Prepare embeddings for injection
        embeddings_flat = np.vstack(self.stored_hidden_states).astype(np.float32).flatten()
        embeddings_b64 = base64.b64encode(embeddings_flat.tobytes()).decode('ascii')

        # Positions where embeddings should be injected
        positions = list(range(
            injection_start_position,
            injection_start_position + len(self.stored_hidden_states)
        ))

        payload = {
            "prompt": prompt,
            "max_length": max_length,
            "temperature": temperature,
            "ouroboros_mode": True,
            "ouroboros_embeddings": embeddings_b64,
            "ouroboros_embd_count": len(self.stored_hidden_states),
            "ouroboros_positions": positions,
            "ouroboros_n_embd": self.n_embd,
            **kwargs
        }

        resp = requests.post(f"{self.base_url}/api/v1/generate", json=payload)
        if resp.status_code == 200:
            return resp.json().get('results', [{}])[0].get('text', '')
        else:
            print(f"Error: {resp.status_code} - {resp.text}")
            return ""


def run_coherence_experiment(client: OuroborosClient, prompt: str, max_tokens: int = 30,
                             seed: int = 42, temperature: float = 0.0):
    """
    Run a controlled experiment comparing:
    1. Normal generation (baseline)
    2. Generation with ouroboros (hidden state injection)

    Tests whether preserving continuous representation improves coherence.

    Args:
        seed: Fixed seed for reproducibility
        temperature: 0.0 for greedy (deterministic), >0 for sampling
    """
    print("=" * 70)
    print("OUROBOROS COHERENCE EXPERIMENT (DETERMINISTIC)")
    print("=" * 70)
    print(f"Prompt: {prompt}")
    print(f"Tokens per phase: {max_tokens}")
    print(f"Seed: {seed}")
    print(f"Temperature: {temperature} {'(greedy/deterministic)' if temperature == 0 else ''}")
    print()

    # Phase 1: Generate initial text with hidden state capture
    print("[Phase 1] Initial Generation (with hidden state extraction)")
    print("-" * 50)
    text1, hidden_states = client.generate_with_hidden_states(
        prompt,
        max_length=max_tokens,
        temperature=temperature,
        seed=seed
    )
    print(f"Generated: {text1}")
    print(f"Captured {len(hidden_states)} hidden states")
    print()

    if not hidden_states:
        print("ERROR: No hidden states captured. Check that output_hidden_states is working.")
        return

    # Store the hidden states
    client.store_hidden_states(hidden_states)

    # Analyze hidden state statistics
    if client.stored_hidden_states:
        embeddings = np.vstack(client.stored_hidden_states)
        print("[Analysis] Hidden state statistics:")
        print(f"  Shape: {embeddings.shape}")
        print(f"  Mean norm: {np.mean(np.linalg.norm(embeddings, axis=1)):.4f}")
        print(f"  Std norm: {np.std(np.linalg.norm(embeddings, axis=1)):.4f}")

        # Inter-token distances
        distances = [np.linalg.norm(embeddings[i] - embeddings[i-1])
                     for i in range(1, len(embeddings))]
        if distances:
            print(f"  Mean inter-token distance: {np.mean(distances):.4f}")
        print()

    # Phase 2: Control - continue with normal token embeddings
    print("[Phase 2] Control Continuation (normal token embedding lookup)")
    print("-" * 50)
    continuation_prompt = prompt + text1 + " Furthermore,"
    print(f"Continuation prompt: ...{text1[-50:]} Furthermore,")
    text2_control = client.generate_with_hidden_states(
        continuation_prompt,
        max_length=max_tokens,
        temperature=temperature,
        seed=seed
    )[0]
    print(f"Continuation: {text2_control}")
    print()

    # Phase 3: Ouroboros - continue with injected hidden states
    print("[Phase 3] Ouroboros Continuation (hidden state injection)")
    print("-" * 50)
    print("(Injecting stored hidden states instead of token embeddings)")

    # For ouroboros, we inject the stored hidden states at the positions
    # corresponding to the model's previous output
    # The prompt structure is: [original_prompt][previous_output][continuation_cue]
    # We inject at the positions where [previous_output] would be

    # Use actual tokenization for accurate position calculation
    prompt_token_count = client.count_tokens(prompt)
    text1_token_count = len(hidden_states)  # We captured one hidden state per generated token
    continuation_cue = " Furthermore,"
    continuation_cue_token_count = client.count_tokens(continuation_cue)

    print(f"  Prompt tokens: {prompt_token_count}")
    print(f"  Generated tokens (with hidden states): {text1_token_count}")
    print(f"  Continuation cue tokens: {continuation_cue_token_count}")

    # Hidden states start right after the prompt
    injection_start = prompt_token_count
    injection_end = injection_start + text1_token_count
    print(f"  Injection positions: [{injection_start}, {injection_end})")
    print()

    try:
        text2_ouroboros = client.generate_with_ouroboros(
            continuation_prompt,
            injection_start_position=injection_start,
            max_length=max_tokens,
            temperature=temperature,
            seed=seed
        )
        print(f"Continuation: {text2_ouroboros}")
    except Exception as e:
        print(f"Ouroboros generation failed: {e}")
        text2_ouroboros = ""
    print()

    # Comparison
    print("[Comparison]")
    print("-" * 50)
    print(f"Control length: {len(text2_control)} chars")
    print(f"Ouroboros length: {len(text2_ouroboros)} chars")
    print()

    if text2_control and text2_ouroboros:
        # Check for exact match
        if text2_control == text2_ouroboros:
            print("RESULT: Outputs are IDENTICAL")
            print("The hidden state injection produced the same output as token embedding.")
            print("This suggests the continuous representation was effectively quantized")
            print("to the same discrete token embeddings by the model.")
        else:
            print("RESULT: Outputs DIFFER")
            print()
            # Find first divergence point
            min_len = min(len(text2_control), len(text2_ouroboros))
            diverge_idx = None
            for i in range(min_len):
                if text2_control[i] != text2_ouroboros[i]:
                    diverge_idx = i
                    break

            if diverge_idx is not None:
                print(f"First divergence at character {diverge_idx}:")
                ctx_start = max(0, diverge_idx - 20)
                ctx_end = min(min_len, diverge_idx + 30)
                print(f"  Control:   ...{text2_control[ctx_start:ctx_end]}...")
                print(f"  Ouroboros: ...{text2_ouroboros[ctx_start:ctx_end]}...")
                print(f"                {' ' * (diverge_idx - ctx_start)}^ divergence")
            elif len(text2_control) != len(text2_ouroboros):
                print(f"Outputs match up to char {min_len}, then lengths differ")

            print()
            # Token-level comparison
            control_tokens = text2_control.split()
            ouroboros_tokens = text2_ouroboros.split()
            print(f"Control tokens: {len(control_tokens)}")
            print(f"Ouroboros tokens: {len(ouroboros_tokens)}")

            # Word-level diff
            matching = 0
            for i in range(min(len(control_tokens), len(ouroboros_tokens))):
                if control_tokens[i] == ouroboros_tokens[i]:
                    matching += 1
                else:
                    break
            print(f"Matching prefix tokens: {matching}")

        print()
        print("INTERPRETATION:")
        print("If outputs differ, the continuous representation contains information")
        print("that affects the model's predictions - the 'texture' is preserved.")
        print("If identical, the model may be robust to the embedding source,")
        print("or the hidden states were close enough to canonical embeddings.")

    return {
        'prompt': prompt,
        'phase1_text': text1,
        'phase1_hidden_states_count': len(hidden_states),
        'phase2_control': text2_control,
        'phase3_ouroboros': text2_ouroboros,
    }


def main():
    parser = argparse.ArgumentParser(description="Test ouroboros via HTTP API")
    parser.add_argument('--url', default='http://localhost:5001',
                        help='KoboldCpp server URL')
    parser.add_argument('--prompt', default="The nature of consciousness involves",
                        help='Test prompt')
    parser.add_argument('--tokens', type=int, default=30,
                        help='Tokens per phase')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--temperature', type=float, default=0.0,
                        help='Sampling temperature (0.0 = greedy/deterministic)')

    args = parser.parse_args()

    client = OuroborosClient(args.url)

    if not client.check_server():
        print(f"Error: Cannot connect to server at {args.url}")
        print("Make sure koboldcpp is running with a loaded model.")
        print("\nRebuild koboldcpp with ouroboros support:")
        print("  make clean && make LLAMA_CUBLAS=1 -j8")
        return

    info = client.get_model_info()
    print(f"Connected to: {info.get('result', 'Unknown model')}")
    print()

    results = run_coherence_experiment(client, args.prompt, args.tokens,
                                        seed=args.seed, temperature=args.temperature)

    print("\n" + "=" * 70)
    print("Experiment complete.")


if __name__ == "__main__":
    main()
