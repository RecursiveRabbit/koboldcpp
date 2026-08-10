#!/usr/bin/env python3
"""
Ouroboros Experiment: Continuous Representation Preservation

This script tests whether feeding back pre-unembedding hidden states (instead of
discrete token embeddings) preserves coherence through the model's generation cycle.

The hypothesis: The hidden state H before unembedding contains richer information than
the discrete token it maps to. By feeding H directly instead of doing embedding lookup
on the sampled token, we preserve the "texture" of the model's thought.

Experiment design:
1. Generate a response with hidden state extraction enabled
2. Store those hidden states in the ouroboros buffer
3. Run two continuations:
   a) CONTROL: Feed back using normal token embedding lookup
   b) OUROBOROS: Feed back using stored hidden states
4. Compare outputs for coherence, consistency, and "texture preservation"
"""

import ctypes
import numpy as np
import json
import sys
import os

# Add koboldcpp to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# We'll load the library directly for testing
def load_koboldcpp_library():
    """Load the koboldcpp shared library."""
    import platform
    if platform.system() == "Windows":
        lib_name = "koboldcpp.dll"
    elif platform.system() == "Darwin":
        lib_name = "koboldcpp.dylib"
    else:
        lib_name = "koboldcpp.so"

    lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), lib_name)
    if not os.path.exists(lib_path):
        raise FileNotFoundError(f"Library not found: {lib_path}")

    return ctypes.CDLL(lib_path)


class OuroborosExperiment:
    """
    Controlled experiment for testing ouroboros (continuous representation preservation).
    """

    def __init__(self, handle):
        self.handle = handle
        self.n_embd = 0
        self.stored_embeddings = []
        self.stored_token_ids = []
        self.stored_positions = []

    def store_generation_hidden_states(self):
        """
        After generation, store all hidden states into our local buffer.
        Returns the number of states stored.
        """
        # Call C++ to store from generated_tokens into g_ouroboros_buffer
        self.handle.ouroboros_store_generation()

        count = self.handle.ouroboros_count()
        self.n_embd = self.handle.ouroboros_n_embd()

        print(f"[Ouroboros] Stored {count} hidden states, n_embd={self.n_embd}")

        # Copy to local Python buffer for manipulation
        self.stored_embeddings = []
        self.stored_token_ids = []
        self.stored_positions = []

        for i in range(count):
            # Get embedding pointer
            embd_ptr = self.handle.ouroboros_embedding(i)
            if embd_ptr:
                # Copy to numpy array
                embd = np.ctypeslib.as_array(embd_ptr, shape=(self.n_embd,)).copy()
                self.stored_embeddings.append(embd)
            else:
                self.stored_embeddings.append(None)

            self.stored_token_ids.append(self.handle.ouroboros_token_id(i))
            self.stored_positions.append(self.handle.ouroboros_position(i))

        return count

    def get_embeddings_for_injection(self, start_idx=0, count=None):
        """
        Prepare embeddings array for injection into generation.
        Returns (embeddings_flat, positions) ready for ctypes.
        """
        if count is None:
            count = len(self.stored_embeddings) - start_idx

        embeddings = self.stored_embeddings[start_idx:start_idx+count]
        positions = self.stored_positions[start_idx:start_idx+count]

        # Filter out None values
        valid_pairs = [(e, p) for e, p in zip(embeddings, positions) if e is not None]
        if not valid_pairs:
            return None, None

        embeddings, positions = zip(*valid_pairs)

        # Flatten embeddings to contiguous array
        embeddings_flat = np.vstack(embeddings).astype(np.float32).flatten()
        positions_arr = np.array(positions, dtype=np.int32)

        return embeddings_flat, positions_arr

    def compute_embedding_stats(self):
        """Compute statistics about stored embeddings for analysis."""
        if not self.stored_embeddings:
            return {}

        valid_embeddings = [e for e in self.stored_embeddings if e is not None]
        if not valid_embeddings:
            return {}

        embeddings_array = np.vstack(valid_embeddings)

        return {
            'count': len(valid_embeddings),
            'n_embd': self.n_embd,
            'mean_norm': float(np.mean(np.linalg.norm(embeddings_array, axis=1))),
            'std_norm': float(np.std(np.linalg.norm(embeddings_array, axis=1))),
            'mean': float(np.mean(embeddings_array)),
            'std': float(np.std(embeddings_array)),
            'min': float(np.min(embeddings_array)),
            'max': float(np.max(embeddings_array)),
        }

    def compute_inter_embedding_distances(self):
        """
        Compute distances between consecutive embeddings.
        Useful for understanding how the hidden state evolves.
        """
        valid_embeddings = [e for e in self.stored_embeddings if e is not None]
        if len(valid_embeddings) < 2:
            return []

        distances = []
        for i in range(1, len(valid_embeddings)):
            dist = np.linalg.norm(valid_embeddings[i] - valid_embeddings[i-1])
            distances.append(float(dist))

        return distances


def run_experiment(model_path, prompt, continuation_prompt="", max_tokens=50):
    """
    Run the full ouroboros experiment.

    Args:
        model_path: Path to the GGUF model file
        prompt: Initial prompt to generate from
        continuation_prompt: Text to prepend before continuing (for control comparison)
        max_tokens: Number of tokens to generate in each phase

    Returns:
        dict with experiment results
    """
    # This is a simplified test - in practice, you'd use koboldcpp's full API
    print("=" * 60)
    print("OUROBOROS EXPERIMENT")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Prompt: {prompt[:100]}...")
    print(f"Max tokens per phase: {max_tokens}")
    print("=" * 60)

    results = {
        'prompt': prompt,
        'continuation_prompt': continuation_prompt,
        'max_tokens': max_tokens,
        'phases': {}
    }

    # Note: This is a template - actual execution requires the model to be loaded
    # and the koboldcpp server to be running. The key demonstration is the API
    # structure for ouroboros mode.

    print("\n[Phase 1] Initial Generation with Hidden State Extraction")
    print("-" * 40)
    print("Would generate text and capture hidden states...")
    print("Hidden states represent the continuous pre-unembedding vectors")

    print("\n[Phase 2] Control Continuation (Normal Token Embedding)")
    print("-" * 40)
    print("Would continue using standard token -> embedding lookup...")
    print("This is the baseline where discrete tokens are re-embedded")

    print("\n[Phase 3] Ouroboros Continuation (Hidden State Injection)")
    print("-" * 40)
    print("Would continue by injecting stored hidden states directly...")
    print("This bypasses the embedding lookup, preserving continuous representation")

    print("\n[Analysis]")
    print("-" * 40)
    print("Comparison metrics:")
    print("- Token overlap between control and ouroboros continuations")
    print("- Semantic similarity of outputs")
    print("- Coherence scores")
    print("- 'Texture' preservation (subjective evaluation)")

    return results


def create_ouroboros_generation_params(base_params, embeddings_flat, positions_arr):
    """
    Modify generation parameters to enable ouroboros mode.

    Args:
        base_params: Base generation parameters dict
        embeddings_flat: Flattened numpy array of embeddings [n_tokens * n_embd]
        positions_arr: Array of positions where embeddings should be injected

    Returns:
        Modified params dict with ouroboros fields set
    """
    params = base_params.copy()

    params['ouroboros_mode'] = True
    params['ouroboros_embeddings'] = embeddings_flat
    params['ouroboros_embd_count'] = len(positions_arr)
    params['ouroboros_positions'] = positions_arr

    return params


# Example usage documentation
USAGE = """
OUROBOROS EXPERIMENT - Usage
============================

This module provides tools for testing continuous representation preservation
in language model generation.

Basic workflow:
1. Load model with koboldcpp
2. Generate text with output_hidden_states=True
3. Call experiment.store_generation_hidden_states()
4. Get embeddings with experiment.get_embeddings_for_injection()
5. Pass embeddings to next generation with ouroboros_mode=True

Example API call structure (JSON for HTTP API):
{
    "prompt": "Your prompt here",
    "max_length": 100,
    "output_hidden_states": true,
    "ouroboros_mode": true,
    "ouroboros_embeddings": [<flattened float array>],
    "ouroboros_embd_count": <number of embeddings>,
    "ouroboros_positions": [<position indices>]
}

Key insight:
-----------
When a model generates the token "Paris", the actual hidden state H before
unembedding isn't exactly at the canonical embedding position for "Paris" -
it's slightly offset, perhaps representing "Paris in a romantic sense" or
"Paris as a memory". When we discretize H -> "Paris" -> E("Paris"), we lose
that offset. Ouroboros mode preserves it by feeding H directly.

Compilation:
-----------
Rebuild koboldcpp with: make LLAMA_CUBLAS=1 -j8

Testing:
--------
python test_ouroboros.py --model /path/to/model.gguf --prompt "Once upon a time"
"""


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Ouroboros Experiment: Test continuous representation preservation",
        epilog=USAGE,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--model', type=str, help='Path to GGUF model file')
    parser.add_argument('--prompt', type=str, default="The nature of consciousness is",
                        help='Initial prompt for generation')
    parser.add_argument('--max-tokens', type=int, default=50,
                        help='Maximum tokens per generation phase')
    parser.add_argument('--api-url', type=str, default='http://localhost:5001',
                        help='KoboldCpp API URL')

    args = parser.parse_args()

    if args.model:
        results = run_experiment(args.model, args.prompt, max_tokens=args.max_tokens)
        print("\n" + "=" * 60)
        print("Experiment complete. Results:")
        print(json.dumps(results, indent=2))
    else:
        print(USAGE)
        print("\nTo run experiment, provide --model path")
        print("\nAPI structures have been updated in:")
        print("  - expose.h (C++ struct definitions)")
        print("  - gpttype_adapter.cpp (ouroboros buffer + decode injection)")
        print("  - expose.cpp (API function exports)")
        print("  - koboldcpp.py (Python ctypes bindings)")
