#!/usr/bin/env python3
"""
Test script for attention extraction in koboldcpp
Tests the push-model token+attention pairing
"""

import sys
import os
import ctypes

def test_attention_extraction():
    """Test basic attention extraction functionality"""

    model_path = "/home/evans/Coding_Projects/Halo_Weave/models/Qwen2.5-VL-7B-Instruct-Q8_0.gguf"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    lib_path = os.path.join(script_dir, "koboldcpp_cublas.so")

    print("="*60)
    print("KOBOLDCPP ATTENTION EXTRACTION TEST")
    print("="*60)
    print(f"Library: {lib_path}")
    print(f"Model: {model_path}")
    print()

    # Load library
    print("[1/5] Loading library...")
    try:
        if not os.path.exists(lib_path):
            print(f"✗ Library not found: {lib_path}")
            return False

        handle = ctypes.CDLL(lib_path)
        print(f"✓ Library loaded successfully")
    except Exception as e:
        print(f"✗ Library load failed: {e}")
        return False

    print()
    print("[2/5] Setting up struct definitions...")
    try:
        # Define attention_outputs struct
        class attention_outputs(ctypes.Structure):
            _fields_ = [("data", ctypes.POINTER(ctypes.c_float)),
                        ("n_layers", ctypes.c_int),
                        ("n_heads", ctypes.c_int),
                        ("seq_len", ctypes.c_int),
                        ("valid", ctypes.c_bool)]

        print("✓ attention_outputs struct defined")
    except Exception as e:
        print(f"✗ Struct definition failed: {e}")
        return False

    print()
    print("[3/5] Binding get_token_attention function...")
    try:
        handle.get_token_attention.argtypes = [ctypes.c_int]
        handle.get_token_attention.restype = attention_outputs
        print("✓ get_token_attention binding successful")
        print(f"  Signature: get_token_attention(int idx) -> attention_outputs")
    except Exception as e:
        print(f"✗ Binding failed: {e}")
        return False

    print()
    print("[4/5] Testing function call (no model loaded)...")
    try:
        # Call get_token_attention with index 0 (should return invalid)
        result = handle.get_token_attention(0)
        print(f"✓ Function call successful")
        print(f"  Result:")
        print(f"    valid: {result.valid}")
        print(f"    n_layers: {result.n_layers}")
        print(f"    n_heads: {result.n_heads}")
        print(f"    seq_len: {result.seq_len}")
        print(f"    data ptr: {result.data}")

        if not result.valid:
            print("  ✓ Expected: valid=False (no tokens generated yet)")
    except Exception as e:
        print(f"✗ Function call failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print()
    print("[5/5] Summary")
    print("="*60)
    print("✅ All API bindings are correctly configured")
    print("✅ Library compiled successfully with attention extraction")
    print("✅ Ready for testing with loaded model")
    print()
    print("NEXT STEPS:")
    print("1. Use koboldcpp.py to load model:")
    print(f"   python koboldcpp.py --model '{model_path}' --usecublas")
    print("2. Make a generation request with output_attentions=True")
    print("3. Verify get_token_attention(idx) returns valid attention data")
    print("4. Check shape: [n_layers, n_heads, seq_len]")
    print("="*60)

    return True

if __name__ == "__main__":
    try:
        success = test_attention_extraction()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
