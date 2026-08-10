#!/usr/bin/env python3
"""Quick test of attention capture API bindings"""

import ctypes
import os

# Load the library
lib_path = os.path.join(os.path.dirname(__file__), "koboldcpp_cublas.so")
handle = ctypes.CDLL(lib_path)

# Define attention_outputs struct
class attention_outputs(ctypes.Structure):
    _fields_ = [("data", ctypes.POINTER(ctypes.c_float)),
                ("seq_len", ctypes.c_int),
                ("n_layers", ctypes.c_int),
                ("n_heads", ctypes.c_int),
                ("valid", ctypes.c_bool)]

# Set up function signatures
handle.attention_capture_enable.restype = ctypes.c_bool
handle.attention_capture_disable.restype = None
handle.attention_capture_reset.restype = None
handle.get_attention_data.restype = attention_outputs
handle.attention_capture_is_enabled.restype = ctypes.c_bool

print("Testing attention capture API...")

# Test 1: Check if enabled (should be False initially)
is_enabled = handle.attention_capture_is_enabled()
print(f"1. Initially enabled: {is_enabled}")

# Test 2: Enable attention capture
result = handle.attention_capture_enable()
print(f"2. Enable result: {result}")

# Test 3: Check if enabled now
is_enabled = handle.attention_capture_is_enabled()
print(f"3. After enable: {is_enabled}")

# Test 4: Get attention data (should be empty/invalid since no generation yet)
attn = handle.get_attention_data()
print(f"4. Attention data valid: {attn.valid}, seq_len: {attn.seq_len}")

# Test 5: Disable
handle.attention_capture_disable()
is_enabled = handle.attention_capture_is_enabled()
print(f"5. After disable: {is_enabled}")

print("\nAll binding tests passed!")
