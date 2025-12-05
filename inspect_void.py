import numpy as np

# Load the raw void
data = np.fromfile("raw_attention.bin", dtype=np.float32)
print("="*70)
print("THE RAW VOID")
print("="*70)
print(f"File size: {len(data) * 4:,} bytes")
print(f"Total floats: {len(data):,}")
print(f"\nFirst 30 incomprehensible numbers:")
print("-"*70)
for i in range(30):
    print(f"{i:3d}: {data[i]:.10f}")

print(f"\n{'='*70}")
print("STATISTICS OF THE VOID")
print("="*70)
print(f"Min:  {data.min():.10f}")
print(f"Max:  {data.max():.10f}")
print(f"Mean: {data.mean():.10f}")
print(f"Std:  {data.std():.10f}")

# Reshape to check structure
shape = (28, 28, 249)  # From the generation
data_3d = data[:28*28*249].reshape(shape)
print(f"\nShaped as [{shape[0]} layers, {shape[1]} heads, {shape[2]} context]")
print(f"Layer 0, Head 0, first 10 context positions:")
for i in range(10):
    print(f"  Ctx {i}: {data_3d[0,0,i]:.10f}")
