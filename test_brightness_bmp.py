#!/usr/bin/env python3
"""
Test script to visualize brightness texture as a BMP image.
Generates text about Plato and outputs the resulting brightness values as a visual.
"""

import requests
import struct
import sys

SERVER = "http://localhost:5001"

def generate_plato_text():
    """Generate ~300 tokens describing Plato"""
    print("Generating text about Plato (300 tokens)...")

    payload = {
        "prompt": "Describe the ancient Greek philosopher Plato, his life, teachings, and influence on Western philosophy.",
        "max_length": 300,
        "temperature": 0.7,
        "top_p": 0.9,
    }

    response = requests.post(f"{SERVER}/api/v1/generate", json=payload)
    if response.status_code != 200:
        print(f"Generation failed: {response.status_code}")
        print(response.text)
        return None

    result = response.json()
    text = result.get("results", [{}])[0].get("text", "")
    print(f"Generated {len(text)} characters")
    print("---")
    print(text[:500] + "..." if len(text) > 500 else text)
    print("---")
    return text

def get_brightness():
    """Fetch brightness texture from server"""
    print("\nFetching brightness data...")

    response = requests.get(f"{SERVER}/api/extra/brightness")
    if response.status_code != 200:
        print(f"Brightness fetch failed: {response.status_code}")
        return None

    data = response.json()
    if not data.get("valid"):
        print("Brightness data not valid")
        return None

    brightness = data.get("data", [])
    ctx_len = data.get("ctx_len", 0)
    sink_pos = data.get("sink_pos", -1)

    print(f"Context length: {ctx_len}")
    print(f"Sink position: {sink_pos}")
    print(f"Brightness values: min={min(brightness):.2f}, max={max(brightness):.2f}, avg={sum(brightness)/len(brightness):.2f}")

    return brightness, sink_pos

def write_bmp(filename, brightness, sink_pos):
    """
    Write brightness data as a BMP image.
    Each token is a vertical column, brightness maps to grayscale.
    Sink position is highlighted in red.
    """
    width = len(brightness)
    height = 64  # Make it tall enough to see

    # Normalize brightness to 0-255 range
    max_b = max(brightness) if brightness else 1.0
    min_b = min(brightness) if brightness else 0.0
    range_b = max_b - min_b if max_b > min_b else 1.0

    # BMP is stored bottom-up, BGR format, rows padded to 4 bytes
    row_size = ((width * 3 + 3) // 4) * 4
    pixel_data_size = row_size * height
    file_size = 54 + pixel_data_size  # 54 byte header

    with open(filename, 'wb') as f:
        # BMP Header (14 bytes)
        f.write(b'BM')                          # Signature
        f.write(struct.pack('<I', file_size))   # File size
        f.write(struct.pack('<HH', 0, 0))       # Reserved
        f.write(struct.pack('<I', 54))          # Pixel data offset

        # DIB Header (40 bytes - BITMAPINFOHEADER)
        f.write(struct.pack('<I', 40))          # Header size
        f.write(struct.pack('<i', width))       # Width
        f.write(struct.pack('<i', height))      # Height (positive = bottom-up)
        f.write(struct.pack('<HH', 1, 24))      # Planes, bits per pixel
        f.write(struct.pack('<I', 0))           # Compression (none)
        f.write(struct.pack('<I', pixel_data_size))  # Image size
        f.write(struct.pack('<i', 2835))        # X pixels per meter
        f.write(struct.pack('<i', 2835))        # Y pixels per meter
        f.write(struct.pack('<I', 0))           # Colors in color table
        f.write(struct.pack('<I', 0))           # Important colors

        # Pixel data (bottom-up, BGR)
        for y in range(height):
            row = bytearray()
            for x in range(width):
                # Normalize brightness to 0-255
                normalized = (brightness[x] - min_b) / range_b
                gray = int(normalized * 255)
                gray = max(0, min(255, gray))

                # Highlight sink position in cyan
                if x == sink_pos:
                    row.extend([255, 255, 0])  # BGR: cyan
                else:
                    row.extend([gray, gray, gray])  # BGR: grayscale

            # Pad row to 4-byte boundary
            while len(row) % 4 != 0:
                row.append(0)

            f.write(row)

    print(f"\nWrote {filename}: {width}x{height} pixels")
    print(f"  - Grayscale intensity = normalized brightness (dark=0, bright=10000)")
    print(f"  - Cyan column = attention sink at position {sink_pos}")

def main():
    # Step 1: Generate text
    text = generate_plato_text()
    if text is None:
        sys.exit(1)

    # Step 2: Get brightness
    result = get_brightness()
    if result is None:
        sys.exit(1)

    brightness, sink_pos = result

    # Step 3: Write BMP
    output_file = "/tmp/plato_brightness.bmp"
    write_bmp(output_file, brightness, sink_pos)

    print(f"\nDone! Open {output_file} to view the brightness texture.")

if __name__ == "__main__":
    main()
