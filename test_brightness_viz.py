#!/usr/bin/env python3
"""
Quick brightness visualization test.
Generates text about Plato, then outputs a BMP showing token brightness.
"""

import requests
import struct
import json

BASE_URL = "http://localhost:5001"

def generate_text(prompt, max_length=100):
    """Generate text using the KoboldCpp API."""
    response = requests.post(
        f"{BASE_URL}/api/v1/generate",
        json={"prompt": prompt, "max_length": max_length, "temperature": 0.7}
    )
    try:
        return response.json()
    except:
        return None

def get_brightness():
    """Get brightness data from the API."""
    response = requests.get(f"{BASE_URL}/api/extra/brightness")
    return response.json()

def brightness_to_rgb24(brightness):
    """Convert brightness to 24-bit RGB.

    Brightness IS the 24-bit RGB value directly:
    - 16777215 (0xFFFFFF) = white = max brightness
    - 16777214 (0xFFFFFE) = one decay
    - 0 = black = fully decayed

    No mapping needed - brightness value is the RGB int.
    """
    rgb_int = max(0, min(0xFFFFFF, int(brightness)))

    r = (rgb_int >> 16) & 0xFF
    g = (rgb_int >> 8) & 0xFF
    b = rgb_int & 0xFF
    return r, g, b

def write_bmp(filename, brightness_data, width=256, height=100):
    """Write brightness data as a BMP file.

    Uses 24-bit RGB encoding for maximum precision:
    - White (255,255,255) = max brightness (10000)
    - Each brightness decrement = -1 in 24-bit RGB space
    """
    ctx_len = len(brightness_data)

    # BMP needs width to be multiple of 4 for row padding
    row_size = (width * 3 + 3) & ~3
    padding = row_size - width * 3

    # Calculate image size
    pixel_data_size = row_size * height
    file_size = 54 + pixel_data_size

    with open(filename, 'wb') as f:
        # BMP Header (14 bytes)
        f.write(b'BM')
        f.write(struct.pack('<I', file_size))
        f.write(struct.pack('<HH', 0, 0))
        f.write(struct.pack('<I', 54))

        # DIB Header (40 bytes)
        f.write(struct.pack('<I', 40))
        f.write(struct.pack('<i', width))
        f.write(struct.pack('<i', height))
        f.write(struct.pack('<HH', 1, 24))
        f.write(struct.pack('<I', 0))
        f.write(struct.pack('<I', pixel_data_size))
        f.write(struct.pack('<i', 2835))
        f.write(struct.pack('<i', 2835))
        f.write(struct.pack('<II', 0, 0))

        # Pixel data (bottom to top in BMP)
        for row in range(height - 1, -1, -1):
            for col in range(width):
                if col < ctx_len:
                    r, g, b = brightness_to_rgb24(brightness_data[col])
                else:
                    r, g, b = 0, 0, 0  # Unused positions are black

                # BGR format for BMP
                f.write(bytes([b, g, r]))

            # Row padding
            f.write(b'\x00' * padding)

    print(f"Wrote {filename}: {width}x{height} BMP")

    # Show some example RGB values for verification
    if ctx_len > 0:
        r, g, b = brightness_to_rgb24(brightness_data[0])
        print(f"  Token 0: brightness={brightness_data[0]:.0f} → RGB({r},{g},{b})")
        if ctx_len > 1:
            r, g, b = brightness_to_rgb24(brightness_data[1])
            print(f"  Token 1: brightness={brightness_data[1]:.0f} → RGB({r},{g},{b})")

def main():
    print("Testing brightness visualization...")

    # Generate some text about Plato
    print("\n1. Generating text about Plato...")
    result = generate_text(
        "Plato was an ancient Greek philosopher who founded the Academy in Athens. "
        "His philosophical works include The Republic, in which he describes",
        max_length=50
    )
    print(f"   Generation result: {result}")

    # Get brightness data
    print("\n2. Fetching brightness data...")
    brightness = get_brightness()
    print(f"   valid: {brightness['valid']}")
    print(f"   ctx_len: {brightness['ctx_len']}")
    print(f"   sink_pos: {brightness['sink_pos']}")

    if brightness['valid'] and brightness['data']:
        data = brightness['data']
        print(f"   brightness range: {min(data):.1f} - {max(data):.1f}")
        print(f"   first 10 values: {[round(v, 1) for v in data[:10]]}")

        # Write BMP
        print("\n3. Writing BMP visualization...")
        write_bmp("/tmp/plato_brightness.bmp", data)

        # Also write token index visualization
        print("\nDone! Open /tmp/plato_brightness.bmp to see the visualization.")
        print("Bright = important tokens, Dark = less important tokens")
    else:
        print("   No brightness data available!")

if __name__ == "__main__":
    main()
