#!/usr/bin/env python3
"""
Debug OCR to see exactly what text is being extracted
"""

import pytesseract
from PIL import Image
from pathlib import Path

screenshots_dir = Path(__file__).parent / 'screenshots'

# Process one screenshot to see raw lines
screenshot = list(screenshots_dir.glob("*.PNG"))[0]
print(f"\n🔍 Processing: {screenshot.name}")
print("=" * 50)

# Extract text
image = Image.open(screenshot)
text = pytesseract.image_to_string(image)

# Show each line with character codes
print("\n📝 RAW OCR LINES:")
print("=" * 50)
lines = text.split('\n')
for i, line in enumerate(lines):
    if line.strip():
        print(f"\nLine {i+1}: '{line}'")
        print(f"Chars: {[ord(c) for c in line]}")
