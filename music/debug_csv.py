#!/usr/bin/env python3
import csv
from pathlib import Path

csv_path = Path(__file__).parent / 'my_music_library.csv'

print("📄 CSV Debug Info")
print("=" * 40)

with open(csv_path, 'r') as f:
    lines = f.readlines()
    print(f"Total lines: {len(lines)}")
    print(f"First line: '{lines[0].strip()}'")
    print(f"Second line: '{lines[1].strip()}'")
    print(f"Third line: '{lines[2].strip()}'")

print("\n📊 Attempting to read as CSV:")
with open(csv_path, 'r') as f:
    # Try reading with different approaches
    f.seek(0)
    reader = csv.reader(f)
    for i, row in enumerate(reader):
        if i < 5:  # Show first 5 rows
            print(f"Row {i}: {row}")
