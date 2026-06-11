#!/usr/bin/env python3
"""
API call configured to match successful web interface generations
"""

import requests
import time
import json
from pathlib import Path

API_KEY = "ecd6b773-0236-4125-9f9a-35ca67ca4a61"
HEADERS = {
    "accept": "application/json",
    "authorization": f"Bearer {API_KEY}",
    "content-type": "application/json"
}

# Try different model IDs that might match web interface
MODELS_TO_TEST = [
    "6bef9f1b-29c0-4735-886d-060b0b7b2b3b",  # Leonardo PhotoReal
    "aa77f04e-3eec-4034-9c07-d0f619dffdbe",  # Leonardo Kino XL
    "1e60896f-3c6a-47a0-9b56-0d45c7d5e9c2",  # Leonardo Vision XL
    "b24e16ff-06e3-43eb-9d33-4410c1f77f6b",  # Leonardo Diffusion XL
]

# The prompt that works in web interface
PROMPT = """Black and white line art coloring page for children. Clean bold lines, large spaces for coloring, pure black on white, no shading, no gray tones. Professional children's coloring book style. Subject: a cute unicorn standing in a meadow. Simple but engaging."""

# Web interface default settings
PAYLOAD_TEMPLATE = {
    "prompt": PROMPT,
    "width": 1024,
    "height": 1328,
    "num_images": 1,
    "photoReal": False,
    "presetStyle": "NONE",
    "alchemy": False,
    "guidance_scale": 7,
    "steps": 30,
}

print("="*60)
print("🎨 Testing API with Web Interface Settings")
print("="*60)

for model_id in MODELS_TO_TEST:
    print(f"\n📝 Testing model: {model_id}")
    payload = PAYLOAD_TEMPLATE.copy()
    payload["modelId"] = model_id
    
    resp = requests.post(
        "https://cloud.leonardo.ai/api/rest/v1/generations",
        headers=HEADERS,
        json=payload
    )
    
    if resp.status_code != 200:
        print(f"   ❌ Model {model_id[:8]}... failed: {resp.status_code}")
        continue
    
    generation_id = resp.json()["sdGenerationJob"]["generationId"]
    print(f"   ✅ Generation started: {generation_id[:8]}...")
    
    for attempt in range(45):
        time.sleep(2)
        status_resp = requests.get(
            f"https://cloud.leonardo.ai/api/rest/v1/generations/{generation_id}",
            headers=HEADERS
        )
        if status_resp.status_code == 200:
            data = status_resp.json()
            if data["generations_by_pk"]["status"] == "COMPLETE":
                images = data["generations_by_pk"]["generated_images"]
                if images:
                    url = images[0]["url"]
                    img_data = requests.get(url).content
                    output_path = Path(f"data/test_model_{model_id[:8]}.png")
                    with open(output_path, "wb") as f:
                        f.write(img_data)
                    print(f"   ✅ Saved to: {output_path}")
                    break
            elif data["generations_by_pk"]["status"] == "FAILED":
                print(f"   ❌ Generation failed")
                break

print("\n" + "="*60)
print("✅ Testing complete. Check which model produced good results.")
print("="*60)
