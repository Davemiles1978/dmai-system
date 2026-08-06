"""
DMAI AvatarEngine — Self-contained procedural avatar generation.
Pure Python + PIL. No GPU, no external APIs, no models.
Creates consistent human faces using parametric facial geometry.
"""

import math, random, io, base64
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Tuple
from PIL import Image, ImageDraw, ImageFilter


class FaceParams:
    def __init__(self, seed: int = None):
        rng = random.Random(seed or 42)
        self.head_width = rng.uniform(0.85, 1.15)
        self.head_height = rng.uniform(0.90, 1.10)
        self.jaw_width = rng.uniform(0.75, 0.95)
        self.chin_shape = rng.uniform(0.3, 0.7)
        self.eye_size = rng.uniform(0.85, 1.15)
        self.eye_spacing = rng.uniform(0.90, 1.10)
        self.eye_height = rng.uniform(0.42, 0.48)
        self.nose_length = rng.uniform(0.8, 1.2)
        self.nose_width = rng.uniform(0.8, 1.2)
        self.mouth_width = rng.uniform(0.8, 1.2)
        self.mouth_height = rng.uniform(0.55, 0.62)
        self.lip_thickness = rng.uniform(0.7, 1.3)
        self.hair_color = rng.choice([(20,15,10),(60,40,20),(120,80,40),(200,160,100),(40,30,20),(180,50,30),(30,30,30),(80,80,80),(200,180,160)])
        self.skin_tone = rng.choice([(255,220,190),(240,200,170),(220,180,150),(200,160,130),(180,140,110),(255,230,210)])
        variation = rng.uniform(-10, 10)
        self.skin_tone = tuple(max(0,min(255,int(c+variation))) for c in self.skin_tone)


class AvatarEngine:
    def __init__(self, output_dir: str = "data/generated_content"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.generation_count = 0
        self.params = FaceParams()
    
    def _shade(self, color, amount):
        return tuple(max(0, min(255, int(c + amount))) for c in color)
    
    def generate_face(self, expression="neutral", width=512, height=512):
        p = self.params
        img = Image.new("RGB", (width, height), (240, 240, 245))
        draw = ImageDraw.Draw(img)
        cx, cy = width//2, height//2
        head_w = int(160 * p.head_width)
        head_h = int(210 * p.head_height)
        head_top = cy - head_h
        head_bottom = cy + head_h
        jaw_w = int(head_w * p.jaw_width)
        
        # Head shape
        pts = []
        for a in range(-90, 90, 3):
            r = math.radians(a)
            pts.append((cx + int(head_w*0.95*math.cos(r)), cy - int(head_h*math.sin(r))-20))
        for a in range(90, 270, 3):
            r = math.radians(a)
            pts.append((cx + int(jaw_w*math.cos(r)*(0.7+0.3*p.chin_shape)), cy - int(head_h*0.8*math.sin(r))+30))
        draw.polygon(pts, fill=p.skin_tone, outline=self._shade(p.skin_tone, -30))
        
        # Hair
        hair_pts = []
        for a in range(-100, 260, 3):
            r = math.radians(a)
            hair_pts.append((cx+int(head_w*1.05*math.cos(r)), cy-int(head_h*1.02*math.sin(r))-15))
        draw.polygon(hair_pts, fill=p.hair_color)
        
        # Eyes
        eye_y = cy - int(head_h * p.eye_height)
        eye_sp = int(55 * p.eye_spacing)
        eye_w = int(35 * p.eye_size)
        eye_h = int(18 * p.eye_size) if expression != "surprised" else int(25 * p.eye_size)
        for side in [-1, 1]:
            ex = cx + side * eye_sp
            draw.ellipse([ex-eye_w, eye_y-eye_h, ex+eye_w, eye_y+eye_h], fill=(255,255,255), outline=(80,80,80))
            ir = eye_h - 2
            draw.ellipse([ex-ir, eye_y-ir, ex+ir, eye_y+ir], fill=(60,120,200))
            pr = ir//2
            draw.ellipse([ex-pr, eye_y-pr, ex+pr, eye_y+pr], fill=(0,0,0))
            draw.ellipse([ex+ir//2-2, eye_y-ir//2-2, ex+ir//2+2, eye_y-ir//2+2], fill=(255,255,255))
            draw.arc([ex-eye_w, eye_y-eye_h-5, ex+eye_w, eye_y-eye_h+10], 0, 180, fill=(40,30,30), width=2)
        
        # Eyebrows
        brow_y = eye_y - eye_h - 15
        if expression == "angry": brow_y -= 5
        elif expression == "surprised": brow_y -= 10
        for side in [-1, 1]:
            bx = cx + side * eye_sp
            for t in range(-2, 3):
                draw.line([(bx-eye_w-5, brow_y+t), (bx, brow_y-5+t), (bx+eye_w+5, brow_y+t)], fill=(50,35,25), width=2)
        
        # Nose
        nose_y = eye_y + eye_h + 10
        nose_b = nose_y + int(50*p.nose_length)
        nose_w = int(25*p.nose_width)
        draw.line([(cx, nose_y), (cx, nose_b)], fill=self._shade(p.skin_tone,-20), width=3)
        for side in [-1, 1]:
            nx = cx + side*nose_w
            draw.ellipse([nx-8, nose_b-5, nx+8, nose_b+10], fill=self._shade(p.skin_tone,-15), outline=self._shade(p.skin_tone,-40))
        
        # Mouth
        mouth_y = nose_b + int(30*p.mouth_height)
        mouth_w = int(45*p.mouth_width)
        lip_t = int(10*p.lip_thickness)
        if expression == "happy":
            pts = [(cx+x, mouth_y+int(-(x**2)/(mouth_w*4))) for x in range(-mouth_w, mouth_w+1, 2)]
            draw.line(pts, fill=(180,80,80), width=3)
        elif expression == "sad":
            pts = [(cx+x, mouth_y+int((x**2)/(mouth_w*4))) for x in range(-mouth_w, mouth_w+1, 2)]
            draw.line(pts, fill=(150,70,70), width=3)
        elif expression == "surprised":
            draw.ellipse([cx-mouth_w//2, mouth_y-lip_t, cx+mouth_w//2, mouth_y+lip_t*2], fill=(40,20,20))
            draw.ellipse([cx-mouth_w//3, mouth_y-lip_t//2, cx+mouth_w//3, mouth_y+lip_t], fill=(255,255,250))
        elif expression == "angry":
            draw.line([(cx-mouth_w, mouth_y), (cx+mouth_w, mouth_y)], fill=(180,60,60), width=3)
        else:
            draw.line([(cx-mouth_w, mouth_y), (cx+mouth_w, mouth_y)], fill=(190,100,100), width=3)
        
        # Neck & shoulders
        neck_w = int(jaw_w*0.6)
        draw.rectangle([cx-neck_w, head_bottom-20, cx+neck_w, height], fill=self._shade(p.skin_tone,-10))
        draw.polygon([(0,height),(0,head_bottom+30),(cx-neck_w*2,head_bottom+30),(cx-neck_w,head_bottom),(cx+neck_w,head_bottom),(cx+neck_w*2,head_bottom+30),(width,head_bottom+30),(width,height)], fill=(60,60,80))
        
        img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
        return img
    
    def generate_avatar(self, expression="neutral", width=512, height=512) -> Dict:
        img = self.generate_face(expression, width, height)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"dma_avatar_{expression}_{ts}_{self.generation_count}.png"
        filepath = self.output_dir / filename
        img.save(str(filepath), "PNG")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        self.generation_count += 1
        return {"ok":True,"image_base64":f"data:image/png;base64,{b64}","view_url":f"/api/content/view/{filename}","file":str(filepath),"filename":filename,"expression":expression,"generator":"DMAI AvatarEngine"}


if __name__ == "__main__":
    os.makedirs("data/generated_content", exist_ok=True)
    engine = AvatarEngine()
    for expr in ["neutral","happy","sad","surprised","angry"]:
        r = engine.generate_avatar(expression=expr)
        print(f"  {expr}: {r['filename']}")
