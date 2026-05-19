"""
Avatar Consistency System - Maintain consistent character across all generated content
"""

import json
import base64
import requests
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

class AvatarConsistencySystem:
    """Maintain consistent avatar identity across all generated images"""
    
    def __init__(self):
        self.avatar_config = self.load_avatar_config()
        self.character_presets = {}
        
    def load_avatar_config(self) -> Dict:
        """Load avatar configuration from file"""
        config_path = Path("data/avatar_config.json")
        if config_path.exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        
        # Default configuration
        return {
            "character_name": "DMAI",
            "base_prompt": "",
            "negative_prompt": "ugly, deformed, blurry, low quality, cartoon, anime, drawing, sketch, watermark, text, signature, bad anatomy",
            "styles": {
                "professional": {},
                "casual": {},
                "glamour": {},
                "fitness": {},
                "business": {}
            }
        }
    
    def save_avatar_config(self):
        """Save avatar configuration to file"""
        config_path = Path("data/avatar_config.json")
        with open(config_path, 'w') as f:
            json.dump(self.avatar_config, f, indent=2)
    
    def set_reference_images(self, image_paths: List[str]) -> Dict:
        """Set reference images for consistent avatar generation"""
        
        reference_images = []
        for img_path in image_paths:
            path = Path(img_path)
            if path.exists():
                # Convert to base64 for storage
                with open(path, 'rb') as f:
                    img_base64 = base64.b64encode(f.read()).decode('utf-8')
                reference_images.append({
                    "path": str(path),
                    "data": img_base64[:100] + "...",  # Truncated for display
                    "original_name": path.name
                })
        
        self.avatar_config["reference_images"] = reference_images
        self.save_avatar_config()
        
        # Generate base prompt from reference images
        self.avatar_config["base_prompt"] = self._extract_character_description(reference_images)
        self.save_avatar_config()
        
        return {
            "success": True,
            "reference_count": len(reference_images),
            "base_prompt": self.avatar_config["base_prompt"]
        }
    
    def _extract_character_description(self, reference_images: List) -> str:
        """Extract consistent character description from reference images"""
        
        # This would ideally use CLIP or similar to analyze the images
        # For now, return a template that you can customize
        return """
        Beautiful woman, consistent facial features, same person across all images,
        professional photography, high quality, 4k, natural skin texture,
        same hairstyle, same eye color, consistent body type, natural expression
        """
    
    def generate_consistent_avatar(self, style: str = "professional", 
                                   action: str = "pose", 
                                   custom_prompt: str = None) -> Dict:
        """Generate avatar maintaining consistency with reference images"""
        
        base_prompt = self.avatar_config.get("base_prompt", "")
        negative_prompt = self.avatar_config.get("negative_prompt", "")
        
        style_prompts = {
            "professional": f"{base_prompt} professional headshot, studio lighting, business attire, confident expression, clean background, {action}",
            "casual": f"{base_prompt} casual everyday look, natural lighting, relaxed pose, street fashion, candid moment, {action}",
            "glamour": f"{base_prompt} glamorous photoshoot, dramatic lighting, elegant dress, magazine quality, red carpet look, {action}",
            "fitness": f"{base_prompt} fitness model, gym attire, athletic pose, motivated expression, workout setting, {action}",
            "business": f"{base_prompt} corporate professional, suit, office background, executive presence, confident stance, {action}",
            "adult": f"{base_prompt} tasteful artistic nude, boudoir photography, soft lighting, sensual but classy, professional quality, {action}"
        }
        
        prompt = custom_prompt or style_prompts.get(style, style_prompts["professional"])
        
        # Return the generation request (to be sent to Stable Diffusion/ComfyUI)
        return {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "style": style,
            "action": action,
            "reference_images": self.avatar_config.get("reference_images", []),
            "width": 768,
            "height": 1024,
            "num_outputs": 4,
            "guidance_scale": 7.5,
            "steps": 30,
            "seed": None  # Random seed for variety
        }
    
    def batch_generate(self, styles: List[str], actions: List[str]) -> List[Dict]:
        """Generate multiple avatars across styles and actions"""
        
        results = []
        for style in styles:
            for action in actions[:2]:  # Limit actions per style
                result = self.generate_consistent_avatar(style, action)
                results.append({
                    "style": style,
                    "action": action,
                    "prompt": result["prompt"],
                    "negative_prompt": result["negative_prompt"]
                })
        return results
    
    def create_content_calendar(self, days: int = 30) -> Dict:
        """Create a content calendar for consistent posting"""
        
        calendar = {}
        
        # Define posting schedule
        schedule = {
            "monday": {"style": "professional", "action": "talking"},
            "tuesday": {"style": "casual", "action": "laughing"},
            "wednesday": {"style": "fitness", "action": "working out"},
            "thursday": {"style": "business", "action": "presenting"},
            "friday": {"style": "glamour", "action": "posing"},
            "saturday": {"style": "casual", "action": "relaxing"},
            "sunday": {"style": "professional", "action": "thinking"}
        }
        
        for i in range(days):
            from datetime import datetime, timedelta
            date = datetime.now() + timedelta(days=i)
            day_name = date.strftime("%A").lower()
            
            if day_name in schedule:
                calendar[date.strftime("%Y-%m-%d")] = {
                    "style": schedule[day_name]["style"],
                    "action": schedule[day_name]["action"],
                    "platform": "instagram",
                    "caption": f"Day {i+1} of 30 - {schedule[day_name]['style']} shoot",
                    "hashtags": "#DMAI #contentcreator #model"
                }
        
        return calendar

class ImageUploader:
    """Handle image uploads and storage"""
    
    def __init__(self, upload_dir: str = "data/avatars"):
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
    
    def save_uploaded_image(self, image_data: bytes, filename: str) -> Dict:
        """Save uploaded image to disk"""
        filepath = self.upload_dir / filename
        with open(filepath, 'wb') as f:
            f.write(image_data)
        
        return {
            "success": True,
            "path": str(filepath),
            "filename": filename,
            "size": len(image_data)
        }
    
    def get_avatar_gallery(self) -> List[Dict]:
        """Get all generated avatars"""
        avatars = []
        for img_path in self.upload_dir.glob("*.jpg"):
            avatars.append({
                "filename": img_path.name,
                "path": str(img_path),
                "modified": datetime.fromtimestamp(img_path.stat().st_mtime).isoformat()
            })
        return sorted(avatars, key=lambda x: x["modified"], reverse=True)

def initialize_avatar_system():
    """Initialize the avatar consistency system"""
    return {
        "consistency": AvatarConsistencySystem(),
        "uploader": ImageUploader()
    }
