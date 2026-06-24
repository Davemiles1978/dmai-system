"""
Alex Riviera Avatar System - CANONICAL SOURCE OF TRUTH
Based on the master description provided by user
Facial features LOCKED | Hairstyle flexible (colour locked: platinum-blonde)
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List

class AlexRivieraAvatar:
    """
    Alex Riviera's official avatar - CANONICAL SOURCE OF TRUTH
    Facial features LOCKED. Hairstyle flexible (colour only: platinum-blonde)
    """

    def __init__(self):
        _root = Path(__file__).resolve().parents[2]
        self.final_avatar_reference = _root / "data" / "avatars" / "reference_images" / "ChatGPT Image Jun 4, 2026, 03_41_53 PM.png"
        if self.final_avatar_reference.exists():
            print(f"✅ Final avatar reference loaded: {self.final_avatar_reference.name}")

        self.avatar_dir = _root / "data" / "avatars" / "canonical"
        self.avatar_dir.mkdir(parents=True, exist_ok=True)
        
        # Load the canonical profile
        self.profile = self._load_canonical_profile()
        
        # === HAIRSTYLE OPTIONS (colour locked: platinum-blonde) ===
        self.hairstyle_options = {
            "loose_wavy": "platinum-blonde hair worn loose, soft waves",
            "loose_straight": "platinum-blonde hair worn straight and sleek",
            "ponytail": "platinum-blonde hair in high ponytail",
            "low_ponytail": "platinum-blonde hair in low ponytail",
            "messy_bun": "platinum-blonde hair in casual messy bun",
            "elegant_bun": "platinum-blonde hair in elegant updo bun",
            "twin_braids": "platinum-blonde hair in twin braids (Viking style)",
            "single_braid": "platinum-blonde hair in single side braid",
            "half_up": "platinum-blonde hair half up, half down",
            "curls": "platinum-blonde hair in soft curls",
            "beach_waves": "platinum-blonde hair in beach waves",
            "professional": "platinum-blonde hair in professional style",
            "fitness": "platinum-blonde hair in high ponytail or bun for fitness"
        }
        
        self.default_hairstyle = "loose_wavy"
        
        self._save_canonical_profile()
        print("✅ Alex Riviera Canonical Avatar Profile Loaded (SOURCE OF TRUTH)")
    
    @staticmethod
    def _resolve_profile_path() -> Path:
        """Resolve profile path against project root, not cwd (cwd varies on Render)."""
        # AvatarSystem.py lives at components/media/AvatarSystem.py
        # Project root is two directories up.
        project_root = Path(__file__).resolve().parents[2]
        return project_root / "data" / "avatars" / "canonical" / "alex_riviera_master_profile.json"

    def _load_canonical_profile(self) -> Dict:
        """Load the canonical master profile"""
        profile_file = self._resolve_profile_path()
        if profile_file.exists():
            with open(profile_file) as f:
                return json.load(f)
        else:
            raise FileNotFoundError(f"Canonical profile not found at {profile_file}.")

    def _save_canonical_profile(self):
        """Save any updates to the canonical profile"""
        profile_file = self._resolve_profile_path()
        profile_file.parent.mkdir(parents=True, exist_ok=True)
        with open(profile_file, 'w') as f:
            json.dump(self.profile, f, indent=2)
    
    def get_master_prompt(self, hairstyle: str = None, context: str = None, clothing: str = None, environment: str = None, pose: str = None) -> str:
        """Generate master prompt with optional hairstyle choice"""
        
        # Start with base description
        prompt = self.profile["master_prompt_ultra_short"]
        
        # Add hairstyle if specified
        if hairstyle and hairstyle in self.hairstyle_options:
            prompt = prompt.replace("platinum-blonde hair", f"platinum-blonde hair, {self.hairstyle_options[hairstyle]}")
        
        # Add context-specific details
        if context == "business_coach":
            prompt += " Professional business setting, leadership presence."
        elif context == "fitness_coach":
            prompt += " Fitness environment, motivational energy."
        elif context == "confidence_coach":
            prompt += " Classroom or coaching setting, warm encouraging atmosphere."
        elif context == "viking_theme":
            prompt += " Nordic Viking environment, warrior presence."
        
        if clothing:
            prompt += f" Wearing {clothing}."
        if environment:
            prompt += f" Environment: {environment}."
        if pose:
            prompt += f" Pose: {pose}."
        
        # Append identity lock (CRITICAL for consistency)
        prompt += self.profile["identity_lock_prompt"]
        
        return prompt
    
    def get_identity_lock_prompt(self) -> str:
        """Get the identity lock prompt to append to every generation"""
        return self.profile["identity_lock_prompt"]
    
    def get_ultra_short_prompt(self) -> str:
        """Get the ultra-short version for generators with prompt limits"""
        return self.profile["master_prompt_ultra_short"]
    
    def get_hairstyle_options(self) -> Dict:
        """Get available hairstyle options"""
        return self.hairstyle_options
    
    def get_example_prompts(self) -> Dict:
        """Get example prompts for different use cases"""
        return self.profile["example_use_cases"]
    
    def get_canonical_description(self) -> str:
        """Get the full canonical description"""
        return f"""
ALEX RIVIERA - CANONICAL AVATAR DESCRIPTION

Name: {self.profile['identity']['name']}
Age: {self.profile['identity']['age']}
Nationality: {self.profile['identity']['nationality']}

PHYSICAL APPEARANCE:
- Face: Heart-shaped, high cheekbones, soft jawline, straight nose
- Eyes: Ice-blue
- Skin: Healthy fair complexion, subtle natural warmth
- Hair: Platinum-blonde (styles flexible: {', '.join(list(self.hairstyle_options.keys())[:5])}...)
- Body: Athletic, healthy physique, excellent posture

PERSONALITY: {', '.join(self.profile['personality'])}

ROLES: {', '.join(self.profile['professional_roles'])}

CONSISTENCY: Facial features LOCKED. Hairstyle flexible (colour only: platinum-blonde).
"""


# Run the avatar system
if __name__ == "__main__":
    print("=" * 70)
    print("👤 ALEX RIVIERA - CANONICAL AVATAR SYSTEM")
    print("   SOURCE OF TRUTH - Facial features LOCKED")
    print("=" * 70)
    
    avatar = AlexRivieraAvatar()
    
    print("\n📋 CANONICAL PROFILE LOADED:")
    print(f"   Name: {avatar.profile['identity']['name']}")
    print(f"   Age: {avatar.profile['identity']['age']}")
    print(f"   Nationality: {avatar.profile['identity']['nationality']}")
    
    print("\n🎭 LOCKED FACIAL FEATURES:")
    for key, value in avatar.profile['physical_appearance']['face'].items():
        print(f"   {key}: {value}")
    
    print("\n💇 HAIRSTYLE OPTIONS (colour locked: platinum-blonde):")
    for style, desc in list(avatar.hairstyle_options.items())[:8]:
        print(f"   • {style}: {desc}")
    print(f"   ... and {len(avatar.hairstyle_options) - 8} more")
    
    print("\n🔒 IDENTITY LOCK PROMPT (append to every generation):")
    print(f"   {avatar.get_identity_lock_prompt()[:150]}...")
    
    print("\n📝 EXAMPLE USE CASES:")
    for case, prompt in avatar.get_example_prompts().items():
        print(f"   {case}: {prompt[:80]}...")
    
    print("\n" + "=" * 70)
    print("✅ CANONICAL AVATAR PROFILE ACTIVE")
    print("   - Facial features: LOCKED (never change)")
    print("   - Hair colour: LOCKED (platinum-blonde)")
    print("   - Hairstyle: FLEXIBLE (13 options available)")
    print("   - Identity Lock: Applied to every generation")
    print("=" * 70)
    
    # Save a quick reference file
    ref_file = Path("data/avatars/QUICK_REFERENCE_PROMPT.txt")
    with open(ref_file, 'w') as f:
        f.write("=== ALEX RIVIERA - MASTER PROMPT ===\n\n")
        f.write(avatar.get_ultra_short_prompt())
        f.write("\n\n=== IDENTITY LOCK (MUST APPEND) ===\n")
        f.write(avatar.get_identity_lock_prompt())
        f.write("\n\n=== HAIRSTYLE OPTIONS ===\n")
        for style, desc in avatar.hairstyle_options.items():
            f.write(f"{style}: {desc}\n")
    
    print(f"\n📄 Quick reference saved to: {ref_file}")
