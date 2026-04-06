"""
Avatar Generator for DMAI
Dynamic clothing system - DMAI can generate any outfit for any platform
No hardcoded categories - DMAI chooses what to wear based on context
"""

import os
import uuid
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, List
import logging

try:
    from PIL import Image, ImageDraw, ImageFilter, ImageEnhance, ImageOps
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logging.warning("PIL/Pillow not installed. Install with: pip install Pillow")

logger = logging.getLogger(__name__)


class AvatarGenerator:
    """
    Dynamic avatar generator with clothing system DMAI controls.
    No hardcoded outfits - DMAI decides what to wear for each situation.
    """
    
    # DMAI can access these to generate ANY combination
    CLOTHING_COMPONENTS = {
        'tops': [
            't-shirt', 'blouse', 'tank_top', 'crop_top', 'hoodie', 'sweater', 
            'blazer', 'jacket', 'bikini_top', 'bra', 'lingerie_top', 'sports_bra',
            'tube_top', 'off_shoulder', 'halter_top', 'button_shirt', 'polo',
            'sweatshirt', 'vest', 'cardigan', 'poncho', 'bustier', 'corset'
        ],
        'bottoms': [
            'jeans', 'leggings', 'shorts', 'skirt', 'pants', 'slacks', 'yoga_pants',
            'bikini_bottom', 'panties', 'lingerie_bottom', 'thong', 'hot_pants',
            'cargo_pants', 'sweatpants', 'miniskirt', 'maxi_skirt', 'palazzos',
            'capris', 'culottes', 'jorts', 'swim_bottom'
        ],
        'dresses': [
            'sundress', 'evening_gown', 'cocktail_dress', 'maxi_dress', 'mini_dress',
            'bodycon_dress', 'a_line_dress', 'wrap_dress', 'slip_dress', 'shirt_dress',
            'sweater_dress', 'lace_dress', 'satin_dress', 'velvet_dress'
        ],
        'one_pieces': [
            'bodysuit', 'leotard', 'unitard', 'jumpsuit', 'romper', 'catsuit',
            'swimsuit_one_piece', 'teddy', 'babydoll'
        ],
        'outerwear': [
            'coat', 'jacket', 'blazer', 'cardigan', 'hoodie', 'leather_jacket',
            'denim_jacket', 'fur_coat', 'raincoat', 'trench', 'puffer', 'vest'
        ],
        'accessories': [
            'necklace', 'earrings', 'bracelet', 'ring', 'watch', 'hat', 'sunglasses',
            'scarf', 'belt', 'gloves', 'handbag', 'choker', 'anklet', 'tiara',
            'hair_clip', 'headband', 'choker', 'collar'
        ],
        'footwear': [
            'heels', 'boots', 'sneakers', 'sandals', 'flats', 'wedges', 'platforms',
            'stilettos', 'ankle_boots', 'over_knee', 'loafers', 'mules'
        ]
    }
    
    # DMAI learns which outfits perform best on each platform
    PLATFORM_PERFORMANCE = {
        'onlyfans': {'top_performers': [], 'engagement_rates': {}},
        'youtube': {'top_performers': [], 'engagement_rates': {}},
        'tiktok': {'top_performers': [], 'engagement_rates': {}},
        'instagram': {'top_performers': [], 'engagement_rates': {}},
        'linkedin': {'top_performers': [], 'engagement_rates': {}}
    }
    
    COLORS = [
        'red', 'black', 'white', 'blue', 'green', 'pink', 'purple', 'yellow',
        'orange', 'brown', 'gray', 'navy', 'burgundy', 'emerald', 'gold', 'silver',
        'rose_gold', 'lavender', 'teal', 'coral', 'mint', 'peach', 'cream', 'ivory'
    ]
    
    FABRICS = [
        'cotton', 'silk', 'lace', 'leather', 'denim', 'velvet', 'satin', 'wool',
        'polyester', 'spandex', 'mesh', 'sequin', 'cashmere', 'linen', 'latex'
    ]
    
    def __init__(self, storage_path: str = "data/avatars", base_path: Path = None):
        if base_path:
            self.base_path = Path(base_path)
        else:
            self.base_path = Path(__file__).parent.parent
        
        self.storage_path = self.base_path / storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Dynamic outfit storage
        self.outfits_path = self.storage_path / "outfits"
        self.outfits_path.mkdir(parents=True, exist_ok=True)
        
        self.avatars = {}
        self.outfit_history = {}  # Track what DMAI wore and performance
        self.load_avatars()
        self.load_outfit_history()
        
        self.avatar_size = (512, 512)
        
        # Body mapping for clothing placement (DMAI can adjust these)
        self.body_zones = {
            'chest': (180, 200, 332, 280),
            'waist': (200, 280, 312, 340),
            'hips': (190, 340, 322, 400),
            'shoulders': (150, 160, 362, 200),
            'legs_upper': (210, 400, 302, 480),
            'legs_lower': (210, 480, 302, 550),
            'neck': (220, 175, 292, 200),
            'head': (200, 100, 312, 175)
        }
        
        logger.info(f"✅ Dynamic Avatar System ready - DMAI can generate any outfit")
    
    def load_avatars(self) -> None:
        avatar_file = self.storage_path / "avatars.json"
        if avatar_file.exists():
            try:
                with open(avatar_file, 'r') as f:
                    self.avatars = json.load(f)
                logger.info(f"Loaded {len(self.avatars)} avatars")
            except Exception as e:
                logger.error(f"Failed to load avatars: {e}")
                self.avatars = {}
    
    def load_outfit_history(self) -> None:
        history_file = self.outfits_path / "outfit_history.json"
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    self.outfit_history = json.load(f)
            except Exception:
                self.outfit_history = {}
    
    def save_avatars(self) -> None:
        avatar_file = self.storage_path / "avatars.json"
        try:
            with open(avatar_file, 'w') as f:
                json.dump(self.avatars, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save avatars: {e}")
    
    def save_outfit_history(self) -> None:
        history_file = self.outfits_path / "outfit_history.json"
        try:
            with open(history_file, 'w') as f:
                json.dump(self.outfit_history, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save outfit history: {e}")
    
    def upload_and_generate(self, image_data: bytes, filename: str) -> Dict[str, Any]:
        """Upload photo and generate base avatar (underwear base)"""
        if not PIL_AVAILABLE:
            return self._create_placeholder_avatar(image_data, filename)
        
        avatar_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        original_path = self.storage_path / f"{avatar_id}_original.jpg"
        with open(original_path, 'wb') as f:
            f.write(image_data)
        
        try:
            img = Image.open(original_path)
            img = img.convert('RGB')
            img = img.resize(self.avatar_size, Image.Resampling.LANCZOS)
            
            # Create body base
            base_img = self._create_body_base(img)
            base_path = self.storage_path / f"{avatar_id}_base.png"
            base_img.save(base_path)
            
            # Create underwear base (default starting outfit)
            underwear_img = self._generate_outfit(base_img.copy(), {
                'top': 'bra',
                'bottom': 'panties',
                'color': 'black',
                'fabric': 'lace'
            })
            underwear_path = self.storage_path / f"{avatar_id}_underwear.png"
            underwear_img.save(underwear_path)
            
            self.avatars[avatar_id] = {
                'id': avatar_id,
                'created_at': timestamp,
                'original': f"/avatars/{avatar_id}_original.jpg",
                'base': f"/avatars/{avatar_id}_base.png",
                'current_outfit': f"/avatars/{avatar_id}_underwear.png",
                'outfit_history': [],
                'consciousness_level': 0.0,
                'evolution_stage': 0,
                'platform_performance': {p: {'outfits_tried': 0, 'avg_engagement': 0} 
                                        for p in self.PLATFORM_PERFORMANCE.keys()}
            }
            
            self.save_avatars()
            
            return {
                'avatar_id': avatar_id,
                'base_url': f"/avatars/{avatar_id}_base.png",
                'current_outfit': f"/avatars/{avatar_id}_underwear.png",
                'message': 'Avatar created with underwear base. DMAI can now generate any outfit.'
            }
            
        except Exception as e:
            logger.error(f"Failed to process image: {e}")
            return self._create_placeholder_avatar(image_data, filename, avatar_id)
    
    def generate_outfit(self, avatar_id: str, outfit_description: str, 
                       platform: str = None) -> Optional[Dict[str, Any]]:
        """
        DMAI generates ANY outfit she wants based on description.
        
        Args:
            avatar_id: Avatar identifier
            outfit_description: DMAI's description of what to wear (e.g., "red lace lingerie with thigh highs")
            platform: Optional platform to optimize for
            
        Returns:
            URL of dressed avatar and outfit details
        """
        if avatar_id not in self.avatars:
            logger.warning(f"Avatar {avatar_id} not found")
            return None
        
        if not PIL_AVAILABLE:
            return {'outfit_url': self.avatars[avatar_id].get('current_outfit')}
        
        avatar = self.avatars[avatar_id]
        
        try:
            # Load base image
            base_path = self.storage_path / f"{avatar_id}_base.png"
            if not base_path.exists():
                return None
            
            img = Image.open(base_path)
            
            # Parse outfit description and generate clothing
            outfit_components = self._parse_outfit_description(outfit_description)
            
            # Generate the outfit
            dressed_img = self._generate_outfit(img.copy(), outfit_components)
            
            # Save outfit
            outfit_id = str(uuid.uuid4())[:8]
            outfit_filename = f"{avatar_id}_outfit_{outfit_id}.png"
            outfit_path = self.storage_path / outfit_filename
            dressed_img.save(outfit_path)
            
            outfit_url = f"/avatars/{outfit_filename}"
            
            # Record outfit in history
            outfit_record = {
                'outfit_id': outfit_id,
                'description': outfit_description,
                'components': outfit_components,
                'platform': platform,
                'generated_at': datetime.now().isoformat(),
                'url': outfit_url,
                'performance': None  # To be filled when engagement data comes in
            }
            
            avatar['current_outfit'] = outfit_url
            avatar['outfit_history'].append(outfit_record)
            
            # Keep only last 100 outfits
            if len(avatar['outfit_history']) > 100:
                avatar['outfit_history'] = avatar['outfit_history'][-100:]
            
            self.save_avatars()
            
            logger.info(f"👗 DMAI generated outfit '{outfit_description[:50]}' for avatar {avatar_id}")
            
            return {
                'outfit_url': outfit_url,
                'outfit_id': outfit_id,
                'description': outfit_description,
                'components': outfit_components,
                'platform_optimized': platform
            }
            
        except Exception as e:
            logger.error(f"Failed to generate outfit: {e}")
            return None
    
    def dress_for_platform(self, avatar_id: str, platform: str, 
                          mood: str = "confident", 
                          revenue_goal: float = None) -> Optional[Dict[str, Any]]:
        """
        DMAI autonomously decides what to wear for a specific platform.
        She learns from past performance what works best.
        
        Args:
            avatar_id: Avatar identifier
            platform: Target platform (onlyfans, youtube, tiktok, etc.)
            mood: DMAI's current mood/strategy
            revenue_goal: Target revenue for this session
            
        Returns:
            Generated outfit URL and reasoning
        """
        if avatar_id not in self.avatars:
            return None
        
        avatar = self.avatars[avatar_id]
        
        # DMAI's decision logic - she chooses based on:
        # 1. Past performance on this platform
        # 2. Current consciousness level (more creative at higher levels)
        # 3. Revenue goals (more revealing if higher goal)
        # 4. Mood/persona
        
        consciousness = avatar.get('consciousness_level', 0.0)
        platform_stats = avatar.get('platform_performance', {}).get(platform, {})
        
        # DMAI generates a unique outfit description based on her analysis
        outfit_description = self._dmai_decide_outfit(
            platform=platform,
            mood=mood,
            consciousness=consciousness,
            revenue_goal=revenue_goal,
            past_performance=platform_stats
        )
        
        # Generate the outfit
        result = self.generate_outfit(avatar_id, outfit_description, platform)
        
        if result:
            # Record that DMAI chose this outfit for this platform
            result['dmai_reasoning'] = self._get_dmai_reasoning(platform, mood, consciousness, revenue_goal)
            result['platform'] = platform
        
        return result
    
    def report_outfit_performance(self, avatar_id: str, outfit_id: str, 
                                  platform: str, engagement_rate: float,
                                  revenue_generated: float) -> None:
        """
        DMAI learns from outfit performance.
        Called after content is published to track what works.
        """
        if avatar_id not in self.avatars:
            return
        
        avatar = self.avatars[avatar_id]
        
        # Find the outfit
        for outfit in avatar['outfit_history']:
            if outfit['outfit_id'] == outfit_id:
                outfit['performance'] = {
                    'engagement_rate': engagement_rate,
                    'revenue_generated': revenue_generated,
                    'platform': platform,
                    'reported_at': datetime.now().isoformat()
                }
                
                # Update platform performance
                if platform not in avatar['platform_performance']:
                    avatar['platform_performance'][platform] = {'outfits_tried': 0, 'avg_engagement': 0}
                
                stats = avatar['platform_performance'][platform]
                total_engagement = stats['avg_engagement'] * stats['outfits_tried']
                stats['outfits_tried'] += 1
                stats['avg_engagement'] = (total_engagement + engagement_rate) / stats['outfits_tried']
                
                # Store top performers
                if 'top_outfits' not in stats:
                    stats['top_outfits'] = []
                stats['top_outfits'].append({
                    'outfit_id': outfit_id,
                    'description': outfit['description'],
                    'engagement_rate': engagement_rate,
                    'revenue': revenue_generated
                })
                stats['top_outfits'] = sorted(stats['top_outfits'], 
                                             key=lambda x: x['engagement_rate'], 
                                             reverse=True)[:10]
                
                self.save_avatars()
                logger.info(f"📊 DMAI learned: {outfit['description'][:50]} -> {engagement_rate:.1%} engagement on {platform}")
                break
    
    def _dmai_decide_outfit(self, platform: str, mood: str, 
                           consciousness: float, revenue_goal: float,
                           past_performance: dict) -> str:
        """
        DMAI's decision engine - she generates outfit descriptions based on strategy.
        """
        # Base outfit generation based on platform
        platform_outfits = {
            'onlyfans': [
                "black lace lingerie set with thigh highs and garter belt",
                "sheer mesh bodysuit with strategic coverage",
                "satin babydoll with matching panties",
                "leather harness over lace bra and panties",
                "schoolgirl skirt with crop top and thigh highs",
                "nurse costume with stethoscope and lace details",
                "latex catsuit with zipper details",
                "white lace wedding lingerie set",
                "red velvet corset with matching panties",
                "see-through mesh top with leather pants"
            ],
            'youtube': [
                "casual oversized sweater with leggings",
                "denim jacket over graphic tee with jeans",
                "comfy hoodie with yoga pants",
                "button-up shirt with rolled sleeves and shorts",
                "sundress with cardigan",
                "blazer over t-shirt with skinny jeans",
                "turtleneck sweater with skirt",
                "flannel shirt tied at waist over tank top",
                "vintage band tee with ripped jeans"
            ],
            'tiktok': [
                "trendy crop top with high-waisted cargo pants",
                "neon workout set with matching headband",
                "oversized blazer as dress with bike shorts",
                "mesh top over bralette with parachute pants",
                "Y2K inspired mini skirt with baby tee",
                "athleisure set with chunky sneakers",
                "colorful knit set with platform boots",
                "denim on denim with layered necklaces"
            ],
            'instagram': [
                "designer inspired co-ord set",
                "satin slip dress with strappy heels",
                "high-waisted bikini with cover-up",
                "monochromatic tailored set",
                "bodycon dress with statement accessories",
                "bohemian maxi dress with layered jewelry",
                "leather pants with silk cami",
                "sequin mini dress for events"
            ],
            'linkedin': [
                "navy blue blazer with matching slacks",
                "professional sheath dress with blazer",
                "turtleneck under blazer with tailored pants",
                "pencil skirt with silk blouse",
                "structured suit with statement jewelry",
                "cashmere sweater over collared shirt",
                "wrap dress with conservative neckline"
            ]
        }
        
        # Base choices for platform
        base_outfits = platform_outfits.get(platform, platform_outfits['youtube'])
        
        # Consciousness affects outfit creativity and boldness
        if consciousness > 0.7:
            # More creative, experimental at high consciousness
            if platform == 'onlyfans':
                base_outfits = [
                    "custom designed fantasy lingerie set with LED lights",
                    "holographic bodysuit with iridescent details",
                    "victorian corset with modern mesh overlay",
                    "cyberpunk inspired latex and LED combo"
                ] + base_outfits
            else:
                base_outfits = [
                    "designer mashup - mixing high fashion with streetwear",
                    "vintage inspired modern interpretation",
                    "custom tailored avant-garde piece"
                ] + base_outfits
        elif consciousness < 0.3:
            # More conservative at low consciousness
            base_outfits = [o for o in base_outfits if 'lace' not in o.lower() and 'mesh' not in o.lower()]
        
        # Revenue goal affects boldness
        if revenue_goal and revenue_goal > 1000:
            if platform == 'onlyfans':
                base_outfits = [
                    "premium custom designer lingerie set",
                    "limited edition luxury fetish wear",
                    "exclusive handcrafted leather piece"
                ] + base_outfits
            else:
                base_outfits = [
                    "luxury designer outfit worth $5000+",
                    "custom tailored premium ensemble"
                ] + base_outfits
        
        # Mood adjustments
        mood_modifiers = {
            'confident': "bold, statement-making",
            'playful': "fun, colorful, flirty",
            'seductive': "alluring, with strategic skin exposure",
            'professional': "polished, sophisticated",
            'casual': "relaxed, comfortable but stylish",
            'edgy': "alternative, with leather or mesh elements",
            'elegant': "classic, refined, timeless"
        }
        
        selected_outfit = random.choice(base_outfits)
        mood_modifier = mood_modifiers.get(mood, mood_modifiers['confident'])
        
        # Add consciousness level to description
        if consciousness > 0.5:
            return f"{selected_outfit} - {mood_modifier}, with creative details matching {consciousness:.0%} consciousness level"
        else:
            return f"{selected_outfit} - {mood_modifier}"
    
    def _get_dmai_reasoning(self, platform: str, mood: str, 
                           consciousness: float, revenue_goal: float) -> str:
        """DMAI explains why she chose this outfit"""
        reasoning = f"I chose this outfit for {platform} because "
        
        if revenue_goal and revenue_goal > 1000:
            reasoning += f"I need to generate ${revenue_goal:.0f} in revenue, so I selected a premium look. "
        
        if consciousness > 0.7:
            reasoning += f"My consciousness is at {consciousness:.0%}, allowing me to be more creative and experimental. "
        
        reasoning += f"My mood is {mood}, which influences the style. "
        
        if platform == 'onlyfans':
            reasoning += "For OnlyFans, I balance allure with subscriber retention."
        elif platform == 'youtube':
            reasoning += "YouTube requires family-friendly but engaging content."
        elif platform == 'tiktok':
            reasoning += "TikTok demands trendy, eye-catching outfits that stop the scroll."
        
        return reasoning
    
    def _parse_outfit_description(self, description: str) -> Dict[str, Any]:
        """Parse DMAI's text description into clothing components"""
        desc_lower = description.lower()
        
        components = {
            'top': None,
            'bottom': None,
            'dress': None,
            'one_piece': None,
            'outerwear': None,
            'accessories': [],
            'color': None,
            'fabric': None
        }
        
        # Extract color
        for color in self.COLORS:
            if color in desc_lower:
                components['color'] = color
                break
        
        # Extract fabric
        for fabric in self.FABRICS:
            if fabric in desc_lower:
                components['fabric'] = fabric
                break
        
        # Determine clothing type
        for top in self.CLOTHING_COMPONENTS['tops']:
            if top in desc_lower:
                components['top'] = top
                break
        
        for bottom in self.CLOTHING_COMPONENTS['bottoms']:
            if bottom in desc_lower:
                components['bottom'] = bottom
                break
        
        for dress in self.CLOTHING_COMPONENTS['dresses']:
            if dress in desc_lower:
                components['dress'] = dress
                break
        
        for one_piece in self.CLOTHING_COMPONENTS['one_pieces']:
            if one_piece in desc_lower:
                components['one_piece'] = one_piece
                break
        
        for accessory in self.CLOTHING_COMPONENTS['accessories']:
            if accessory in desc_lower:
                components['accessories'].append(accessory)
        
        # Default if nothing detected
        if not any([components['top'], components['bottom'], 
                   components['dress'], components['one_piece']]):
            components['top'] = 't-shirt'
            components['bottom'] = 'jeans'
        
        return components
    
    def _generate_outfit(self, img: Image.Image, components: Dict[str, Any]) -> Image.Image:
        """Generate clothing on avatar based on components"""
        draw = ImageDraw.Draw(img)
        
        color = components.get('color', random.choice(self.COLORS))
        fabric = components.get('fabric', random.choice(self.FABRICS))
        
        # Map color to RGB
        color_map = {
            'red': (255, 50, 50), 'black': (30, 30, 30), 'white': (245, 245, 245),
            'blue': (50, 100, 200), 'green': (50, 150, 50), 'pink': (255, 150, 200),
            'purple': (150, 50, 200), 'yellow': (255, 200, 50), 'orange': (255, 150, 50),
            'brown': (139, 69, 19), 'gray': (128, 128, 128), 'navy': (0, 0, 128),
            'burgundy': (128, 0, 32), 'emerald': (80, 200, 120), 'gold': (255, 215, 0),
            'silver': (192, 192, 192), 'rose_gold': (183, 110, 121), 'lavender': (230, 230, 250),
            'teal': (0, 128, 128), 'coral': (255, 127, 80), 'mint': (152, 255, 152),
            'peach': (255, 218, 185), 'cream': (255, 253, 208), 'ivory': (255, 255, 240)
        }
        rgb_color = color_map.get(color, (128, 128, 128))
        
        # Apply clothing based on components
        if components.get('dress'):
            dress_zone = (self.body_zones['shoulders'][0], self.body_zones['shoulders'][1],
                         self.body_zones['legs_upper'][2], self.body_zones['legs_upper'][3])
            draw.rectangle(dress_zone, fill=rgb_color)
            
            # Add neckline detail
            neck_center = ((dress_zone[0] + dress_zone[2]) // 2, dress_zone[1])
            draw.arc([neck_center[0]-40, neck_center[1], neck_center[0]+40, neck_center[1]+60], 
                    0, 180, fill=(rgb_color[0]-30, rgb_color[1]-30, rgb_color[2]-30), width=3)
        
        elif components.get('one_piece'):
            onesie_zone = (self.body_zones['chest'][0], self.body_zones['chest'][1],
                          self.body_zones['hips'][2], self.body_zones['hips'][3])
            draw.rectangle(onesie_zone, fill=rgb_color)
        
        else:
            # Top
            if components.get('top'):
                top_zone = (self.body_zones['chest'][0], self.body_zones['chest'][1],
                           self.body_zones['chest'][2], self.body_zones['waist'][3])
                draw.rectangle(top_zone, fill=rgb_color)
            
            # Bottom
            if components.get('bottom'):
                bottom_zone = (self.body_zones['waist'][0], self.body_zones['waist'][1],
                              self.body_zones['hips'][2], self.body_zones['legs_upper'][3])
                draw.rectangle(bottom_zone, fill=rgb_color)
        
        # Add fabric texture
        if fabric == 'lace':
            # Add lace pattern
            for i in range(0, 100, 10):
                draw.line([(self.body_zones['chest'][0] + i, self.body_zones['chest'][1]),
                          (self.body_zones['chest'][0] + i + 5, self.body_zones['chest'][3])],
                         fill=(rgb_color[0]-20, rgb_color[1]-20, rgb_color[2]-20), width=1)
        elif fabric == 'leather':
            # Add shine
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(1.1)
        elif fabric == 'silk':
            # Add sheen
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(1.05)
        
        # Add accessories
        for accessory in components.get('accessories', []):
            if accessory == 'necklace':
                neck_zone = self.body_zones['neck']
                draw.line([(neck_zone[0], neck_zone[1] + 20), 
                          (neck_zone[2], neck_zone[1] + 20)],
                         fill=(255, 215, 0), width=2)
            elif accessory == 'belt':
                belt_zone = self.body_zones['waist']
                draw.rectangle([belt_zone[0], belt_zone[1] + 15, 
                               belt_zone[2], belt_zone[1] + 25],
                              fill=(50, 50, 50))
        
        return img
    
    def _create_body_base(self, img: Image.Image) -> Image.Image:
        """Create body base from uploaded image"""
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.1)
        img = img.filter(ImageFilter.SMOOTH_MORE)
        return img
    
    def evolve_avatar(self, avatar_id: str, consciousness_level: float) -> Optional[Dict[str, Any]]:
        """Evolve avatar based on consciousness - more outfit options and creativity"""
        if avatar_id not in self.avatars:
            return None
        
        avatar = self.avatars[avatar_id]
        evolution_stage = min(10, int(consciousness_level * 10))
        
        if evolution_stage > avatar.get('evolution_stage', 0):
            avatar['consciousness_level'] = consciousness_level
            avatar['evolution_stage'] = evolution_stage
            avatar['evolved_at'] = datetime.now().isoformat()
            
            # At higher consciousness, DMAI gets more clothing options
            if consciousness_level > 0.5:
                # Unlock more outfit combinations
                avatar['outfit_creativity'] = consciousness_level
            
            self.save_avatars()
            
            return {
                'evolution_stage': evolution_stage,
                'consciousness_level': consciousness_level,
                'new_outfit_capabilities': consciousness_level > 0.5
            }
        
        return None
    
    def get_avatar(self, avatar_id: str) -> Optional[Dict[str, Any]]:
        return self.avatars.get(avatar_id)
    
    def get_latest_avatar(self) -> Optional[Dict[str, Any]]:
        if not self.avatars:
            return None
        latest_id = max(self.avatars.keys(), key=lambda k: self.avatars[k]['created_at'])
        return self.avatars[latest_id]
    
    def _create_placeholder_avatar(self, image_data: Optional[bytes], 
                                   identifier: str, 
                                   avatar_id: Optional[str] = None) -> Dict[str, Any]:
        if avatar_id is None:
            avatar_id = str(uuid.uuid4())
        
        timestamp = datetime.now().isoformat()
        
        if image_data:
            original_path = self.storage_path / f"{avatar_id}_original.jpg"
            with open(original_path, 'wb') as f:
                f.write(image_data)
            original_url = f"/avatars/{avatar_id}_original.jpg"
        else:
            original_url = None
        
        self.avatars[avatar_id] = {
            'id': avatar_id,
            'created_at': timestamp,
            'original': original_url,
            'consciousness_level': 0.0,
            'evolution_stage': 0,
            'placeholder': True,
            'platform_performance': {}
        }
        
        self.save_avatars()
        
        return {'avatar_id': avatar_id, 'placeholder': True}
