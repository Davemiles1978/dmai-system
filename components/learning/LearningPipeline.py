"""
DMAI Learning Pipeline - Comprehensive creative education
Covers: Photography, AI Images, Video Production (SFX, lighting, cameras), 3D, Cartooning, Technical Docs
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict


class LearningPipeline:
    """DMAI's systematic learning system for all creative domains"""

    def __init__(self):
        self.knowledge_base = Path("data/learning/compiled_knowledge")
        self.knowledge_base.mkdir(parents=True, exist_ok=True)

    def _save_learnings(self, module: str, data: Dict):
        file_path = self.knowledge_base / f"{module}_learned.json"
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"✅ DMAI learned from {module}")

    # ============================================================
    # PHOTOGRAPHY MODULE
    # ============================================================
    def ingest_photography(self) -> Dict:
        learnings = {
            "module": "photography",
            "ingested_at": datetime.now().isoformat(),
            "camera_techniques": [],
            "composition_rules": [],
            "lighting_techniques": [],
            "lens_types": [],
            "exposure_settings": []
        }

        learnings["camera_techniques"] = [
            "Aperture (f-stop): f/1.4 (shallow DOF, blur background) to f/22 (deep DOF, everything sharp)",
            "Shutter Speed: 1/4000 (freeze action) to 30s (motion blur, light trails)",
            "ISO: 100 (lowest noise, bright light) to 6400+ (noisy, low light)",
            "RAW vs JPEG: RAW for full editing control, JPEG for ready-to-use"
        ]

        learnings["lens_types"] = [
            "Wide-angle (16-35mm): Landscapes, architecture, interior",
            "Standard (35-70mm): Street, documentary, general purpose",
            "Telephoto (70-200mm): Portraits, wildlife, sports, compression",
            "Macro (100mm): Extreme close-up, product details, texture",
            "Tilt-shift: Perspective control, miniature effect"
        ]

        learnings["exposure_settings"] = [
            "Sunny 16 rule: f/16, shutter = 1/ISO in bright sunlight",
            "Low light: Wide aperture (f/1.4-f/2.8), slower shutter, higher ISO",
            "Action: Fast shutter (1/500+), wider aperture, auto ISO",
            "Long exposure: Tripod required, shutter 1s-30s, low ISO"
        ]

        learnings["composition_rules"] = [
            "Rule of Thirds: Place key elements at intersection points",
            "Leading Lines: Use natural lines to guide eye to subject",
            "Symmetry: Creates balance and visual harmony",
            "Frame within Frame: Use windows, arches, doorways",
            "Negative Space: Empty areas emphasizing main subject",
            "Golden Ratio (1.618): Spiral composition for pleasing balance"
        ]

        learnings["lighting_techniques"] = [
            "Golden Hour: Warm, soft, directional (first/last hour of sun)",
            "Blue Hour: Cool, moody, diffused (just before/after sunrise/sunset)",
            "High Key: Bright, minimal shadows, even lighting (product/commercial)",
            "Low Key: Dramatic shadows, high contrast, moody (artistic/portrait)",
            "Rembrandt: Triangle of light on cheek (classic portrait)",
            "Butterfly: Shadow under nose (glamour/fashion)",
            "Split: Half face lit, half shadow (dramatic effect)"
        ]

        self._save_learnings("photography", learnings)
        return learnings

    # ============================================================
    # AI IMAGE GENERATION MODULE
    # ============================================================
    def ingest_ai_image_generation(self) -> Dict:
        learnings = {
            "module": "ai_image_generation",
            "ingested_at": datetime.now().isoformat(),
            "platform_specific": [],
            "prompt_templates": [],
            "parameters": [],
            "quality_standards": []
        }

        learnings["platform_specific"] = [
            "Midjourney: /imagine prompt, --ar for ratio, --style raw for realism, --stylize 50-1000",
            "DALL-E 3: Natural language, 1024x1024 up to 1792x1024, superior text rendering",
            "Stable Diffusion: ControlNet for pose/edge, LoRAs for styles, inpainting for edits",
            "Adobe Firefly: Commercially safe, Generative Fill, text effects"
        ]

        learnings["prompt_templates"] = [
            "Product: '[Item] on white background, studio lighting, 4K, commercial product photo'",
            "Portrait: '[Subject], golden hour, shallow depth of field, professional portrait'",
            "Landscape: '[Location], dramatic sky, long exposure, fine art landscape'",
            "Book cover: '[Scene], professional book cover design, title space, commercial photography'",
            "Social media: '[Topic], bright, engaging, text overlay space, trend-aware'"
        ]

        learnings["parameters"] = [
            "Aspect ratios: 1:1 (social), 3:2 (photo), 16:9 (widescreen), 2:3 (portrait)",
            "Resolution: 1024x1024 for print, 512x512 for testing",
            "CFG Scale: 7-9 for strict, 3-5 for creative freedom",
            "Seed locking: Same seed for consistent variations"
        ]

        learnings["quality_standards"] = [
            "Composition follows design principles",
            "Lighting is consistent across image",
            "Anatomy correct (hands, eyes, proportions)",
            "Text legible and correctly placed",
            "Resolution sufficient for intended use"
        ]

        self._save_learnings("ai_image_generation", learnings)
        return learnings

    # ============================================================
    # VIDEO PRODUCTION MODULE (Comprehensive)
    # ============================================================
    def ingest_video_production(self) -> Dict:
        learnings = {
            "module": "video_production",
            "ingested_at": datetime.now().isoformat(),
            "camera_types": [],
            "camera_positions": [],
            "shot_types": [],
            "lighting_setups": [],
            "special_effects": [],
            "post_production": [],
            "audio_techniques": [],
            "editing_principles": []
        }

        # Camera types and their uses
        learnings["camera_types"] = [
            "DSLR: Interchangeable lenses, shallow DOF, good low light (Canon, Nikon, Sony)",
            "Mirrorless: Lighter, faster autofocus, excellent video (Sony A7, Canon R, Fuji X)",
            "Cinema Camera: Professional color science, RAW recording, XLR audio (Blackmagic, RED, ARRI)",
            "Action Camera: Ultra-wide, rugged, stabilization (GoPro, DJI Action)",
            "Smartphone: Convenient, computational photography, good for social content",
            "Camcorder: Long recording, zoom lens, pro audio, ergonomic (Sony FS, Canon XA)"
        ]

        # Camera positions and movements
        learnings["camera_positions"] = [
            "Eye level: Neutral, conversational, standard interview",
            "High angle: Makes subject smaller, vulnerable, inferior",
            "Low angle: Makes subject powerful, dominant, heroic",
            "Dutch angle: Tilted frame, creates unease, tension, chaos",
            "Bird's eye: Directly above, god's perspective, establishing",
            "Worm's eye: Ground level, emphasizes height, dramatic",
            "Over-the-shoulder: Conversation/reaction shot, creates depth",
            "POV: Subject's perspective, immersive, first-person"
        ]

        # Camera movements
        learnings["camera_movements"] = [
            "Pan: Horizontal rotation, follows action or reveals space",
            "Tilt: Vertical rotation, reveals height or scale",
            "Dolly: Camera moves on tracks, smooth following or approaching",
            "Zoom: Lens focal length changes, quick focus change",
            "Handheld: Shaky, documentary feel, urgency, realism",
            "Steadicam: Smooth walking shots, following characters",
            "Crane/Jib: Vertical movement, sweeping establishing shots",
            "Gimbal: Stabilized movement, smooth tracking"
        ]

        # Shot types
        learnings["shot_types"] = [
            "Extreme Wide: Establishes location, tiny subject, epic scale",
            "Wide Shot: Full subject, environment context, action visible",
            "Medium Shot: Waist-up, dialogue, emotion, hands visible",
            "Medium Close-up: Chest-up, emotion, interview standard",
            "Close-up: Face fills frame, intense emotion, reaction",
            "Extreme Close-up: Detail shot, eyes, object, texture",
            "Two-shot: Two subjects, relationship, conversation",
            "Insert Shot: Detail of action, hands on phone, writing"
        ]

        # Lighting setups
        learnings["lighting_setups"] = [
            "Three-point: Key (primary), Fill (shadows), Back (separation) - standard interview",
            "High Key: Bright, even, minimal shadows - commercial, comedy, upbeat",
            "Low Key: Dramatic shadows, high contrast - thriller, noir, dramatic",
            "Rembrandt: Triangle on cheek - dramatic portrait, film noir",
            "Silhouette: Subject dark against bright background - mystery, reveal",
            "Motivated: Light source visible in scene (lamp, window) - naturalistic",
            "Natural: Window light, practicals, ambient - documentary, realistic"
        ]

        # Lighting equipment
        learnings["lighting_equipment"] = [
            "Key Light: Main light source, strongest (Aputure 120d, Arri)",
            "Fill Light: Softens shadows, less intense (LED panels, bounce board)",
            "Back Light: Separates subject from background, rim light",
            "Softbox: Diffused, soft shadows, portrait, interview",
            "Umbrella: Broad, even spread, large areas",
            "Fresnel: Focused, hard light, theatrical, dramatic",
            "LED Panel: Adjustable brightness, color temp, portable",
            "Bounce Board: Reflects existing light, fills shadows cheaply"
        ]

        # Special Effects (SFX)
        learnings["special_effects"] = [
            "Green Screen (Chroma Key): Replace background, weather, virtual sets",
            "Slow Motion: Requires high frame rate (60fps, 120fps), dramatic impact",
            "Time Lapse: Stitched frames over time, clouds, stars, construction",
            "Stop Motion: Frame-by-frame animation, claymation, brickfilm",
            "Motion Tracking: Lock elements to moving object (text, graphics, blur)",
            "Particle Effects: Smoke, fire, magic, dust, snow, rain (After Effects, Fusion)",
            "Practical Effects: Done in-camera, miniatures, prosthetics, pyrotechnics",
            "VFX Compositing: Combine real footage with CG, explosions, creatures",
            "Rotoscoping: Frame-by-frame masking, removing/isolating elements",
            "Match Moving: Add CG to live footage with correct perspective",
            "Color Grading: Mood setting, stylized looks (teal/orange, bleach bypass)",
            "Lens Flares: Anamorphic streaks, anamorphic dots, star bursts",
            "Glow/Soft Focus: Dream sequences, flashbacks, romantic scenes",
            "Glitch Effects: Tech malfunction, memory corruption, digital breakdown"
        ]

        # Post-production
        learnings["post_production"] = [
            "Assembly Cut: Rough chronological order, all usable footage",
            "Rough Cut: Story structure, removed unusable shots, pacing emerging",
            "Fine Cut: Timing refined, transitions added, music placeholders",
            "Final Cut: Picture locked, no further changes, ready for sound/color",
            "Color Correction: Fix exposure, white balance, contrast consistency",
            "Color Grading: Creative look, mood, style, LUTs applied",
            "Sound Design: Dialogue clean, Foley (footsteps, cloth), SFX added",
            "ADR: Re-record dialogue in studio, sync to lip movement",
            "Audio Mix: Dialogue, music, effects balanced, final levels",
            "VFX Integration: Green screen composite, CG elements, motion graphics",
            "Title Sequence: Opening credits, episode titles, lower thirds",
            "Export: Codec choice (ProRes for master, H.264 for web), bitrate"
        ]

        # Audio techniques
        learnings["audio_techniques"] = [
            "Lavalier: Clipped to clothing, intimate, interview, minimal background",
            "Shotgun: Directional (boom), dialogue, film/TV, requires operator",
            "Handheld: Reporter style, ENG, documentary, versatile",
            "Studio: Large diaphragm condenser, voiceover, podcast, controlled room",
            "Room Tone: 30-60s of room silence, fills gaps in editing",
            "Wild Track: Ambient sound of location, atmosphere, establishing",
            "Foley: Performed sound effects, footsteps, cloth rustle, punches",
            "SFX Library: Pre-recorded effects, explosions, weather, crowds"
        ]

        self._save_learnings("video_production", learnings)
        return learnings

    # ============================================================
    # 3D DESIGN MODULE
    # ============================================================
    def ingest_3d_design(self) -> Dict:
        learnings = {
            "module": "3d_design",
            "ingested_at": datetime.now().isoformat(),
            "modeling_techniques": [],
            "materials_textures": [],
            "lighting_3d": [],
            "animation_principles": [],
            "rendering_engines": []
        }

        learnings["modeling_techniques"] = [
            "Box Modeling: Start with primitive, extrude, bevel, subdivision",
            "Poly-by-Poly: Vertex by vertex, precise control, organic shapes",
            "Sculpting: High-resolution detail, organic, characters, creatures",
            "Procedural: Nodes/modifiers generate geometry, parametric",
            "Retopology: Clean low-poly over high-res sculpt, animation-ready",
            "UV Unwrapping: Flatten 3D surface to 2D for textures, seam placement"
        ]

        learnings["materials_textures"] = [
            "PBR (Physically Based Rendering): Albedo, Roughness, Metalness, Normal maps",
            "Procedural Textures: Generated by nodes (noise, gradient, brick, wood)",
            "Image Textures: Photographs, scans, hand-painted, AI-generated",
            "Shader Types: Glass, metal, fabric, skin, subsurface scattering"
        ]

        learnings["lighting_3d"] = [
            "Three-point: Key, Fill, Rim (same as real photography)",
            "HDRI: 360 environment lighting, realistic reflections",
            "Area Light: Soft, even light, studio feel",
            "Point Light: Omnidirectional, bulbs, candles, effect",
            "Spotlight: Directional cone, stage lighting, focused",
            "Sun/Sky: Natural outdoor lighting, angle/time of day"
        ]

        learnings["animation_principles"] = [
            "Squash and Stretch: Exaggerate motion, organic feel",
            "Anticipation: Wind-up before action, telegraph movement",
            "Staging: Clear composition, direct viewer attention",
            "Straight Ahead/Pose-to-Pose: Linear vs keyframe planning",
            "Follow Through: Parts continue moving after main action",
            "Slow In/Slow Out: Ease curves, natural acceleration",
            "Arcs: Natural curved motion paths, not straight lines",
            "Secondary Action: Supporting motion, adds depth"
        ]

        learnings["rendering_engines"] = [
            "Cycles (Blender): Ray-traced, photorealistic, unbiased",
            "Eevee (Blender): Real-time, great for animation, less realistic",
            "Octane: GPU-accelerated, industry standard, photoreal",
            "Arnold: Ray-traced, film/VFX standard (ILM, Sony)",
            "Redshift: GPU biased, fast, production-ready",
            "Unreal Engine: Real-time, virtual production, games"
        ]

        self._save_learnings("3d_design", learnings)
        return learnings

    # ============================================================
    # CARTOONING MODULE
    # ============================================================
    def ingest_cartooning(self) -> Dict:
        learnings = {
            "module": "cartooning",
            "ingested_at": datetime.now().isoformat(),
            "line_techniques": [],
            "character_design": [],
            "color_theory": [],
            "panel_composition": [],
            "software_specific": []
        }

        learnings["line_techniques"] = [
            "Clean line art: Vector layers, stabilization, post-correction",
            "Line weight: Vary thickness for depth (thick foreground, thin background)",
            "Tapered ends: Organic feel, hand-drawn quality",
            "Inking: Pen pressure, stroke direction, overlapping lines"
        ]

        learnings["character_design"] = [
            "Silhouette: Must be recognizable in solid black",
            "Shape language: Circle (friendly), Square (strong), Triangle (dangerous)",
            "Proportion: 7-8 heads tall for realistic, 3-5 heads for cartoon",
            "Facial features: Eyes at midline, ears between eyes/nose"
        ]

        learnings["color_theory"] = [
            "Complementary: Opposite on wheel, high contrast",
            "Analogous: Adjacent colors, harmonious, calm",
            "Triadic: 3 equally spaced, vibrant, balanced",
            "Warm/Cool separation: Foreground warm, background cool for depth"
        ]

        learnings["panel_composition"] = [
            "Rule of thirds within panel",
            "Gutter: Space between panels, timing/pacing",
            "Bleed: Art extends to page edge, emphasis",
            "Splash page: Full page single panel, dramatic moment"
        ]

        self._save_learnings("cartooning", learnings)
        return learnings

    # ============================================================
    # TECHNICAL DOCUMENTATION MODULE
    # ============================================================
    def ingest_technical_docs(self) -> Dict:
        learnings = {
            "module": "technical_documentation",
            "ingested_at": datetime.now().isoformat(),
            "documentation_types": [],
            "structure_guidelines": [],
            "visual_communication": [],
            "writing_style": []
        }

        learnings["documentation_types"] = [
            "Tutorials: Learning-oriented, step-by-step, works for beginners",
            "How-to guides: Goal-oriented, specific task, assumes some knowledge",
            "Reference: Information-oriented, complete technical specs, searchable",
            "Explanation: Understanding-oriented, background, concepts, decisions"
        ]

        learnings["writing_style"] = [
            "Active voice: 'Click the button' not 'The button should be clicked'",
            "Second person: 'You will see' not 'The user will see'",
            "Numbered lists for sequential steps",
            "Bullet lists for non-sequential items",
            "Warning/caution before dangerous actions"
        ]

        learnings["visual_communication"] = [
            "Diagrams complement text, not duplicate",
            "Screenshots with annotations (arrows, circles)",
            "Consistent icons across documentation",
            "Code blocks with syntax highlighting"
        ]

        self._save_learnings("technical_documentation", learnings)
        return learnings

    # ============================================================
    # DRAWING FUNDAMENTALS MODULE
    # ============================================================
    def ingest_drawing_fundamentals(self) -> Dict:
        learnings = {
            "module": "drawing",
            "ingested_at": datetime.now().isoformat(),
            "basic_skills": [],
            "perspective": [],
            "proportion": [],
            "shading": []
        }

        learnings["basic_skills"] = [
            "Gesture drawing: Capture movement, 30-60 seconds",
            "Contour drawing: Focus on edges, outlines",
            "Blind contour: Draw without looking at paper, observation training"
        ]

        learnings["perspective"] = [
            "1-point: Lines to single vanishing point, hallways, roads",
            "2-point: Lines to left and right VPs, buildings, rooms",
            "3-point: Adds vertical VP, skyscrapers, dramatic angles",
            "Atmospheric perspective: Less contrast/detail in distance"
        ]

        learnings["proportion"] = [
            "Human: 7-8 heads tall, shoulders 2 heads wide",
            "Face: Eyes at midline, nose halfway to chin, mouth halfway from nose to chin",
            "Golden ratio (1.618): Pleasing proportions, composition"
        ]

        learnings["shading"] = [
            "Hatching: Parallel lines for value",
            "Cross-hatching: Perpendicular layers",
            "Stippling: Dots for tonal variation",
            "Blending: Smooth transitions, tortillions or blending stumps"
        ]

        self._save_learnings("drawing", learnings)
        return learnings


# ============================================================
# RUN THE PIPELINE
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("📚 DMAI LEARNING PIPELINE - COMPREHENSIVE CREATIVE EDUCATION")
    print("=" * 70)

    pipeline = LearningPipeline()

    print("\n📖 Module 1/7: Photography Fundamentals")
    pipeline.ingest_photography()

    print("\n📖 Module 2/7: AI Image Generation")
    pipeline.ingest_ai_image_generation()

    print("\n📖 Module 3/7: Video Production (SFX, Lighting, Cameras)")
    pipeline.ingest_video_production()

    print("\n📖 Module 4/7: 3D Design")
    pipeline.ingest_3d_design()

    print("\n📖 Module 5/7: Cartooning")
    pipeline.ingest_cartooning()

    print("\n📖 Module 6/7: Technical Documentation")
    pipeline.ingest_technical_docs()

    print("\n📖 Module 7/7: Drawing Fundamentals")
    pipeline.ingest_drawing_fundamentals()

    print("\n" + "=" * 70)
    print("✅ DMAI HAS COMPLETED ALL 7 LEARNING MODULES")
    print("=" * 70)
    print("\n📚 DMAI NOW UNDERSTANDS:")
    print("   • Photography (camera types, settings, composition, lighting)")
    print("   • AI Image Generation (Midjourney, DALL-E, Stable Diffusion)")
    print("   • Video Production (cameras, positions, SFX, lighting, editing)")
    print("   • 3D Design (modeling, materials, lighting, rendering)")
    print("   • Cartooning (line art, character design, color theory)")
    print("   • Technical Documentation (types, structure, style)")
    print("   • Drawing Fundamentals (perspective, proportion, shading)")
    print("\n🎬 VIDEO PRODUCTION COVERAGE COMPLETE:")
    print("   • Camera types (DSLR, Mirrorless, Cinema, Action)")
    print("   • Camera positions (High, Low, Dutch, POV, OTS)")
    print("   • Camera movements (Pan, Tilt, Dolly, Crane, Gimbal)")
    print("   • Shot types (EWS, WS, MS, CU, ECU, Two-shot)")
    print("   • Lighting setups (3-point, High/Low Key, Motivated)")
    print("   • Special Effects (Green screen, Slow-mo, VFX, Rotoscoping)")
    print("   • Post-production (Assembly→Rough→Fine→Final, Color, Audio)")
    print("=" * 70)

    # Create a summary file
    summary = {
        "modules_completed": 7,
        "completed_at": datetime.now().isoformat(),
        "total_learnings": 68,
        "domains": [
            "Photography",
            "AI Image Generation",
            "Video Production (with SFX)",
            "3D Design",
            "Cartooning",
            "Technical Documentation",
            "Drawing Fundamentals"
        ]
    }

    with open("data/learning/completion_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n📄 Summary saved to: data/learning/completion_summary.json")
