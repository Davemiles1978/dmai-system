"""
graph_wallpaper.py — renders the DMAI knowledge graph as:
  - A PNG at iPhone 15 Pro resolution (1179×2556) for use as a wallpaper
  - A compact SVG for use as a Widgetsmith home-screen widget

Both are generated entirely server-side using only Pillow (already in requirements).
No browser, no headless Chrome, no extra dependencies.

Flask routes (registered in dmai_core_complete.py):
  GET /wallpaper          → PNG, iPhone resolution
  GET /wallpaper?size=mini → PNG, 400×400 (quick preview)
  GET /graph-widget       → SVG, 155×155 (Widgetsmith small widget)
  GET /graph-widget?size=medium → SVG, 329×155 (medium widget)
"""

import json
import math
import os
import io
import time
import hashlib
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("GraphWallpaper")

# ── Config ────────────────────────────────────────────────────────────────────
GRAPH_SCHEMA_PATH = os.getenv(
    "GRAPH_SCHEMA_PATH",
    "aevora-training/dashboard/data/graph_schema.json"
)

CLUSTER_COLORS = {
    "core":      "#6c63ff",
    "learning":  "#00d4aa",
    "research":  "#ffa502",
    "knowledge": "#a29bfe",
    "providers": "#74b9ff",
    "revenue":   "#ff4757",
}

BG_DARK  = (10, 10, 15)
BG_LIGHT = (244, 244, 248)

# Cache rendered images for 5 minutes to avoid re-rendering on every request
_cache: dict = {}
_CACHE_TTL = 300  # seconds


def _hex_to_rgb(h: str) -> tuple:
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def _load_schema() -> dict:
    p = Path(GRAPH_SCHEMA_PATH)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception as e:
            logger.warning("Could not load graph schema: %s", e)
    return {"neurons": [], "synapses": [], "clusters": {}, "evolution_cycle": 0}


def _layout_neurons(neurons: list, width: int, height: int, cx: int, cy: int) -> dict:
    """
    Place neurons in concentric rings by cluster.
    Returns {neuron_id: (x, y)} positions.
    """
    cluster_order = ["core", "learning", "research", "knowledge", "providers", "revenue"]
    by_cluster: dict = {}
    for n in neurons:
        c = n.get("cluster", "core")
        by_cluster.setdefault(c, []).append(n)

    positions = {}
    # Core cluster at centre, others in rings
    radii = [0, width * 0.18, width * 0.33, width * 0.44]
    ring_clusters = [["core"], ["learning", "research"], ["knowledge", "providers"], ["revenue"]]

    for ring_idx, cluster_list in enumerate(ring_clusters):
        r = radii[ring_idx]
        all_in_ring = []
        for cl in cluster_list:
            all_in_ring.extend(by_cluster.get(cl, []))
        n_total = len(all_in_ring)
        if n_total == 0:
            continue
        for i, neuron in enumerate(all_in_ring):
            if r == 0:
                positions[neuron["id"]] = (cx, cy)
            else:
                angle = (2 * math.pi * i / n_total) - math.pi / 2
                # Small angular offset per cluster for visual separation
                cl_offset = cluster_list.index(neuron.get("cluster", cluster_list[0])) * 0.3
                angle += cl_offset
                x = cx + r * math.cos(angle)
                y = cy + r * math.sin(angle)
                positions[neuron["id"]] = (x, y)

    # Any neurons not in known clusters get placed in outer ring
    placed = set(positions.keys())
    extras = [n for n in neurons if n["id"] not in placed]
    for i, neuron in enumerate(extras):
        angle = 2 * math.pi * i / max(len(extras), 1)
        r = width * 0.47
        positions[neuron["id"]] = (cx + r * math.cos(angle), cy + r * math.sin(angle))

    return positions


# ── PNG wallpaper ─────────────────────────────────────────────────────────────

def render_wallpaper_png(
    width: int = 1179,
    height: int = 2556,
    dark: bool = True,
    quality: int = 85,
) -> bytes:
    """Render the knowledge graph as a PNG at the given resolution."""
    cache_key = f"png_{width}_{height}_{dark}"
    cached = _cache.get(cache_key)
    if cached and time.time() - cached["ts"] < _CACHE_TTL:
        return cached["data"]

    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        logger.error("Pillow not installed")
        return b""

    schema = _load_schema()
    neurons     = schema.get("neurons", [])
    synapses    = schema.get("synapses", [])
    cycle       = schema.get("evolution_cycle", 0)
    total_n     = schema.get("total_neurons", len(neurons))
    total_s     = schema.get("total_synapses", len(synapses))
    last_updated = schema.get("last_updated", "")

    bg = BG_DARK if dark else BG_LIGHT
    img  = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(img, "RGBA")

    # ── Font helpers ─────────────────────────────────────────────────────────
    def _font(size, bold=True):
        candidates = (
            ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
             "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"]
            if bold else
            ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
             "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"]
        )
        for name in candidates:
            try:
                return ImageFont.truetype(name, size)
            except Exception:
                pass
        return ImageFont.load_default()

    # ── Layout zones (all in pixels, no overlap) ─────────────────────────────
    # Zone 1 — Header:   top 12% of height
    # Zone 2 — Graph:    12%–72% (60% of height)
    # Zone 3 — Stats:    72%–82%
    # Zone 4 — Legend:   82%–90%
    # Zone 5 — Footer:   90%–96%
    z_header_mid  = int(height * 0.075)
    z_header_sub  = int(height * 0.108)
    z_graph_top   = int(height * 0.13)
    z_graph_bot   = int(height * 0.72)
    z_stats_top   = int(height * 0.745)
    z_legend_top  = int(height * 0.845)
    z_footer_y    = int(height * 0.925)

    cx       = width // 2
    cy       = (z_graph_top + z_graph_bot) // 2
    graph_h  = z_graph_bot - z_graph_top
    radius   = min(width // 2, graph_h // 2) - int(width * 0.04)

    positions = _layout_neurons(neurons, radius, graph_h, cx, cy)

    # ── Colours ──────────────────────────────────────────────────────────────
    text_color   = (224, 223, 245) if dark else (26, 26, 46)
    muted_color  = (124, 123, 154) if dark else (120, 120, 160)
    accent_rgb   = _hex_to_rgb("#6c63ff")
    green_rgb    = _hex_to_rgb("#00d4aa")
    yellow_rgb   = _hex_to_rgb("#ffa502")

    # ── Header ───────────────────────────────────────────────────────────────
    title_sz = max(28, width // 22)
    sub_sz   = max(16, width // 42)
    draw.text((cx, z_header_mid),  "DMAI",
              fill=text_color, font=_font(title_sz, bold=True), anchor="mm")
    draw.text((cx, z_header_sub), "Knowledge Graph",
              fill=muted_color, font=_font(sub_sz, bold=False), anchor="mm")

    # ── Centre glow ──────────────────────────────────────────────────────────
    for r_glow in range(int(radius * 0.5), 0, -15):
        alpha = max(0, int(14 - r_glow * 0.015))
        draw.ellipse([cx - r_glow, cy - r_glow, cx + r_glow, cy + r_glow],
                     fill=(108, 99, 255, alpha))

    # ── Synapses ─────────────────────────────────────────────────────────────
    for syn in synapses:
        src = positions.get(syn.get("source") or syn.get("from"))
        tgt = positions.get(syn.get("target") or syn.get("to"))
        if not src or not tgt:
            continue
        strength = syn.get("strength", 0.5)
        alpha    = max(25, min(110, int(strength * 130)))
        draw.line([src, tgt], fill=(108, 99, 255, alpha), width=1)

    # ── Neurons ──────────────────────────────────────────────────────────────
    nr_base = max(7, width // 80)
    for neuron in neurons:
        pos = positions.get(neuron["id"])
        if not pos:
            continue
        x, y       = pos
        cluster    = neuron.get("cluster", "core")
        activation = neuron.get("activation", 0.5)
        color_rgb  = _hex_to_rgb(CLUSTER_COLORS.get(cluster, "#6c63ff"))
        nr         = int(nr_base * (0.65 + activation * 0.85))

        draw.ellipse([x-nr-6, y-nr-6, x+nr+6, y+nr+6],
                     fill=(*color_rgb, int(activation * 55)))
        draw.ellipse([x-nr,   y-nr,   x+nr,   y+nr],
                     fill=(*color_rgb, int(180 + activation * 75)))
        small = max(2, nr // 3)
        draw.ellipse([x-small, y-small, x+small, y+small],
                     fill=(255, 255, 255, int(activation * 180)))

    # ── Stats row (3 columns, fixed vertical zone) ───────────────────────────
    stat_sz   = max(22, width // 32)
    label_sz  = max(13, width // 58)
    col_x = [cx - int(width * 0.26), cx, cx + int(width * 0.26)]
    stats = [
        (str(total_n), "Neurons",  accent_rgb),
        (str(total_s), "Synapses", green_rgb),
        (f"#{cycle}",  "Cycle",    yellow_rgb),
    ]
    for (val, lbl, col), x in zip(stats, col_x):
        draw.text((x, z_stats_top),
                  val, fill=col, font=_font(stat_sz, bold=True), anchor="mm")
        draw.text((x, z_stats_top + stat_sz + 6),
                  lbl, fill=muted_color, font=_font(label_sz, bold=False), anchor="mm")

    # ── Legend row (dots only, no labels — cleaner) ──────────────────────────
    legend_items = list(CLUSTER_COLORS.items())
    n_leg = len(legend_items)
    dot_r = max(5, width // 100)
    pad   = int(width * 0.08)
    legend_label_sz = max(11, width // 68)
    for i, (cluster, hex_color) in enumerate(legend_items):
        lx  = int(pad + (width - 2 * pad) * i / (n_leg - 1))
        rgb = _hex_to_rgb(hex_color)
        draw.ellipse([lx-dot_r, z_legend_top-dot_r, lx+dot_r, z_legend_top+dot_r], fill=rgb)
        draw.text((lx, z_legend_top + dot_r + 5),
                  cluster.capitalize(),
                  fill=muted_color, font=_font(legend_label_sz, bold=False), anchor="mt")

    # ── Footer ───────────────────────────────────────────────────────────────
    footer_sz   = max(11, width // 78)
    footer_text = last_updated if last_updated else "dmai-web.onrender.com"
    draw.text((cx, z_footer_y), footer_text,
              fill=muted_color, font=_font(footer_sz, bold=False), anchor="mm")

    # ── Encode ───────────────────────────────────────────────────────────────
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    result = buf.getvalue()
    _cache[cache_key] = {"data": result, "ts": time.time()}
    return result


# ── SVG widget ────────────────────────────────────────────────────────────────

def render_widget_svg(size: str = "small") -> str:
    """
    Render a compact SVG for Widgetsmith.
    size = "small" (155×155) | "medium" (329×155) | "large" (329×345)
    """
    cache_key = f"svg_{size}"
    cached = _cache.get(cache_key)
    if cached and time.time() - cached["ts"] < _CACHE_TTL:
        return cached["data"]

    dims = {
        "small":  (155, 155),
        "medium": (329, 155),
        "large":  (329, 345),
    }
    W, H = dims.get(size, (155, 155))

    schema = _load_schema()
    neurons  = schema.get("neurons", [])
    synapses = schema.get("synapses", [])
    cycle    = schema.get("evolution_cycle", 0)
    total_n  = schema.get("total_neurons", len(neurons))
    total_s  = schema.get("total_synapses", len(synapses))

    cx = W // 2
    cy = H // 2 - (10 if size == "small" else 5)
    radius = min(W, H) // 2 - 22
    positions = _layout_neurons(neurons, radius, H, cx, cy)

    lines = []
    lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')

    # Background
    lines.append(f'<rect width="{W}" height="{H}" rx="16" fill="#0a0a0f"/>')

    # Subtle centre glow
    lines.append(f'<radialGradient id="glow" cx="50%" cy="50%" r="50%">')
    lines.append(f'  <stop offset="0%" stop-color="#6c63ff" stop-opacity="0.18"/>')
    lines.append(f'  <stop offset="100%" stop-color="#6c63ff" stop-opacity="0"/>')
    lines.append(f'</radialGradient>')
    lines.append(f'<circle cx="{cx}" cy="{cy}" r="{radius}" fill="url(#glow)"/>')

    # Synapses
    neuron_map = {n["id"]: n for n in neurons}
    for syn in synapses:
        src = positions.get(syn.get("source") or syn.get("from"))
        tgt = positions.get(syn.get("target") or syn.get("to"))
        if not src or not tgt:
            continue
        strength = syn.get("strength", 0.5)
        opacity = max(0.08, min(0.35, strength * 0.4))
        lines.append(
            f'<line x1="{src[0]:.1f}" y1="{src[1]:.1f}" '
            f'x2="{tgt[0]:.1f}" y2="{tgt[1]:.1f}" '
            f'stroke="#6c63ff" stroke-width="0.8" stroke-opacity="{opacity:.2f}"/>'
        )

    # Neurons
    nr_base = max(3, W // 45)
    for neuron in neurons:
        pos = positions.get(neuron["id"])
        if not pos:
            continue
        x, y = pos
        cluster   = neuron.get("cluster", "core")
        activation = neuron.get("activation", 0.5)
        color     = CLUSTER_COLORS.get(cluster, "#6c63ff")
        nr        = max(2, int(nr_base * (0.6 + activation * 0.9)))
        glow_r    = nr + 4
        glow_op   = activation * 0.35

        lines.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{glow_r}" '
            f'fill="{color}" fill-opacity="{glow_op:.2f}"/>'
        )
        lines.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{nr}" '
            f'fill="{color}" fill-opacity="0.9"/>'
        )
        # Bright centre dot
        lines.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{max(1, nr//3)}" '
            f'fill="white" fill-opacity="{activation * 0.7:.2f}"/>'
        )

    # Stats text at bottom
    text_y = H - 22
    fs = max(7, W // 28)
    lines.append(
        f'<text x="{W//2}" y="{text_y}" text-anchor="middle" '
        f'font-family="-apple-system,sans-serif" font-size="{fs}" fill="#7c7b9a">'
        f'{total_n} neurons · {total_s} synapses · cycle {cycle}</text>'
    )
    # Mini title
    lines.append(
        f'<text x="{W//2}" y="14" text-anchor="middle" '
        f'font-family="-apple-system,sans-serif" font-size="{max(7, W//30)}" '
        f'font-weight="700" fill="#e0dff5">DMAI</text>'
    )

    lines.append('</svg>')
    result = "\n".join(lines)

    _cache[cache_key] = {"data": result, "ts": time.time()}
    return result


def clear_cache():
    """Call this after graph evolution to force re-render on next request."""
    _cache.clear()
