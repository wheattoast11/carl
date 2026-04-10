"""Generate the CARL hero animation: chaos → crystallization → agency.

Creates a GIF showing particles morphing from entropy to crystal structure,
with cheeky labels. Used as the README hero and social preview.

Requires: pip install Pillow numpy
Output: assets/carl-hero.gif
"""
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 1200, 630  # OG image dimensions (social preview standard)
BG = (8, 8, 12)
CYAN = (6, 182, 212)
PURPLE = (168, 85, 247)
AMBER = (245, 158, 11)
GREEN = (16, 185, 129)
WHITE = (240, 240, 240)
DIM = (80, 80, 90)
N_PARTICLES = 120
FRAMES = 80
FPS = 20

# Crystal lattice target positions (hexagonal grid centered)
def hex_grid(n, cx, cy, spacing=28):
    pts = []
    cols = int(math.sqrt(n) * 1.2)
    rows = int(n / cols) + 1
    ox = cx - (cols * spacing) / 2
    oy = cy - (rows * spacing * 0.866) / 2
    for r in range(rows):
        for c in range(cols):
            x = ox + c * spacing + (spacing / 2 if r % 2 else 0)
            y = oy + r * spacing * 0.866
            pts.append((x, y))
            if len(pts) >= n:
                return pts
    return pts

# Phase timeline: (frame_start, frame_end, phase_name, label, phi)
PHASES = [
    (0, 20, "chaos", "before: \"idk lol\"", 0.03),
    (20, 35, "melting", "entropy go brrr", 0.01),
    (35, 42, "snap", "phase transition.", 0.65),
    (42, 55, "crystal", "\"I got this\"", 0.99),
    (55, 80, "agency", "alignment you can see.", 0.99),
]

def get_phase(frame):
    for start, end, name, label, phi in PHASES:
        if start <= frame < end:
            t = (frame - start) / max(end - start, 1)
            return name, label, phi, t
    return PHASES[-1][2], PHASES[-1][3], PHASES[-1][4], 1.0

def lerp(a, b, t):
    return a + (b - a) * t

def ease_out_elastic(t):
    if t <= 0: return 0
    if t >= 1: return 1
    p = 0.3
    return pow(2, -10 * t) * math.sin((t - p/4) * (2 * math.pi) / p) + 1

def draw_phi_bar(draw, phi, y, label_text):
    bar_x, bar_w, bar_h = 60, 300, 16
    # Background
    draw.rounded_rectangle([bar_x, y, bar_x + bar_w, y + bar_h], radius=4, fill=(30, 30, 35))
    # Fill
    fill_w = int(bar_w * phi)
    if fill_w > 2:
        color = CYAN if phi < 0.5 else GREEN if phi < 0.9 else PURPLE
        draw.rounded_rectangle([bar_x, y, bar_x + fill_w, y + bar_h], radius=4, fill=color)
    # Label
    draw.text((bar_x + bar_w + 12, y - 2), f"Phi: {phi:.2f}", fill=DIM, font=None)

def generate_frames():
    random.seed(42)
    np.random.seed(42)

    # Initial random positions
    init_pos = [(random.uniform(150, W - 150), random.uniform(120, H - 180)) for _ in range(N_PARTICLES)]
    # Crystal target positions
    crystal_pos = hex_grid(N_PARTICLES, W // 2, H // 2 - 20, spacing=32)
    # Random velocities for chaos phase
    velocities = [(random.uniform(-3, 3), random.uniform(-3, 3)) for _ in range(N_PARTICLES)]

    positions = list(init_pos)
    frames = []

    try:
        font_large = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 42)
        font_med = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 22)
        font_small = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 14)
        font_title = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 56)
    except (OSError, IOError):
        font_large = ImageFont.load_default()
        font_med = font_large
        font_small = font_large
        font_title = font_large

    for frame in range(FRAMES):
        img = Image.new("RGB", (W, H), BG)
        draw = ImageDraw.Draw(img)

        phase, label, phi, t = get_phase(frame)

        # Update particle positions based on phase
        for i in range(N_PARTICLES):
            px, py = positions[i]
            tx, ty = crystal_pos[i]
            vx, vy = velocities[i]

            if phase == "chaos":
                # Random jitter
                px += vx + random.uniform(-1.5, 1.5)
                py += vy + random.uniform(-1.5, 1.5)
                # Bounce off walls
                if px < 100 or px > W - 100: vx = -vx
                if py < 80 or py > H - 150: vy = -vy
                velocities[i] = (vx, vy)

            elif phase == "melting":
                # CHAOS INTENSIFIES
                px += random.uniform(-6, 6)
                py += random.uniform(-6, 6)

            elif phase == "snap":
                # Elastic snap to crystal position
                ease = ease_out_elastic(t)
                px = lerp(px, tx, min(ease * 0.3, 1.0))
                py = lerp(py, ty, min(ease * 0.3, 1.0))

            elif phase in ("crystal", "agency"):
                # Hold crystal position with subtle breathing
                breath = math.sin(frame * 0.15 + i * 0.1) * 1.5
                px = tx + breath
                py = ty + breath * 0.5

            # Clamp
            px = max(40, min(W - 40, px))
            py = max(40, min(H - 80, py))
            positions[i] = (px, py)

        # Draw connections (crystal phase only)
        if phase in ("crystal", "agency", "snap") and t > 0.3:
            alpha = min(1.0, (t - 0.3) * 2) if phase == "snap" else 1.0
            conn_color = tuple(int(c * alpha * 0.3) for c in CYAN)
            for i in range(N_PARTICLES):
                for j in range(i + 1, min(i + 8, N_PARTICLES)):
                    x1, y1 = positions[i]
                    x2, y2 = positions[j]
                    dist = math.hypot(x2 - x1, y2 - y1)
                    if dist < 50:
                        draw.line([(x1, y1), (x2, y2)], fill=conn_color, width=1)

        # Draw particles
        for i, (px, py) in enumerate(positions):
            if phase == "chaos":
                color = tuple(random.randint(c - 30, c + 30) for c in DIM)
                r = random.uniform(2, 4)
            elif phase == "melting":
                # Red-shifting entropy
                r = random.uniform(2, 5)
                color = (min(255, 180 + random.randint(0, 75)), random.randint(40, 80), random.randint(20, 50))
            elif phase == "snap":
                r = 3
                blend = ease_out_elastic(t)
                color = tuple(int(lerp(200, c, blend)) for c in CYAN)
            else:
                r = 3
                # Cycle colors in agency phase
                if phase == "agency":
                    hue_shift = (frame * 0.05 + i * 0.02) % 1.0
                    if hue_shift < 0.33:
                        color = CYAN
                    elif hue_shift < 0.66:
                        color = PURPLE
                    else:
                        color = GREEN
                else:
                    color = CYAN

            draw.ellipse([px - r, py - r, px + r, py + r], fill=color)

        # Title
        draw.text((W // 2, 28), "CARL", fill=WHITE, font=font_title, anchor="mt")

        # Subtitle
        draw.text((W // 2, 82), "Coherence-Aware Reinforcement Learning", fill=DIM, font=font_small, anchor="mt")

        # Phase label (bottom, with attitude)
        draw.text((W // 2, H - 70), label, fill=WHITE, font=font_large, anchor="mm")

        # Phi bar
        draw_phi_bar(draw, phi, H - 35, "Phi")

        # Phase indicator
        phase_colors = {"chaos": DIM, "melting": (220, 80, 60), "snap": AMBER, "crystal": CYAN, "agency": PURPLE}
        dot_color = phase_colors.get(phase, DIM)
        draw.ellipse([W - 50, H - 38, W - 34, H - 22], fill=dot_color)

        # terminals.tech watermark
        draw.text((W - 30, 15), "terminals.tech", fill=(40, 40, 45), font=font_small, anchor="rt")

        frames.append(img)

    return frames


def main():
    out_dir = Path(__file__).parent.parent / "assets"
    out_dir.mkdir(exist_ok=True)

    print("Generating frames...")
    frames = generate_frames()

    # Save as GIF
    gif_path = out_dir / "carl-hero.gif"
    print(f"Saving {gif_path}...")
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=1000 // FPS,
        loop=0,
        optimize=True,
    )
    print(f"Done: {gif_path} ({gif_path.stat().st_size / 1024:.0f} KB)")

    # Also save a static frame for social preview (the crystal moment)
    preview_path = out_dir / "carl-social-preview.png"
    frames[48].save(preview_path)
    print(f"Social preview: {preview_path}")


if __name__ == "__main__":
    main()
