"""Render authored diagrams and optional GIFs from one CARL visual grammar."""

from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "assets/brand"
COLORS = {
    "paper": "#FAF6EE",
    "ink": "#45424B",
    "attention": "#9D591C",
    "confirmed": "#2E6351",
    "derived": "#635186",
    "muted": "#6B6570",
    "night_ink": "#E9F0F0",
    "night_secondary": "#B9CCCC",
    "night_attention": "#F2B35E",
    "night_confirmed": "#9ECFD0",
    "night_muted": "#8AA3A3",
}
TITLES = {
    "carl-overview": "Improve models. Keep the evidence.",
    "interpretation": "A correction changes the next interpretation.",
    "representation": "Compute once. Reuse the full representation.",
    "boundaries": "One product. Explicit ownership boundaries.",
}


class Diagram:
    """Shared SVG and raster drawing primitives."""

    def __init__(self, title: str, active: int | None) -> None:
        self.active = active
        self.image = Image.new("RGB", (1280, 760), COLORS["paper"])
        self.draw = ImageDraw.Draw(self.image)
        self.svg = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="760" viewBox="0 0 1280 760" role="img"><title>{html.escape(title)}</title><desc>Authored illustration, not a measured model result.</desc><rect width="1280" height="760" fill="{COLORS["paper"]}"/>'
        ]
        self.text(64, 42, "CARL", 30)
        self.text(870, 54, "Intuition Labs LLC · terminals", 20)
        self.text(64, 120, title, 38)
        self.text(64, 188, "Coherence-Aware Reinforcement Learning", 22, "muted")

    def text(self, x: int, y: int, value: str, size: int = 22, role: str = "ink") -> None:
        font = ImageFont.truetype("NotoSans[wght].ttf", size)
        self.draw.text((x, y), value, font=font, fill=COLORS[role])
        self.svg.append(
            f'<text x="{x}" y="{y + size}" font-family="system-ui,sans-serif" font-size="{size}" fill="{COLORS[role]}">{html.escape(value)}</text>'
        )

    def box(self, x: int, y: int, w: int, h: int, role: str, index: int = -1) -> None:
        width = 4 if self.active == index else 2
        self.draw.rounded_rectangle(
            (x, y, x + w, y + h), radius=12, fill=COLORS["paper"], outline=COLORS[role], width=width
        )
        self.svg.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{COLORS["paper"]}" stroke="{COLORS[role]}" stroke-width="{width}"/>'
        )

    def line(self, x1: int, y1: int, x2: int, y2: int) -> None:
        self.draw.line((x1, y1, x2, y2), fill=COLORS["muted"], width=2)
        self.svg.append(
            f'<path d="M{x1} {y1} L{x2} {y2}" fill="none" stroke="{COLORS["muted"]}" stroke-width="2"/>'
        )

    def finish(self, footer: str) -> None:
        self.text(64, 704, footer, 17, "muted")
        self.svg.append("</svg>")


def scene(name: str, active: int | None) -> Diagram:
    d = Diagram(TITLES[name], active)
    if name == "representation":
        d.box(64, 250, 1152, 76, "derived", 0)
        d.text(88, 270, "Raw carrier · 768 dimensions · retained once", 24, "derived")
        for i, dim in enumerate((128, 256, 512, 768)):
            x = 64 + i * 294
            d.line(x + 135, 326, x + 135, 370)
            d.box(x, 370, 270, 210, "derived", i)
            d.text(x + 20, 390, f"{dim} dimensions", 24, "derived")
            d.text(x + 20, 440, "Raw prefix retained", 19)
            d.text(x + 20, 478, "Normalize for scoring", 19)
            for j in range(12):
                d.box(x + 20 + j * 18, 530, 12, 24, "derived" if j < dim / 64 else "muted")
        d.text(64, 624, "Initial recipes: text 128; multimodal 256. Refine at 512 and 768.", 22)
        d.finish(
            "Initial binding · same carrier · selected resolution is separate from model identity"
        )
    elif name == "boundaries":
        for i, (label, rows, role) in enumerate(
            [
                (
                    "Public packages",
                    ["carl-core", "carl-encoders", "carl-studio", "MIT source notices"],
                    "ink",
                ),
                (
                    "Model artifacts",
                    [
                        "Weights + heads",
                        "Processor + evidence",
                        "Artifact-specific terms",
                        "GGUF qualification",
                    ],
                    "derived",
                ),
                (
                    "Private runtime",
                    [
                        "Separate installation",
                        "Private algorithms",
                        "Authorized resolver",
                        "No source in public wheel",
                    ],
                    "ink",
                ),
                (
                    "Operator harness",
                    [
                        "Rights evidence",
                        "Destination-bound grant",
                        "Release and rollback",
                        "Outside public CARL",
                    ],
                    "ink",
                ),
            ]
        ):
            x = 64 + i * 294
            d.box(x, 260, 270, 340, role, i)
            d.text(x + 18, 280, label, 23, role)
            for j, row in enumerate(rows):
                d.text(x + 18, 348 + j * 52, row, 18)
        d.text(
            64,
            634,
            "Capture permission, training permission and publication permission stay separate.",
            21,
        )
        d.finish("Ownership inventory · code, weights, data and marks retain separate rights")
    elif name == "interpretation":
        d.text(
            64,
            244,
            "Fictional request: “Lighten it.”   Goal: reduce the application download size.",
            22,
        )
        rows = [
            ("Proposal", "Use pale colors.", "attention"),
            ("Explicit correction", "“I mean fewer bytes.”", "confirmed"),
            (
                "Successor interpretation",
                "Remove unused assets. Supersedes the proposal.",
                "derived",
            ),
            (
                "Action and replay",
                "Measure bundle size. Commit once; retries reuse the artifact.",
                "ink",
            ),
        ]
        for i, (label, detail, role) in enumerate(rows):
            y = 296 + i * 88
            d.box(64, y, 1152, 72, role, i)
            d.text(88, y + 20, label, 22, role)
            d.text(395, y + 20, detail, 21)
        d.finish(
            "Fictional example · explicit feedback changes meaning · action success still requires measurement"
        )
    else:
        rows = [
            ("Prepare", "Bind sources, goal, grader and limits.", "attention"),
            ("Learn", "Train heads or encoder adapters within the declared budget.", "derived"),
            ("Compare", "Measure held-out action and required retention slices.", "derived"),
            ("Decide", "Accepted / rejected / inconclusive. Keep the predecessor.", "ink"),
        ]
        for i, (label, detail, role) in enumerate(rows):
            y = 260 + i * 95
            d.box(64, y, 1152, 76, role, i)
            d.text(90, y + 23, f"{i + 1}. {label}", 24, role)
            d.text(340, y + 23, detail, 22)
        d.finish("Illustrated workflow · training updates do not establish held-out improvement")
    return d


def render() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    files = {}
    for name in TITLES:
        diagram = scene(name, None)
        (OUTPUT / f"{name}.svg").write_text("\n".join(diagram.svg) + "\n")
        diagram.image.save(OUTPUT / f"{name}.png")
        frames = [scene(name, active).image for active in range(4)]
        frames[0].save(
            OUTPUT / f"{name}.gif",
            save_all=True,
            append_images=frames[1:],
            duration=[1400, 1400, 1400, 2200],
            loop=0,
            optimize=False,
        )
        for suffix in ("svg", "png", "gif"):
            p = OUTPUT / f"{name}.{suffix}"
            files[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    (OUTPUT / "tokens.json").write_text(json.dumps(COLORS, indent=2) + "\n")
    (OUTPUT / "provenance.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "license": "MIT",
                "copyright": "2026 Intuition Labs LLC",
                "kind": "authored explanatory diagrams",
                "performance_claim": False,
                "fonts_redistributed": False,
                "source": "scripts/render_brand.py",
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "files": files,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    render()
