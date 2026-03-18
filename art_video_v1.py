from __future__ import annotations

"""
Art Video V1
============

Generate a short promotional video from:
- one HD artwork image
- one explanatory text file

Pipeline:
1. Load image
2. Detect visually interesting regions with OpenCV saliency + fallback heuristics
3. Split long text into short overlay blocks
4. Build a cinematic sequence:
   - opening full view
   - several detail shots with slow zoom / pan
   - closing full view
5. Render mp4 with MoviePy

Install (example):
    pip install moviepy opencv-contrib-python pillow numpy

FFmpeg must also be available on the system path.

Example:
    python art_video_v1.py \
        --image painting.jpg \
        --text description.txt \
        --output output.mp4 \
        --title "Blue Tide" \
        --artist "Justine" \
        --fps 30

Notes:
- This V1 stays fully local.
- No TTS.
- Music is optional and not included yet.
"""

import argparse
import math
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy import (
    AudioFileClip,
    CompositeVideoClip,
    ImageClip,
    VideoClip,
    concatenate_videoclips,
)
from moviepy.video.fx.FadeIn import FadeIn
from moviepy.video.fx.FadeOut import FadeOut


# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Region:
    """A crop region in image pixel coordinates."""

    x: int
    y: int
    w: int
    h: int
    score: float


@dataclass(frozen=True)
class ShotSpec:
    """A single video shot to render."""

    region: Region
    duration: float
    text: str
    zoom_start: float
    zoom_end: float


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def ease_in_out(t: float) -> float:
    """Smoothstep easing in [0,1]."""
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(value, high))


def read_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Text file is empty: {path}")
    return text


# -----------------------------------------------------------------------------
# Text segmentation
# -----------------------------------------------------------------------------


def split_text_into_blocks(
    text: str,
    min_chars: int = 110,
    max_chars: int = 220,
    max_blocks: int = 6,
) -> List[str]:
    """
    Split long explanatory text into readable blocks.

    Strategy:
    - first split by paragraphs
    - then split by sentences
    - accumulate until a comfortable reading size
    """
    raw_paragraphs = [
        p.strip() for p in text.replace("\r\n", "\n").split("\n") if p.strip()
    ]
    if not raw_paragraphs:
        return [text.strip()]

    sentence_candidates: List[str] = []
    for para in raw_paragraphs:
        # Simple sentence splitting heuristic.
        buff = []
        current = ""
        for token in para.split(" "):
            current = f"{current} {token}".strip()
            if token.endswith((".", "!", "?", ";", ":")):
                buff.append(current)
                current = ""
        if current:
            buff.append(current)
        sentence_candidates.extend(s.strip() for s in buff if s.strip())

    blocks: List[str] = []
    current_block = ""
    for sentence in sentence_candidates:
        proposal = f"{current_block} {sentence}".strip()
        if not current_block:
            current_block = sentence
            continue
        if len(proposal) <= max_chars:
            current_block = proposal
        else:
            blocks.append(current_block)
            current_block = sentence
    if current_block:
        blocks.append(current_block)

    # Merge tiny blocks with the next one if possible.
    merged: List[str] = []
    i = 0
    while i < len(blocks):
        blk = blocks[i]
        if len(blk) < min_chars and i + 1 < len(blocks):
            candidate = f"{blk} {blocks[i + 1]}"
            if len(candidate) <= max_chars + 60:
                merged.append(candidate)
                i += 2
                continue
        merged.append(blk)
        i += 1

    if len(merged) > max_blocks:
        # Coarsen by regrouping evenly.
        per_group = math.ceil(len(merged) / max_blocks)
        regrouped = []
        for start in range(0, len(merged), per_group):
            regrouped.append(" ".join(merged[start : start + per_group]))
        merged = regrouped

    return merged[:max_blocks]


# -----------------------------------------------------------------------------
# Region detection
# -----------------------------------------------------------------------------


def non_max_suppression(
    regions: Sequence[Region], iou_threshold: float = 0.35
) -> List[Region]:
    """Keep diverse regions by suppressing heavily overlapping boxes."""

    def iou(a: Region, b: Region) -> float:
        x1 = max(a.x, b.x)
        y1 = max(a.y, b.y)
        x2 = min(a.x + a.w, b.x + b.w)
        y2 = min(a.y + a.h, b.y + b.h)
        inter_w = max(0, x2 - x1)
        inter_h = max(0, y2 - y1)
        inter = inter_w * inter_h
        union = a.w * a.h + b.w * b.h - inter
        return inter / union if union > 0 else 0.0

    kept: List[Region] = []
    for region in sorted(regions, key=lambda r: r.score, reverse=True):
        if all(iou(region, existing) < iou_threshold for existing in kept):
            kept.append(region)
    return kept


def detect_candidate_regions(
    image_bgr: np.ndarray,
    num_regions: int = 4,
    crop_scale: float = 0.42,
) -> List[Region]:
    """
    Detect interesting regions using OpenCV saliency, with robust fallbacks.
    Returns pixel regions suitable for detail shots.
    """
    h, w = image_bgr.shape[:2]
    crop_w = max(160, int(w * crop_scale))
    crop_h = max(160, int(h * crop_scale))

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    texture_map = np.abs(lap)
    texture_map = cv2.GaussianBlur(texture_map, (0, 0), 5)

    saliency_map = None
    if hasattr(cv2, "saliency"):
        try:
            saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
            success, saliency_map = saliency.computeSaliency(image_bgr)
            if not success:
                saliency_map = None
        except Exception:
            saliency_map = None

    if saliency_map is None:
        # Fallback: use normalized texture map as a pseudo-saliency map.
        saliency_map = texture_map.copy()
        denom = float(saliency_map.max()) or 1.0
        saliency_map = saliency_map / denom

    saliency_map = saliency_map.astype(np.float32)
    saliency_map = cv2.GaussianBlur(saliency_map, (0, 0), 7)

    # Create a grid of candidate crops.
    candidates: List[Region] = []
    step_x = max(20, crop_w // 5)
    step_y = max(20, crop_h // 5)

    sat_hsv = (
        cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)[:, :, 1].astype(np.float32) / 255.0
    )
    sat_hsv = cv2.GaussianBlur(sat_hsv, (0, 0), 5)

    for y in range(0, max(1, h - crop_h + 1), step_y):
        for x in range(0, max(1, w - crop_w + 1), step_x):
            saliency_score = float(
                np.mean(saliency_map[y : y + crop_h, x : x + crop_w])
            )
            texture_score = float(np.mean(texture_map[y : y + crop_h, x : x + crop_w]))
            saturation_score = float(np.mean(sat_hsv[y : y + crop_h, x : x + crop_w]))

            # Light centrality bonus: details too close to borders are often less elegant.
            cx = x + crop_w / 2
            cy = y + crop_h / 2
            dx = abs(cx - w / 2) / (w / 2 + 1e-6)
            dy = abs(cy - h / 2) / (h / 2 + 1e-6)
            centrality_bonus = 1.0 - 0.35 * (dx + dy)

            score = (
                0.55 * saliency_score
                + 0.30 * (texture_score / (float(texture_map.max()) + 1e-6))
                + 0.15 * saturation_score
            ) * centrality_bonus

            candidates.append(Region(x=x, y=y, w=crop_w, h=crop_h, score=score))

    # Add a few handcrafted candidates for robustness.
    handcrafted = [
        Region(int(w * 0.08), int(h * 0.08), crop_w, crop_h, 0.01),
        Region(int(w * 0.50 - crop_w / 2), int(h * 0.20), crop_w, crop_h, 0.01),
        Region(int(w * 0.20), int(h * 0.50 - crop_h / 2), crop_w, crop_h, 0.01),
        Region(int(w * 0.55), int(h * 0.55), crop_w, crop_h, 0.01),
        Region(
            int(w * 0.50 - crop_w / 2), int(h * 0.50 - crop_h / 2), crop_w, crop_h, 0.01
        ),
    ]

    def fit(region: Region) -> Region:
        x = int(clamp(region.x, 0, w - crop_w))
        y = int(clamp(region.y, 0, h - crop_h))
        return Region(x=x, y=y, w=crop_w, h=crop_h, score=region.score)

    all_candidates = [fit(c) for c in candidates] + [fit(c) for c in handcrafted]
    diverse = non_max_suppression(all_candidates, iou_threshold=0.28)

    if not diverse:
        return [Region(0, 0, w, h, 1.0)]

    return diverse[:num_regions]


# -----------------------------------------------------------------------------
# Text rendering with Pillow
# -----------------------------------------------------------------------------


def load_font(font_size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Try a few common fonts, then fallback to PIL default."""
    font_candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/Library/Fonts/Arial.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
    ]
    for path in font_candidates:
        try:
            return ImageFont.truetype(path, font_size)
        except Exception:
            continue
    return ImageFont.load_default()


def wrap_text_for_width(
    draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int
) -> str:
    words = text.split()
    if not words:
        return ""

    lines: List[str] = []
    current = words[0]
    for word in words[1:]:
        proposal = f"{current} {word}"
        bbox = draw.textbbox((0, 0), proposal, font=font)
        width = bbox[2] - bbox[0]
        if width <= max_width:
            current = proposal
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return "\n".join(lines)


def make_text_overlay_frame(
    size: Tuple[int, int],
    text: str,
    title: str | None = None,
    artist: str | None = None,
    opacity: int = 185,
) -> np.ndarray:
    """Create an RGBA overlay with a bottom text panel."""
    width, height = size
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    title_font = load_font(max(24, width // 28))
    body_font = load_font(max(22, width // 38))
    meta_font = load_font(max(18, width // 50))

    panel_margin = int(width * 0.05)
    panel_width = width - 2 * panel_margin
    body_max_width = int(panel_width * 0.90)

    wrapped_title = title or ""
    wrapped_body = wrap_text_for_width(draw, text, body_font, body_max_width)
    wrapped_artist = artist or ""

    # Estimate text block height.
    cursor_y = 0
    top_padding = 20
    line_gap = 10

    bbox_title = (
        draw.multiline_textbbox((0, 0), wrapped_title, font=title_font, spacing=6)
        if wrapped_title
        else (0, 0, 0, 0)
    )
    bbox_body = draw.multiline_textbbox((0, 0), wrapped_body, font=body_font, spacing=8)
    bbox_artist = (
        draw.multiline_textbbox((0, 0), wrapped_artist, font=meta_font, spacing=4)
        if wrapped_artist
        else (0, 0, 0, 0)
    )

    title_h = bbox_title[3] - bbox_title[1] if wrapped_title else 0
    body_h = bbox_body[3] - bbox_body[1]
    artist_h = bbox_artist[3] - bbox_artist[1] if wrapped_artist else 0

    panel_height = (
        top_padding
        + title_h
        + (line_gap if title_h else 0)
        + body_h
        + (line_gap if artist_h else 0)
        + artist_h
        + 24
    )
    panel_y = height - panel_height - int(height * 0.04)

    draw.rounded_rectangle(
        [panel_margin, panel_y, panel_margin + panel_width, panel_y + panel_height],
        radius=24,
        fill=(12, 12, 12, opacity),
    )

    text_x = panel_margin + int(panel_width * 0.05)
    cursor_y = panel_y + top_padding

    if wrapped_title:
        draw.multiline_text(
            (text_x, cursor_y),
            wrapped_title,
            font=title_font,
            fill=(255, 255, 255, 255),
            spacing=6,
        )
        cursor_y += title_h + line_gap

    draw.multiline_text(
        (text_x, cursor_y),
        wrapped_body,
        font=body_font,
        fill=(240, 240, 240, 255),
        spacing=8,
    )
    cursor_y += body_h

    if wrapped_artist:
        cursor_y += line_gap
        draw.multiline_text(
            (text_x, cursor_y),
            wrapped_artist,
            font=meta_font,
            fill=(200, 200, 200, 255),
            spacing=4,
        )

    return np.array(img)


# -----------------------------------------------------------------------------
# Shot rendering
# -----------------------------------------------------------------------------


def make_animated_crop_clip(
    image_rgb: np.ndarray,
    region: Region,
    output_size: Tuple[int, int],
    duration: float,
    zoom_start: float,
    zoom_end: float,
) -> VideoClip:
    """
    Create a shot from a large image by animating inside a region.

    The animation is a gentle zoom over the selected region.
    """
    source_h, source_w = image_rgb.shape[:2]
    out_w, out_h = output_size

    # Base region center.
    cx = region.x + region.w / 2
    cy = region.y + region.h / 2

    # Target crop aspect ratio must match output.
    aspect = out_w / out_h
    base_w = region.w
    base_h = region.h
    if base_w / base_h > aspect:
        base_h = base_w / aspect
    else:
        base_w = base_h * aspect

    base_w = min(base_w, source_w)
    base_h = min(base_h, source_h)

    def get_frame(t: float) -> np.ndarray:
        alpha = ease_in_out(t / duration if duration > 0 else 1.0)
        zoom = zoom_start + (zoom_end - zoom_start) * alpha

        crop_w = int(base_w / zoom)
        crop_h = int(base_h / zoom)
        crop_w = max(64, min(crop_w, source_w))
        crop_h = max(64, min(crop_h, source_h))

        x1 = int(round(cx - crop_w / 2))
        y1 = int(round(cy - crop_h / 2))
        x1 = int(clamp(x1, 0, source_w - crop_w))
        y1 = int(clamp(y1, 0, source_h - crop_h))
        x2 = x1 + crop_w
        y2 = y1 + crop_h

        crop = image_rgb[y1:y2, x1:x2]
        resized = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
        return resized

    return VideoClip(frame_function=get_frame, duration=duration)


def build_shot_clip(
    image_rgb: np.ndarray,
    shot: ShotSpec,
    output_size: Tuple[int, int],
    title: str | None,
    artist: str | None,
    fadein_duration: float = 0.4,
    fadeout_duration: float = 0.4,
) -> CompositeVideoClip:
    base = make_animated_crop_clip(
        image_rgb=image_rgb,
        region=shot.region,
        output_size=output_size,
        duration=shot.duration,
        zoom_start=shot.zoom_start,
        zoom_end=shot.zoom_end,
    )

    overlay_img = make_text_overlay_frame(
        size=output_size,
        text=shot.text,
        title=title,
        artist=artist,
    )

    overlay = ImageClip(overlay_img, transparent=True).with_duration(shot.duration)

    clip = CompositeVideoClip([base, overlay], size=output_size)
    clip = clip.with_duration(shot.duration)
    clip = clip.with_effects([FadeIn(fadein_duration), FadeOut(fadeout_duration)])
    return clip


# -----------------------------------------------------------------------------
# Sequence building
# -----------------------------------------------------------------------------


def build_intro_outro_region(image_w: int, image_h: int) -> Region:
    return Region(0, 0, image_w, image_h, 1.0)


def build_shot_specs(
    image_shape: Tuple[int, int, int],
    text_blocks: Sequence[str],
    detail_regions: Sequence[Region],
    intro_duration: float = 2.8,
    detail_duration: float = 4.2,
    outro_duration: float = 2.8,
) -> List[ShotSpec]:
    h, w = image_shape[:2]
    full_region = build_intro_outro_region(w, h)

    shots: List[ShotSpec] = []

    # Opening full view uses first block if present.
    if text_blocks:
        shots.append(
            ShotSpec(
                region=full_region,
                duration=intro_duration,
                text=text_blocks[0],
                zoom_start=1.00,
                zoom_end=1.06,
            )
        )

    # Middle details use subsequent text blocks.
    middle_texts = list(text_blocks[1:]) if len(text_blocks) > 1 else []
    if not middle_texts:
        middle_texts = (
            [text_blocks[0]] * min(len(detail_regions), 3)
            if text_blocks
            else [""] * min(len(detail_regions), 3)
        )

    for region, text in zip(detail_regions, middle_texts):
        shots.append(
            ShotSpec(
                region=region,
                duration=detail_duration,
                text=text,
                zoom_start=1.00,
                zoom_end=1.12,
            )
        )

    # Closing full frame returns to overall composition.
    closing_text = ""
    if len(text_blocks) >= 2:
        closing_text = text_blocks[-1]
    elif text_blocks:
        closing_text = text_blocks[0]

    shots.append(
        ShotSpec(
            region=full_region,
            duration=outro_duration,
            text=closing_text,
            zoom_start=1.04,
            zoom_end=1.00,
        )
    )

    return shots


# -----------------------------------------------------------------------------
# Main render function
# -----------------------------------------------------------------------------


def generate_video(
    image_path: Path,
    text_path: Path,
    output_path: Path,
    title: str | None,
    artist: str | None,
    fps: int = 30,
    width: int = 1080,
    height: int = 1920,
    num_detail_shots: int = 4,
    music_path: Path | None = None,
) -> None:
    # Load image
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    text = read_text(text_path)
    text_blocks = split_text_into_blocks(text, max_blocks=num_detail_shots + 2)
    detail_regions = detect_candidate_regions(image_bgr, num_regions=num_detail_shots)

    shots = build_shot_specs(
        image_shape=image_rgb.shape,
        text_blocks=text_blocks,
        detail_regions=detail_regions,
    )

    shot_clips = [
        build_shot_clip(
            image_rgb=image_rgb,
            shot=shot,
            output_size=(width, height),
            title=title if i == 0 else None,
            artist=artist if i == len(shots) - 1 else None,
        )
        for i, shot in enumerate(shots)
    ]

    final = concatenate_videoclips(shot_clips, method="compose", padding=-0.2)

    if music_path is not None:
        audio = AudioFileClip(str(music_path))
        if audio.duration < final.duration:
            # V1 simple behavior: cut or loop would come later; for now just use what exists.
            audio = audio.with_duration(min(audio.duration, final.duration))
        else:
            audio = audio.subclipped(0, final.duration)
        final = final.with_audio(audio)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    final.write_videofile(
        str(output_path),
        fps=fps,
        codec="libx264",
        audio_codec="aac" if music_path else None,
        preset="medium",
    )


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a cinematic artwork promo video from one image and one text."
    )
    parser.add_argument(
        "--image", type=Path, required=True, help="Path to the artwork image"
    )
    parser.add_argument(
        "--text", type=Path, required=True, help="Path to the explanatory text file"
    )
    parser.add_argument("--output", type=Path, required=True, help="Output mp4 path")
    parser.add_argument(
        "--title", type=str, default=None, help="Optional artwork title"
    )
    parser.add_argument("--artist", type=str, default=None, help="Optional artist name")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--width", type=int, default=1080, help="Output width")
    parser.add_argument("--height", type=int, default=1920, help="Output height")
    parser.add_argument(
        "--num-detail-shots", type=int, default=4, help="Number of detail shots"
    )
    parser.add_argument(
        "--music", type=Path, default=None, help="Optional background music file"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_video(
        image_path=args.image,
        text_path=args.text,
        output_path=args.output,
        title=args.title,
        artist=args.artist,
        fps=args.fps,
        width=args.width,
        height=args.height,
        num_detail_shots=args.num_detail_shots,
        music_path=args.music,
    )


if __name__ == "__main__":
    main()
