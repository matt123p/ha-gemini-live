"""Generate the Gemini Live brand icon using only the Python standard library."""

from __future__ import annotations

import math
from pathlib import Path
import struct
import zlib


SIZE = 256
SCALE = 4
CANVAS = SIZE * SCALE
OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "custom_components"
    / "gemini_live"
    / "brand"
    / "icon.png"
)


def blend(pixel: bytearray, color: tuple[int, int, int, int]) -> None:
    """Alpha-composite color over one RGBA pixel."""
    source_alpha = color[3] / 255
    target_alpha = pixel[3] / 255
    output_alpha = source_alpha + target_alpha * (1 - source_alpha)
    if output_alpha == 0:
        return
    for channel in range(3):
        pixel[channel] = round(
            (
                color[channel] * source_alpha
                + pixel[channel] * target_alpha * (1 - source_alpha)
            )
            / output_alpha
        )
    pixel[3] = round(output_alpha * 255)


def set_pixel(
    image: list[bytearray], x: int, y: int, color: tuple[int, int, int, int]
) -> None:
    """Set one pixel with alpha blending."""
    if not (0 <= x < CANVAS and 0 <= y < CANVAS):
        return
    offset = x * 4
    target = bytearray(image[y][offset : offset + 4])
    blend(target, color)
    image[y][offset : offset + 4] = target


def draw_disc(
    image: list[bytearray],
    center_x: float,
    center_y: float,
    radius: float,
    color: tuple[int, int, int, int],
) -> None:
    """Draw a filled disc."""
    for y in range(max(0, int(center_y - radius)), min(CANVAS, int(center_y + radius) + 1)):
        span = math.sqrt(max(0, radius * radius - (y - center_y) ** 2))
        for x in range(max(0, int(center_x - span)), min(CANVAS, int(center_x + span) + 1)):
            set_pixel(image, x, y, color)


def draw_line(
    image: list[bytearray],
    start: tuple[float, float],
    end: tuple[float, float],
    width: float,
    color: tuple[int, int, int, int],
) -> None:
    """Draw a rounded line from overlapping discs."""
    distance = math.dist(start, end)
    steps = max(1, math.ceil(distance / max(1, width / 3)))
    for step in range(steps + 1):
        amount = step / steps
        x = start[0] + (end[0] - start[0]) * amount
        y = start[1] + (end[1] - start[1]) * amount
        draw_disc(image, x, y, width / 2, color)


def png_chunk(kind: bytes, data: bytes) -> bytes:
    """Build one PNG chunk."""
    return (
        struct.pack(">I", len(data))
        + kind
        + data
        + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    )


def write_png(image: list[bytearray]) -> None:
    """Downsample the antialiased canvas and write a PNG."""
    rows = []
    for output_y in range(SIZE):
        row = bytearray([0])
        for output_x in range(SIZE):
            totals = [0, 0, 0, 0]
            for sample_y in range(output_y * SCALE, (output_y + 1) * SCALE):
                for sample_x in range(output_x * SCALE, (output_x + 1) * SCALE):
                    offset = sample_x * 4
                    for channel in range(4):
                        totals[channel] += image[sample_y][offset + channel]
            row.extend(round(total / (SCALE * SCALE)) for total in totals)
        rows.append(bytes(row))

    header = struct.pack(">IIBBBBB", SIZE, SIZE, 8, 6, 0, 0, 0)
    payload = zlib.compress(b"".join(rows), level=9)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", header)
        + png_chunk(b"IDAT", payload)
        + png_chunk(b"IEND", b"")
    )


def main() -> None:
    """Render a four-color live waveform surrounding a central sparkle."""
    image = [bytearray(CANVAS * 4) for _ in range(CANVAS)]
    colors = [
        (66, 133, 244, 255),
        (52, 168, 83, 255),
        (251, 188, 4, 255),
        (234, 67, 53, 255),
    ]

    center = CANVAS / 2
    ring_radius = 378
    ring_width = 62
    segments = 132
    for segment in range(segments):
        angle_a = -math.pi / 2 + segment * math.tau / segments
        angle_b = -math.pi / 2 + (segment + 0.72) * math.tau / segments
        color_position = (segment / segments) * len(colors)
        color_index = int(color_position) % len(colors)
        next_color = colors[(color_index + 1) % len(colors)]
        mix = color_position - int(color_position)
        color = tuple(
            round(colors[color_index][channel] * (1 - mix) + next_color[channel] * mix)
            for channel in range(4)
        )
        start = (
            center + math.cos(angle_a) * ring_radius,
            center + math.sin(angle_a) * ring_radius,
        )
        end = (
            center + math.cos(angle_b) * ring_radius,
            center + math.sin(angle_b) * ring_radius,
        )
        draw_line(image, start, end, ring_width, color)

    waveform = [
        (-250, 0, 44),
        (-190, 0, 44),
        (-145, -88, 44),
        (-80, 112, 44),
        (0, -165, 44),
        (80, 112, 44),
        (145, -88, 44),
        (190, 0, 44),
        (250, 0, 44),
    ]
    points = [(center + x, center + y) for x, y, _width in waveform]
    for index, (start, end) in enumerate(zip(points, points[1:])):
        draw_line(image, start, end, waveform[index][2], (255, 255, 255, 255))

    write_png(image)
    print(f"Generated {OUTPUT}")


if __name__ == "__main__":
    main()
