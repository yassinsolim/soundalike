"""Draws the Soundalike app icon: two sound waves stacked into an "≈" (sounds alike),
the lower one an echo of the upper, on a violet-to-coral gradient.

python3 make_icon.py  ->  Assets.xcassets/AppIcon.appiconset/icon.png (1024 px)
"""
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

S = 4096  # drawn at 4x, downsampled for clean edges
OUT = Path(__file__).parent / "Assets.xcassets/AppIcon.appiconset/icon.png"


def lerp(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def gradient():
    top_left, middle, bottom_right = (18, 10, 52), (78, 26, 150), (168, 44, 190)
    small = Image.new("RGB", (256, 256))
    px = small.load()
    for y in range(256):
        for x in range(256):
            t = (x * 0.45 + y * 0.55) / 255
            px[x, y] = lerp(top_left, middle, t / 0.55) if t < 0.55 else lerp(middle, bottom_right, (t - 0.55) / 0.45)
    return small.resize((S, S), Image.BICUBIC)


def wave(center_y, amplitude, phase, cycles=1.6, margin=0.17):
    points = []
    left, right = S * margin, S * (1 - margin)
    for i in range(1201):
        t = i / 1200
        x = left + (right - left) * t
        # Taper the ends so each wave fades in and out like a sound.
        envelope = math.sin(math.pi * t) ** 0.8
        y = center_y + amplitude * envelope * math.sin(2 * math.pi * cycles * t + phase)
        points.append((x, y))
    return points


def stroke(points, width, color):
    """A round-capped stroke, stamped as overlapping discs so wide curves stay smooth."""
    mask = Image.new("L", (S, S), 0)
    draw = ImageDraw.Draw(mask)
    r = width / 2
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        steps = max(1, int(math.hypot(x1 - x0, y1 - y0) / 4))
        for k in range(steps + 1):
            x, y = x0 + (x1 - x0) * k / steps, y0 + (y1 - y0) * k / steps
            draw.ellipse((x - r, y - r, x + r, y + r), fill=255)
    layer = Image.new("RGBA", (S, S), color[:3] + (0,))
    layer.putalpha(mask.point(lambda v: v * color[3] // 255))
    return layer


def main():
    icon = gradient().convert("RGBA")

    # A violet bloom behind the waves gives depth without greying the corners.
    bloom = Image.new("L", (S, S), 0)
    ImageDraw.Draw(bloom).ellipse((S * 0.12, S * 0.2, S * 0.88, S * 0.82), fill=255)
    bloom = bloom.filter(ImageFilter.GaussianBlur(S * 0.16)).point(lambda v: v * 120 // 255)
    icon = Image.composite(Image.new("RGBA", (S, S), (150, 70, 235, 255)), icon, bloom)

    upper = wave(S * 0.43, S * 0.085, 0)
    lower = wave(S * 0.60, S * 0.085, 0)
    width = int(S * 0.062)

    # The echo: the same wave, lower, in warm coral fading to pink across its length.
    echo_mask = stroke(lower, width, (255, 255, 255, 255)).getchannel("A")
    warm = Image.linear_gradient("L").rotate(90).resize((S, S))
    coral = Image.merge("RGB", (
        warm.point(lambda v: 255),
        warm.point(lambda v: 96 + (v * 40) // 255),
        warm.point(lambda v: 86 + (v * 90) // 255),
    )).convert("RGBA")
    coral.putalpha(echo_mask)
    halo = Image.new("RGBA", (S, S), (255, 120, 170, 0))
    halo.putalpha(echo_mask.filter(ImageFilter.GaussianBlur(S * 0.03)).point(lambda v: v * 110 // 255))
    # A soft glow and shadow lift the main wave off the background.
    main_alpha = stroke(upper, width, (255, 255, 255, 255)).getchannel("A")
    glow_layer = Image.new("RGBA", (S, S), (200, 170, 255, 0))
    glow_layer.putalpha(main_alpha.filter(ImageFilter.GaussianBlur(S * 0.035)).point(lambda v: v * 120 // 255))
    main_wave = stroke(upper, width, (255, 255, 255, 255))

    for layer in (halo, coral, glow_layer, main_wave):
        icon = Image.alpha_composite(icon, layer)

    icon.convert("RGB").resize((1024, 1024), Image.LANCZOS).save(OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
