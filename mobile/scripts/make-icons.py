"""Generates the Soundalike app icons.

The mark matches the web app's brand: a diamond glyph on a 135 degree
green to violet gradient, the same gradient used by the .logo tile in
webapp/index.html.
"""

from __future__ import annotations

import pathlib

from PIL import Image, ImageDraw

GREEN = (29, 185, 84)
VIOLET = (139, 108, 255)
INK = (10, 13, 18)

ASSETS = pathlib.Path(__file__).resolve().parent.parent / "assets"
SS = 4  # supersample factor for clean edges


def gradient(size: int, visible: float = 1.0) -> Image.Image:
    """135 degree linear gradient, green at top left to violet at bottom right.

    `visible` is the fraction of the image that survives masking. Android crops
    an adaptive icon to its centre, so the ramp is widened to keep both brand
    colours inside the part people actually see.
    """
    base = Image.new("RGB", (size, size))
    pixels = base.load()
    span = (size - 1) * visible
    start = (size - 1 - span) / 2
    for y in range(size):
        for x in range(size):
            t = ((x + y) / 2 - start) / span if span else 0.0
            t = min(1.0, max(0.0, t))
            pixels[x, y] = (
                round(GREEN[0] + (VIOLET[0] - GREEN[0]) * t),
                round(GREEN[1] + (VIOLET[1] - GREEN[1]) * t),
                round(GREEN[2] + (VIOLET[2] - GREEN[2]) * t),
            )
    return base


def diamond(cx: float, cy: float, radius: float) -> list[tuple[float, float]]:
    return [(cx, cy - radius), (cx + radius, cy), (cx, cy + radius), (cx - radius, cy)]


def draw_mark(size: int, colour: tuple[int, int, int], scale: float) -> Image.Image:
    """The diamond-in-diamond glyph, drawn as geometry so it needs no font."""
    big = size * SS
    layer = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    pen = ImageDraw.Draw(layer)

    centre = big / 2
    outer = big * scale / 2
    stroke = outer * 0.17

    pen.polygon(diamond(centre, centre, outer), fill=colour + (255,))
    pen.polygon(diamond(centre, centre, outer - stroke), fill=(0, 0, 0, 0))
    pen.polygon(diamond(centre, centre, outer * 0.42), fill=colour + (255,))

    return layer.resize((size, size), Image.LANCZOS)


def write(image: Image.Image, name: str) -> None:
    path = ASSETS / name
    image.save(path, "PNG", optimize=True)
    print(f"{name:32} {path.stat().st_size / 1024:7.1f} KB  {image.size[0]}x{image.size[1]}")


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)

    # iOS and the store listing want a full-bleed square. iOS applies its own mask.
    icon = gradient(1024).convert("RGBA")
    icon.alpha_composite(draw_mark(1024, INK, 0.52))
    write(icon.convert("RGB"), "icon.png")

    # Android draws the foreground inside a safe circle, so the mark sits smaller.
    foreground = Image.new("RGBA", (1024, 1024), (0, 0, 0, 0))
    foreground.alpha_composite(draw_mark(1024, INK, 0.34))
    write(foreground, "android-icon-foreground.png")

    write(gradient(1024, visible=72 / 108).convert("RGB"), "android-icon-background.png")

    # Themed icons are tinted by the system, so this is a plain silhouette.
    write(draw_mark(1024, (255, 255, 255), 0.34), "android-icon-monochrome.png")

    splash = Image.new("RGBA", (1024, 1024), (0, 0, 0, 0))
    splash.alpha_composite(draw_mark(1024, (255, 255, 255), 0.44))
    write(splash, "splash-icon.png")

    favicon = gradient(96).convert("RGBA")
    favicon.alpha_composite(draw_mark(96, INK, 0.56))
    write(favicon.convert("RGB"), "favicon.png")


if __name__ == "__main__":
    main()
