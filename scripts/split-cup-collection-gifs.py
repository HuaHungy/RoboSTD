from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageSequence


def half_delay(duration: int) -> int:
    """Return a GIF delay for 2x playback, in milliseconds."""
    return max(10, (max(10, duration // 2) // 10) * 10)


def split_gif(source: Path, destination: Path, parts: int = 3) -> None:
    # GIF frames may be delta frames. Reading each frame through PIL and
    # copying the converted image materializes the complete composited canvas.
    # This is essential when a segment starts in the middle of the animation.
    with Image.open(source) as gif:
        frames: list[Image.Image] = []
        durations: list[int] = []
        for frame in ImageSequence.Iterator(gif):
            frames.append(frame.convert("RGB").copy())
            durations.append(half_delay(int(frame.info.get("duration", 30))))

    base, remainder = divmod(len(frames), parts)
    destination.mkdir(parents=True, exist_ok=True)
    start = 0
    for part in range(1, parts + 1):
        length = base + (1 if part <= remainder else 0)
        segment_frames = frames[start : start + length]
        segment_durations = durations[start : start + length]
        output = destination / f"{source.stem}-{part}.gif"
        first, *rest = segment_frames
        first.save(
            output,
            format="GIF",
            save_all=True,
            append_images=rest,
            duration=segment_durations,
            loop=0,
            optimize=True,
            disposal=2,
        )
        print(f"{source.name} -> {output.name} ({length} composited frames, 2x speed)", flush=True)
        start += length


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    for source in sorted(args.source.glob("collect_cups_*.gif")):
        split_gif(source, args.destination)


if __name__ == "__main__":
    main()
