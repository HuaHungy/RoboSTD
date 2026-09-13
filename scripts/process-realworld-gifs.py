from __future__ import annotations

import argparse
from pathlib import Path


def process_gif(source: Path, destination: Path) -> None:
    """Copy a GIF while halving every Graphic Control Extension delay.

    Editing the GIF timing bytes directly keeps every original frame intact,
    so the robot motion remains smooth and no re-encoding artifacts or frame
    drops are introduced. GIF delays are stored in centiseconds.
    """
    data = bytearray(source.read_bytes())
    if data[:6] not in (b"GIF87a", b"GIF89a"):
        raise ValueError(f"Not a GIF file: {source}")

    def skip_subblocks(position: int) -> int:
        while position < len(data):
            size = data[position]
            position += 1
            if size == 0:
                return position
            position += size
        raise ValueError(f"Truncated GIF sub-blocks: {source}")

    # Logical Screen Descriptor, followed by an optional global color table.
    position = 6
    packed = data[position + 4]
    position += 7
    if packed & 0x80:
        position += 3 * (2 ** ((packed & 0x07) + 1))

    changed = 0
    while position < len(data):
        marker = data[position]
        if marker == 0x3B:  # Trailer.
            break
        if marker == 0x21:  # Extension.
            label = data[position + 1]
            if label == 0xF9:  # Graphic Control Extension: real frame timing.
                if data[position + 2] != 4:
                    raise ValueError(f"Unexpected Graphic Control Extension: {source}")
                delay_offset = position + 4
                delay = data[delay_offset] | (data[delay_offset + 1] << 8)
                new_delay = max(1, delay // 2)
                data[delay_offset] = new_delay & 0xff
                data[delay_offset + 1] = (new_delay >> 8) & 0xff
                changed += 1
                position += 8  # introducer, label, size, payload, terminator
            else:
                position = skip_subblocks(position + 2)
            continue
        if marker == 0x2C:  # Image descriptor.
            local_packed = data[position + 9]
            position += 10
            if local_packed & 0x80:
                position += 3 * (2 ** ((local_packed & 0x07) + 1))
            position += 1  # LZW minimum code size.
            position = skip_subblocks(position)
            continue
        raise ValueError(f"Unexpected GIF block 0x{marker:02x}: {source}")

    if changed == 0:
        raise ValueError(f"No GIF frame timing blocks found in {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)
    print(f"{source.name} -> {destination.name} ({changed} frames, 2x speed)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()

    files = sorted(args.source.glob("*.gif"))
    if not files:
        raise SystemExit(f"No GIF files found in {args.source}")
    for source in files:
        process_gif(source, args.destination / source.name)


if __name__ == "__main__":
    main()
