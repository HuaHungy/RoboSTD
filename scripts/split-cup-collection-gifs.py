from __future__ import annotations

import argparse
from pathlib import Path


def skip_subblocks(data: bytearray, position: int) -> int:
    while position < len(data):
        size = data[position]
        position += 1
        if size == 0:
            return position
        position += size
    raise ValueError("Truncated GIF data")


def read_frame_chunks(source: Path) -> tuple[bytes, list[bytes]]:
    data = bytearray(source.read_bytes())
    if data[:6] not in (b"GIF87a", b"GIF89a"):
        raise ValueError(f"Not a GIF file: {source}")

    position = 6
    packed = data[position + 4]
    position += 7
    if packed & 0x80:
        position += 3 * (2 ** ((packed & 0x07) + 1))

    frame_chunks: list[bytes] = []
    pending_start = position
    frame_start: int | None = None
    first_frame_start: int | None = None

    while position < len(data):
        marker = data[position]
        if marker == 0x3B:
            break
        if marker == 0x21:
            label = data[position + 1]
            extension_start = position
            if label == 0xF9:
                if data[position + 2] != 4:
                    raise ValueError(f"Unexpected Graphic Control Extension: {source}")
                position += 8
                frame_start = extension_start
            else:
                position = skip_subblocks(data, position + 2)
            continue
        if marker == 0x2C:
            local_packed = data[position + 9]
            position += 10
            if local_packed & 0x80:
                position += 3 * (2 ** ((local_packed & 0x07) + 1))
            position += 1
            image_end = skip_subblocks(data, position)
            chunk_start = frame_start if frame_start is not None else pending_start
            if first_frame_start is None:
                first_frame_start = chunk_start
            frame_chunks.append(bytes(data[chunk_start:image_end]))
            pending_start = image_end
            frame_start = None
            position = image_end
            continue
        raise ValueError(f"Unexpected GIF block 0x{marker:02x}: {source}")

    if not frame_chunks:
        raise ValueError(f"No image frames found in {source}")
    assert first_frame_start is not None
    prefix = bytes(data[:first_frame_start])
    return prefix, frame_chunks


def split_gif(source: Path, destination: Path, parts: int = 3) -> None:
    prefix, frames = read_frame_chunks(source)
    base, remainder = divmod(len(frames), parts)
    destination.mkdir(parents=True, exist_ok=True)
    start = 0
    for part in range(1, parts + 1):
        length = base + (1 if part <= remainder else 0)
        output = destination / f"{source.stem}-{part}.gif"
        output.write_bytes(prefix + b"".join(frames[start : start + length]) + b"\x3b")
        print(f"{source.name} -> {output.name} ({length} frames)", flush=True)
        start += length


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    for source in sorted(args.source.glob("cup-collection-*.gif")):
        split_gif(source, args.destination)


if __name__ == "__main__":
    main()
