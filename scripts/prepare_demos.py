from pathlib import Path

from PIL import Image, ImageSequence


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "gif_output_分视角_五段" / "mid"
OUTPUT = Path(__file__).resolve().parents[1] / "public" / "media" / "demos"

DEMOS = {
    "bowl-placement.gif": "bowl_basket_right_mirror_mid_part_03.gif",
    "towel-storage.gif": "towel_basket_right_mirror_mid_part_03.gif",
    "flower-arrangement.gif": "flower_arrange_right_mirror_mid_part_03.gif",
    "collect-cups-1.gif": "collect_cups_1_mid_part_03.gif",
    "collect-cups-2.gif": "collect_cups_2_mid_part_03.gif",
    "collect-cups-3.gif": "collect_cups_3_mid_part_03.gif",
    "cup-collection.gif": "collect_cups_both_mid_part_03.gif",
    "sandwich-making.gif": "make_sandwich_dual_mid_part_03.gif",
}


def optimize_gif(source: Path, destination: Path) -> None:
    image = Image.open(source)
    original_duration = image.info.get("duration", 30)
    sample_step = 3
    frames = []

    for index, frame in enumerate(ImageSequence.Iterator(image)):
        if index % sample_step:
            continue
        converted = frame.convert("RGB")
        if converted.width > 360:
            height = round(converted.height * 360 / converted.width)
            converted = converted.resize((360, height), Image.Resampling.LANCZOS)
        frames.append(converted.quantize(colors=128, method=Image.Quantize.MEDIANCUT))

    if not frames:
        raise ValueError(f"No frames extracted from {source}")

    frames[0].save(
        destination,
        save_all=True,
        append_images=frames[1:],
        duration=original_duration * sample_step,
        loop=0,
        optimize=True,
        disposal=2,
    )


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for output_name, source_name in DEMOS.items():
        optimize_gif(SOURCE / source_name, OUTPUT / output_name)


if __name__ == "__main__":
    main()
