# src/image_mirror.py
import cv2
from pathlib import Path

def mirror_image_file(input_path: str, output_path: str, lossless: bool = True):
    """
    水平镜像单张图片，保证无损输出（默认保存为PNG）。
    :param input_path:  输入图片路径
    :param output_path: 输出图片路径
    :param lossless:    True 则强制输出为PNG格式（无损），False 则按原格式保存
    """
    img = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)  # 保留原始通道（含alpha）
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {input_path}")

    # 水平翻转（镜像），0表示垂直翻转，1表示水平翻转，-1表示同时
    mirrored = cv2.flip(img, 1)

    if lossless:
        # 确保输出为PNG无损格式
        out = Path(output_path)
        out = out.with_suffix('.png') if out.suffix.lower() not in ['.png'] else out
        cv2.imwrite(str(out), mirrored, [cv2.IMWRITE_PNG_COMPRESSION, 0])  # 0 为无压缩但无损
    else:
        cv2.imwrite(str(output_path), mirrored)

def mirror_image_dir(input_dir: str, output_dir: str, pattern: str = "*.jpg", lossless: bool = True):
    """
    对目录下所有匹配的图片执行水平镜像，保持原文件名，输出到另一目录。
    :param input_dir:  输入图片文件夹
    :param output_dir: 输出图片文件夹
    :param pattern:    glob 匹配模式，如 "*.jpg", "frame_*.png"
    :param lossless:   是否无损输出
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for img_file in sorted(input_path.glob(pattern)):
        in_file = str(img_file)
        out_file = str(output_path / img_file.name)
        mirror_image_file(in_file, out_file, lossless)