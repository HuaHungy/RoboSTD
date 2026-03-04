import cv2
import numpy as np

def reverse_video(input_path, output_path):
    """
    Reverses the video playback direction.
    
    Args:
        input_path (str): Path to input video.
        output_path (str): Path to save output video.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_path}")
        return False

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Reading {total_frames} frames into memory... (Warning: Large videos may consume high RAM)")
    
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()

    if not frames:
        print("Error: No frames read.")
        return False

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Writing frames in reverse order...")
    for i, frame in enumerate(reversed(frames)):
        out.write(frame)
        if i % 30 == 0:
            print(f"Processing frame {i+1}/{len(frames)}...", end='\r')
            
    out.release()
    print(f"\nReverse complete: {output_path}")
    return True
