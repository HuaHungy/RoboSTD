import cv2

def mirror_video(video_path, output_path):
    """
    Mirrors the video along the vertical axis (left-right flip).
    
    Args:
        video_path (str): Path to input video.
        output_path (str): Path to save output video.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return False

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        mirrored_frame = cv2.flip(frame, 1)
        out.write(mirrored_frame)
        frame_count += 1
        
        if frame_count % 30 == 0:
            print(f"Processing frame {frame_count}...", end='\r')

    cap.release()
    out.release()
    print(f"\nMirroring complete: {output_path}")
    return True
