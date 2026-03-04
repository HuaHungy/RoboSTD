import cv2
import numpy as np
import os
import sys

def crop_video_advanced(video_path, output_path, roi=None):
    """
    Allows user to select a region to crop using mouse drag.
    The selected region is kept, and the rest is filled with white background.
    
    Args:
        video_path (str): Input video path.
        output_path (str): Output video path.
        roi (tuple, optional): (x, y, w, h). If provided, skips GUI selection.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return False

    # Read first frame to get dimensions or for selection
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read video frame")
        cap.release()
        return False

    if roi is None:
        # Check for display environment (GUI requirement)
        if sys.platform.startswith('linux') and not os.environ.get('DISPLAY'):
            print("Error: This tool requires a graphical display (GUI) which is not available in this environment.")
            print("Please provide 'roi' argument explicitly for headless operation.")
            cap.release()
            return False

        # Variables for mouse callback
        drawing = False
        ix, iy = -1, -1
        selection = None # (x, y, w, h)

        def draw_rectangle(event, x, y, flags, param):
            nonlocal drawing, ix, iy, selection
            if event == cv2.EVENT_LBUTTONDOWN:
                drawing = True
                ix, iy = x, y
            elif event == cv2.EVENT_MOUSEMOVE:
                if drawing:
                    w = x - ix
                    h = y - iy
                    selection = (ix, iy, w, h)
            elif event == cv2.EVENT_LBUTTONUP:
                drawing = False
                w = x - ix
                h = y - iy
                selection = (ix, iy, w, h)

        cv2.namedWindow('Select Region')
        cv2.setMouseCallback('Select Region', draw_rectangle)

        print("Draw a rectangle with your mouse. Press 'c' to confirm, 'r' to reset, 'q' to quit.")

        while True:
            display_frame = frame.copy()
            if selection:
                x, y, w, h = selection
                # Ensure coordinates are ordered correctly (handling negative width/height)
                x_start = min(x, x + w)
                y_start = min(y, y + h)
                x_end = max(x, x + w)
                y_end = max(y, y + h)
                cv2.rectangle(display_frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
            
            cv2.imshow('Select Region', display_frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('c'): # Confirm
                if selection:
                    roi = selection
                    break
                else:
                    print("Please select a region first.")
            elif key == ord('r'): # Reset
                selection = None
                drawing = False
            elif key == ord('q'): # Quit
                cap.release()
                cv2.destroyAllWindows()
                return False
        
        cv2.destroyAllWindows()
    
    # Process video with ROI
    if not roi:
        print("No region selected.")
        cap.release()
        return False
        
    x, y, w, h = roi
    # Normalize ROI
    x_start = max(0, min(x, x + w))
    y_start = max(0, min(y, y + h))
    x_end = min(frame.shape[1], max(x, x + w))
    y_end = min(frame.shape[0], max(y, y + h))
    
    # Check if ROI is valid
    if x_end <= x_start or y_end <= y_start:
        print("Invalid region selected.")
        cap.release()
        return False

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Reset to beginning
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
        
        white_bg = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Copy ROI
        roi_frame = frame[y_start:y_end, x_start:x_end]
        white_bg[y_start:y_end, x_start:x_end] = roi_frame
        
        out.write(white_bg)
        frame_count += 1
        
        if frame_count % 30 == 0:
            print(f"Processing frame {frame_count}...", end='\r')

    cap.release()
    out.release()
    print(f"\nAdvanced cropping complete: {output_path}")
    return True
