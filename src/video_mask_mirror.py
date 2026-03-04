import cv2
import numpy as np
import sys
import os

def mask_mirror_video(video_path, output_path, mask_protected=None, mask_source=None, source_side=None):
    """
    Two-step masking and mirroring tool.
    
    Step 1: Select a "Protected Region" (remains unchanged).
    Step 2: Select a "Source Region" (from one half of the screen).
            This source region is mirrored and overlaid onto the OTHER half.
    
    Args:
        video_path (str): Path to input video.
        output_path (str): Path to save output video.
        mask_protected (np.array, optional): Boolean mask for protected region.
        mask_source (np.array, optional): Boolean mask for source region.
        source_side (str, optional): 'left' or 'right'.
    """
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return False

    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read video frame")
        cap.release()
        return False
        
    height, width = frame.shape[:2]
    mid_x = width // 2
    
    is_interactive = (mask_protected is None or mask_source is None or source_side is None)

    if is_interactive:
        # Check for display environment
        if sys.platform.startswith('linux') and not os.environ.get('DISPLAY'):
            print("Error: This tool requires a graphical display (GUI) which is not available in this environment.")
            print("Please provide masks explicitly for headless operation.")
            cap.release()
            return False

        # --- Helper Function for Drawing Masks ---
        def get_user_mask(window_name, prompt, restriction=None):
            """
            restriction: None, 'left', or 'right'
            """
            drawing = False
            pts = []
            
            def draw_polygon(event, x, y, flags, param):
                nonlocal drawing, pts
                if event == cv2.EVENT_LBUTTONDOWN:
                    drawing = True
                    pts.append((x, y))
                elif event == cv2.EVENT_MOUSEMOVE:
                    if drawing:
                        pts.append((x, y))
                elif event == cv2.EVENT_LBUTTONUP:
                    drawing = False
                    pts.append((x, y))

            cv2.namedWindow(window_name)
            cv2.setMouseCallback(window_name, draw_polygon)
            
            print(f"\n--- {window_name} ---")
            print(prompt)
            print("Draw (click & drag). Press 'c' to confirm, 'r' to reset, 'q' to quit.")

            mask = None
            while True:
                display_frame = frame.copy()
                
                # Draw restriction line
                if restriction == 'left':
                    cv2.line(display_frame, (mid_x, 0), (mid_x, height), (0, 0, 255), 2)
                    cv2.putText(display_frame, "DRAW HERE ->", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(display_frame, "<- NO DRAW", (mid_x + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                elif restriction == 'right':
                    cv2.line(display_frame, (mid_x, 0), (mid_x, height), (0, 0, 255), 2)
                    cv2.putText(display_frame, "NO DRAW ->", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.putText(display_frame, "<- DRAW HERE", (mid_x + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                if len(pts) > 0:
                    pts_np = np.array(pts, np.int32)
                    pts_np = pts_np.reshape((-1, 1, 2))
                    cv2.polylines(display_frame, [pts_np], False, (0, 255, 0), 2)
                
                cv2.imshow(window_name, display_frame)
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('c'):
                    if len(pts) > 2:
                        temp_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                        pts_np = np.array(pts, np.int32)
                        cv2.fillPoly(temp_mask, [pts_np], 255)
                        
                        # Apply restriction
                        if restriction == 'left':
                            temp_mask[:, mid_x:] = 0 # Clear right side
                        elif restriction == 'right':
                            temp_mask[:, :mid_x] = 0 # Clear left side
                            
                        mask = temp_mask
                        break
                    else:
                        if "Protected" in window_name:
                             mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                             break
                        print("Please draw a valid region.")
                elif key == ord('r'):
                    pts = []
                elif key == ord('q'):
                    cv2.destroyAllWindows()
                    return None
            
            cv2.destroyAllWindows()
            return mask

        # --- Step 1: Select Protected Region ---
        if mask_protected is None:
            print("\nStep 1: Select Protected Region (Area that will NOT be changed).")
            mask_protected = get_user_mask('Step 1: Protected Region', 
                                          "Draw area to PROTECT from changes.")
            if mask_protected is None:
                cap.release()
                return False

        # --- Step 2: Select Source Side ---
        if source_side is None:
            print("\nStep 2: Select Source Side.")
            print("Which side contains the object you want to mirror?")
            print("1. Left (will mirror to Right)")
            print("2. Right (will mirror to Left)")
            while True:
                choice = input("Enter 1 or 2: ").strip()
                if choice == '1':
                    source_side = 'left'
                    break
                elif choice == '2':
                    source_side = 'right'
                    break
                print("Invalid choice.")

        # --- Step 3: Select Source Region ---
        if mask_source is None:
            print(f"\nStep 3: Select Source Region (Draw on the {source_side} side).")
            mask_source = get_user_mask('Step 3: Source Region', 
                                       f"Draw the object on the {source_side.upper()} side to mirror.",
                                       restriction=source_side)
            if mask_source is None:
                cap.release()
                return False
                
    # --- Processing ---
    print("\nProcessing video...")
    
    # Calculate the Mirror Mask (Where the mirrored content will go)
    mask_mirror_dest = cv2.flip(mask_source, 1)
    
    # Remove Protected Area from Destination Mask
    # We only overwrite pixels that are in Destination Mask AND NOT in Protected Mask
    mask_effective = cv2.bitwise_and(mask_mirror_dest, cv2.bitwise_not(mask_protected))
    
    # Invert for background
    mask_effective_inv = cv2.bitwise_not(mask_effective)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 1. Get Source Content
        # We want the content from 'mask_source' region flipped.
        # So we flip the whole frame first.
        frame_flipped = cv2.flip(frame, 1)
        
        # 2. Compose
        # Background: Original frame (where mask_effective is 0)
        bg = cv2.bitwise_and(frame, frame, mask=mask_effective_inv)
        
        # Foreground: Flipped frame (where mask_effective is 1)
        # Note: frame_flipped correctly maps pixels from Left->Right or Right->Left
        fg = cv2.bitwise_and(frame_flipped, frame_flipped, mask=mask_effective)
        
        output_frame = cv2.add(bg, fg)
        
        out.write(output_frame)
        frame_count += 1
        
        if frame_count % 30 == 0:
            print(f"Processing frame {frame_count}...", end='\r')

    cap.release()
    out.release()
    print(f"\nMasked mirroring complete: {output_path}")
    return True
