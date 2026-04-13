import cv2
import os
import sys

def extract_frames(video_path, start_frame=0):
    output_folder = "outputs/calibration_frames"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Define a CONSTANT window name so OpenCV reuses the same window
    win_name = "Frame Picker - Calibration"
    cv2.namedWindow(win_name)

    print(f"\nControls:\n [D] -> +1 Frame | [A] -> -1 Frame")
    print(f" [Right Arrow] -> +10 | [Left Arrow] -> -10\n [S] -> SAVE | [Q] -> Quit")

    while True:
        current_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        ret, frame = cap.read()
        
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
            continue

        # Resize for display
        show_width = 800
        show_height = int(frame.shape[0] * (show_width / frame.shape[1]))
        small_frame = cv2.resize(frame, (show_width, show_height))
        
        # Draw the frame info directly on the image so the window stays still
        info_text = f"Frame: {current_pos} / {total_frames}"
        cv2.putText(small_frame, info_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Show the frame in the constant window
        cv2.imshow(win_name, small_frame)
        
        # Reset pointer because read() moved it forward
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)

        key = cv2.waitKeyEx(0) # waitKeyEx handles arrow keys better on some systems

        if key == ord('d'):
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos + 1)
        elif key == ord('a'):
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, current_pos - 1))
        # Support for both standard and extended arrow key codes
        elif key in [ord('s'), 83, 65363]: # 's' or Right Arrow
            if key == ord('s'): # Manual check for 's' to save
                filename = f"{output_folder}/frame_{current_pos:04d}.png"
                cv2.imwrite(filename, frame)
                print(f"Saved: {filename}")
            else: # It was the right arrow
                cap.set(cv2.CAP_PROP_POS_FRAMES, min(total_frames - 1, current_pos + 10))
        elif key in [81, 65361]: # Left Arrow
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, current_pos - 10))
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 frame_picker.py <video_file> <start_frame>")
    else:
        start = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        extract_frames(sys.argv[1], start_frame=start)