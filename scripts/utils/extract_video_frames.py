import cv2
import os
import argparse
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description="Extract frames from a video and save them into Step1-Step6 folders.")
    parser.add_argument("video_path", help="Path to the input video file (e.g., my_video.mp4)")
    parser.add_argument("session_dir", help="Path to the session directory (e.g., data/sessions/session_003_...)")
    args = parser.parse_args()

    video_path = args.video_path
    session_dir = args.session_dir

    if not os.path.exists(video_path):
        print(f"[ERROR] Video not found: {video_path}")
        return

    # Create Step folders
    for i in range(1, 7):
        os.makedirs(os.path.join(session_dir, f"Step{i}"), exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30 # Default if unable to read

    window_name = "Video Frame Extractor (Press 1-6 to save, Space to pause/play, A/D to seek, Q to quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    paused = True
    print("=" * 60)
    print("  Video Frame Extractor")
    print("=" * 60)
    print("Controls:")
    print("  1-6   : Save current frame to Step1 - Step6")
    print("  Space : Pause / Play video")
    print("  D / ] : Skip forward 3 seconds")
    print("  A / [ : Skip backward 3 seconds")
    print("  Q     : Quit")
    print("=" * 60)

    # Read first frame
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Video is empty or cannot be read.")
        return

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] Reached end of video.")
                break
        
        display_frame = frame.copy()
        
        # Add UI text
        status_text = "PAUSED" if paused else "PLAYING"
        cv2.putText(display_frame, f"Status: {status_text}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(display_frame, "Press 1-6 to Save. Q to Quit", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow(window_name, display_frame)

        # Wait for key press (0 if paused to wait indefinitely, else normal wait based on fps)
        delay = 0 if paused else int(1000 / fps)
        key = cv2.waitKey(delay) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
        elif key in [ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6')]:
            step_num = chr(key)
            step_dir = os.path.join(session_dir, f"Step{step_num}")
            
            # Use video filename as prefix to make tracing easier
            video_basename = os.path.splitext(os.path.basename(video_path))[0]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
            save_path = os.path.join(step_dir, f"{video_basename}_{timestamp}.jpg")
            
            # Save the ORIGINAL frame, not the one with UI text
            cv2.imwrite(save_path, frame)
            print(f"[SAVED] Extracted frame to {save_path}")
            
            # Briefly show a flash or text to confirm save
            flash_frame = frame.copy()
            cv2.putText(flash_frame, f"SAVED TO STEP {step_num}!", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
            cv2.imshow(window_name, flash_frame)
            cv2.waitKey(300) # Show flash for 300ms

        elif key == ord('d') or key == ord(']'):
            # Skip forward 3 seconds
            current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame + (fps * 3))
            ret, frame = cap.read() # Read the new frame
        elif key == ord('a') or key == ord('['):
            # Skip backward 3 seconds
            current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            new_frame = max(0, current_frame - (fps * 3))
            cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
            ret, frame = cap.read() # Read the new frame

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Extraction complete.")

if __name__ == "__main__":
    main()
