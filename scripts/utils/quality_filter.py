import cv2
import os
import argparse
import glob
import shutil
from PIL import Image
import imagehash
import numpy as np

def is_blurry(image, threshold=2.0):
    """
    Detect if an image is corrupted, black, or severely blurry using Variance of Laplacian.
    
    Args:
        image: BGR numpy image array.
        threshold: Variance of Laplacian threshold (default: 2.0). Indoor matte origami paper
                   naturally scores between 5.0 and 25.0 due to smooth white surfaces; threshold=2.0
                   safely rejects only empty/black or completely corrupted frames without discarding
                   valid paper folding shots.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    return fm < threshold, fm

def filter_quality(session_dir, blur_threshold=2.0, hash_threshold=5):
    """Filter blur and duplicates, moving them to Rejected folder."""
    rejected_base = os.path.join(session_dir, "Rejected")
    
    total_processed = 0
    total_blurred = 0
    total_dupes = 0
    
    for step in range(1, 7):
        step_name = f"Step{step}"
        step_dir = os.path.join(session_dir, step_name)
        
        if not os.path.exists(step_dir):
            continue
            
        rejected_step_dir = os.path.join(rejected_base, step_name)
        os.makedirs(rejected_step_dir, exist_ok=True)
        
        images = glob.glob(os.path.join(step_dir, "*.jpg")) + glob.glob(os.path.join(step_dir, "*.png"))
        
        # Sort images to process in order (useful for hashing sequential frames)
        images.sort()
        
        previous_hash = None
        
        for img_path in images:
            total_processed += 1
            filename = os.path.basename(img_path)
            rejected_path = os.path.join(rejected_step_dir, filename)
            
            # Read for blur detection
            cv_img = cv2.imread(img_path)
            if cv_img is None:
                # Fix Bug: Move corrupted files instead of leaving them in the dataset
                shutil.move(img_path, rejected_path)
                print(f"[CORRUPT] Moved {filename} (OpenCV could not read)")
                total_blurred += 1 # Count as discarded
                continue
                
            blurry, fm = is_blurry(cv_img, blur_threshold)
            if blurry:
                shutil.move(img_path, rejected_path)
                print(f"[BLUR] Moved {filename} (score: {fm:.2f})")
                total_blurred += 1
                continue
                
            # Read for hash detection
            current_hash = None
            try:
                # Fix Bug: Use context manager to prevent Windows PermissionError (File Lock)
                with Image.open(img_path) as pil_img:
                    current_hash = imagehash.phash(pil_img)
            except Exception as e:
                # Fix Bug: Move corrupted files that OpenCV missed but PIL caught
                shutil.move(img_path, rejected_path)
                print(f"[CORRUPT] Moved {filename} (PIL Error: {e})")
                total_blurred += 1 # Count as discarded
                continue
                
            if previous_hash is not None and current_hash is not None:
                hash_diff = current_hash - previous_hash
                if hash_diff < hash_threshold:
                    # Move to rejected because it's too similar to the previous frame
                    shutil.move(img_path, rejected_path)
                    print(f"[DUPE] Moved {filename} (diff: {hash_diff})")
                    total_dupes += 1
                    continue
                    
            # If it passes both, update previous_hash
            previous_hash = current_hash

    print("\n" + "="*40)
    print("Quality Filter Summary")
    print("="*40)
    print(f"Total images processed : {total_processed}")
    print(f"Removed (Blurry)       : {total_blurred}")
    print(f"Removed (Duplicates)   : {total_dupes}")
    print(f"Remaining              : {total_processed - total_blurred - total_dupes}")
    print("="*40)

def main():
    parser = argparse.ArgumentParser(description="Filter blurry and duplicate images.")
    parser.add_argument("--session-dir", required=True, help="Path to session directory")
    parser.add_argument("--blur-threshold", type=float, default=2.0, help="Laplacian variance threshold (default: 2.0)")
    parser.add_argument("--hash-threshold", type=int, default=5, help="Hamming distance threshold for pHash (default: 5)")
    args = parser.parse_args()

    if not os.path.exists(args.session_dir):
        print(f"[ERROR] Directory not found: {args.session_dir}")
        return

    filter_quality(args.session_dir, args.blur_threshold, args.hash_threshold)

if __name__ == "__main__":
    main()
