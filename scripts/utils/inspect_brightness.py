"""
inspect_brightness.py — Find overly bright and washed-out images
"""
import cv2
import numpy as np
from pathlib import Path

target = Path('data/background_candidates')
stats = []

for d in target.glob('*/*'):
    img_dir = d / 'images'
    if img_dir.exists():
        for p in img_dir.glob('*.jpg'):
            img = cv2.imread(str(p))
            if img is not None:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                mean_val = float(gray.mean())
                ov240 = float((gray >= 235).mean() * 100)
                ov250 = float((gray >= 248).mean() * 100)
                stats.append((mean_val, ov240, ov250, p))

stats.sort(key=lambda x: x[0], reverse=True)
print(f"Total images checked: {len(stats)}")
print("Top 40 Brightest Images:")
header = "Mean   | % >=235 | % >=248 | Path"
print(header)
print("-" * 80)
for m, ov, vov, p in stats[:40]:
    rel = str(p.relative_to(target)).replace('\\', '/')
    print(f"{m:6.1f} | {ov:7.1f}% | {vov:7.1f}% | {rel}")
