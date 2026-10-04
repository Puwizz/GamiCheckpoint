"""
preprocessing.py — ฟังก์ชัน Image Preprocessing
================================================
รวมฟังก์ชันสำหรับ preprocess ภาพก่อนใส่ model
"""

import cv2
import numpy as np
from PIL import Image
from torchvision import transforms


DEFAULT_MARGIN_RATIO = 0.18
WHITE_PAPER_AREA_WEIGHT = 0.35
WHITE_PAPER_BRIGHTNESS_WEIGHT = 0.30
WHITE_PAPER_CENTER_WEIGHT = 0.35

WHITE_PAPER_PERCENTILE = 82
WHITE_PAPER_MIN_THRESH = 100
WHITE_PAPER_MAX_THRESH = 230
WHITE_PAPER_BRIGHTNESS_DIVISOR = 80.0


class AutoCropForeground(object):
    """
    Custom PyTorch Transform เพื่อครอปภาพบริเวณกระดาษให้อัตโนมัติระหว่างโหลด Dataset (On-the-fly)
    แก้ปัญหา Mismatch ระหว่างตอนเทรนกับตอนรันเว็บจริง
    """
    def __init__(self, margin_ratio=DEFAULT_MARGIN_RATIO):
        self.margin_ratio = margin_ratio

    def __call__(self, img):
        # Convert PIL (RGB) -> OpenCV (BGR)
        cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        # Crop foreground origami region
        cropped_cv, _ = crop_to_foreground(cv_img, margin_ratio=self.margin_ratio)
        
        # Convert back OpenCV (BGR) -> PIL (RGB)
        return Image.fromarray(cv2.cvtColor(cropped_cv, cv2.COLOR_BGR2RGB))


# === Transform สำหรับ Training (พร้อม Augmentation) ===
train_transform = transforms.Compose([
    AutoCropForeground(margin_ratio=DEFAULT_MARGIN_RATIO),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(
        brightness=0.35,
        contrast=0.35,
        saturation=0.40,
        hue=0.15
    ),
    transforms.RandomGrayscale(p=0.20),
    transforms.RandomAdjustSharpness(sharpness_factor=2.0, p=0.30),
    transforms.RandomAutocontrast(p=0.20),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.10, 0.10),
        scale=(0.85, 1.15),
        shear=5
    ),
    transforms.RandomPerspective(distortion_scale=0.15, p=0.25),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],   # ImageNet mean
        std=[0.229, 0.224, 0.225]      # ImageNet std
    )
])

# === Transform สำหรับ Validation/Test (ไม่มี Augmentation) ===
val_transform = transforms.Compose([
    AutoCropForeground(margin_ratio=DEFAULT_MARGIN_RATIO),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# === Transform มาตรฐานสำหรับภาพ PIL (Resize -> ToTensor -> Normalize) โดยไม่มี AutoCrop ซ้ำซ้อน ===
tensor_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# === Transform สำหรับ Inference จาก Webcam (เมื่อ input เป็น numpy array) ===
inference_transform = transforms.Compose([
    transforms.ToPILImage(),
    *tensor_transform.transforms
])


def _expand_bbox(x, y, w, h, image_width, image_height, margin_ratio=DEFAULT_MARGIN_RATIO):
    """ขยายกรอบ crop เผื่อขอบกระดาษเล็กน้อย"""
    margin_x = int(w * margin_ratio)
    margin_y = int(h * margin_ratio)
    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(image_width, x + w + margin_x)
    y2 = min(image_height, y + h + margin_y)
    return x1, y1, x2, y2


def _calculate_center_score(x, y, w, h, image_center, max_distance):
    """Calculate proximity score (0.0 to 1.0) of a bounding box center to the frame center."""
    center = np.array([x + w / 2.0, y + h / 2.0], dtype=np.float32)
    center_distance = np.linalg.norm(center - image_center) / max_distance
    return max(0.0, 1.0 - center_distance)


def _detect_bbox_by_edges(frame, min_area_ratio=0.015):
    """
    หา bounding box ของวัตถุเด่นในภาพด้วย edges/contours
    รองรับกระดาษสีต่าง ๆ ดีกว่า white-threshold แบบเดิม
    """
    original_h, original_w = frame.shape[:2]
    scale = 1.0
    work = frame

    if max(original_h, original_w) > 512:
        scale = 512 / max(original_h, original_w)
        work = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    work_h, work_w = work.shape[:2]
    work_area = float(work_h * work_w)

    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    median = np.median(blurred)
    lower = int(max(0, 0.66 * median))
    upper = int(min(255, 1.33 * median + 20))
    edges = cv2.Canny(blurred, lower, upper)

    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    image_center = np.array([work_w / 2.0, work_h / 2.0], dtype=np.float32)
    max_distance = np.linalg.norm(image_center) + 1e-6
    best_bbox = None
    best_score = -1.0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < work_area * min_area_ratio:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        box_area = float(w * h)
        if box_area <= 0 or box_area > work_area * 0.97:
            continue

        fill_ratio = area / box_area
        center_score = _calculate_center_score(x, y, w, h, image_center, max_distance)
        area_score = area / work_area
        score = (0.65 * area_score) + (0.20 * min(fill_ratio, 1.0)) + (0.15 * center_score)

        if score > best_score:
            best_score = score
            best_bbox = (x, y, w, h)

    if best_bbox is None:
        return None

    x, y, w, h = best_bbox
    return (
        int(x / scale),
        int(y / scale),
        int(w / scale),
        int(h / scale),
    )


def _detect_bbox_by_grabcut(frame, min_area_ratio=0.03):
    """
    Fallback ด้วย GrabCut เมื่อ edge-based contour หาไม่เจอ
    อาศัยสมมติฐานว่ากระดาษอยู่กลางเฟรมเป็นหลัก
    """
    original_h, original_w = frame.shape[:2]
    scale = 1.0
    work = frame

    if max(original_h, original_w) > 256:
        scale = 256 / max(original_h, original_w)
        work = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    work_h, work_w = work.shape[:2]
    work_area = float(work_h * work_w)
    pad_x = max(8, int(work_w * 0.08))
    pad_y = max(8, int(work_h * 0.08))
    rect_w = work_w - (2 * pad_x)
    rect_h = work_h - (2 * pad_y)

    if rect_w <= 0 or rect_h <= 0:
        return None

    mask = np.zeros(work.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(
            work,
            mask,
            (pad_x, pad_y, rect_w, rect_h),
            bgd_model,
            fgd_model,
            1,
            cv2.GC_INIT_WITH_RECT
        )
    except cv2.error:
        return None

    foreground_mask = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
        255,
        0
    ).astype("uint8")

    kernel = np.ones((5, 5), np.uint8)
    foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(
        foreground_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < work_area * min_area_ratio:
        return None

    x, y, w, h = cv2.boundingRect(largest)
    if (w * h) > work_area * 0.97:
        return None

    return (
        int(x / scale),
        int(y / scale),
        int(w / scale),
        int(h / scale),
    )


def _detect_bbox_by_white_paper(frame, min_area_ratio=0.02):
    """
    Detect white origami paper by finding the dominant bright region
    contrasted against darker table/floor backgrounds, excluding border artifacts.
    """
    original_h, original_w = frame.shape[:2]
    scale = 1.0
    work = frame

    if max(original_h, original_w) > 512:
        scale = 512 / max(original_h, original_w)
        work = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    wh, ww = work.shape[:2]
    work_area = float(wh * ww)

    # Exclude extreme top and bottom 5% from search (border reflection artifacts)
    mask_valid = np.zeros((wh, ww), dtype=np.uint8)
    pad_top = int(wh * 0.05)
    pad_bot = int(wh * 0.95)
    mask_valid[pad_top:pad_bot, :] = 255

    # Split BGR channels: true white paper requires high intensity in Blue, Green, and Red simultaneously.
    # Colored surfaces (orange tables, wood desks, warm lamp reflections) have diminished Blue/Green values.
    # np.minimum(np.minimum(b, g), r) isolates achromatic brightness (pure whiteness), completely immune
    # to warm orange/brown background reflections.
    b, g, r = cv2.split(work)
    whiteness = np.minimum(np.minimum(b, g), r)
    blurred = cv2.GaussianBlur(whiteness, (7, 7), 0)

    middle_pixels = blurred[mask_valid == 255]
    if len(middle_pixels) == 0:
        return None
    p_high = np.percentile(middle_pixels, WHITE_PAPER_PERCENTILE)
    median_val = np.median(middle_pixels)
    thresh_val = int(max(p_high, median_val + 20))
    thresh_val = int(np.clip(thresh_val, WHITE_PAPER_MIN_THRESH, WHITE_PAPER_MAX_THRESH))
    _, binary = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY)
    binary = cv2.bitwise_and(binary, mask_valid)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    image_center = np.array([ww / 2.0, wh / 2.0], dtype=np.float32)
    max_distance = np.linalg.norm(image_center) + 1e-6
    best_bbox = None
    best_score = -1.0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < work_area * min_area_ratio:
            continue
        x, y, w_box, h_box = cv2.boundingRect(contour)

        # Reject background wall or floor strip spanning wide across image or taking excessive area
        if w_box > ww * 0.95 or h_box > wh * 0.95:
            continue
        box_area = float(w_box * h_box)
        # Reject oversized areas (desks, large wall backgrounds)
        if box_area > work_area * 0.52 or area > work_area * 0.48:
            continue
        if y <= pad_top and w_box > ww * 0.45:
            continue
        if (y + h_box) >= pad_bot and w_box > ww * 0.45:
            continue

        aspect = w_box / float(h_box)
        if aspect > 5.0 or aspect < 0.15:
            continue

        center_score = _calculate_center_score(x, y, w_box, h_box, image_center, max_distance)

        c_mask = np.zeros(work.shape[:2], dtype=np.uint8)
        cv2.drawContours(c_mask, [contour], -1, 255, -1)
        mean_val = cv2.mean(blurred, mask=c_mask)[0]
        brightness_score = min(1.0, max(0.0, (mean_val - median_val) / WHITE_PAPER_BRIGHTNESS_DIVISOR))

        # Normalized area score: optimal between 5-35% of frame, gracefully accommodating up to 60%
        area_ratio = area / work_area
        if area_ratio <= 0.25:
            norm_area = min(1.0, area_ratio / 0.10)
        else:
            norm_area = max(0.4, 1.0 - ((area_ratio - 0.25) / 0.50))

        score = (
            (WHITE_PAPER_AREA_WEIGHT * norm_area)
            + (WHITE_PAPER_BRIGHTNESS_WEIGHT * brightness_score)
            + (WHITE_PAPER_CENTER_WEIGHT * center_score)
        )
        if score > best_score:
            best_score = score
            best_bbox = (x, y, w_box, h_box)

    if best_bbox is None:
        return None

    return (
        int(best_bbox[0] / scale),
        int(best_bbox[1] / scale),
        int(best_bbox[2] / scale),
        int(best_bbox[3] / scale),
    )


def crop_to_foreground(frame, margin_ratio=DEFAULT_MARGIN_RATIO):
    """
    Crop เฉพาะบริเวณกระดาษ/ตัวพับเพื่อลดอิทธิพลของพื้นหลัง

    Args:
        frame: ภาพ BGR จาก OpenCV
        margin_ratio: สัดส่วนขยายกรอบ crop

    Returns:
        cropped_frame, meta
    """
    image_h, image_w = frame.shape[:2]
    bbox = _detect_bbox_by_white_paper(frame)
    method = "white_paper"

    if bbox is None:
        bbox = _detect_bbox_by_edges(frame)
        method = "edges"

    if bbox is None:
        bbox = _detect_bbox_by_grabcut(frame)
        method = "grabcut"

    if bbox is None:
        return frame, {
            "applied": False,
            "method": "full-frame",
            "bbox": [0, 0, image_w, image_h],
        }

    x, y, w, h = bbox
    x1, y1, x2, y2 = _expand_bbox(x, y, w, h, image_w, image_h, margin_ratio=margin_ratio)
    cropped = frame[y1:y2, x1:x2]

    if cropped.size == 0:
        return frame, {
            "applied": False,
            "method": "full-frame",
            "bbox": [0, 0, image_w, image_h],
        }

    return cropped, {
        "applied": True,
        "method": method,
        "bbox": [int(x1), int(y1), int(x2), int(y2)],
    }
