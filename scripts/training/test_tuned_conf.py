import cv2
from pathlib import Path
import sys
sys.path.append('src')
from yolo_verifier import YoloStepVerifier

verifier = YoloStepVerifier('checkpoints/yolo_runs/origami_yolo_run1/weights/best.pt', conf_threshold=0.22)
img = cv2.imread('outputs/audit_user_tests/Test_YOLO_with_white_paper_frame_4.jpg')
res = verifier.verify_bgr(img)
print('Tuned detection result:')
print('Recognized:', res.recognized)
print('Prediction:', res.prediction, 'Confidence:', res.confidence)
print('ROI BBox:', res.roi.get('bbox'))
