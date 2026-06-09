# test_sam.py
from ultralytics import SAM
import cv2
import numpy as np

# 1. 모델 로드
model = SAM("sam_b.pt")

# 2. 가짜 프레임 생성 (테스트용)
dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)

# 3. 테스트용 BBox (Batch)
test_bboxes = [[100, 100, 300, 300], [400, 400, 600, 600]]

print("SAM 추론 시작...")
results = model(dummy_frame, bboxes=test_bboxes, verbose=True)

if results[0].masks is not None:
    print(f"성공! 마스크 개수: {len(results[0].masks)}")
else:
    print("실패! 마스크가 추출되지 않았습니다.")