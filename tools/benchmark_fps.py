"""
[프로덕션 FPS 실측] 각 무거운 모델(YOLO/SAM/Depth) 추론 시간을 측정하고,
캐싱 스케줄(DEPTH_INTERVAL, SAM_INTERVAL)을 반영해 실효 FPS를 계산한다.

카메라 불필요 — 추론 시간은 프레임 내용과 무관하게 거의 일정하므로 합성 프레임으로 측정한다.
실효 프레임시간 ≈ t_yolo + t_sam/SAM_INTERVAL + t_depth/DEPTH_INTERVAL + 경량처리
"""

import time
import numpy as np
import cv2

from vision.detector import ObjectDetector
from vision.segmentation.segmenter import ObjectSegmenter, SAM_INTERVAL, DEPTH_INTERVAL
from vision.depth.depth_estimator import DepthEstimator

WARMUP = 2
ITERS = 6
H, W = 720, 1280
BBOX = [[440, 200, 760, 620]]   # SAM용 더미 박스 1개


def _time(fn, iters=ITERS):
    for _ in range(WARMUP):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters


def main():
    # 내용 무관하지만 현실적인 질감의 합성 프레임(그라디언트+노이즈)
    grad = np.tile(np.linspace(0, 255, W, dtype=np.uint8), (H, 1))
    frame = cv2.merge([grad, np.roll(grad, 100, 1), np.roll(grad, 200, 1)])
    frame = cv2.add(frame, (np.random.default_rng(0).integers(0, 40, (H, W, 3))).astype(np.uint8))

    print("모델 로드 중...")
    detector = ObjectDetector()
    segmenter = ObjectSegmenter()
    depth = DepthEstimator()

    print("추론 시간 측정 중...")
    t_yolo = _time(lambda: detector.detect(frame))
    t_depth = _time(lambda: depth.get_depth_map(frame))
    t_sam = _time(lambda: segmenter.model(frame, bboxes=BBOX, device=segmenter.device, verbose=False))

    print("\n=== 모델별 추론 시간 (프레임당) ===")
    print(f"  YOLO  : {t_yolo*1000:7.1f} ms")
    print(f"  SAM   : {t_sam*1000:7.1f} ms  (박스 {len(BBOX)}개)")
    print(f"  Depth : {t_depth*1000:7.1f} ms")

    def eff(sam_iv, depth_iv, label):
        # 매 프레임 YOLO, SAM은 sam_iv마다, Depth는 depth_iv마다
        per = t_yolo + t_sam / sam_iv + t_depth / depth_iv
        fps = 1.0 / per
        print(f"  {label:26} → 프레임 {per*1000:6.1f} ms = {fps:5.1f} fps "
              f"(SAM 1/{sam_iv}, Depth 1/{depth_iv})")
        return fps

    print("\n=== 스케줄별 실효 FPS ===")
    eff(1, 1, "검증도구(캐싱 OFF)")
    eff(SAM_INTERVAL, DEPTH_INTERVAL, "segmenter 프로덕션")
    eff(3, DEPTH_INTERVAL, "server_websocket")

    print("\n주: Depth가 가장 무거우므로 캐싱(1/10)이 FPS를 좌우함. 하드웨어별로 값이 다름.")


if __name__ == "__main__":
    main()
