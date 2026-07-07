"""
===========================================================================
[7-3단계] 의미-기하 융합 구조 임계값(Threshold) 실험 스크립트
===========================================================================
실험 목적:
  Mask_norm = (mask_area * target_z^2) / A_real 의 유효성 검증
  held_in_hand / elevated 상태 분리를 위한 최적 임계값 도출

사용법:
  1. 아래 제어 변수를 실험 조건에 맞게 설정한다.
     - OBJECT_LABEL   : "cell phone" 또는 "bottle"
     - OBJECT_STATE   : "held_in_hand" 또는 "elevated"
     - DISTANCE_LABEL : "1m", "2m", "3m", "4m"
     - MAX_FRAMES     : 최대 수집 프레임 수 (100~200 권장)
  2. python -m tests.test_mask_norm 으로 실행한다.
  3. 수집된 로그는 data/mask_norm_log.csv 에 누적 저장된다.
  4. 'q'를 누르거나 MAX_FRAMES에 도달하면 자동 종료된다.
===========================================================================
"""

import cv2
import numpy as np
import csv
import os
import time
import copy
import torch
from ultralytics import YOLO, SAM
from transformers import pipeline
from PIL import Image

# =====================================================================
# ★ 실험 제어 변수 (매 세션마다 수정)
# =====================================================================
OBJECT_LABEL   = "keyboard"       # "cell phone" | "bottle" | "mouse" | "keyboard"
OBJECT_STATE   = "held_in_hand"      # "held_in_hand" | "elevated"
DISTANCE_LABEL = "2000"           # "500" | "1000" | "1500" | "2000"  (단위: mm)
MAX_FRAMES     = 150             # 100~200 권장

# =====================================================================
# 고정 설정값
# =====================================================================
SEMANTIC_PRIOR_DB = {
    "cell phone": {"real_area": 12773.16},   # 163.6mm × 78.1mm (실측)
    "bottle":     {"real_area": 13975.0},    # 215mm × 65mm (실측)
    "mouse":      {"real_area": 6893.0},     # 113mm × 61mm (실측)
    "keyboard":   {"real_area": 64500.0},    # 430mm × 150mm (실측)
}

SAM_INTERVAL   = 5
DEPTH_INTERVAL = 10

CSV_OUTPUT_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "data", "mask_norm_log.csv")
)

# =====================================================================
# 카메라 캘리브레이션값
# =====================================================================
# 재캘리브레이션값 (RMS 0.2619 — 기존 0.8에서 개선)
CAMERA_MATRIX = np.array([
    [964.86361287, 0.0,          636.84364465],
    [0.0,          964.45199507, 359.34677934],
    [0.0,          0.0,          1.0         ]
], dtype=np.float32)

DIST_COEFFS = np.array([
    -0.00996911, -0.0233743, -0.00077566, -0.00061024, 0.05377734
], dtype=np.float32)


# =====================================================================
# 바닥 검출기 (Dual Edge ROI + P10 — 탁자 가림 대응)
# =====================================================================
class FloorDetector:
    def __init__(self, roi_ratio=0.35, edge_ratio=0.20, percentile=10):
        self.roi_ratio  = roi_ratio    # 하단 35% ROI
        self.edge_ratio = edge_ratio   # 좌우 각 20% 슬릿
        self.percentile = percentile   # P10: 가장 먼 픽셀(실제 바닥)

    def get_floor_depth(self, depth_map):
        if depth_map is None:
            return 0.0
        h, w = depth_map.shape[:2]
        roi_y = int(h * (1.0 - self.roi_ratio))
        ex    = int(w * self.edge_ratio)

        # 좌우 슬릿 — 탁자 중앙을 피해 바닥만 포착
        left  = depth_map[roi_y:h, :ex]
        right = depth_map[roi_y:h, w - ex:]
        valid = np.concatenate([left.ravel(), right.ravel()])
        valid = valid[valid > 0]

        if valid.size == 0:
            # fallback: 하단 ROI 전체 P10
            roi   = depth_map[roi_y:h, :]
            valid = roi[roi > 0]
            return float(np.percentile(valid, self.percentile)) if valid.size > 0 else 0.0

        return float(np.percentile(valid, self.percentile))


# =====================================================================
# CSV 초기화 (헤더 최초 1회만 기록)
# =====================================================================
def init_csv(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "label",
                "distance_label",
                "object_state",
                "mask_area",
                "target_z",
                "floor_margin",
                "mask_norm",
                "timestamp"
            ])
        print(f"[CSV] 새 로그 파일 생성: {path}")
    else:
        print(f"[CSV] 기존 로그 파일에 누적 저장: {path}")


def append_csv(path, rows):
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


# =====================================================================
# 메인 루프
# =====================================================================
def main():
    print("=" * 65)
    print("  [7-3단계] Mask_norm 임계값 실험 데이터 수집기")
    print("=" * 65)
    print(f"  대상 객체  : {OBJECT_LABEL}")
    print(f"  객체 상태  : {OBJECT_STATE}")
    print(f"  촬영 거리  : {DISTANCE_LABEL}")
    print(f"  수집 목표  : {MAX_FRAMES} 프레임")
    print(f"  CSV 경로   : {CSV_OUTPUT_PATH}")
    print("=" * 65)

    # 장치 선택
    if torch.cuda.is_available():
        device = "cuda:0"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"[장치] {device}")

    # 모델 로드
    print("[모델 로드] YOLO, SAM, Depth Anything V2 초기화 중...")
    yolo       = YOLO("yolov8n.pt")
    sam        = SAM("sam_b.pt")
    depth_pipe = pipeline(
        task="depth-estimation",
        model="depth-anything/Depth-Anything-V2-Base-hf",
        device=device
    )
    print("[모델 로드] 완료.")

    # 카메라
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    floor_detector = FloorDetector()
    init_csv(CSV_OUTPUT_PATH)

    frame_count   = 0   # 전체 처리 프레임
    collect_count = 0   # 유효 수집 프레임 (대상 객체 탐지 성공)

    cached_depth = None
    last_masks   = []
    last_labels  = []
    csv_buffer   = []

    prior_area = SEMANTIC_PRIOR_DB.get(OBJECT_LABEL, {}).get("real_area", None)
    if prior_area is None:
        print(f"[오류] '{OBJECT_LABEL}'에 대한 Semantic Prior가 없습니다.")
        cap.release()
        return

    print(f"\n[실험 시작] 'q'를 누르거나 {MAX_FRAMES}프레임 수집 시 자동 종료됩니다.\n")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("[오류] 카메라 스트림 오류")
            break

        frame_count += 1
        h, w = frame.shape[:2]

        # 왜곡 보정
        frame = cv2.undistort(frame, CAMERA_MATRIX, DIST_COEFFS)

        # Depth 추론 (DEPTH_INTERVAL마다)
        if frame_count == 1 or frame_count % DEPTH_INTERVAL == 0:
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            out     = depth_pipe(pil_img)
            pred    = out["predicted_depth"]
            if not isinstance(pred, torch.Tensor):
                pred = torch.from_numpy(np.array(pred)).float()
            cached_depth = torch.nn.functional.interpolate(
                pred.unsqueeze(0).unsqueeze(0),
                size=(h, w), mode="bilinear", align_corners=False
            ).squeeze().cpu().numpy()

        # YOLO 탐지
        yolo_res = yolo(frame, device=device, verbose=False, conf=0.10)[0]
        boxes    = yolo_res.boxes
        labels   = [yolo_res.names[int(b.cls[0].item())] for b in boxes]
        bboxes   = [list(map(int, b.xyxy[0].tolist())) for b in boxes]

        # 10프레임마다 탐지 현황 출력 (대상 미검출 시 원인 파악용)
        if frame_count % 10 == 0:
            if labels:
                confs = [round(float(b.conf[0].item()), 2) for b in boxes]
                detected_str = ", ".join(f"{l}({c})" for l, c in zip(labels, confs))
            else:
                detected_str = "없음"
            print(f"  [YOLO 탐지] frame={frame_count} → {detected_str}")

        # 대상 객체 인덱스 탐색
        target_indices = [i for i, lbl in enumerate(labels) if lbl == OBJECT_LABEL]

        # SAM 세그멘테이션 (SAM_INTERVAL마다 또는 라벨 변경 시)
        can_cache = (
            frame_count % SAM_INTERVAL != 0 and
            len(last_masks) == len(labels) and
            labels == last_labels
        )

        mask_overlay = np.zeros_like(frame)

        if can_cache:
            all_masks = last_masks
        else:
            all_masks = []
            if bboxes:
                try:
                    sam_res = sam(frame, bboxes=bboxes, device=device, verbose=False)[0]
                    if sam_res.masks is not None:
                        for i in range(len(labels)):
                            if i < len(sam_res.masks.data):
                                m = sam_res.masks.data[i].cpu().numpy().astype(bool)
                                all_masks.append(m)
                            else:
                                all_masks.append(None)
                        last_masks  = all_masks
                        last_labels = copy.deepcopy(labels)
                except Exception as e:
                    print(f"[SAM 오류] {e}")

        # 바닥 깊이 계산
        floor_depth = floor_detector.get_floor_depth(cached_depth)

        # 대상 객체별 mask_norm 계산 및 수집
        collected_this_frame = False

        for ti in target_indices:
            if ti >= len(all_masks) or all_masks[ti] is None:
                continue

            m_bool    = all_masks[ti]
            mask_area = int(np.sum(m_bool))

            if mask_area == 0:
                continue

            # SAM 마스크 시각화 (청록색)
            mask_overlay[m_bool] = [0, 200, 255]

            # 깊이값 추출
            if cached_depth is not None:
                roi_pix  = cached_depth[m_bool]
                roi_pix  = roi_pix[roi_pix > 0]
                target_z = float(np.mean(roi_pix)) if roi_pix.size > 0 else 0.0
            else:
                target_z = 0.0

            # floor_margin
            floor_margin = round(float(floor_depth - target_z), 4) if target_z > 0 else 0.0

            # mask_norm 계산
            if target_z > 0:
                mask_norm = (mask_area * (target_z ** 2)) / prior_area
            else:
                mask_norm = None

            # CSV 버퍼 적재
            if mask_norm is not None:
                csv_buffer.append([
                    OBJECT_LABEL,
                    DISTANCE_LABEL,
                    OBJECT_STATE,
                    mask_area,
                    round(target_z, 4),
                    floor_margin,
                    round(mask_norm, 6),
                    round(time.time(), 3)
                ])
                collected_this_frame = True

                # 콘솔 실시간 출력
                print(
                    f"[{collect_count+1:>4}/{MAX_FRAMES}] "
                    f"{OBJECT_LABEL} | {OBJECT_STATE} | {DISTANCE_LABEL} | "
                    f"mask={mask_area:>6} | z={target_z:.3f} | "
                    f"floor_m={floor_margin:.3f} | "
                    f"mask_norm={mask_norm:.6f}"
                )

        if collected_this_frame:
            collect_count += 1

        # CSV 50프레임마다 플러시
        if frame_count % 50 == 0 and csv_buffer:
            append_csv(CSV_OUTPUT_PATH, csv_buffer)
            csv_buffer.clear()

        # 화면 오버레이
        vis = cv2.addWeighted(frame, 1.0, mask_overlay, 0.45, 0)

        # HUD 정보 표시
        status_color = (0, 255, 120) if collected_this_frame else (80, 80, 80)
        cv2.rectangle(vis, (0, 0), (700, 36), (0, 0, 0), -1)
        cv2.putText(
            vis,
            f"[{OBJECT_LABEL}] {OBJECT_STATE} | {DISTANCE_LABEL} | "
            f"Collected: {collect_count}/{MAX_FRAMES}",
            (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, status_color, 2
        )

        # 진행 바
        bar_w = int((collect_count / MAX_FRAMES) * w)
        cv2.rectangle(vis, (0, h - 8), (bar_w, h), (0, 200, 100), -1)

        cv2.imshow("[7-3] Mask_norm Experiment", vis)

        # 종료 조건
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n[종료] 사용자 요청으로 종료합니다.")
            break

        if collect_count >= MAX_FRAMES:
            print(f"\n[완료] {MAX_FRAMES}프레임 수집 완료. 자동 종료합니다.")
            break

    # 잔여 버퍼 저장
    if csv_buffer:
        append_csv(CSV_OUTPUT_PATH, csv_buffer)
        print(f"[CSV] 잔여 {len(csv_buffer)}행 저장 완료.")

    cap.release()
    cv2.destroyAllWindows()

    print("=" * 65)
    print(f"  수집 결과 요약")
    print(f"  - 전체 처리 프레임 : {frame_count}")
    print(f"  - 유효 수집 프레임 : {collect_count}")
    print(f"  - CSV 저장 경로    : {CSV_OUTPUT_PATH}")
    print("=" * 65)


if __name__ == "__main__":
    main()
