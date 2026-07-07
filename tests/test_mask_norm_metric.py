"""
===========================================================================
[7-3단계 · Metric 재촬영판] 원근 왜곡 보정 가설 검증 스크립트
===========================================================================
목적:
  가설 "거리에 따른 원근 왜곡을 수식적으로 보정 → mask_norm 거리 불변"을
  Metric Depth 모델로 재검증한다.

  기존판(test_mask_norm.py)과의 차이 (3가지):
    ① 깊이 모델: relative(disparity) → METRIC(미터). target_z ∝ Z 가 됨.
    ② 원본 프레임 + 깊이맵 저장: data/frames/ (재추론 대비 — 다신 재촬영 안 하도록)
    ③ CSV 경로 분리: mask_norm_metric_log.csv (기존 disparity 데이터와 격리)

사용법:
  1. 아래 제어 변수를 실험 조건에 맞게 설정한다.
     - OBJECT_LABEL   : "cell phone" | "bottle" | "mouse" | "keyboard"
     - OBJECT_STATE   : "held_in_hand" | "elevated"
     - DISTANCE_LABEL : "500" | "1000" | "1500" | "2000"  (단위: mm, 줄자로 정확히)
     - MAX_FRAMES     : 100~200 권장
  2. python -m tests.test_mask_norm_metric 으로 실행.
  3. 로그는 data/mask_norm_metric_log.csv 에 누적, 프레임은 data/frames/ 에 저장.
  4. 'q' 또는 MAX_FRAMES 도달 시 자동 종료.

★ 촬영 전 30초 SANITY CHECK (필수):
  아무 거리 2개(예: 500·2000)만 잠깐 찍어 콘솔의 z= 값을 확인:
    ✅ 거리 멀수록 z 증가(≈미터 단위 0.5~2.0)  → metric 정상, 전체 촬영 진행
    ❌ 거리 멀수록 z 감소                       → 아직 disparity, 모델 출력 재확인
  깊이 로드 직후 [DEPTH RANGE] 로그로 min/max도 함께 출력됨.
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

# 바닥 검출은 원본 파이프라인 로직을 그대로 사용 (복사본 금지 — 단일 출처)
from vision.spatial.floor_detector import FloorPlaneDetector

# =====================================================================
# ★ 실험 제어 변수 (매 세션마다 수정)
# =====================================================================
OBJECT_LABEL   = "book"              # "cell phone" | "bottle" | "mouse" | "keyboard" | "book"
OBJECT_STATE   = "elevated"      # "held_in_hand" | "elevated"
DISTANCE_LABEL = "1.5"               # "1.0" | "1.5" | "2.0" | "2.5"  (단위: m, 1m부터 0.5m 간격)
MAX_FRAMES     = 150                 # 100~200 권장

# =====================================================================
# ★ [변경 ①] Metric Depth 모델 (미터 깊이 직접 출력)
# =====================================================================
#   기존: "depth-anything/Depth-Anything-V2-Base-hf"          (relative/disparity)
#   교체: 아래 Metric-Indoor (실내 0.5~2m 범위에 적합)
#   대안: "Intel/zoedepth-nyu-kitti" (ZoeDepth)
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf"

# =====================================================================
# ★ [변경 ②] 원본 프레임 / 깊이맵 저장 경로
# =====================================================================
SAVE_FRAMES = True                               # 원본 jpg 저장
SAVE_DEPTH  = True                               # 깊이맵 npy 저장 (재추론/디버깅용)
FRAME_DIR   = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "data", "frames")
)

# =====================================================================
# 고정 설정값
# =====================================================================
SEMANTIC_PRIOR_DB = {
    "cell phone": {"real_area": 12773.16},   # 163.6mm × 78.1mm (실측)
    "bottle":     {"real_area": 13975.0},    # 215mm × 65mm (실측)
    "mouse":      {"real_area": 6893.0},     # 113mm × 61mm (실측)
    "keyboard":   {"real_area": 64500.0},    # 430mm × 150mm (실측)
    "book":       {"real_area": 58910.0},    # 274mm × 215mm (실측)
}

SAM_INTERVAL   = 5
DEPTH_INTERVAL = 10

# ★ [변경 ③] CSV 경로 분리 (기존 disparity 데이터와 격리)
CSV_OUTPUT_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "data", "mask_norm_metric_log.csv")
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


# 바닥 검출기: 로컬 복사본을 폐기하고 원본 파이프라인(FloorPlaneDetector)을 import해 사용.
#   → floor_detector.py 한 곳만 고치면 실험/배포가 동일 로직을 공유(단일 출처).
#   현재 원본 로직: 하단 ROI median. (개선 필요 시 RANSAC 평면 피팅은 원본에서 처리)


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
    print("  [7-3 · Metric] 원근 보정 가설 재검증 데이터 수집기")
    print("=" * 65)
    print(f"  대상 객체  : {OBJECT_LABEL}")
    print(f"  객체 상태  : {OBJECT_STATE}")
    print(f"  촬영 거리  : {DISTANCE_LABEL} mm")
    print(f"  수집 목표  : {MAX_FRAMES} 프레임")
    print(f"  깊이 모델  : {DEPTH_MODEL}")
    print(f"  CSV 경로   : {CSV_OUTPUT_PATH}")
    print(f"  프레임 저장: {FRAME_DIR} (frames={SAVE_FRAMES}, depth={SAVE_DEPTH})")
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
    print("[모델 로드] YOLO, SAM, Metric Depth 초기화 중...")
    yolo       = YOLO("yolov8n.pt")
    sam        = SAM("sam_b.pt")
    depth_pipe = pipeline(
        task="depth-estimation",
        model=DEPTH_MODEL,
        device=device
    )
    print("[모델 로드] 완료.")

    # 카메라
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    floor_detector = FloorPlaneDetector()
    init_csv(CSV_OUTPUT_PATH)
    if SAVE_FRAMES or SAVE_DEPTH:
        os.makedirs(FRAME_DIR, exist_ok=True)

    frame_count   = 0   # 전체 처리 프레임
    collect_count = 0   # 유효 수집 프레임 (대상 객체 탐지 성공)
    depth_logged  = False  # DEPTH RANGE 최초 1회 출력용

    cached_depth = None
    last_masks   = []
    last_labels  = []
    csv_buffer   = []

    prior_area = SEMANTIC_PRIOR_DB.get(OBJECT_LABEL, {}).get("real_area", None)
    if prior_area is None:
        print(f"[오류] '{OBJECT_LABEL}'에 대한 Semantic Prior가 없습니다.")
        cap.release()
        return

    print(f"\n[실험 시작] 'q'를 누르거나 {MAX_FRAMES}프레임 수집 시 자동 종료됩니다.")
    print("★ SANITY CHECK: [DEPTH RANGE]와 z= 값이 미터 단위(멀수록 증가)인지 확인하세요.\n")

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

            # ★ SANITY CHECK: 깊이 스케일 진단 (metric이면 대략 미터 단위)
            valid_d = cached_depth[cached_depth > 0]
            if valid_d.size > 0:
                print(f"  [DEPTH RANGE] frame={frame_count} "
                      f"min={valid_d.min():.3f} p50={np.median(valid_d):.3f} "
                      f"max={valid_d.max():.3f}  (metric이면 미터 단위)")

        # YOLO 탐지
        yolo_res = yolo(frame, device=device, verbose=False, conf=0.10)[0]
        boxes    = yolo_res.boxes
        labels   = [yolo_res.names[int(b.cls[0].item())] for b in boxes]
        bboxes   = [list(map(int, b.xyxy[0].tolist())) for b in boxes]

        # 10프레임마다 탐지 현황 출력
        if frame_count % 10 == 0:
            if labels:
                confs = [round(float(b.conf[0].item()), 2) for b in boxes]
                detected_str = ", ".join(f"{l}({c})" for l, c in zip(labels, confs))
            else:
                detected_str = "없음"
            print(f"  [YOLO 탐지] frame={frame_count} → {detected_str}")

        # 대상 객체 인덱스 탐색
        target_indices = [i for i, lbl in enumerate(labels) if lbl == OBJECT_LABEL]

        # SAM 세그멘테이션
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
        # 원본 파이프라인과 동일하게 detect_floor 사용 (입력: {"raw_depth": ...})
        floor_depth = floor_detector.detect_floor({"raw_depth": cached_depth}).get("floor_depth", 0.0)

        # 대상 객체별 mask_norm 계산 및 수집
        collected_this_frame = False

        for ti in target_indices:
            if ti >= len(all_masks) or all_masks[ti] is None:
                continue

            m_bool    = all_masks[ti]
            mask_area = int(np.sum(m_bool))

            if mask_area == 0:
                continue

            mask_overlay[m_bool] = [0, 200, 255]

            # 깊이값 추출 (metric: 미터 단위, 물체 마스크 영역 평균)
            if cached_depth is not None:
                roi_pix  = cached_depth[m_bool]
                roi_pix  = roi_pix[roi_pix > 0]
                target_z = float(np.mean(roi_pix)) if roi_pix.size > 0 else 0.0
            else:
                target_z = 0.0

            # floor_margin (metric: 부호가 기존 disparity와 반대 방향)
            floor_margin = round(float(floor_depth - target_z), 4) if target_z > 0 else 0.0

            # mask_norm 계산 = mask_area * target_z^2 / A_real
            #   metric에선 target_z ∝ Z 이므로 거리 불변으로 수렴해야 함 (가설)
            if target_z > 0:
                mask_norm = (mask_area * (target_z ** 2)) / prior_area
            else:
                mask_norm = None

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

                # ★ [변경 ②] 원본 프레임 / 깊이맵 저장
                if SAVE_FRAMES or SAVE_DEPTH:
                    base = os.path.join(
                        FRAME_DIR,
                        f"{OBJECT_LABEL}_{OBJECT_STATE}_{DISTANCE_LABEL}_{frame_count:04d}"
                    )
                    if SAVE_FRAMES:
                        cv2.imwrite(base + ".jpg", frame)
                    if SAVE_DEPTH and cached_depth is not None:
                        np.save(base + "_depth.npy", cached_depth.astype(np.float32))

                print(
                    f"[{collect_count+1:>4}/{MAX_FRAMES}] "
                    f"{OBJECT_LABEL} | {OBJECT_STATE} | {DISTANCE_LABEL}mm | "
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

        status_color = (0, 255, 120) if collected_this_frame else (80, 80, 80)
        cv2.rectangle(vis, (0, 0), (760, 36), (0, 0, 0), -1)
        cv2.putText(
            vis,
            f"[{OBJECT_LABEL}] {OBJECT_STATE} | {DISTANCE_LABEL}mm | "
            f"Collected: {collect_count}/{MAX_FRAMES} | METRIC",
            (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, status_color, 2
        )

        bar_w = int((collect_count / MAX_FRAMES) * w)
        cv2.rectangle(vis, (0, h - 8), (bar_w, h), (0, 200, 100), -1)

        cv2.imshow("[7-3 Metric] Mask_norm Hypothesis Re-test", vis)

        if cached_depth is not None:
            # 깊이맵 시각화를 위해 정규화 및 컬러맵(Magma/Inferno) 적용
            valid_d = cached_depth[cached_depth > 0]
            if valid_d.size > 0:
                d_min, d_max = valid_d.min(), valid_d.max()
                if d_max > d_min:
                    depth_norm = (cached_depth - d_min) / (d_max - d_min)
                else:
                    depth_norm = np.zeros_like(cached_depth)
                
                depth_uint8 = (depth_norm * 255).astype(np.uint8)
                # 컬러맵 적용 (첨부해주신 이미지와 유사한 시각화)
                depth_color = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_MAGMA)
                cv2.imshow("CV_AR - Depth Anything V2 (Validated)", depth_color)

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
    print(f"  - 프레임 저장 경로 : {FRAME_DIR}")
    print("=" * 65)


if __name__ == "__main__":
    main()
