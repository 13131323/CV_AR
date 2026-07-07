"""
===========================================================================
[floor 재계산 실험] 물체 발밑 지지면(footing) 방식으로 floor_margin 재산출
===========================================================================
목적:
  기존 floor 검출("화면 하단 전체" median/P90)은 물체 위치를 안 따라가서
  floor_margin이 '높이'가 아니라 '거리'를 재는 문제가 있었음.
  → 여기선 "물체 bbox 바로 아래 밴드"의 깊이를 지지면으로 잡아
    floor_margin = footing_floor - target_z 를 재계산한다.

특징:
  - 본문 파이프라인(vision/spatial/floor_detector.py)은 건드리지 않음 (실험 전용)
  - 저장된 프레임(data/frames/*.jpg + *_depth.npy)을 재처리 → 재촬영 X
  - 결과: data/floor_refit_log.csv + 세션별 held/elevated 방향 요약 출력

물리적 기대:
  - elevated(탁자 위): 물체 아래 = 탁자면(물체와 비슷한 깊이) → margin ≈ 0
  - held(손에 듬):     물체 아래 = 손/빈공간 뒤 먼 바닥      → margin 큼(양수)
  → 방향이 모든 거리에서 일관되면 임계값 확정 가능.

실행:
  python -m tests.test_floor_refit
===========================================================================
"""
import cv2
import numpy as np
import glob
import os
import csv
import torch
from ultralytics import YOLO

FRAME_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "frames")
OUT_CSV   = os.path.join(os.path.dirname(__file__), "..", "data", "floor_refit_log.csv")
FOOTING_H = 45          # 물체 아래 지지면 밴드 높이(px)
CENTER_RATIO = 0.5      # 물체 깊이 추출 시 bbox 중앙 비율(배경 배제)


def parse_name(path):
    """book_held_in_hand_1.0_0002.jpg → (label, state, dist)"""
    name = os.path.basename(path).replace(".jpg", "")
    p = name.split("_")
    label = p[0]
    dist  = p[-2]
    state = "_".join(p[1:-2])
    return label, state, dist


def object_depth(depth, box):
    """bbox 중앙 영역(배경 배제)의 median 깊이 = target_z 근사"""
    x1, y1, x2, y2 = box
    mw = int((x2 - x1) * (1 - CENTER_RATIO) / 2)
    mh = int((y2 - y1) * (1 - CENTER_RATIO) / 2)
    roi = depth[y1 + mh:y2 - mh, x1 + mw:x2 - mw]
    v = roi[roi > 0]
    return float(np.median(v)) if v.size else 0.0


def footing_depth(depth, box):
    """bbox 바로 아래 밴드의 median 깊이 = 물체가 놓인/떠 있는 지지면"""
    x1, y1, x2, y2 = box
    h, w = depth.shape[:2]
    by1 = min(int(y2), h - 1)
    by2 = min(int(y2) + FOOTING_H, h)
    band = depth[by1:by2, x1:x2]
    v = band[band > 0]
    return float(np.median(v)) if v.size else 0.0


def main():
    device = ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
              else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    print(f"[장치] {device}")
    yolo = YOLO("yolov8n.pt")

    files = sorted(glob.glob(os.path.join(FRAME_DIR, "*.jpg")))
    print(f"[재처리 대상] {len(files)} 프레임")
    if not files:
        print("[오류] data/frames/ 에 jpg가 없습니다.")
        return

    rows = []
    from collections import defaultdict
    agg = defaultdict(lambda: {"fm": [], "tz": [], "foot": []})

    for i, jpg in enumerate(files):
        label, state, dist = parse_name(jpg)
        npy = jpg.replace(".jpg", "_depth.npy")
        if not os.path.exists(npy):
            continue
        frame = cv2.imread(jpg)
        depth = np.load(npy)
        if frame is None or depth is None:
            continue

        res = yolo(frame, device=device, verbose=False, conf=0.10)[0]
        # 대상 라벨 bbox 중 가장 큰 것 선택
        best = None
        best_area = 0
        for b in res.boxes:
            if res.names[int(b.cls[0].item())] != label:
                continue
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            area = (x2 - x1) * (y2 - y1)
            if area > best_area:
                best_area = area
                best = (x1, y1, x2, y2)
        if best is None:
            continue

        tz   = object_depth(depth, best)
        foot = footing_depth(depth, best)
        if tz <= 0 or foot <= 0:
            continue
        fm = foot - tz  # 새 floor_margin

        rows.append([label, dist, state, round(tz, 4), round(foot, 4), round(fm, 4)])
        key = (label, state, dist)
        agg[key]["fm"].append(fm)
        agg[key]["tz"].append(tz)
        agg[key]["foot"].append(foot)

        if (i + 1) % 200 == 0:
            print(f"  진행 {i+1}/{len(files)}  (수집 {len(rows)})")

    # CSV 저장
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["label", "distance_label", "object_state",
                    "target_z", "footing_floor", "floor_margin"])
        w.writerows(rows)
    print(f"\n[CSV] {OUT_CSV} 저장 ({len(rows)}행)")

    # 세션별 held vs elevated 방향 요약
    print("\n" + "=" * 66)
    print("새 floor_margin (footing) — held vs elevated 방향")
    print("=" * 66)
    print(f"{'obj':>7}{'dist':>6}{'held_fm':>10}{'elev_fm':>10}{'방향':>10}")
    for lab in sorted(set(k[0] for k in agg)):
        for d in ["1.0", "1.5", "2.0", "2.5"]:
            h = agg.get((lab, "held_in_hand", d))
            e = agg.get((lab, "elevated", d))
            if not h or not e or not h["fm"] or not e["fm"]:
                continue
            hfm = float(np.median(h["fm"]))
            efm = float(np.median(e["fm"]))
            arrow = "held>" if hfm > efm else "held<"
            print(f"{lab:>7}{d:>6}{hfm:>10.3f}{efm:>10.3f}{arrow:>10}")
    print("\n→ 방향(held>/held<)이 모든 거리에서 일관되면 임계값 확정 가능.")


if __name__ == "__main__":
    main()
