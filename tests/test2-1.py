"""VLM 입력 이미지 가로폭을 원본의 30/30부터 1/30까지 줄이는 실험."""

from __future__ import annotations

import csv
import json
import sys
import threading
import time
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = PROJECT_ROOT / "test_res"
TEXT_RESULT = RESULT_DIR / "test2_res.txt"
CSV_RESULT = RESULT_DIR / "test2_res.csv"
EXPERIMENT_COUNT = 30


def width_for_experiment(original_width: int, experiment_index: int) -> int:
    """0-based 실험 번호에 대응하는 30/30 .. 1/30 가로폭을 반환한다."""
    remaining_steps = EXPERIMENT_COUNT - experiment_index
    return max(1, round(original_width * remaining_steps / EXPERIMENT_COUNT))


def initialise_result_files() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    TEXT_RESULT.write_text(
        "test2: VLM image-width experiment (30/30 -> 1/30)\n"
        f"started_at: {started_at}\n\n",
        encoding="utf-8",
    )
    with CSV_RESULT.open("w", newline="", encoding="utf-8-sig") as file:
        csv.writer(file).writerow(
            [
                "experiment",
                "original_width",
                "vlm_image_max_width",
                "width_ratio",
                "vlm_inference_seconds",
                "timestamp",
                "json",
            ]
        )


def append_result(
    experiment: int,
    original_width: int,
    target_width: int,
    elapsed: float,
    result: dict,
) -> None:
    timestamp = datetime.now().astimezone().isoformat(timespec="milliseconds")
    width_ratio = target_width / original_width
    logged_result = {
        "vlm_inference_seconds": round(elapsed, 6),
        "vlm_result": result,
    }
    compact_json = json.dumps(logged_result, ensure_ascii=False, separators=(",", ":"))
    pretty_json = json.dumps(result, ensure_ascii=False, indent=2)

    with TEXT_RESULT.open("a", encoding="utf-8") as file:
        file.write(
            f"[experiment {experiment:02d}]\n"
            f"original_width: {original_width}\n"
            f"vlm_image_max_width: {target_width}\n"
            f"width_ratio: {width_ratio:.6f}\n"
            f"vlm_inference_seconds: {elapsed:.6f}\n"
            f"timestamp: {timestamp}\n"
            f"json:\n{pretty_json}\n\n"
        )

    with CSV_RESULT.open("a", newline="", encoding="utf-8-sig") as file:
        csv.writer(file).writerow(
            [
                experiment,
                original_width,
                target_width,
                f"{width_ratio:.6f}",
                f"{elapsed:.6f}",
                timestamp,
                compact_json,
            ]
        )


def main() -> int:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    import cv2

    import llm.interpreter as interpreter
    import llm.server_websocket as server
    from vision.stream import WebcamStream

    initialise_result_files()
    experiment_done = threading.Event()
    result_lock = threading.Lock()
    next_result_index = 0
    experiment_original_width: int | None = None
    original_interpret_batch = server.interpret_batch

    def measured_interpret_batch(batch_input, image=None):
        nonlocal next_result_index, experiment_original_width

        if image is None:
            raise RuntimeError("test2에는 원본 웹캠 이미지가 필요합니다.")

        with result_lock:
            if next_result_index >= EXPERIMENT_COUNT:
                return original_interpret_batch(batch_input, image=image)
            if experiment_original_width is None:
                experiment_original_width = image.width
            original_width = experiment_original_width
            experiment = next_result_index + 1
            target_width = width_for_experiment(original_width, next_result_index)

        interpreter.ENABLE_VLM_IMAGE_DOWNSAMPLING = True
        interpreter.VLM_IMAGE_MAX_WIDTH = target_width
        print(
            f"[test2] 실험 {experiment}/{EXPERIMENT_COUNT} 시작: "
            f"원본 {original_width}px -> VLM 최대 {target_width}px"
        )

        started_at = time.perf_counter()
        output = original_interpret_batch(batch_input, image=image)
        elapsed = time.perf_counter() - started_at
        result = output.model_dump(mode="json")

        with result_lock:
            append_result(experiment, original_width, target_width, elapsed, result)
            next_result_index += 1
            print(f"[test2] 실험 {experiment}/{EXPERIMENT_COUNT} 저장 완료 ({elapsed:.3f}초)")
            if next_result_index == EXPERIMENT_COUNT:
                experiment_done.set()
        return output

    server.interpret_batch = measured_interpret_batch
    # 정적인 장면에서도 각 해상도 조건을 차례로 실행한다.
    server.is_significant_change = lambda _previous, _current: True

    threading.Thread(target=server.start_websocket_server, daemon=True).start()
    threading.Thread(target=server.ai_worker_thread, daemon=True).start()
    threading.Thread(target=server.vlm_worker_thread, daemon=True).start()

    stream = WebcamStream()
    print("[test2] WebSocket/Vision/VLM 시작. q를 누르면 중단합니다.")
    try:
        while not experiment_done.is_set():
            ret, frame = stream.get_frame()
            if not ret:
                time.sleep(0.01)
                continue

            with server.frame_lock:
                server.latest_frame = frame
            with server.annotated_lock:
                display = (
                    server.annotated_frame_to_display
                    if server.annotated_frame_to_display is not None
                    else frame
                )
            cv2.imshow("test2 - VLM image width 30/30 to 1/30", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[test2] 사용자 요청으로 실험을 중단합니다.")
                return 1
    except KeyboardInterrupt:
        print("\n[test2] 사용자 요청으로 실험을 중단합니다.")
        return 1
    finally:
        stream.release()
        cv2.destroyAllWindows()

    print(f"[test2] 30개 실험 완료: {TEXT_RESULT} / {CSV_RESULT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
