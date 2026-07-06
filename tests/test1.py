"""VLM reasoning 단어 수(30 -> 0) 실험 실행기.

프로젝트 루트에서 다음 명령으로 실행한다.

    python -m tests.test1

각 단어 제한에서 성공한 VLM 추론을 한 번씩 수집하고 종료한다.
"""

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
TEXT_RESULT = RESULT_DIR / "test1_res.txt"
CSV_RESULT = RESULT_DIR / "test1_res.csv"
WORD_LIMITS = tuple(range(30, -1, -1))


def reasoning_prompt(word_limit: int) -> str:
    if word_limit == 0:
        return """
[Micro CoT 출력 제한 - 실험 조건]
각 객체의 `reasoning`은 반드시 빈 문자열(`""`)로 출력하라.
`reasoning` 외에는 JSON 스키마가 요구하는 값만 출력하라.
"""
    return f"""
[Micro CoT 출력 제한 - 실험 조건]
내부 판단 과정이나 배경을 길게 설명하지 마라.
각 객체의 `reasoning`은 핵심 시각 근거와 결론만 담아 반드시 한국어 {word_limit}단어 이내로 작성하라.
단어 수는 공백으로 구분하여 계산한다.
`reasoning` 외에는 JSON 스키마가 요구하는 값만 출력하라.
"""


def initialise_result_files() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    TEXT_RESULT.write_text(
        "test1: reasoning word-limit experiment (30 -> 0)\n"
        f"started_at: {started_at}\n\n",
        encoding="utf-8",
    )
    with CSV_RESULT.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["experiment", "word_limit", "vlm_inference_seconds", "timestamp", "json"]
        )


def append_result(experiment: int, word_limit: int, elapsed: float, result: dict) -> None:
    timestamp = datetime.now().astimezone().isoformat(timespec="milliseconds")
    logged_result = {
        "vlm_inference_seconds": round(elapsed, 6),
        "vlm_result": result,
    }
    compact_json = json.dumps(logged_result, ensure_ascii=False, separators=(",", ":"))
    pretty_json = json.dumps(result, ensure_ascii=False, indent=2)

    with TEXT_RESULT.open("a", encoding="utf-8") as file:
        file.write(
            f"[experiment {experiment:02d}]\n"
            f"word_limit: {word_limit}\n"
            f"vlm_inference_seconds: {elapsed:.6f}\n"
            f"timestamp: {timestamp}\n"
            f"json:\n{pretty_json}\n\n"
        )

    with CSV_RESULT.open("a", newline="", encoding="utf-8-sig") as file:
        csv.writer(file).writerow(
            [experiment, word_limit, f"{elapsed:.6f}", timestamp, compact_json]
        )


def main() -> int:
    # 직접 파일로 실행해도 프로젝트 패키지를 찾을 수 있게 한다.
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
    original_interpret_batch = server.interpret_batch

    def measured_interpret_batch(batch_input, image=None):
        nonlocal next_result_index

        with result_lock:
            word_limit = WORD_LIMITS[next_result_index]
            experiment = next_result_index + 1

        # server_websocket이 가져온 함수도 내부적으로 이 모듈의 전역 프롬프트를 읽는다.
        interpreter.USE_MICRO_COT = True
        interpreter.MICRO_COT_PROMPT = reasoning_prompt(word_limit)
        print(f"[test1] 실험 {experiment}/31 시작: reasoning 최대 {word_limit}단어")

        started_at = time.perf_counter()
        output = original_interpret_batch(batch_input, image=image)
        elapsed = time.perf_counter() - started_at
        result = output.model_dump(mode="json")

        with result_lock:
            # 성공한 추론만 다음 제한값으로 진행한다. API 오류는 같은 조건으로 재시도된다.
            append_result(experiment, word_limit, elapsed, result)
            next_result_index += 1
            print(f"[test1] 실험 {experiment}/31 저장 완료 ({elapsed:.3f}초)")
            if next_result_index == len(WORD_LIMITS):
                experiment_done.set()
        return output

    server.interpret_batch = measured_interpret_batch
    # 같은 물체를 유지해도 매 조건을 연속 실험할 수 있게 delta trigger만 해제한다.
    server.is_significant_change = lambda _previous, _current: True

    threading.Thread(target=server.start_websocket_server, daemon=True).start()
    threading.Thread(target=server.ai_worker_thread, daemon=True).start()
    threading.Thread(target=server.vlm_worker_thread, daemon=True).start()

    stream = WebcamStream()
    print("[test1] WebSocket/Vision/VLM 시작. q를 누르면 중단합니다.")
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
            cv2.imshow("test1 - reasoning word limit 30 to 0", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[test1] 사용자 요청으로 실험을 중단합니다.")
                return 1
    except KeyboardInterrupt:
        print("\n[test1] 사용자 요청으로 실험을 중단합니다.")
        return 1
    finally:
        stream.release()
        cv2.destroyAllWindows()

    print(f"[test1] 31개 실험 완료: {TEXT_RESULT} / {CSV_RESULT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
