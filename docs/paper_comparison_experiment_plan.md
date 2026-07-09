# 외부 베이스라인 비교평가 계획 — Ours vs {GPT-4o · LLaVA · Relative Depth}

> **핵심 태스크: 도달성(Reachability) 판단** — "이 물체를 아바타가 손 닿아 상호작용할 수 있나?"
> metric 거리(몇 m)를 직접 묻지 않는다(§2 참조). 목적(에이전트-상대 PPS 도달 어포던스)에 맞춰
> **이진 도달 판단 정확도**를 주 지표로 한다.
> ※ Ablation(−Cal/−Robust/−Stab)은 **다른 팀원 담당** → 본 계획서는 **외부 베이스라인**만.
> 프레임워크: `baseline_eval/` (본문 무수정, import만).

## 0. 비교 대상

| 대상 | 유형 | 실행 | 도달 판단 방식 |
|---|---|---|---|
| **Ours** | 기하 metric depth + 3D + PPS 0.7 m | 로컬 | metric 거리 ≤ 0.7 m → reachable |
| **GPT-4o** | image-only VLM (API, 비공개) | API 질의 | VLM에 도달성 직접 질의 |
| **LLaVA-1.5** | image-only VLM (오픈) | M4 로컬 fp16 | VLM에 도달성 직접 질의 |
| **Relative Depth** | 단안 상대깊이 (MiDaS/DPT류) | 로컬 | **순서만 가능**(§2-3) |

- GPT-4o: Ours 내부와 같은 VLM → "3D 주입 유무"만 다른 통제 비교.
- LLaVA: 오픈·재현성(SpatialVLM이 쓴 것과 동일 베이스라인).
- Relative Depth: **metric이 아니라 상대깊이만** → "상대깊이로는 절대 도달 판단 불가"를 보이는 대비군.
- SpatialVLM: 비공개 → Related Work 포지셔닝·지표 인용만(표 밖).

## 1. 평가 태스크 (질문 설계)

**주 태스크 — 도달성 이진 판단.** 각 방법에 아래를 판단시킨다:
- "손이 닿을까? / 집을 수 있을까?" (reachable / unreachable)
- **GT**: 줄자 실측 거리로 결정. **거리 ≤ 0.7 m → reachable(1), > 0.7 m → unreachable(0)** (PPS 0.7 m 기준).

**보조 태스크 — 순서(ordinal).** 두 물체 중 어느 것이 더 가까운가 → Relative Depth가 참여 가능한 유일한 절대-무관 태스크.

**증거용 probe — metric 거리(선택).** "거리 몇 m?"를 여전히 물어 **VLM의 metric 실패를 정량화**(NA-rate). 주 지표 아님.

## 2. 왜 "몇 m"를 주 질문에서 뺐나

- VLM은 metric 거리를 grounding 못 함(예비: "0.5 m 상수 응답") → 미터 질문은 **"숫자 grounding 실패"를 재는 것**이지 공간추론/도달성이 아님.
- 논문 목적은 **도달 어포던스 판단** → 도달성 이진 판단이 목적과 직결.
- VLM에도 공정(정성 판단은 잘함).

### 2-3. Relative Depth의 판단 방식 (명시)
- Relative Depth는 **순서(ordinal)만** 출력 → **절대 0.7 m 임계 판단 불가**.
- **방식 A 채택**: 도달성 이진 판단은 **"불가(N/A)"로 보고** → "상대깊이만으로는 PPS 도달 판단 불가, **metric depth가 필요**"라는 **negative result**로 활용.
- 순서 태스크(near/far)에서는 정상 참여.
- (대안 B = 상대깊이 값에 임계 학습 → 쓸 경우 "Relative+learned-threshold"로 **반드시 명시**.)

## 3. 평가 지표

| 층 | 지표 | 비고 |
|---|---|---|
| **주 (도달성)** | **Reachability Decision Accuracy**, **False-Trigger rate**(FP: 안 닿는데 reachable), **Miss rate**(FN: 닿는데 unreachable), **F1** | GT = 0.7 m PPS. False-Trigger가 안전상 핵심 |
| 보조 (순서) | near/far ordinal accuracy (2-object) | Relative Depth 참여 |
| 증거 (metric) | 거리 **NA/refusal rate**, 숫자 응답 시 MAE/RMSE/AbsRel | VLM metric 실패 정량화 |
| 안정성/비용 | (참고) 프레임 flip-rate, 지연, 호출 수 | Ours geometry=0 |
| 검정 | McNemar(도달 판단), 부트스트랩 95% CI | |

### 3-1. NA / Parsing Rule (논문 명시 필수)
- **응답 파싱**: 정규식으로 숫자+단위 추출, cm→m 변환, "yes/닿음/reachable"→1 매핑 규칙 문서화.
- **NA 처리**: 도달 판단 미응답/모호 → 별도 **NA 집계**(정확도 계산서 제외하되 **NA-rate 자체를 보고**).
- 재현성 위해 **파싱 규칙 전문을 부록에 수록**.

## 4. 데이터 수집

- person + 객체(cell phone/cup/bottle), 거리 **0.3/0.5/0.7/0.9/1.2 m**(PPS 0.7 m 경계 촘촘).
- **도달/비도달 균형**: 0.7 m 아래(reachable) / 위(unreachable) 표본 균형.
- 조건당 20프레임+, 카메라·배경 고정, person·object 좌우 분리, 정지.
- GT: 줄자 거리 + 도달성 라벨(≤0.7 m→reachable).
- 도구: `baseline_eval/collect_frames.py`.

## 5. 실행 순서

```bash
cd baseline_eval && source ../.venv/bin/activate
python metrics.py                                                    # 지표 검증
python collect_frames.py --distance 0.30 --label "cell phone" --shots 20   # 데이터(거리별)
python run_ours.py     --manifest data/manifest.csv                 # Ours (PPS 0.7m 도달 판단)
python vlm_baseline.py --manifest data/manifest.csv --backend llava --task reach   # LLaVA (무료)
python vlm_baseline.py --manifest data/manifest.csv --backend gpt4o --task reach   # GPT-4o (크레딧)
python run_relative_depth.py --manifest data/manifest.csv           # Relative Depth (순서/NA)
python compare.py --manifest data/manifest.csv --results results/   # 비교표+그림
```
> `--task reach` = 도달성 질문. metric probe는 `--task dist`로 별도(증거용).

## 6. 산출물

- **Main Table**: `Ours` vs `GPT-4o` vs `LLaVA` vs `Relative Depth`
  × {**Reach Accuracy, False-Trigger, Miss, F1**, (증거)NA-rate, (순서)ordinal Acc, 지연} + 유의성.
- **Figure**: 거리 층화 도달정확도(near/boundary 0.5~0.7/far), False-Trigger 비교, VLM NA-rate.
- **정성**: VLM "0.9 m를 reachable로 오판(False Trigger)" 사례, VLM "거리 숫자 거부" 사례.
- **Related Work**: SpatialVLM[cite] 포지셔닝(재학습 vs 추론시 주입) + in-range 지표 정렬 서술(표 밖).

## 7. 블로커 & 정직성

- GPT-4o만 크레딧 필요 → **LLaVA + Relative Depth는 지금 무료 선행 가능.**
- LLaVA는 **fp16 full-precision**(양자화 X, 베이스라인 약화 방지).
- Relative Depth 도달 판단은 "불가(N/A)"로 정직 처리(방식 A), 임계 학습 시 명시.
- SpatialVLM 수치는 결과표에 넣지 않음(다른 벤치마크).
