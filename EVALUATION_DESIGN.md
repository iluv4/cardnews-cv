# 평가·실험 설계 문서 — cardnews-cv 탑티어 보강

> 목적: 현재 "의사 레이블 자기참조 평가 + 정성적 생성 결과"에서, **(A) 골드 테스트셋, (B) 생성 정량 평가, (C) 109 vs 687 마무리**로 끌어올려 CVPR/ICDAR/WACV급 엄밀성을 확보한다. 본 문서는 일반론이 아니라 **현재 코드/데이터에 바로 꽂히는 실행 설계**다.

---

## 0. 현재 상태 & 핵심 갭

| 구성요소 | 상태 | 근거(파일) |
|---|---|---|
| 검출기(YOLOv8n, e15) | ✅ mAP@50-95 **0.718**, mAP@50 0.854, P 0.785, R 0.842 / 5-fold 0.611±0.049 | `results/ablation.csv`, `results/e15_long300_card/` |
| 검출 평가 기준 | ⚠️ **전부 의사 레이블(EasyOCR)** — 사람 정답 0장 | `autolabel.py`, `build_datasets.py` |
| 클래스 | ⚠️ title/body만 실제 존재. **logo/underlay는 정의만 있고 미라벨**(혼동행렬에서 빈 칸) | `dataset.yaml`, `confusion_matrix.png` |
| 109 vs 687 | ⚠️ 코드는 있으나 **미실행**(`build_datasets.py` 안 돌림) + **덱 단위 분할 누수 버그** | `build_datasets.py:189-192`, `compare_data_scale.py` |
| 생성 정량 평가 | ❌ **0** — 템플릿 엔진·DS-GAN 모두 정성 결과뿐 | `gen/`, `service/cardgen/` |

**3대 갭 → 3대 작업(A/B/C).** 추가로 두 가지 교차 이슈: **(i) logo/underlay 실라벨 부재**, **(ii) 덱 누수**. 둘 다 A/C에서 함께 해결한다.

---

## Part A. 골드(사람) 테스트셋 라벨링 프로토콜

### A.1 왜 필요한가 (리뷰어 설득 포인트)
지금은 *EasyOCR이 만든 라벨로 학습한 검출기를, 같은 방식의 EasyOCR 라벨로 평가*한다 → **순환 평가(self-referential)**. 리뷰어 1순위 지적. 골드셋은 두 가지를 가능하게 한다:
1. **검출기의 절대 성능**을 사람 정답 기준으로 보고 (진짜 mAP).
2. **의사 레이블 자체의 품질 정량화** — 골드 대비 EasyOCR 라벨의 P/R. 이건 단점이 아니라 *기여*가 된다: "라벨 노이즈가 X% 있음에도 검출기가 Y 성능"이라는 서사.

### A.2 표본 설계
- **크기: 250장** (권장). 근거: title/body 박스가 장당 평균 ~8.5개(284+641 / 109) → 약 2,000박스로 클래스별 mAP 신뢰구간이 안정. logo/underlay는 희소하므로 250장은 되어야 의미 있는 양성 표본 확보.
- **층화 추출(stratified)** — 세 축으로 균형:
  - **출처**: 공공 데이터 / 직접 제작 / 인스타 RapidAPI 크롤링
  - **아키타입**: cover / statement / list / body / mixed (이미 `service/library/templates.json`의 `archetype` 필드로 분류 가능)
  - **톤**: dark / light (`templates.json`의 `is_dark`, 또는 `reflib/features.py`의 `dark`)
- **★ 덱 단위 분리**: 같은 `series_folder`(덱)는 통째로 한쪽에만. 골드 테스트셋에 들어간 덱은 **어떤 학습 풀에도 등장 금지**. (`images_new/_mapping_new.csv`의 `series_folder`로 그룹핑)
- **통합 설계**: 이 골드셋이 곧 **Part C의 공통 테스트셋**이 되도록 한다(아래 C). → 의사 레이블 공통 테스트(현재 86장)를 골드로 교체.

### A.3 클래스 정의 & 라벨링 가이드
| 클래스 | 정의 | 경계 규칙 |
|---|---|---|
| title | 페이지의 주 제목/헤드라인 블록 | 줄이 아니라 **블록 단위**(의사 레이블과 동일 기준). 시각적으로 한 덩어리인 제목은 1박스 |
| body | 본문/설명/리스트 텍스트 블록 | 문단/리스트 항목 묶음을 블록으로 |
| logo | 브랜드/계정 로고·워터마크 | 텍스트형 로고 포함, 페이지번호와 구분 |
| underlay | 텍스트 가독성을 위한 받침판/스크림/말풍선/패널 | 순수 장식(배경 일러스트)과 구분 |

- **엣지 케이스 명문화**: 페이지 번호(제외), CTA 버튼(underlay+body), 말풍선(underlay), 표/도표 내부 텍스트(body로 묶되 도표 축 라벨은 제외), 이모지/아이콘(제외).
- **파일럿 20장**으로 가이드 보정 후 본 라벨링.

### A.4 도구 & 산출물
- **도구**: Label Studio(권장, 무료, YOLO export) 또는 Roboflow. 검수는 기존 `viz_labels.py` 재사용(빨강=title, 초록=body 오버레이) — logo/underlay 색만 추가.
- **산출물**: `data_gold/images/`, `data_gold/labels/`(YOLO txt), `dataset_gold.yaml`(4클래스), `data_gold/iaa_report.md`.

### A.5 품질관리 — 어노테이터 일치도(IAA)
- **2인 독립 라벨** → 매칭(IoU≥0.5) 후:
  - **박스 일치율**(detection-style): 매칭된 박스 비율
  - **클래스 일치 κ**(Cohen's kappa): 매칭 박스의 클래스 라벨 일치
  - 목표: κ ≥ 0.8(title/body), logo/underlay는 희소하므로 별도 보고.
- **불일치 조정(adjudication)**: 제3자 또는 합의로 최종 골드 확정.
- 논문에 IAA 보고 → 데이터 신뢰성 근거.

### A.6 평가에 쓰는 법 (핵심 산출 2개)
1. **검출기 절대 성능**: e15 모델을 `dataset_gold.yaml`로 평가 → "진짜" mAP. 의사-라벨 mAP(0.718)와의 **gap**을 보고.
2. **의사 레이블 품질**: 골드를 기준으로 EasyOCR 의사 레이블의 P/R/F1 측정 → 라벨 노이즈 정량화(서사 자산).

---

## Part B. 생성 정량 평가 설계

> 생성은 "그림만 좋아 보인다"가 가장 큰 약점. **자동지표(무참조) + VLM-judge + 인간 스터디**의 3단으로 설계한다. 기존에 *이미 계산되는 신호*(saliency map, gradient band, palette, 박스)를 재활용해 구현 비용을 최소화한다.

### B.1 평가 조건(비교군)
| # | 조건 | 출처 |
|---|---|---|
| 1 | **템플릿 엔진(ours)** | `service/cardgen/`, `gen/copy_layout.py` |
| 2 | DS-GAN (PosterLayout 적응) | `gen/build_posterlayout_dataset.py` + posterlayout 학습 |
| 3 | PosterLayout 원본(사전학습 가중치) | 가능 시 — 상한 비교 |
| 4 | 휴리스틱/랜덤 배치 | 하한(lower bound) 베이스라인 — **반드시 포함**(지표 보정용) |
| 5 | 실제 카드(real) | 상한/기준(upper bound) |

동일 **콘텐츠 세트**(같은 제목/본문 텍스트 N개)를 모든 조건에 통과시켜 카드 생성 → 공정 비교.

### B.2 자동 레이아웃 지표 (무참조, 사람 불필요)
PosterLayout 평가 관례(가독성/겹침/정렬/하단정렬 등)와 **정렬되게** 설계해 리뷰어 친숙도 확보. 신규 모듈 `eval_gen/metrics.py`.

| 지표 | 정의 | 재활용 신호 |
|---|---|---|
| **M1 명암비(WCAG)** | 텍스트 박스 전경/배경 상대휘도비, **AA 4.5:1 통과율**. 현재 `region_luma()`의 단순 임계(luma<130)를 정식 WCAG 공식으로 교체 | `service/cardgen/placement.py::region_luma` → WCAG 상대휘도 |
| **M2 돌출영역 회피** | 텍스트 박스 내 평균 saliency / 전체 평균. 낮을수록 좋음(얼굴·핵심 가림 회피) | 이미 생성됨: `Dataset/.../saliencymaps_basnet`(u2net), `saliencymaps_pfpn`(isnet) |
| **M3 요소 겹침(overlap)** | 박스 쌍 간 IoU/중첩 면적 비율. 0이 이상 | 생성 박스(템플릿 블록 / DS-GAN `boxes-*.pt`) |
| **M4 정렬(alignment)** | 좌/중/우 엣지 정렬 일관성(엣지 좌표 분산 최소화) | 생성 박스 좌표 |
| **M5 여백/점유율** | 캔버스 텍스트 점유율, 가장자리 침범량 | 생성 박스 |
| **M6 가독성 프록시** | 폰트 크기/행간/문자폭 대비 박스 면적(읽기 난이도) | `components.fit_lines()` 결과 |

- **검증(sanity)**: 조건 4(랜덤) ≪ 조건 1/5 이어야 지표가 의미 있음. 조건 5(real)가 상한.
- **WCAG 공식**(M1): `L = 0.2126·R'+0.7152·G'+0.0722·B'`(선형화 후), 명암비 `(L_light+0.05)/(L_dark+0.05)`, AA=4.5, AAA=7. 팀 Pixstagram 보고서도 WCAG 4.5:1을 썼으므로 **일관**.

### B.3 VLM-judge (자동 프록시 — 팀 자산 활용)
- **gpt-4.1-mini Vision**으로 쌍대 비교(A/B): 루브릭 = 가독성 / 디자인 완성도 / "실제 카드 같은가".
- **편향 제거**: 좌우 순서 무작위 + 양방향(A·B / B·A) 평가 후 평균, 위치편향 보고.
- 용도: 인간 스터디 **사전 스크리닝 + 스케일업**. 반드시 **인간 결과와의 상관(Spearman ρ)** 보고 → judge 신뢰성 입증.

### B.4 인간 선호도 스터디 (CVPR 설득력의 핵심)
- **형식**: 2AFC pairwise(두 카드 중 선택), forced choice.
- **규모**: 평가자 **20명**, 조건쌍 = C(5,2)=10쌍 × 콘텐츠 30개 = 300쌍, 각 쌍 ≥5명 중복 → 통계력 확보. (전부는 과하면 ours vs {DS-GAN, real, 랜덤} 4쌍으로 축소 가능)
- **차원**: ① 가독성 ② 디자인 품질 ③ 실제같음(realism).
- **분석**: 조건별 **win-rate**, **Bradley–Terry** 능력점수, 유의성(부트스트랩 CI / 이항검정), 평가자 일치도 **Krippendorff α**.
- **모집**: Prolific(유료) 또는 수업 동료/교내. 카드·쌍 수와 1인 소요시간(≈15분) 사전 계산.
- **산출물**: `eval_gen/human_study/`(자극 세트, 응답 CSV, 분석 노트북).

### B.5 DS-GAN 정직한 처리
DS-GAN은 한글 글리프 렌더링 실패·불안정으로 **배포 불가**. 평가에서 (a) **박스 레벨 지표(M3/M4)** 로는 비교하되, (b) **렌더 텍스트 가독성(M1)·인간 스터디**에서 명확히 열세임을 보여 "왜 템플릿 엔진인가"를 정량 입증. 이게 논문의 핵심 비교 서사.

---

## Part C. 109 vs 687 마무리 (+ 덱 누수 수정)

### C.1 ★ 먼저 고칠 버그 — 덱 단위 분할
현재 `build_datasets.py:189-192`:
```python
new_shuf = new[:]
random.Random(SEED).shuffle(new_shuf)      # ← 이미지 단위 셔플
test_new = set(new_shuf[:n_test])
```
578장은 66개 **덱**(덱당 4~11페이지)이고 같은 덱은 템플릿·색·스타일이 동일 → **이미지 단위 분할 시 같은 덱이 train/test 양쪽에 = 누수 → mAP 과대평가.**

**수정**: `images_new/_mapping_new.csv`의 `series_folder`로 묶어 **덱 단위로 hold-out**.
```python
import csv, collections, random
rows = list(csv.DictReader(open("images_new/_mapping_new.csv", encoding="utf-8")))
deck_of = {r["new_name"]: r["series_folder"] for r in rows}
decks = collections.defaultdict(list)
for p in new:                                   # p: images_new/<new_name>
    decks[deck_of[os.path.basename(p)]].append(p)
deck_ids = sorted(decks)
random.Random(SEED).shuffle(deck_ids)           # ← 덱 단위 셔플
n_test_decks = max(1, round(len(deck_ids) * TEST_FRAC))   # 66*0.15 ≈ 10 덱
test_decks = set(deck_ids[:n_test_decks])
test_new = {p for d in test_decks for p in decks[d]}
train_new = [p for p in new if p not in test_new]
```
→ 약 10개 덱(≈80~90장)이 공통 테스트, 나머지 덱이 학습. 누수 0.

### C.2 골드 테스트셋과 통합
- **공통 테스트셋 = Part A 골드셋**(또는 골드셋 중 인스타/신규 출처 부분)으로 교체. 즉 `dataset_base109.yaml` / `dataset_full687.yaml`의 `val`을 `data_gold/`로 지정.
- 효과: 109 vs 687 비교도 **사람 정답 기준**으로 측정 → 결과 신뢰성 급상승. (의사 레이블 공통 테스트는 보조 지표로 병행 가능)

### C.3 학습 곡선(data scaling) 확장
단일 109 vs 687 비교를 넘어 **데이터 규모 곡선**으로:
- 학습 풀 크기: **109 / 250 / 450 / 687**(또는 덱 수 기준 등간) — 덱 단위 서브샘플링.
- 각 점 × **seed 3개**(0,1,2) → mAP 평균±표준편차 곡선.
- 해석: 포화(saturation) 여부 → "더 모으면 도움되나"에 정량 답. CVPR에서 강한 그림.

### C.4 실행(클라우드)
```bash
# RunPod RTX 4090 (~$0.4-0.7/hr). 로컬 RTX3050은 torch 미설치/메모리 부족 → 클라우드 필수
pip install -U ultralytics easyocr pillow numpy pandas matplotlib
python build_datasets.py            # ← C.1 덱 분할 패치 적용 후
EPOCHS=300 SEEDS=0,1,2 BATCH=16 WORKERS=8 AMP=1 python compare_data_scale.py
# 학습 곡선용: 서브샘플 풀별로 반복(스크립트에 POOL_SIZES 인자 추가 권장)
```
- 비용/시간: 점 4개 × seed 3 × 300epoch ≈ 12런, 4090에서 ~수 시간(런당 1.4분 학습이지만 imgsz·검증 포함 시 여유). A100이면 더 빠름.
- 산출물: `results/data_scale_comparison.csv`, `COMPARISON.md`, `results/data_scale_chart.png`.

---

## Part D. 통합 순서·의존성·프레이밍

### D.1 의존성 & 우선순위
```
A. 골드 250장 라벨링 ─┬─→ (검출기 절대성능 / 의사라벨 품질)
                      └─→ C.2 공통 테스트 교체 ─→ C.3 학습곡선(109..687)
C.1 덱분할 패치 ─→ build_datasets.py ─→ compare_data_scale.py
B. 자동지표 구현(eval_gen/metrics.py) ─→ B.3 VLM-judge ─→ B.4 인간 스터디
```
**권장 착수 순서**(임팩트 대비 비용):
1. **C.1 덱분할 패치**(반나절, 코드만) — 가장 싸고, 안 고치면 모든 수치가 의심받음.
2. **A 골드셋 250장**(가장 큰 임팩트) — 라벨링이 시간 소요라 **지금 바로 시작**.
3. **C.2+C.3 109 vs 687 + 학습곡선**(A 일부 완료되면 즉시).
4. **B.2 자동지표**(코드, saliency·WCAG 재활용으로 빠름).
5. **B.4 인간 스터디**(가장 설득력, 모집·시간 필요 → 일찍 설계 시작).

### D.2 논문 프레이밍 (현실적 venue)
- **1순위 현실 타깃**: **ICDAR**(문서/레이아웃/OCR에 정확히 부합), **CVPR/ICCV 워크샵**(graphic design, vision-for-UI/document), **WACV**(응용 친화).
- **포지셔닝**: "한국어 카드뉴스 **레이아웃 이해 벤치마크 + 검출-기반 템플릿 생성**" — (i) 골드 라벨 벤치마크, (ii) 의사라벨→골드 전이의 라벨노이즈 분석, (iii) 생성 정량 평가로 GAN 대비 템플릿 엔진 우위 입증, (iv) 덱 시계열 분석. 데이터·평가가 갖춰지면 메인 트랙도 시도 가치.
- **신규성 보강**: 단순 "엔지니어링"으로 안 보이게 — *돌출/명암비 인지 배치를 정량 지표로 최적화*하고 *위치-조건부(표지/내지) 템플릿*을 시계열 분석과 연결하면 방법적 기여가 선다.

### D.3 리스크 & 정직성
- 소규모 데이터(687) → 벤치마크/분석 논문 프레이밍이 안전. "SOTA 경쟁"보다 "새 도메인·새 평가".
- DS-GAN 비현실성 → 숨기지 말고 *정량적으로* 보여 동기로 전환.
- 골드 라벨 일관성(logo/underlay 희소) → IAA·표본수로 방어.

---

## 부록. 즉시 실행 체크리스트

- [ ] **C.1** `build_datasets.py` 덱 단위 분할 패치(위 코드) + 로컬 스모크(`EPOCHS=3 SEEDS=0`).
- [ ] **A** 라벨링 가이드(클래스 정의·엣지케이스) 1쪽 확정 → 파일럿 20장 → IAA 점검 → 250장.
- [ ] **A** Label Studio 프로젝트 셋업, `viz_labels.py`에 logo/underlay 색 추가.
- [ ] **C.2** `dataset_*.yaml`의 `val`을 `data_gold/`로 교체.
- [ ] **C.3** `compare_data_scale.py`에 `POOL_SIZES=109,250,450,687` 서브샘플 인자 추가(덱 단위).
- [ ] **B.2** `eval_gen/metrics.py` 신설: M1(WCAG)·M2(saliency)·M3(overlap)·M4(align)·M5·M6.
- [ ] **B.3** VLM-judge 스크립트 + 인간 상관 측정.
- [ ] **B.4** 인간 스터디 프로토콜·자극세트·IRB/동의 문구.
- [ ] RunPod 런북 갱신(`RUNPOD.md`)에 위 명령 추가.

### 핵심 파일 참조
- 라벨/검출: `autolabel.py`, `build_datasets.py`, `compare_data_scale.py`, `train_detector.py`, `dataset.yaml`, `images_new/_mapping_new.csv`
- 생성/신호: `service/cardgen/{placement,components,render,from_template}.py`, `gen/{copy_layout,extract_templates,build_posterlayout_dataset}.py`, `service/library/templates.json`, `Dataset/.../saliencymaps_*`
- 결과: `results/ablation.csv`, `results/e15_long300_card/`, `results/kfold_summary.txt`
