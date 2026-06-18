# 다음 할 일 (NEXT STEPS)

> 상세 설계는 [`EVALUATION_DESIGN.md`](EVALUATION_DESIGN.md) 참고. 이 파일은 "다음에 어디서 시작" 빠른 포인터.

## 지금까지 (이번 세션)
- 국문 2단 기말 보고서 완성: `paper/cardnews_report_ko.pdf` (빌더: `paper/build_report_ko.py`)
- 탑티어 보강용 평가·실험 설계 문서 작성: `EVALUATION_DESIGN.md`

## 다음 착수 순서 (임팩트 대비 비용)
1. **[코드·반나절] 덱 단위 분할 패치** — `build_datasets.py:189-192`가 이미지 단위 셔플이라 같은 덱이 train/test 양쪽에 = **누수**. `EVALUATION_DESIGN.md` Part C.1에 패치 코드 있음. **이거 먼저 안 고치면 109 vs 687 수치 전부 의심받음.**
2. **[라벨링·시간소요 → 지금 시작] 골드 테스트셋 250장** — Part A. Label Studio, 4클래스(logo/underlay 첫 실라벨), 층화+덱 분리, 2인 IAA(κ).
3. **[클라우드] 109/250/450/687 학습곡선** — Part C. RunPod 4090. `build_datasets.py`(패치 후) → `compare_data_scale.py`. 공통 테스트를 골드로 교체.
4. **[코드] `eval_gen/metrics.py`** — 생성 자동지표: WCAG 명암비 / saliency 회피 / overlap / alignment (기존 `saliencymaps_*`, `region_luma` 재활용).
5. **[설계 일찍] 생성 인간 스터디(2AFC) + VLM-judge** — Part B. 모집·시간 필요하니 프로토콜 먼저.

## 빠른 재시작 명령
```bash
py -3 paper/build_report_ko.py        # 국문 보고서 재빌드
# RunPod: pip install -U ultralytics easyocr ... ; python build_datasets.py ; python compare_data_scale.py
```

## 메모
- DS-GAN/생성 약점은 숨기지 말고 **정량적으로 정직하게** → "왜 템플릿 엔진인가"의 근거로.
- 현실 venue: **ICDAR / CVPR·ICCV 워크샵 / WACV**. 데이터·평가 갖춰지면 메인 트랙.
- `slides/`(별도 PPT 작업)는 이번 커밋에 미포함 — 원하면 따로 푸시.
