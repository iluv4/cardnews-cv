# 배포 가이드 (Railway + Supabase)

팀원이 URL로 접속해 테스트하고, 평가(👍/👎)가 중앙 DB에 쌓여 데이터셋이 됩니다.

## 구성
- **백엔드/UI**: FastAPI (`service/app.py`) — `Dockerfile.web`(torch 없는 경량 이미지)로 Railway 배포. `railway.json`이 이 Dockerfile을 가리킴.
- **DB**: Supabase Postgres `public.feedback` 테이블 (서울 리전). 앱이 서버사이드에서 익명 키로 기록 → 브라우저엔 키 노출 안 됨.
- **검색**: lexical + 색 + 필터 + 유사도(저장된 임베딩, torch 불필요)로 동작. 한국어 의미검색(텍스트→임베딩)은 torch가 필요해 배포본엔 빠짐(추후 임베딩 API/워커로 추가).

## Railway 배포 (≈5분)
1. https://railway.app → **New Project → Deploy from GitHub repo** → `iluv4/cardnews-cv` 선택.
2. Railway가 `railway.json`을 읽어 `Dockerfile.web`로 빌드 (포트는 Railway가 `$PORT`로 주입, CMD가 사용).
3. **Variables**에 환경변수 2개 추가:
   - `SUPABASE_URL` = `https://athcliegcpfbwjvzriqa.supabase.co`
   - `SUPABASE_ANON_KEY` = (Supabase 대시보드 → Project Settings → **API → anon/publishable key** 복사)
   > 키는 보안상 깃에 커밋하지 않음. 대시보드에서 복사해 Railway 변수로만 넣으세요.
4. **Settings → Networking → Generate Domain** → 나온 URL을 팀에 공유.

환경변수가 없으면 앱은 자동으로 로컬 JSONL(`reflib/data/feedback.jsonl`)에 기록합니다(로컬 개발용).

## 데이터셋 export
```powershell
$env:SUPABASE_URL="https://athcliegcpfbwjvzriqa.supabase.co"
$env:SUPABASE_ANON_KEY="<anon key>"
py -3 scripts\export_feedback.py     # -> data_export\feedback.csv (+ net-per-ref 요약)
```

## 성능 업그레이드 (배포 후 언제든)
GPU 1회 패스로 #1(의미검색)·#2(전 레퍼런스 레이아웃) 해결 — `runpod_ml.sh` 참고:
```bash
bash runpod_ml.sh    # 687 템플릿 + CLIP 임베딩 -> reflib_artifacts.tar.gz
```
산출물을 repo 루트에 풀고 재배포하면 폴백이 사라지고 유사검색이 의미 기반으로 승격됩니다.

## 피드백이 성능을 올리는 방식
`/api/feedback`에 쌓인 👍/👎 net 점수가 검색 랭킹에 부스팅으로 반영(`_apply_feedback_boost`)되고, 같은 데이터를 `export_feedback.py`로 뽑아 재랭킹/평가 데이터셋으로 사용합니다.
