# cardnews-cv — 라이브 데모 런처 (발표용)
#   PowerShell에서:  .\run_demo.ps1
# 서버를 띄우고 3초 뒤 브라우저로 http://127.0.0.1:8000/ 를 엽니다. Ctrl+C로 종료.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# 의존성(최초 1회): pip install fastapi "uvicorn[standard]"
python -c "import fastapi, uvicorn" 2>$null
if ($LASTEXITCODE -ne 0) {
  Write-Host "FastAPI/uvicorn 설치 중..." -ForegroundColor Yellow
  py -3 -m pip install -q fastapi "uvicorn[standard]"
}

# 인덱스가 없으면 생성(검색 라이브러리)
if (-not (Test-Path "reflib\data\index.json")) {
  Write-Host "레퍼런스 인덱스 생성 중..." -ForegroundColor Yellow
  py -3 reflib\build_index.py
  py -3 reflib\cluster.py --k 8
}

Start-Job { Start-Sleep 3; Start-Process "http://127.0.0.1:8000/" } | Out-Null
Write-Host "데모: http://127.0.0.1:8000/   (종료: Ctrl+C)" -ForegroundColor Green
py -3 -m uvicorn service.app:app --port 8000
