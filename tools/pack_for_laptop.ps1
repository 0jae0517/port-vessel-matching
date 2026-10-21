# 노트북으로 옮길 git 제외 파일(데이터·모델·캐시·Claude 메모리)을 외장 드라이브 등으로 복사한다.
#   실행: powershell -ExecutionPolicy Bypass -File tools\pack_for_laptop.ps1 -Dest E:\port-vessel-transfer
#   -Minimal: 필수만 (fold 0 실험용 모델 제외, 약 21GB) / 기본: 전부 (약 35GB)
param([Parameter(Mandatory = $true)][string]$Dest, [switch]$Minimal)

$root = Split-Path -Parent $PSScriptRoot
$models = @("e5-small", "e5-small-ft", "e5-small-fth0", "e5-small-fth1", "labse", "kanana2b",
            "kanana-lora", "kanana-lorah0", "kanana-lorah1")
if (-not $Minimal) {
    # 스트레스 진단(fold 0)용. 노트북에서 몇 분이면 다시 만들 수 있다
    $models += @("e5-small-ft0", "e5-small-ft0h0", "e5-small-ft0h1", "kanana-lora0-light", "kanana-lora0h0", "kanana-lora0h1")
}
$dirs = @("open") + ($models | ForEach-Object { "models\$_" }) + @("outputs\llm", "outputs\emb", "outputs\translit", "outputs\synth", "outputs\analysis")
if (-not $Minimal) { $dirs += @("outputs\cv") }

foreach ($d in $dirs) {
    $src = Join-Path $root $d
    if (-not (Test-Path $src)) { Write-Host "건너뜀 (없음): $d"; continue }
    Write-Host "복사: $d"
    robocopy $src (Join-Path $Dest $d) /E /XD .cache /NFL /NDL /NJH /NJS /NP | Out-Null
}
# 최종 후보 제출본 (v25, zip만)
foreach ($v in @("v25")) {
    $zip = Join-Path $root "outputs\submissions\$v\submit_$v.zip"
    if (Test-Path $zip) {
        New-Item -ItemType Directory -Force (Join-Path $Dest "outputs\submissions\$v") | Out-Null
        Copy-Item $zip (Join-Path $Dest "outputs\submissions\$v\")
    }
}
# Claude 메모리 (노트북에서 프로젝트 경로에 맞는 폴더로 옮길 것 — HANDOFF.md 참고)
$mem = Join-Path $env:USERPROFILE ".claude\projects\D--dev-dacon-port-vessel-matching\memory"
if (Test-Path $mem) { robocopy $mem (Join-Path $Dest "claude-memory") /E /NFL /NDL /NJH /NJS /NP | Out-Null }

$gb = (Get-ChildItem $Dest -Recurse -File | Measure-Object Length -Sum).Sum / 1GB
Write-Host ("완료: {0} ({1:N1} GB)" -f $Dest, $gb)
