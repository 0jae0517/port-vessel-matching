# LLM 재정렬 학습용 RunPod 묶음 → outputs\rerank_bundle.zip
#   실행: powershell -ExecutionPolicy Bypass -File tools\make_rerank_bundle.ps1
# 대회 데이터(open/, train 행 후보)가 들어 있으므로 개인 Pod 에만 올리고, 작업 후 Pod 와 함께 삭제할 것
$root = Split-Path -Parent $PSScriptRoot
$stage = Join-Path $env:TEMP "rerank_bundle"
if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
$files = @(
    "tools\rerank_train.py", "tools\download_models.py", "tools\runpod_rerank.sh",
    "open\train.csv", "open\vessel_catalog.csv",
    "outputs\rerank\train_cands.csv", "outputs\rerank\fold0_diag.csv"
)
foreach ($f in $files) {
    $dst = Join-Path $stage $f
    New-Item -ItemType Directory -Force (Split-Path $dst) | Out-Null
    Copy-Item (Join-Path $root $f) $dst
}
$sh = Join-Path $stage "tools\runpod_rerank.sh"
[IO.File]::WriteAllText($sh, ([IO.File]::ReadAllText($sh) -replace "`r`n", "`n"), (New-Object System.Text.UTF8Encoding $false))
$zip = Join-Path $root "outputs\rerank_bundle.zip"
if (Test-Path $zip) { Remove-Item $zip }
& (Join-Path $root ".venv\Scripts\python.exe") -c "import sys, zipfile, pathlib; s = pathlib.Path(sys.argv[1]); z = zipfile.ZipFile(sys.argv[2], 'w', zipfile.ZIP_DEFLATED); [z.write(f, f.relative_to(s).as_posix()) for f in s.rglob('*') if f.is_file()]; z.close()" $stage $zip
Remove-Item -Recurse -Force $stage
"{0} ({1:N1} MB)" -f $zip, ((Get-Item $zip).Length / 1MB)
