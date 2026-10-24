# 임대 GPU(RunPod 등)에서 카탈로그 한글 표기표를 만들 때 올릴 최소 파일 묶음 → outputs\runpod_bundle.zip
#   실행: powershell -ExecutionPolicy Bypass -File tools\make_runpod_bundle.ps1
# 대회 데이터(open/)가 들어 있으므로 개인 Pod 에만 올리고, 작업 후 Pod 와 함께 삭제할 것
$root = Split-Path -Parent $PSScriptRoot
$stage = Join-Path $env:TEMP "runpod_bundle"
if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
$files = @(
    "solution\match_lib.py", "solution\augment.py",
    "analysis\synth_hangul.py", "analysis\pinyin_hangul.py",
    "tools\translit_catalog.py", "tools\download_models.py", "tools\runpod_translit.sh", "tools\runpod_translit8b.sh",
    "outputs\translit\suspects.txt",
    "open\train.csv", "open\vessel_catalog.csv"
)
foreach ($f in $files) {
    $dst = Join-Path $stage $f
    New-Item -ItemType Directory -Force (Split-Path $dst) | Out-Null
    Copy-Item (Join-Path $root $f) $dst
}
# 리눅스에서 실행할 셸 스크립트는 LF 줄바꿈으로
foreach ($s in "tools\runpod_translit.sh", "tools\runpod_translit8b.sh") {
    $sh = Join-Path $stage $s
    [IO.File]::WriteAllText($sh, ([IO.File]::ReadAllText($sh) -replace "`r`n", "`n"), (New-Object System.Text.UTF8Encoding $false))
}
$zip = Join-Path $root "outputs\runpod_bundle.zip"
if (Test-Path $zip) { Remove-Item $zip }
# 리눅스에서 풀 수 있게 / 경로로 압축 (PowerShell 5.1 Compress-Archive 는 \ 경로를 씀)
& (Join-Path $root ".venv\Scripts\python.exe") -c "import sys, zipfile, pathlib; s = pathlib.Path(sys.argv[1]); z = zipfile.ZipFile(sys.argv[2], 'w', zipfile.ZIP_DEFLATED); [z.write(f, f.relative_to(s).as_posix()) for f in s.rglob('*') if f.is_file()]; z.close()" $stage $zip
Remove-Item -Recurse -Force $stage
"{0} ({1:N1} MB)" -f $zip, ((Get-Item $zip).Length / 1MB)
