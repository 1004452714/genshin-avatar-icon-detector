param(
    [string]$Python = "C:\ProgramData\anaconda3\envs\avatardetect\python.exe",
    [string]$Config = "configs\train.yaml",
    [switch]$SkipCleanup,
    [switch]$NoPause
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$OutputEncoding = [System.Text.UTF8Encoding]::new()
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

$RepoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
Set-Location -LiteralPath $RepoRoot

function Zh {
    param([string]$Text)
    return [System.Text.RegularExpressions.Regex]::Unescape($Text)
}

function Pause-IfNeeded {
    if (-not $NoPause) {
        Write-Host ""
        $null = Read-Host (Zh "\u6309 Enter \u9000\u51fa")
    }
}

trap {
    Write-Host ""
    Write-Host ((Zh "\u811a\u672c\u6267\u884c\u5931\u8d25: ") + $_.Exception.Message) -ForegroundColor Red
    Pause-IfNeeded
    exit 1
}

if (-not (Test-Path -LiteralPath $Python)) {
    throw ((Zh "\u627e\u4e0d\u5230 Python: ") + $Python)
}

function Invoke-Step {
    param(
        [string]$Name,
        [scriptblock]$Script
    )
    Write-Host ""
    Write-Host "==> $Name"
    & $Script
    if ($LASTEXITCODE -ne 0) {
        throw ($Name + (Zh " \u5931\u8d25\uff0c\u9000\u51fa\u7801=") + $LASTEXITCODE)
    }
}

function Remove-TrainingArtifact {
    param([string]$Path)

    $outputs = [System.IO.Path]::GetFullPath((Join-Path $RepoRoot "outputs"))
    $fullPath = [System.IO.Path]::GetFullPath($Path)
    $prefix = $outputs.TrimEnd([System.IO.Path]::DirectorySeparatorChar) + [System.IO.Path]::DirectorySeparatorChar
    if (-not $fullPath.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw ((Zh "\u62d2\u7edd\u5220\u9664 outputs \u4e4b\u5916\u7684\u8def\u5f84: ") + $fullPath)
    }
    if (Test-Path -LiteralPath $fullPath) {
        Remove-Item -LiteralPath $fullPath -Recurse -Force
        Write-Host ((Zh "\u5df2\u5220\u9664: ") + $fullPath)
    }
}

Invoke-Step (Zh "\u751f\u6210 labels.csv") {
    & $Python scripts\generate_labels.py
}

Invoke-Step (Zh "\u8bed\u6cd5\u68c0\u67e5") {
    & $Python -m compileall -q scripts src
}

Invoke-Step (Zh "\u6570\u636e\u6821\u9a8c") {
    & $Python scripts\validate_data.py --config $Config
}

if (-not $SkipCleanup) {
    Write-Host ""
    Write-Host (Zh "==> \u6e05\u7406\u65e7\u8bad\u7ec3\u4ea7\u7269")
    New-Item -ItemType Directory -Force -Path (Join-Path $RepoRoot "outputs") | Out-Null
    Remove-TrainingArtifact (Join-Path $RepoRoot "outputs\checkpoints")
    Remove-TrainingArtifact (Join-Path $RepoRoot "outputs\prototypes.csv")
    Remove-TrainingArtifact (Join-Path $RepoRoot "outputs\avatar.onnx")
    if (Test-Path -LiteralPath (Join-Path $RepoRoot "outputs\previews")) {
        Write-Host ((Zh "\u5df2\u4fdd\u7559: ") + (Join-Path $RepoRoot "outputs\previews"))
    }
}

Invoke-Step (Zh "\u5f00\u59cb\u8bad\u7ec3") {
    & $Python scripts\train.py --config $Config
}

Invoke-Step (Zh "\u751f\u6210 prototype \u5411\u91cf\u5e93") {
    & $Python scripts\build_prototypes.py --config $Config --checkpoint outputs\checkpoints\best.pt --out outputs\prototypes.csv
}

Invoke-Step (Zh "\u5bfc\u51fa ONNX") {
    & $Python scripts\export_onnx.py --config $Config --checkpoint outputs\checkpoints\best.pt --out outputs\avatar.onnx
}

Write-Host ""
Write-Host (Zh "\u8bad\u7ec3\u6d41\u7a0b\u5b8c\u6210\u3002")
Write-Host "checkpoint: outputs\checkpoints\best.pt"
Write-Host "prototypes: outputs\prototypes.csv"
Write-Host "onnx: outputs\avatar.onnx"

Pause-IfNeeded
