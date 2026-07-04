param(
    [string]$Python = "C:\ProgramData\anaconda3\envs\avatardetect\python.exe",
    [string]$Config = "configs\train.yaml",
    [string]$Model = "outputs\avatar.onnx",
    [string]$Prototypes = "outputs\prototypes.csv",
    [string]$RealVal = "data\real_val.csv",
    [ValidateSet("cpu", "cuda")]
    [string]$Provider = "cpu",
    [int]$TopK = 5,
    [switch]$NoPause
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$OutputEncoding = [System.Text.UTF8Encoding]::new()
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

$RepoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
Set-Location -LiteralPath $RepoRoot
$env:PYTHONPATH = Join-Path $RepoRoot "src"

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

$requiredFiles = @($Config, $Model, $Prototypes, $RealVal)
foreach ($path in $requiredFiles) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw ((Zh "\u627e\u4e0d\u5230\u6587\u4ef6: ") + $path)
    }
}

$code = @'
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import onnxruntime as ort
import pandas as pd

from avatardetect.config import load_config
from avatardetect.infer import (
    decode_vector,
    element_probabilities_from_outputs,
    preprocess_inputs,
    rank_with_element_head,
)


def appearance_id_from_row(row: pd.Series) -> str:
    value = str(row.get("appearance_id", "") or "")
    if value:
        return value
    return f"{row['character_id']}_{row['skin_id']}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prototypes", required=True)
    parser.add_argument("--real-val", required=True)
    parser.add_argument("--provider", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    cfg = load_config(args.config)
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if args.provider == "cuda" else ["CPUExecutionProvider"]
    session = ort.InferenceSession(args.model, providers=providers)
    input_names = [item.name for item in session.get_inputs()]

    real = pd.read_csv(args.real_val, dtype=str).fillna("")
    protos = pd.read_csv(args.prototypes, dtype=str).fillna("")
    matrix = np.stack([decode_vector(x) for x in protos["embedding"]]).astype(np.float32)

    correct = 0
    misses = []
    gaps = []
    scores_top1 = []

    for sample_index, row in real.iterrows():
        image, element_image = preprocess_inputs(row["image_path"], cfg)
        feed = {input_names[0]: image}
        if len(input_names) > 1:
            feed[input_names[1]] = element_image
        outputs = session.run(None, feed)
        embedding = outputs[0][0].astype(np.float32)
        embedding = embedding / max(np.linalg.norm(embedding), 1e-12)
        scores = matrix @ embedding
        element_probs, predicted_element = element_probabilities_from_outputs(outputs, protos)
        element_min_probability = float(cfg.get("inference", {}).get("element_min_probability", 0.35))
        ranked_rows = rank_with_element_head(
            protos,
            scores,
            args.top_k,
            element_probs,
            predicted_element,
            element_min_probability,
        )
        pred, score = ranked_rows[0]
        second_score = ranked_rows[1][1] if len(ranked_rows) > 1 else score
        gap = float(score - second_score)
        expected_variant_id = str(row.get("expected_variant_id", "") or "")
        expected_appearance_id = str(row.get("expected_appearance_id", "") or "")
        pred_appearance_id = appearance_id_from_row(pred)
        if expected_variant_id:
            ok = str(pred.get("variant_id", pred_appearance_id)) == expected_variant_id
            expected_text = expected_variant_id
        else:
            ok = pred_appearance_id == expected_appearance_id
            expected_text = expected_appearance_id
        correct += int(ok)
        gaps.append(gap)
        scores_top1.append(score)

        if sample_index > 0:
            print()
        print(f"{Path(row['image_path']).name}\t\u671f\u671b={expected_text}\t{'OK' if ok else 'MISS'}")

        candidates = []
        for rank, (candidate, candidate_score) in enumerate(ranked_rows, start=1):
            candidate_skin_name = candidate.get("skin_name", "")
            candidate_appearance_id = appearance_id_from_row(candidate)
            candidate_variant_id = candidate.get("variant_id", candidate_appearance_id)
            candidate_element_type = candidate.get("element_type", "")
            candidates.append(
                f"{candidate_score:.4f}:{candidate['character_name']}:{candidate_variant_id}"
            )
            print(
                f"\tTop{rank}\t"
                f"\u5206\u6570={candidate_score:.4f}\t"
                f"\u540d\u79f0={candidate['character_name']}\t"
                f"\u76ae\u80a4={candidate_skin_name}\t"
                f"\u5143\u7d20={candidate_element_type}\t"
                f"character_id={candidate['character_id']}\t"
                f"appearance_id={candidate_appearance_id}\t"
                f"variant_id={candidate_variant_id}\t"
                f"element_type={candidate_element_type}\t"
                f"skin_id={candidate['skin_id']}"
            )

        if not ok:
            misses.append((Path(row["image_path"]).name, expected_text, candidates))

    total = len(real)
    print(f"top1={correct}/{total}")
    if total:
        print(f"\u5e73\u5747\u5206\u6570={float(np.mean(scores_top1)):.4f}")
        print(f"\u5e73\u5747\u5dee\u8ddd={float(np.mean(gaps)):.4f}")
        min_gap_idx = int(np.argmin(gaps))
        print(f"\u6700\u5c0f\u5dee\u8ddd={gaps[min_gap_idx]:.4f} \u6837\u672c={Path(real.iloc[min_gap_idx]['image_path']).name}")

    if misses:
        print("MISS_DETAIL")
        for name, expected, candidates in misses:
            print(f"{name}\t\u671f\u671b={expected}\t\u5019\u9009=" + " | ".join(candidates))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'@

$code | & $Python - --config $Config --model $Model --prototypes $Prototypes --real-val $RealVal --provider $Provider --top-k $TopK
if ($LASTEXITCODE -ne 0) {
    throw ((Zh "real_val \u6d4b\u8bd5\u672a\u5168\u90e8\u901a\u8fc7\uff0c\u9000\u51fa\u7801=") + $LASTEXITCODE)
}

Pause-IfNeeded
