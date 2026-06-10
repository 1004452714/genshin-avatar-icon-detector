from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from avatardetect.validate_data import parse_args, validate_data


if __name__ == "__main__":
    args = parse_args()
    try:
        validate_data(args.config)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"错误: {exc}") from None
