from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = "configs/train.yaml"
DEFAULT_MODEL = "outputs/avatar.onnx"
DEFAULT_PROTOTYPES = "outputs/prototypes.csv"
DEFAULT_CHECKPOINT = "outputs/checkpoints/best.pt"
DEFAULT_REAL_VAL = "data/real_val.csv"


def python_env() -> dict[str, str]:
    env = os.environ.copy()
    src = str(ROOT / "src")
    current = env.get("PYTHONPATH")
    env["PYTHONPATH"] = src if not current else src + os.pathsep + current
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env


def run_step(name: str, command: list[str]) -> None:
    print("", flush=True)
    print(f"==> {name}", flush=True)
    result = subprocess.run(command, cwd=ROOT, env=python_env())
    if result.returncode != 0:
        raise SystemExit(f"{name} 失败，退出码={result.returncode}")


def safe_remove(path: Path, allowed_roots: list[Path]) -> None:
    full_path = path.resolve()
    allowed = [root.resolve() for root in allowed_roots]
    if not any(full_path == root or root in full_path.parents for root in allowed):
        raise ValueError(f"拒绝删除非生成目录: {full_path}")
    if not full_path.exists():
        return
    if full_path.is_dir():
        shutil.rmtree(full_path)
    else:
        full_path.unlink()
    print(f"已删除: {full_path}")


def command_prepare(args: argparse.Namespace) -> None:
    run_step("生成 labels.csv", [sys.executable, "scripts/generate_labels.py"])
    run_step("语法检查", [sys.executable, "-m", "compileall", "-q", "scripts", "src", "avatardetect.py"])
    run_step("数据校验", [sys.executable, "scripts/validate_data.py", "--config", args.config])


def remove_training_outputs() -> None:
    outputs = ROOT / "outputs"
    outputs.mkdir(exist_ok=True)
    for path in [
        outputs / "checkpoints",
        outputs / "prototypes.csv",
        outputs / "avatar.onnx",
    ]:
        safe_remove(path, [outputs])


def command_train(args: argparse.Namespace) -> None:
    command_prepare(args)
    if not args.skip_cleanup:
        print("")
        print("==> 清理旧训练产物", flush=True)
        remove_training_outputs()
    run_step("开始训练", [sys.executable, "scripts/train.py", "--config", args.config])
    run_step(
        "生成 prototype 向量库",
        [
            sys.executable,
            "scripts/build_prototypes.py",
            "--config",
            args.config,
            "--checkpoint",
            DEFAULT_CHECKPOINT,
            "--out",
            DEFAULT_PROTOTYPES,
        ],
    )
    run_step(
        "导出 ONNX",
        [
            sys.executable,
            "scripts/export_onnx.py",
            "--config",
            args.config,
            "--checkpoint",
            DEFAULT_CHECKPOINT,
            "--out",
            DEFAULT_MODEL,
        ],
    )
    print("")
    print("训练流程完成。")
    print(f"checkpoint: {DEFAULT_CHECKPOINT}")
    print(f"prototypes: {DEFAULT_PROTOTYPES}")
    print(f"onnx: {DEFAULT_MODEL}")


def command_infer(args: argparse.Namespace) -> None:
    run_step(
        "单图测试",
        [
            sys.executable,
            "scripts/infer.py",
            "--config",
            args.config,
            "--model",
            args.model,
            "--prototypes",
            args.prototypes,
            "--image",
            args.image,
            "--provider",
            args.provider,
            "--top-k",
            str(args.top_k),
        ],
    )


def command_preview(args: argparse.Namespace) -> None:
    command = [
        sys.executable,
        "scripts/preview_compose.py",
        "--config",
        args.config,
        "--out-dir",
        args.out_dir,
        "--count",
        str(args.count),
    ]
    if args.train_augment:
        command.append("--train-augment")
    run_step("生成预览图", command)


def command_prototypes(args: argparse.Namespace) -> None:
    run_step(
        "生成 prototype 向量库",
        [
            sys.executable,
            "scripts/build_prototypes.py",
            "--config",
            args.config,
            "--checkpoint",
            args.checkpoint,
            "--out",
            args.out,
        ],
    )


def command_export(args: argparse.Namespace) -> None:
    run_step(
        "导出 ONNX",
        [
            sys.executable,
            "scripts/export_onnx.py",
            "--config",
            args.config,
            "--checkpoint",
            args.checkpoint,
            "--out",
            args.out,
        ],
    )


def command_real_val(args: argparse.Namespace) -> None:
    shell = "pwsh" if shutil.which("pwsh") else "powershell"
    run_step(
        "真实截图验证",
        [
            shell,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            "scripts/test_real_val.ps1",
            "-Config",
            args.config,
            "-Model",
            args.model,
            "-Prototypes",
            args.prototypes,
            "-RealVal",
            args.real_val,
            "-Provider",
            args.provider,
            "-TopK",
            str(args.top_k),
            "-NoPause",
        ],
    )


def command_clean(_args: argparse.Namespace) -> None:
    print("==> 清理生成物")
    generated_roots = [ROOT / "outputs", ROOT / "data", ROOT / "temp"]
    for path in [
        ROOT / "outputs",
        ROOT / "data" / "generated" / "labels.csv",
        ROOT / "temp",
    ]:
        safe_remove(path, generated_roots)
    for label_path in (ROOT / "data" / "generated").glob("labels_*.csv"):
        safe_remove(label_path, generated_roots)
    cache_roots = [ROOT / "__pycache__", ROOT / "scripts", ROOT / "src", ROOT / "temp"]
    for cache_dir in ROOT.rglob("__pycache__"):
        safe_remove(cache_dir, cache_roots)
    print("清理完成。")


def add_common_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prototypes", default=DEFAULT_PROTOTYPES)
    parser.add_argument("--provider", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--top-k", type=int, default=5)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AvatarDetect 统一入口")
    subparsers = parser.add_subparsers(dest="command")

    prepare = subparsers.add_parser("prepare", help="生成 labels 并校验数据")
    prepare.add_argument("--config", default=DEFAULT_CONFIG)
    prepare.set_defaults(func=command_prepare)

    train = subparsers.add_parser("train", help="从头训练并导出模型")
    train.add_argument("--config", default=DEFAULT_CONFIG)
    train.add_argument("--skip-cleanup", action="store_true", help="训练前不清理旧产物")
    train.set_defaults(func=command_train)

    infer = subparsers.add_parser("infer", help="单图测试")
    add_common_model_args(infer)
    infer.add_argument("--image", required=True)
    infer.set_defaults(func=command_infer)

    real_val = subparsers.add_parser("real-val", help="真实截图验证")
    add_common_model_args(real_val)
    real_val.add_argument("--real-val", default=DEFAULT_REAL_VAL)
    real_val.set_defaults(func=command_real_val)

    preview = subparsers.add_parser("preview", help="生成合成预览图")
    preview.add_argument("--config", default=DEFAULT_CONFIG)
    preview.add_argument("--out-dir", default="outputs/previews")
    preview.add_argument("--count", type=int, default=24)
    preview.add_argument("--train-augment", action="store_true")
    preview.set_defaults(func=command_preview)

    prototypes = subparsers.add_parser("prototypes", help="生成 prototype 向量库")
    prototypes.add_argument("--config", default=DEFAULT_CONFIG)
    prototypes.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    prototypes.add_argument("--out", default=DEFAULT_PROTOTYPES)
    prototypes.set_defaults(func=command_prototypes)

    export = subparsers.add_parser("export", help="导出 ONNX")
    export.add_argument("--config", default=DEFAULT_CONFIG)
    export.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    export.add_argument("--out", default=DEFAULT_MODEL)
    export.set_defaults(func=command_export)

    clean = subparsers.add_parser("clean", help="清理生成物")
    clean.set_defaults(func=command_clean)

    return parser


def prompt_path(message: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    value = input(f"{message}{suffix}: ").strip()
    return value or default


def menu_loop() -> None:
    parser = build_parser()
    while True:
        print("")
        print("==== AvatarDetect ====")
        print("1. 准备数据")
        print("2. 开始完整训练")
        print("3. 单图测试")
        print("4. 真实截图验证")
        print("5. 生成预览图")
        print("6. 清理生成物")
        print("0. 退出")
        choice = input("请选择: ").strip()
        try:
            if choice == "1":
                command_prepare(parser.parse_args(["prepare"]))
            elif choice == "2":
                command_train(parser.parse_args(["train"]))
            elif choice == "3":
                image = prompt_path("请输入图片路径")
                if image:
                    command_infer(parser.parse_args(["infer", "--image", image]))
            elif choice == "4":
                command_real_val(parser.parse_args(["real-val"]))
            elif choice == "5":
                command_preview(parser.parse_args(["preview"]))
            elif choice == "6":
                confirm = input("确认清理 outputs、labels 和缓存？输入 y 确认: ").strip().lower()
                if confirm == "y":
                    command_clean(parser.parse_args(["clean"]))
            elif choice == "0":
                return
            else:
                print("无效选择。")
        except KeyboardInterrupt:
            print("")
            return
        except SystemExit as exc:
            if exc.code not in (None, 0):
                print(exc)
        except Exception as exc:
            print(f"执行失败: {exc}")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        menu_loop()
        return 0
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 1
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
