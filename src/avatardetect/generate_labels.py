from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import load_config


FIELDNAMES = [
    "variant_id",
    "character_id",
    "character_name",
    "skin_id",
    "skin_name",
    "element_type",
    "weapon_type",
    "rarity",
    "image_path",
    "element_icon_path",
    "background_path",
    "split",
]

QUALITY_BACKGROUND = {
    "4": "UI_QUALITY_PURPLE.png",
    "5": "UI_QUALITY_ORANGE.png",
    "105": "UI_QUALITY_RED.png",
}

# 需要补充的角色优先放在 data/metadata/Avatar/custom_*.json。
MANUAL_LABELS: list[dict[str, str]] = []

ELEMENT_ICON_BY_TYPE = {
    "冰": "UI_Buff_Element_Frost.png",
    "风": "UI_Buff_Element_Wind.png",
    "雷": "UI_Buff_Element_Elect.png",
    "水": "UI_Buff_Element_Water.png",
    "火": "UI_Buff_Element_Fire.png",
    "岩": "UI_Buff_Element_Roach.png",
    "草": "UI_Buff_Element_Grass.png",
}

WEAPON_TYPE_BY_ID = {
    "1": "单手剑",
    "10": "法器",
    "11": "双手剑",
    "12": "弓",
    "13": "长柄武器",
}

VARIABLE_ELEMENT_CHARACTER_IDS = {
    "10000005",  # 空
    "10000007",  # 荧
    "10000117",  # 奇偶·男性
    "10000118",  # 奇偶·女性
}

# 这些头像不是可操控角色，不参与训练，也不作为未使用图片输出警告。
IGNORED_AVATAR_IMAGES = {
    "UI_AvatarIcon_Aozi",
    "UI_AvatarIcon_Paimon",
}


@dataclass(frozen=True)
class SkippedLabel:
    character_id: str
    character_name: str
    skin_id: str
    skin_name: str
    icon_name: str
    reason: str


@dataclass(frozen=True)
class UnmatchedAvatarImage:
    icon_name: str
    image_path: Path
    reason: str


def get_value(data: dict[str, Any], *names: str, default: Any = "") -> Any:
    for name in names:
        if name in data:
            return data[name]
    lower_map = {str(key).lower(): key for key in data}
    for name in names:
        key = lower_map.get(name.lower())
        if key is not None:
            return data[key]
    return default


def as_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def is_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def relative_for_csv(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def build_row(
    *,
    root: Path,
    avatar_dir: Path,
    element_dir: Path,
    background_dir: Path,
    use_background: bool,
    character_id: str,
    character_name: str,
    skin_id: str,
    skin_name: str,
    element_type: str,
    weapon_type: str,
    rarity: str,
    icon_name: str,
    split: str,
) -> tuple[dict[str, str] | None, SkippedLabel | None]:
    image_path = avatar_dir / f"{icon_name}.png"
    if not image_path.exists():
        return None, SkippedLabel(
            character_id=character_id,
            character_name=character_name,
            skin_id=skin_id,
            skin_name=skin_name,
            icon_name=icon_name,
            reason=f"找不到图片: {image_path}",
        )

    element_icon_name = ELEMENT_ICON_BY_TYPE.get(element_type)
    if not element_icon_name:
        raise ValueError(f"未配置元素={element_type} 的图标映射")
    element_icon_path = element_dir / element_icon_name
    if not element_icon_path.exists():
        raise FileNotFoundError(f"找不到元素={element_type} 对应的元素图标: {element_icon_path}")

    background_path_text = ""
    if use_background:
        background_name = QUALITY_BACKGROUND.get(rarity)
        if not background_name:
            raise ValueError(f"未配置 Quality={rarity} 的背景映射")
        background_path = background_dir / background_name
        if not background_path.exists():
            raise FileNotFoundError(f"找不到 Quality={rarity} 对应的背景图: {background_path}")
        background_path_text = relative_for_csv(background_path, root)

    appearance_id = f"{character_id}_{skin_id}"
    return {
        "variant_id": f"{appearance_id}_{element_type}",
        "character_id": character_id,
        "character_name": character_name,
        "skin_id": skin_id,
        "skin_name": skin_name,
        "element_type": element_type,
        "weapon_type": weapon_type,
        "rarity": rarity,
        "image_path": relative_for_csv(image_path, root),
        "element_icon_path": relative_for_csv(element_icon_path, root),
        "background_path": background_path_text,
        "split": split,
    }, None


def rows_from_avatar_json(
    json_path: Path,
    *,
    root: Path,
    avatar_dir: Path,
    element_dir: Path,
    background_dir: Path,
    use_background: bool,
    split: str,
) -> tuple[list[dict[str, str]], list[SkippedLabel], set[str]]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    character_id = as_text(get_value(data, "Id", "id"))
    character_name = as_text(get_value(data, "Name", "name"))
    rarity = as_text(get_value(data, "Quality", "quality"))
    weapon_id = as_text(get_value(data, "Weapon", "weapon"))
    weapon_type = WEAPON_TYPE_BY_ID.get(weapon_id)
    default_icon = as_text(get_value(data, "Icon", "icon"))
    fetter_info = get_value(data, "FetterInfo", "fetterInfo", "fetter_info", default={})
    if not isinstance(fetter_info, dict):
        fetter_info = {}
    fixed_element = as_text(get_value(fetter_info, "VisionBefore", "visionBefore", "vision_before"))
    costumes = get_value(data, "Costumes", "costumes", default=[])
    if not isinstance(costumes, list):
        costumes = []

    rows: list[dict[str, str]] = []
    skipped: list[SkippedLabel] = []
    expected_icon_names: set[str] = set()

    for costume in costumes:
        if not isinstance(costume, dict):
            continue
        skin_id = as_text(get_value(costume, "Id", "id"))
        skin_name = as_text(get_value(costume, "Name", "name"))
        front_icon = as_text(get_value(costume, "FrontIcon", "frontIcon", "front_icon"))
        is_default = is_true(get_value(costume, "IsDefault", "isdefault", "isDefault", default=False))

        if not front_icon and (is_default or len(costumes) == 1):
            icon_name = default_icon
        elif front_icon:
            icon_name = front_icon
        else:
            continue

        if icon_name:
            expected_icon_names.add(icon_name)

        if character_id in VARIABLE_ELEMENT_CHARACTER_IDS:
            element_types = list(ELEMENT_ICON_BY_TYPE)
        elif fixed_element:
            element_types = [fixed_element]
        else:
            element_types = []

        if not character_id or not skin_id or not character_name or not rarity or not icon_name or not element_types or not weapon_type:
            reason = f"{json_path.name} 缺少必要字段"
            if not element_types:
                reason = f"{json_path.name} 缺少 FetterInfo.VisionBefore 元素字段"
            elif not weapon_type:
                reason = f"{json_path.name} 缺少或未知 Weapon 武器字段: {weapon_id}"
            skipped.append(
                SkippedLabel(
                    character_id=character_id,
                    character_name=character_name,
                    skin_id=skin_id,
                    skin_name=skin_name,
                    icon_name=icon_name,
                    reason=reason,
                )
            )
            continue

        for element_type in element_types:
            row, skip = build_row(
                root=root,
                avatar_dir=avatar_dir,
                element_dir=element_dir,
                background_dir=background_dir,
                use_background=use_background,
                character_id=character_id,
                character_name=character_name,
                skin_id=skin_id,
                skin_name=skin_name,
                element_type=element_type,
                weapon_type=weapon_type,
                rarity=rarity,
                icon_name=icon_name,
                split=split,
            )
            if row is not None:
                rows.append(row)
            if skip is not None:
                skipped.append(
                    SkippedLabel(
                        character_id=skip.character_id,
                        character_name=skip.character_name,
                        skin_id=skip.skin_id,
                        skin_name=skin_name,
                        icon_name=skip.icon_name,
                        reason=skip.reason,
                    )
                )

    return rows, skipped, expected_icon_names


def find_unmatched_avatar_images(
    *,
    avatar_dir: Path,
    expected_icon_names: set[str],
) -> list[UnmatchedAvatarImage]:
    unmatched: list[UnmatchedAvatarImage] = []
    for image_path in sorted(avatar_dir.glob("*.png"), key=lambda p: p.name.lower()):
        if image_path.stem in IGNORED_AVATAR_IMAGES:
            continue
        if image_path.stem not in expected_icon_names:
            unmatched.append(
                UnmatchedAvatarImage(
                    icon_name=image_path.stem,
                    image_path=image_path,
                    reason="图片存在，但没有 Avatar JSON 或手工标签引用",
                )
            )
    return unmatched


def generate_labels(
    *,
    root: Path,
    json_dir: Path,
    avatar_dir: Path,
    element_dir: Path,
    background_dir: Path,
    out_path: Path,
    split: str,
    use_background: bool = False,
) -> tuple[int, list[SkippedLabel], list[UnmatchedAvatarImage]]:
    if not json_dir.exists():
        raise FileNotFoundError(f"找不到 JSON 目录: {json_dir}")
    if not avatar_dir.exists():
        raise FileNotFoundError(f"找不到角色图目录: {avatar_dir}")
    if not element_dir.exists():
        raise FileNotFoundError(f"找不到元素图标目录: {element_dir}")
    if use_background and not background_dir.exists():
        raise FileNotFoundError(f"找不到背景图目录: {background_dir}")

    all_rows: list[dict[str, str]] = []
    skipped: list[SkippedLabel] = []
    expected_icon_names: set[str] = set()
    for json_path in sorted(json_dir.glob("*.json"), key=lambda p: p.name):
        rows, missing, expected = rows_from_avatar_json(
            json_path,
            root=root,
            avatar_dir=avatar_dir,
            element_dir=element_dir,
            background_dir=background_dir,
            use_background=use_background,
            split=split,
        )
        all_rows.extend(rows)
        skipped.extend(missing)
        expected_icon_names.update(expected)

    existing_variant_ids = {row["variant_id"] for row in all_rows}
    for manual in MANUAL_LABELS:
        expected_icon_names.add(manual["icon_name"])
        row, skip = build_row(
            root=root,
            avatar_dir=avatar_dir,
            element_dir=element_dir,
            background_dir=background_dir,
            use_background=use_background,
            character_id=manual["character_id"],
            character_name=manual["character_name"],
            skin_id=manual["skin_id"],
            skin_name=manual["skin_name"],
            element_type=manual["element_type"],
            weapon_type=manual.get("weapon_type", ""),
            rarity=manual["rarity"],
            icon_name=manual["icon_name"],
            split=split,
        )
        if row is not None and row["variant_id"] not in existing_variant_ids:
            all_rows.append(row)
            existing_variant_ids.add(row["variant_id"])
        if skip is not None:
            skipped.append(
                SkippedLabel(
                    character_id=skip.character_id,
                    character_name=skip.character_name,
                    skin_id=skip.skin_id,
                    skin_name=manual["skin_name"],
                    icon_name=skip.icon_name,
                    reason=skip.reason,
                )
            )

    unmatched_avatar_images = find_unmatched_avatar_images(
        avatar_dir=avatar_dir,
        expected_icon_names=expected_icon_names,
    )

    all_rows.sort(key=lambda row: (int(row["character_id"]), int(row["skin_id"]), row["element_type"]))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_rows)

    return len(all_rows), skipped, unmatched_avatar_images


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="从 Avatar JSON 生成 data/generated/labels.csv")
    parser.add_argument("--config", default=None)
    parser.add_argument("--json-dir", default=str(root / "data" / "metadata" / "Avatar"))
    parser.add_argument("--avatar-dir", default=str(root / "assets" / "icons" / "UI_AvatarIcon"))
    parser.add_argument("--element-dir", default=str(root / "assets" / "icons" / "UI_Buff_Element"))
    parser.add_argument("--background-dir", default=str(root / "assets" / "backgrounds" / "UI_QUALITY"))
    parser.add_argument("--out", default=str(root / "data" / "generated" / "labels.csv"))
    parser.add_argument("--split", default="train")
    parser.add_argument("--use-background", action="store_true")
    parser.add_argument("--no-background", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[2]
    use_background = bool(args.use_background)
    if args.config and not args.use_background and not args.no_background:
        cfg = load_config(args.config)
        use_background = bool(cfg.get("compose", {}).get("use_background", False))
    if args.no_background:
        use_background = False
    count, skipped, unmatched_avatar_images = generate_labels(
        root=root,
        json_dir=Path(args.json_dir),
        avatar_dir=Path(args.avatar_dir),
        element_dir=Path(args.element_dir),
        background_dir=Path(args.background_dir),
        out_path=Path(args.out),
        split=args.split,
        use_background=use_background,
    )
    for item in skipped:
        print(
            "警告: 跳过 "
            f"character_id={item.character_id} 角色={item.character_name} "
            f"skin_id={item.skin_id} 皮肤={item.skin_name} "
            f"图片={item.icon_name}，原因: {item.reason}"
        )
    for item in unmatched_avatar_images:
        print(
            "警告: 未使用图片 "
            f"图片={item.icon_name} 路径={item.image_path}，原因: {item.reason}"
        )
    print(f"生成行数={count}")
    print(f"跳过行数={len(skipped)}")
    print(f"未使用图片数={len(unmatched_avatar_images)}")
    print(f"输出文件={Path(args.out)}")


if __name__ == "__main__":
    main()
