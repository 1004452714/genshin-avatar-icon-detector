from __future__ import annotations

import argparse
import base64
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .checkpoint import load_model_from_checkpoint
from .config import load_config, project_path
from .dataset import AvatarDataset, load_labels


def encode_vector(vec: np.ndarray) -> str:
    return base64.b64encode(vec.astype("<f4", copy=False).tobytes()).decode("ascii")


def build_prototypes(config_path: str | Path, checkpoint_path: str | Path, out_path: str | Path | None) -> None:
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_model_from_checkpoint(checkpoint_path, device)
    mappings = checkpoint["mappings"]
    df = load_labels(project_path(cfg, cfg["data"]["labels_csv"]))
    samples = int(cfg.get("prototype", {}).get("samples_per_appearance", 32))
    rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), ascii=True):
        one = pd.DataFrame([row])
        dataset = AvatarDataset(one, cfg["_project_root"], cfg, mappings, train=samples > 1, prototype=True)
        vectors = []
        with torch.no_grad():
            for _ in range(max(1, samples)):
                item = dataset[0]
                image = item["image"].unsqueeze(0).to(device)
                element_image = item["element_image"].unsqueeze(0).to(device)
                embedding, _, _, _ = model(image, element_image)
                vectors.append(embedding.squeeze(0).detach().cpu().numpy())
        proto = np.mean(np.stack(vectors), axis=0)
        proto = proto / max(np.linalg.norm(proto), 1e-12)
        rows.append(
            {
                "variant_id": row["variant_id"],
                "character_id": row["character_id"],
                "character_name": row["character_name"],
                "skin_id": row["skin_id"],
                "skin_name": row["skin_name"],
                "element_type": row["element_type"],
                "weapon_type": row.get("weapon_type", ""),
                "rarity": row["rarity"],
                "embedding": encode_vector(proto),
            }
        )

    output = Path(out_path) if out_path else project_path(cfg, cfg["prototype"]["output_csv"])
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False, encoding="utf-8")
    print(f"已写入: {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_prototypes(args.config, args.checkpoint, args.out)
