from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mask_label_editor.fits_io import read_fits_image, write_mask_image


@dataclass
class InferTileGroup:
    key: str
    reference: Path | None = None
    aligned: Path | None = None

    @property
    def ready(self) -> bool:
        return self.reference is not None and self.aligned is not None


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_config(config_path: Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def discover_infer_tiles(data_dir: Path) -> list[InferTileGroup]:
    groups: dict[str, InferTileGroup] = {}
    search_dirs = [data_dir]
    tiles_dir = data_dir / "tiles"
    if tiles_dir.is_dir():
        search_dirs.append(tiles_dir)

    for d in search_dirs:
        if not d.exists():
            continue
        for f in d.iterdir():
            if not f.is_file():
                continue
            name_lower = f.name.lower()
            m = re.search(r"^(.+?tile[_]?\d+)", name_lower)
            if m is None:
                continue
            key = m.group(1)
            g = groups.get(key)
            if g is None:
                g = InferTileGroup(key=key)
                groups[key] = g

            if "_1_reference" in name_lower and name_lower.endswith((".fits", ".fit", ".fts")):
                g.reference = f
            elif "_2_aligned" in name_lower and name_lower.endswith((".fits", ".fit", ".fts")):
                g.aligned = f

    return [g for g in sorted(groups.values(), key=lambda x: x.key) if g.ready]


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def build_features(ref: np.ndarray, ali: np.ndarray) -> np.ndarray:
    f1 = ref.reshape(-1).astype(np.float32)
    f2 = ali.reshape(-1).astype(np.float32)
    f3 = (ali - ref).reshape(-1).astype(np.float32)
    f4 = np.abs(f3)
    return np.stack([f1, f2, f3, f4], axis=1)


def predict_single(
    ref_path: Path,
    ali_path: Path,
    weights: np.ndarray,
    bias: float,
    mean: np.ndarray,
    std: np.ndarray,
    threshold: float,
) -> np.ndarray:
    ref_img = np.squeeze(read_fits_image(ref_path).data).astype(np.float32)
    ali_img = np.squeeze(read_fits_image(ali_path).data).astype(np.float32)
    if ref_img.shape != ali_img.shape:
        raise ValueError(f"shape mismatch: reference={ref_img.shape}, aligned={ali_img.shape}")

    x = build_features(ref_img, ali_img)
    x_n = (x - mean) / (std + 1e-6)
    prob = sigmoid(x_n @ weights + float(bias))
    pred = (prob >= threshold).astype(np.uint16).reshape(ref_img.shape)
    return pred


def output_name(reference_file: Path, suffix: str) -> str:
    stem = reference_file.stem
    stem = re.sub(r"_1_reference$", "", stem, flags=re.IGNORECASE)
    return f"{stem}{suffix}"


def main() -> None:
    parser = argparse.ArgumentParser(description="加载 model_logreg.npz 推理并生成预测 mask")
    parser.add_argument(
        "--config",
        type=Path,
        default=repo_root() / "config.json",
        help="配置文件路径，默认仓库根目录 config.json",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="可覆盖配置里的预测数据目录（优先级高于 predict_data_dir / data_dir）",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=repo_root() / "train" / "model_logreg.npz",
        help="训练生成的模型路径",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="预测 mask 输出目录；不传时默认输出到每个 reference 同目录（更兼容训练扫描）",
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="二分类阈值")
    parser.add_argument(
        "--suffix",
        type=str,
        default="_mask.png",
        help="输出文件名后缀（建议包含 _mask，如 _mask.png / _mask.fits）",
    )
    parser.add_argument(
        "--only-keyword",
        type=str,
        default="",
        help="只推理文件名中包含该关键字的 tile（可选）",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    default_predict_dir = cfg.get("predict_data_dir", cfg.get("data_dir", ""))
    data_dir = args.data_dir if args.data_dir is not None else Path(default_predict_dir)
    if not str(data_dir):
        raise ValueError("predict_data_dir/data_dir is empty, please set it in config.json or pass --data-dir")
    if not data_dir.exists():
        raise FileNotFoundError(f"data_dir does not exist: {data_dir}")
    if not args.model.exists():
        raise FileNotFoundError(f"model file not found: {args.model}")

    out_dir = args.output_dir
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    model = np.load(args.model)
    weights = np.asarray(model["weights"], dtype=np.float32)
    bias = float(np.asarray(model["bias"]).item())
    mean = np.asarray(model["mean"], dtype=np.float32)
    std = np.asarray(model["std"], dtype=np.float32)

    tiles = discover_infer_tiles(data_dir)
    if args.only_keyword:
        k = args.only_keyword.lower()
        tiles = [t for t in tiles if k in t.key]

    print(f"[info] model: {args.model}")
    print(f"[info] data_dir: {data_dir}")
    print(f"[info] output_dir: {out_dir if out_dir is not None else '(same as each reference file directory)'}")
    print(f"[info] infer tiles: {len(tiles)}")

    if not tiles:
        raise RuntimeError("No valid tiles for inference. Need *_1_reference.fits and *_2_aligned.fits")

    for i, t in enumerate(tiles, start=1):
        try:
            pred_mask = predict_single(
                ref_path=t.reference,
                ali_path=t.aligned,
                weights=weights,
                bias=bias,
                mean=mean,
                std=std,
                threshold=args.threshold,
            )
            out_name = output_name(t.reference, args.suffix)
            out_path = (out_dir / out_name) if out_dir is not None else (t.reference.parent / out_name)
            write_mask_image(out_path, pred_mask)
            pos_ratio = float(pred_mask.mean())
            print(f"[ok] {i}/{len(tiles)} {t.key} -> {out_path} pos_ratio={pos_ratio:.4f}")
        except Exception as exc:
            print(f"[fail] {i}/{len(tiles)} {t.key}: {exc}")


if __name__ == "__main__":
    main()

