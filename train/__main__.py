from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mask_label_editor.fits_io import read_fits_image, read_mask_image


@dataclass
class TileGroup:
    key: str
    reference: Path | None = None
    aligned: Path | None = None
    mask: Path | None = None

    @property
    def ready(self) -> bool:
        return self.reference is not None and self.aligned is not None and self.mask is not None


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_config(config_path: Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def discover_tiles(data_dir: Path) -> list[TileGroup]:
    groups: dict[str, TileGroup] = {}
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
            group_key = m.group(1)
            g = groups.get(group_key)
            if g is None:
                g = TileGroup(key=group_key)
                groups[group_key] = g

            if "_1_reference" in name_lower and name_lower.endswith((".fits", ".fit", ".fts")):
                g.reference = f
            elif "_2_aligned" in name_lower and name_lower.endswith((".fits", ".fit", ".fts")):
                g.aligned = f
            elif "_mask" in name_lower and name_lower.endswith(
                (".fits", ".fit", ".fts", ".png", ".bmp", ".tif", ".tiff")
            ):
                g.mask = f

    return [g for g in sorted(groups.values(), key=lambda x: x.key) if g.ready]


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def split_train_val(n: int, val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    val_n = int(n * val_ratio)
    val_idx = idx[:val_n]
    train_idx = idx[val_n:]
    return train_idx, val_idx


def build_dataset(tiles: list[TileGroup], max_samples_per_tile: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x_all: list[np.ndarray] = []
    y_all: list[np.ndarray] = []

    for i, t in enumerate(tiles, start=1):
        ref = np.squeeze(read_fits_image(t.reference).data).astype(np.float32)
        ali = np.squeeze(read_fits_image(t.aligned).data).astype(np.float32)
        msk = np.squeeze(read_mask_image(t.mask)).astype(np.int32)

        if ref.shape != ali.shape or ref.shape != msk.shape:
            print(f"[warn] shape mismatch, skip {t.key}: ref={ref.shape}, aligned={ali.shape}, mask={msk.shape}")
            continue

        f1 = ref.reshape(-1)
        f2 = ali.reshape(-1)
        f3 = (ali - ref).reshape(-1)
        f4 = np.abs(f3)
        y = (msk.reshape(-1) > 0).astype(np.float32)

        n = y.size
        if n > max_samples_per_tile:
            sample_idx = rng.choice(n, size=max_samples_per_tile, replace=False)
            f1 = f1[sample_idx]
            f2 = f2[sample_idx]
            f3 = f3[sample_idx]
            f4 = f4[sample_idx]
            y = y[sample_idx]

        x = np.stack([f1, f2, f3, f4], axis=1)
        x_all.append(x)
        y_all.append(y)
        print(f"[data] {i}/{len(tiles)} {t.key} -> +{x.shape[0]} samples")

    if not x_all:
        raise RuntimeError("No valid training samples found.")

    return np.concatenate(x_all, axis=0), np.concatenate(y_all, axis=0)


def train_logreg(
    x: np.ndarray,
    y: np.ndarray,
    epochs: int,
    lr: float,
    val_ratio: float,
    seed: int,
) -> tuple[np.ndarray, float, dict, dict]:
    train_idx, val_idx = split_train_val(len(y), val_ratio=val_ratio, seed=seed)
    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]

    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0) + 1e-6
    x_train_n = (x_train - mean) / std
    x_val_n = (x_val - mean) / std if len(x_val) > 0 else x_val

    w = np.zeros(x_train_n.shape[1], dtype=np.float32)
    b = 0.0

    for ep in range(1, epochs + 1):
        p = sigmoid(x_train_n @ w + b)
        err = p - y_train
        grad_w = (x_train_n.T @ err) / len(y_train)
        grad_b = float(err.mean())
        w -= lr * grad_w
        b -= lr * grad_b

        if ep == 1 or ep % 5 == 0 or ep == epochs:
            train_loss = float((-y_train * np.log(p + 1e-8) - (1.0 - y_train) * np.log(1.0 - p + 1e-8)).mean())
            msg = f"[train] epoch={ep:03d}/{epochs} loss={train_loss:.6f}"
            if len(y_val) > 0:
                pv = sigmoid(x_val_n @ w + b)
                val_loss = float((-y_val * np.log(pv + 1e-8) - (1.0 - y_val) * np.log(1.0 - pv + 1e-8)).mean())
                msg += f" val_loss={val_loss:.6f}"
            print(msg)

    train_metrics = eval_binary(y_train, sigmoid(x_train_n @ w + b))
    val_metrics = eval_binary(y_val, sigmoid(x_val_n @ w + b)) if len(y_val) > 0 else {}
    return w, b, {"mean": mean, "std": std}, {"train": train_metrics, "val": val_metrics}


def eval_binary(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    if len(y_true) == 0:
        return {}
    y_pred = (y_prob >= threshold).astype(np.float32)
    acc = float((y_pred == y_true).mean())
    tp = float(((y_pred == 1) & (y_true == 1)).sum())
    fp = float(((y_pred == 1) & (y_true == 0)).sum())
    fn = float(((y_pred == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
    positive_ratio = float(y_true.mean())
    return {
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "positive_ratio": positive_ratio,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="读取 config.json 并执行简易掩膜训练（二分类逻辑回归）")
    parser.add_argument(
        "--config",
        type=Path,
        default=repo_root() / "config.json",
        help="配置文件路径，默认仓库根目录 config.json",
    )
    parser.add_argument("--data-dir", type=Path, default=None, help="可覆盖配置里的 data_dir")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples-per-tile", type=int, default=50000)
    parser.add_argument("--output", type=Path, default=repo_root() / "train" / "model_logreg.npz")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_dir = args.data_dir if args.data_dir is not None else Path(cfg.get("data_dir", ""))
    if not str(data_dir):
        raise ValueError("data_dir is empty, please set it in config.json or pass --data-dir")
    if not data_dir.exists():
        raise FileNotFoundError(f"data_dir does not exist: {data_dir}")

    print(f"[info] config: {args.config}")
    print(f"[info] data_dir: {data_dir}")
    tiles = discover_tiles(data_dir)
    print(f"[info] found ready tiles: {len(tiles)}")
    if not tiles:
        raise RuntimeError("No training tiles found. Need *_1_reference.fits, *_2_aligned.fits and *_mask.*")

    x, y = build_dataset(tiles, max_samples_per_tile=args.max_samples_per_tile, seed=args.seed)
    print(f"[info] dataset: X={x.shape}, positive_ratio={float(y.mean()):.4f}")

    w, b, norm, metrics = train_logreg(
        x=x,
        y=y,
        epochs=args.epochs,
        lr=args.lr,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        weights=w.astype(np.float32),
        bias=np.float32(b),
        mean=norm["mean"].astype(np.float32),
        std=norm["std"].astype(np.float32),
    )
    print(f"[info] model saved to: {args.output}")
    print(f"[metric] train={metrics['train']}")
    if metrics["val"]:
        print(f"[metric] val={metrics['val']}")


if __name__ == "__main__":
    main()

