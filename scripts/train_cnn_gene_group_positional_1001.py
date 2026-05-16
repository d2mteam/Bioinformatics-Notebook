#!/usr/bin/env python3
"""Train a gene-group CNN with positional encoding on the 1001 bp dataset."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    f1_score,
)
from sklearn.model_selection import GroupShuffleSplit
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm.auto import tqdm


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(
        description="Train 1001 bp ClinVar CNN with gene-group split and positional encoding."
    )
    parser.add_argument("--project-dir", type=Path, default=root)
    parser.add_argument("--length", type=int, default=1001)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--hard-negative-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hard-negative-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--positional-encoding",
        choices=["none", "sinusoidal", "relative"],
        default="sinusoidal",
        help=(
            "Position channels appended to ref/alt/marker. "
            "sinusoidal keeps the previous behavior; relative centers positions "
            "around the mutation site."
        ),
    )
    parser.add_argument("--positional-dim", type=int, default=8)
    parser.add_argument(
        "--architecture",
        choices=[
            "baseline",
            "dilated",
            "dilated_residual",
            "dilated_window_attention",
        ],
        default="baseline",
        help="CNN architecture. baseline preserves the original output names.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--debug-max-train-rows", type=int, default=None)
    parser.add_argument("--debug-max-val-rows", type=int, default=None)
    parser.add_argument("--debug-max-test-rows", type=int, default=None)
    parser.add_argument("--no-hard-negative", action="store_true")
    parser.add_argument(
        "--rc-augment",
        action="store_true",
        help="Randomly reverse-complement training samples with probability 0.5.",
    )
    parser.add_argument(
        "--rc-tta",
        action="store_true",
        help="Average eval/test predictions from original and reverse-complement inputs.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_groups(metadata: pd.DataFrame) -> np.ndarray:
    groups = metadata["GeneSymbol"].fillna("unknown").astype(str).to_numpy(copy=True)
    unknown = (groups == "") | (groups == "unknown") | (groups == "nan")
    row_ids = np.arange(len(groups)).astype(str)
    groups[unknown] = np.char.add("unknown_row_", row_ids[unknown])
    return groups


def maybe_subsample(
    indices: np.ndarray, max_rows: int | None, y: np.ndarray, seed: int
) -> np.ndarray:
    if max_rows is None or len(indices) <= max_rows:
        return indices
    rng = np.random.default_rng(seed)
    selected = []
    for label in [0, 1]:
        label_idx = indices[y[indices] == label]
        n_label = int(round(max_rows * len(label_idx) / len(indices)))
        n_label = min(len(label_idx), max(1, n_label))
        selected.append(rng.choice(label_idx, size=n_label, replace=False))
    out = np.concatenate(selected)
    rng.shuffle(out)
    return out


def make_sinusoidal_position_encoding(length: int, dim: int) -> np.ndarray:
    if dim == 0:
        return np.zeros((length, 0), dtype=np.float32)
    if dim % 2 != 0:
        raise ValueError("positional dim must be even")
    positions = np.arange(length, dtype=np.float32)[:, None]
    div_term = np.exp(np.arange(0, dim, 2, dtype=np.float32) * (-np.log(10000.0) / dim))
    pe = np.zeros((length, dim), dtype=np.float32)
    pe[:, 0::2] = np.sin(positions * div_term)
    pe[:, 1::2] = np.cos(positions * div_term)
    return pe


def make_relative_position_encoding(length: int, dim: int) -> np.ndarray:
    if dim == 0:
        return np.zeros((length, 0), dtype=np.float32)
    if dim % 2 != 0:
        raise ValueError("positional dim must be even")
    center = (length - 1) / 2.0
    positions = (np.arange(length, dtype=np.float32) - center)[:, None]
    div_term = np.exp(np.arange(0, dim, 2, dtype=np.float32) * (-np.log(10000.0) / dim))
    pe = np.zeros((length, dim), dtype=np.float32)
    pe[:, 0::2] = np.sin(positions * div_term)
    pe[:, 1::2] = np.cos(positions * div_term)
    return pe


def make_position_encoding(kind: str, length: int, dim: int) -> np.ndarray:
    if dim < 0:
        raise ValueError("positional dim must be non-negative")
    if kind == "none":
        return np.zeros((length, 0), dtype=np.float32)
    if kind == "sinusoidal":
        return make_sinusoidal_position_encoding(length, dim)
    if kind == "relative":
        return make_relative_position_encoding(length, dim)
    raise ValueError(f"Unknown positional encoding: {kind}")


def reverse_complement_ref_alt_marker(x: np.ndarray) -> np.ndarray:
    """Reverse-complement the ref/alt one-hot channels and reverse the marker."""
    rc = x[::-1].copy()
    rc[:, 0:4] = rc[:, [3, 2, 1, 0]]
    rc[:, 4:8] = rc[:, [7, 6, 5, 4]]
    return rc


class ClinVarSequenceDataset(Dataset):
    def __init__(
        self,
        x_array: np.ndarray,
        y_array: np.ndarray,
        indices: np.ndarray,
        positional_encoding: np.ndarray,
        random_reverse_complement: bool = False,
        reverse_complement: bool = False,
    ) -> None:
        self.x_array = x_array
        self.y_array = y_array
        self.indices = np.asarray(indices, dtype=np.int64)
        self.positional_encoding = positional_encoding.astype(np.float32, copy=False)
        self.random_reverse_complement = random_reverse_complement
        self.reverse_complement = reverse_complement
        if self.random_reverse_complement and self.reverse_complement:
            raise ValueError("Use either random or deterministic reverse-complement, not both.")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        idx = int(self.indices[item])
        x = self.x_array[idx].astype(np.float32, copy=True)
        if self.reverse_complement or (
            self.random_reverse_complement and random.random() < 0.5
        ):
            x = reverse_complement_ref_alt_marker(x)
        x = np.concatenate([x, self.positional_encoding], axis=1)
        x = np.ascontiguousarray(x.T)
        y = np.float32(self.y_array[idx])
        return torch.from_numpy(x), torch.tensor(y)


class ClinVarCNN(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x)).squeeze(1)


class DilatedClinVarCNN(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv1d(in_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        ]
        for dilation in [1, 2, 4, 8, 16, 32, 64]:
            layers.extend(
                [
                    nn.Conv1d(
                        64,
                        64,
                        kernel_size=7,
                        padding=3 * dilation,
                        dilation=dilation,
                    ),
                    nn.BatchNorm1d(64),
                    nn.ReLU(),
                    nn.Dropout(0.05),
                ]
            )
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Linear(64 * 3, 96),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(96, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        center = features[:, :, features.shape[-1] // 2]
        global_max = torch.amax(features, dim=-1)
        global_mean = torch.mean(features, dim=-1)
        pooled = torch.cat([center, global_max, global_mean], dim=1)
        return self.classifier(pooled).squeeze(1)


class WindowAttentionBlock1D(nn.Module):
    def __init__(
        self,
        channels: int,
        window_size: int = 51,
        num_heads: int = 4,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError("channels must be divisible by num_heads")
        self.window_size = window_size
        self.left_pad = window_size // 2
        self.norm_attn = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_ffn = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 2, channels),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, length = x.shape
        x_seq = x.transpose(1, 2)

        right_pad = (-length - self.left_pad) % self.window_size
        if self.left_pad or right_pad:
            x_seq = F.pad(x_seq, (0, 0, self.left_pad, right_pad))

        padded_length = x_seq.shape[1]
        windows = x_seq.reshape(
            batch_size,
            padded_length // self.window_size,
            self.window_size,
            channels,
        )
        windows = windows.reshape(-1, self.window_size, channels)

        attn_input = self.norm_attn(windows)
        attn_output, _ = self.attn(
            attn_input, attn_input, attn_input, need_weights=False
        )
        windows = windows + attn_output
        windows = windows + self.ffn(self.norm_ffn(windows))

        x_seq = windows.reshape(
            batch_size,
            padded_length // self.window_size,
            self.window_size,
            channels,
        )
        x_seq = x_seq.reshape(batch_size, padded_length, channels)
        x_seq = x_seq[:, self.left_pad : self.left_pad + length]
        return x_seq.transpose(1, 2)


class DilatedWindowAttentionClinVarCNN(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv1d(in_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        ]
        for dilation in [1, 2, 4, 8, 16, 32, 64]:
            layers.extend(
                [
                    nn.Conv1d(
                        64,
                        64,
                        kernel_size=7,
                        padding=3 * dilation,
                        dilation=dilation,
                    ),
                    nn.BatchNorm1d(64),
                    nn.ReLU(),
                    nn.Dropout(0.05),
                ]
            )
        self.features = nn.Sequential(*layers)
        self.window_attention = WindowAttentionBlock1D(
            channels=64,
            window_size=51,
            num_heads=4,
            dropout=0.10,
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 3, 96),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(96, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        features = self.window_attention(features)
        center = features[:, :, features.shape[-1] // 2]
        global_max = torch.amax(features, dim=-1)
        global_mean = torch.mean(features, dim=-1)
        pooled = torch.cat([center, global_max, global_mean], dim=1)
        return self.classifier(pooled).squeeze(1)


class DilatedResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=7,
            padding=3 * dilation,
            dilation=dilation,
        )
        self.norm = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.05)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv(x)
        out = self.norm(out)
        out = self.relu(out)
        out = self.dropout(out)
        return self.relu(out + residual)


class DilatedResidualClinVarCNN(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(
            *[
                DilatedResidualBlock(64, dilation)
                for dilation in [1, 2, 4, 8, 16, 32, 64]
            ]
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 3, 96),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(96, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.blocks(self.stem(x))
        center = features[:, :, features.shape[-1] // 2]
        global_max = torch.amax(features, dim=-1)
        global_mean = torch.mean(features, dim=-1)
        pooled = torch.cat([center, global_max, global_mean], dim=1)
        return self.classifier(pooled).squeeze(1)


def make_model(architecture: str, in_channels: int) -> nn.Module:
    if architecture == "baseline":
        return ClinVarCNN(in_channels)
    if architecture == "dilated":
        return DilatedClinVarCNN(in_channels)
    if architecture == "dilated_residual":
        return DilatedResidualClinVarCNN(in_channels)
    if architecture == "dilated_window_attention":
        return DilatedWindowAttentionClinVarCNN(in_channels)
    raise ValueError(f"Unknown architecture: {architecture}")


def make_weighted_sampler(indices: np.ndarray, y: np.ndarray) -> WeightedRandomSampler:
    labels = y[indices].astype(int)
    counts = np.bincount(labels, minlength=2)
    weights_by_class = 1.0 / np.sqrt(np.maximum(counts, 1))
    sample_weights = weights_by_class[labels]
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )


def safe_auc(y_true: np.ndarray, y_score: np.ndarray, kind: str) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    if kind == "roc":
        return float(roc_auc_score(y_true, y_score))
    return float(average_precision_score(y_true, y_score))


def metrics_from_probs(
    y_true: np.ndarray, y_prob: np.ndarray, loss: float
) -> dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "loss": loss,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "roc_auc": safe_auc(y_true, y_prob, "roc"),
        "pr_auc": safe_auc(y_true, y_prob, "pr"),
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    label: str,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    train = optimizer is not None
    model.train(train)
    losses: list[float] = []
    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    iterator = tqdm(loader, desc=label, leave=False)
    for x_batch, y_batch in iterator:
        x_batch = x_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            if train:
                loss.backward()
                optimizer.step()

        probs = torch.sigmoid(logits).detach().cpu().numpy()
        labels = y_batch.detach().cpu().numpy()
        losses.append(float(loss.item()) * len(labels))
        all_probs.append(probs)
        all_labels.append(labels)

    y_true = np.concatenate(all_labels).astype(int)
    y_prob = np.concatenate(all_probs)
    metrics = metrics_from_probs(y_true, y_prob, float(np.sum(losses) / len(y_true)))
    return metrics, y_prob, y_true


def run_eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    rc_loader: DataLoader | None,
    criterion: nn.Module,
    device: torch.device,
    label: str,
    rc_tta: bool,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    metrics, probs, labels = run_epoch(model, loader, criterion, None, device, label)
    if not rc_tta:
        return metrics, probs, labels
    if rc_loader is None:
        raise ValueError("rc_loader is required when rc_tta=True")

    rc_metrics, rc_probs, rc_labels = run_epoch(
        model, rc_loader, criterion, None, device, f"{label} rc"
    )
    if not np.array_equal(labels, rc_labels):
        raise ValueError("Original and reverse-complement loaders produced different labels")

    tta_probs = (probs + rc_probs) / 2.0
    tta_loss = float((metrics["loss"] + rc_metrics["loss"]) / 2.0)
    return metrics_from_probs(labels, tta_probs, tta_loss), tta_probs, labels


def make_loader(
    x: np.ndarray,
    y: np.ndarray,
    indices: np.ndarray,
    positional_encoding: np.ndarray,
    batch_size: int,
    num_workers: int,
    sampler: WeightedRandomSampler | None = None,
    shuffle: bool = False,
    random_reverse_complement: bool = False,
    reverse_complement: bool = False,
) -> DataLoader:
    dataset = ClinVarSequenceDataset(
        x,
        y,
        indices,
        positional_encoding,
        random_reverse_complement=random_reverse_complement,
        reverse_complement=reverse_complement,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def threshold_table(y_true: np.ndarray, y_prob: np.ndarray) -> pd.DataFrame:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    table = pd.DataFrame(
        {
            "threshold": thresholds,
            "precision": precision[:-1],
            "recall": recall[:-1],
        }
    )
    denom_f1 = table["precision"] + table["recall"]
    table["f1"] = np.where(
        denom_f1 > 0, 2 * table["precision"] * table["recall"] / denom_f1, 0.0
    )
    beta2 = 4.0
    denom_f2 = beta2 * table["precision"] + table["recall"]
    table["f2"] = np.where(
        denom_f2 > 0,
        (1.0 + beta2) * table["precision"] * table["recall"] / denom_f2,
        0.0,
    )
    return table


def metrics_at_threshold(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float
) -> dict[str, object]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_pathogenic": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_pathogenic": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_pathogenic": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.random_state)

    project_dir = args.project_dir.resolve()
    processed_dir = project_dir / "processed"
    model_dir = project_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    length = args.length
    positional_name = {
        "none": "no_positional_encoding",
        "sinusoidal": "positional_encoding",
        "relative": "relative_positional_encoding",
    }[args.positional_encoding]
    safe_name = f"cnn_gene_group_{positional_name}_{length}"
    if args.architecture != "baseline":
        safe_name = f"{safe_name}_{args.architecture}"
    if args.rc_augment:
        safe_name = f"{safe_name}_rcaug"
    if args.rc_tta:
        safe_name = f"{safe_name}_rctta"
    if args.random_state != 42:
        safe_name = f"{safe_name}_seed{args.random_state}"
    x_path = processed_dir / f"X_ref_alt_marker_{length}.npy"
    y_path = processed_dir / f"y_{length}.npy"
    metadata_path = processed_dir / f"clinvar_training_metadata_{length}.parquet"
    model_path = model_dir / f"clinvar_{safe_name}_pytorch.pt"
    prediction_path = processed_dir / f"{safe_name}_test_predictions_pytorch.parquet"
    history_path = processed_dir / f"{safe_name}_training_history_pytorch.parquet"
    threshold_path = processed_dir / f"{safe_name}_threshold_table_pytorch.parquet"
    val_threshold_path = processed_dir / f"{safe_name}_val_threshold_table_pytorch.parquet"
    metrics_path = processed_dir / f"{safe_name}_metrics.json"

    outputs = [
        model_path,
        prediction_path,
        history_path,
        threshold_path,
        val_threshold_path,
        metrics_path,
    ]
    if any(path.exists() for path in outputs) and not args.force:
        existing = "\n".join(str(path) for path in outputs if path.exists())
        raise FileExistsError(
            "Output already exists. Use --force to overwrite:\n" + existing
        )

    print("Project:", project_dir)
    print("X:", x_path)
    print("y:", y_path)
    print("metadata:", metadata_path)
    x = np.load(x_path, mmap_mode="r")
    y = np.load(y_path, mmap_mode="r")
    metadata = pd.read_parquet(metadata_path)
    print("X shape:", x.shape, x.dtype)
    print("y shape:", y.shape, y.dtype)
    print("metadata shape:", metadata.shape)
    print("label counts:", dict(zip(*np.unique(y, return_counts=True))))

    groups = make_groups(metadata)
    all_idx = np.arange(len(y))
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=args.random_state)
    train_idx, temp_idx = next(gss1.split(all_idx, y, groups=groups))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=args.random_state)
    val_rel, test_rel = next(gss2.split(temp_idx, y[temp_idx], groups=groups[temp_idx]))
    val_idx = temp_idx[val_rel]
    test_idx = temp_idx[test_rel]

    train_idx = maybe_subsample(train_idx, args.debug_max_train_rows, y, args.random_state)
    val_idx = maybe_subsample(val_idx, args.debug_max_val_rows, y, args.random_state + 1)
    test_idx = maybe_subsample(test_idx, args.debug_max_test_rows, y, args.random_state + 2)

    print(
        f"train={len(train_idx):,}, val={len(val_idx):,}, test={len(test_idx):,}"
    )
    print("train labels:", dict(zip(*np.unique(y[train_idx], return_counts=True))))
    print("val labels:", dict(zip(*np.unique(y[val_idx], return_counts=True))))
    print("test labels:", dict(zip(*np.unique(y[test_idx], return_counts=True))))
    print("gene overlap train/test:", len(set(groups[train_idx]) & set(groups[test_idx])))

    positional_encoding = make_position_encoding(
        args.positional_encoding, x.shape[1], args.positional_dim
    )
    input_channels = x.shape[2] + positional_encoding.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    print("input_channels:", input_channels)
    print("batch_size:", args.batch_size)
    print("architecture:", args.architecture)
    print("positional_encoding:", args.positional_encoding)
    print("positional_dim:", args.positional_dim)
    print("rc_augment:", args.rc_augment)
    print("rc_tta:", args.rc_tta)

    train_loader = make_loader(
        x,
        y,
        train_idx,
        positional_encoding,
        args.batch_size,
        args.num_workers,
        sampler=make_weighted_sampler(train_idx, y),
        random_reverse_complement=args.rc_augment,
    )
    train_eval_loader = make_loader(
        x,
        y,
        train_idx,
        positional_encoding,
        args.batch_size,
        args.num_workers,
    )
    train_eval_rc_loader = (
        make_loader(
            x,
            y,
            train_idx,
            positional_encoding,
            args.batch_size,
            args.num_workers,
            reverse_complement=True,
        )
        if args.rc_tta
        else None
    )
    val_loader = make_loader(
        x, y, val_idx, positional_encoding, args.batch_size, args.num_workers
    )
    val_rc_loader = (
        make_loader(
            x,
            y,
            val_idx,
            positional_encoding,
            args.batch_size,
            args.num_workers,
            reverse_complement=True,
        )
        if args.rc_tta
        else None
    )
    test_loader = make_loader(
        x, y, test_idx, positional_encoding, args.batch_size, args.num_workers
    )
    test_rc_loader = (
        make_loader(
            x,
            y,
            test_idx,
            positional_encoding,
            args.batch_size,
            args.num_workers,
            reverse_complement=True,
        )
        if args.rc_tta
        else None
    )

    labels_train = y[train_idx].astype(int)
    counts = np.bincount(labels_train, minlength=2)
    pos_weight_value = float(np.sqrt(counts[0] / max(counts[1], 1)))
    model = make_model(args.architecture, input_channels).to(device)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight_value, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    history: list[dict[str, float | int | str]] = []
    best_val_pr_auc = -np.inf
    best_state: dict[str, torch.Tensor] | None = None
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        train_metrics, _, _ = run_epoch(
            model, train_loader, criterion, optimizer, device, f"train {epoch}"
        )
        val_metrics, _, _ = run_eval_epoch(
            model,
            val_loader,
            val_rc_loader,
            criterion,
            device,
            f"val {epoch}",
            args.rc_tta,
        )
        row = {"phase": "main", "epoch": epoch}
        row.update({f"train_{k}": v for k, v in train_metrics.items()})
        row.update({f"val_{k}": v for k, v in val_metrics.items()})
        history.append(row)
        print(row)
        if val_metrics["pr_auc"] > best_val_pr_auc:
            best_val_pr_auc = val_metrics["pr_auc"]
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            print("New best val_pr_auc:", f"{best_val_pr_auc:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    if not args.no_hard_negative and args.hard_negative_epochs > 0:
        print("Hard-negative mining on train set...")
        _, train_probs, train_labels = run_eval_epoch(
            model,
            train_eval_loader,
            train_eval_rc_loader,
            criterion,
            device,
            "predict train",
            args.rc_tta,
        )
        train_indices_ordered = train_idx
        pos_idx = train_indices_ordered[train_labels == 1]
        hard_neg_idx = train_indices_ordered[
            (train_labels == 0) & (train_probs >= 0.50)
        ]
        easy_neg_idx = train_indices_ordered[
            (train_labels == 0) & (train_probs < 0.50)
        ]
        max_easy = int((len(pos_idx) + len(hard_neg_idx)) * 1.0)
        rng = np.random.default_rng(args.random_state)
        if len(easy_neg_idx) > max_easy:
            easy_neg_idx = rng.choice(easy_neg_idx, size=max_easy, replace=False)
        hard_train_idx = np.concatenate([pos_idx, hard_neg_idx, easy_neg_idx])
        rng.shuffle(hard_train_idx)
        print("positive train rows:", f"{len(pos_idx):,}")
        print("hard negative rows:", f"{len(hard_neg_idx):,}")
        print("easy negative rows:", f"{len(easy_neg_idx):,}")
        print("fine-tune rows:", f"{len(hard_train_idx):,}")

        hard_loader = make_loader(
            x,
            y,
            hard_train_idx,
            positional_encoding,
            args.batch_size,
            args.num_workers,
            shuffle=True,
            random_reverse_complement=args.rc_augment,
        )
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.hard_negative_lr, weight_decay=args.weight_decay
        )
        for epoch in range(1, args.hard_negative_epochs + 1):
            print(f"Hard-negative epoch {epoch}/{args.hard_negative_epochs}")
            train_metrics, _, _ = run_epoch(
                model, hard_loader, criterion, optimizer, device, f"hard train {epoch}"
            )
            val_metrics, _, _ = run_eval_epoch(
                model,
                val_loader,
                val_rc_loader,
                criterion,
                device,
                f"hard val {epoch}",
                args.rc_tta,
            )
            row = {"phase": "hard_negative", "epoch": epoch}
            row.update({f"train_{k}": v for k, v in train_metrics.items()})
            row.update({f"val_{k}": v for k, v in val_metrics.items()})
            history.append(row)
            print(row)
            if val_metrics["pr_auc"] > best_val_pr_auc:
                best_val_pr_auc = val_metrics["pr_auc"]
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                print("New best val_pr_auc:", f"{best_val_pr_auc:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    final_val_metrics, final_val_probs, final_val_labels = run_eval_epoch(
        model,
        val_loader,
        val_rc_loader,
        criterion,
        device,
        "final val",
        args.rc_tta,
    )
    test_metrics, test_probs, test_labels = run_eval_epoch(
        model,
        test_loader,
        test_rc_loader,
        criterion,
        device,
        "test",
        args.rc_tta,
    )
    test_pred = (test_probs >= 0.5).astype(int)
    print("Test metrics:", test_metrics)
    print(
        classification_report(
            test_labels,
            test_pred,
            target_names=["benign", "pathogenic"],
            zero_division=0,
        )
    )
    print("Confusion matrix:")
    print(confusion_matrix(test_labels, test_pred))

    predictions = metadata.iloc[test_idx].copy()
    predictions["y_true"] = test_labels
    predictions["pred_proba_pathogenic"] = test_probs
    predictions["pred_label"] = test_pred
    predictions.to_parquet(prediction_path, index=False)

    thresholds = threshold_table(test_labels, test_probs)
    thresholds.to_parquet(threshold_path, index=False)
    val_thresholds = threshold_table(final_val_labels, final_val_probs)
    val_thresholds.to_parquet(val_threshold_path, index=False)
    history_df = pd.DataFrame(history)
    history_df.to_parquet(history_path, index=False)

    metrics = {
        "model_name": safe_name,
        "length": length,
        "input_channels": input_channels,
        "positional_encoding": args.positional_encoding,
        "positional_dim": args.positional_dim,
        "architecture": args.architecture,
        "rc_augment": args.rc_augment,
        "rc_tta": args.rc_tta,
        "split": "gene_group",
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "hard_negative_epochs": 0 if args.no_hard_negative else args.hard_negative_epochs,
        "pos_weight": pos_weight_value,
        "train_rows": int(len(train_idx)),
        "val_rows": int(len(val_idx)),
        "test_rows": int(len(test_idx)),
        "final_val_metrics": final_val_metrics,
        "test_metrics": test_metrics,
        "test_precision_pathogenic_05": float(
            precision_score(test_labels, test_pred, zero_division=0)
        ),
        "test_recall_pathogenic_05": float(
            recall_score(test_labels, test_pred, zero_division=0)
        ),
        "test_f1_pathogenic_05": float(
            f1_score(test_labels, test_pred, zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(test_labels, test_pred).tolist(),
        "best_val_pr_auc": float(best_val_pr_auc),
        "runtime_minutes": float((time.time() - start_time) / 60.0),
    }
    best_f1 = thresholds.loc[thresholds["f1"].idxmax()].to_dict()
    best_f2 = thresholds.loc[thresholds["f2"].idxmax()].to_dict()
    best_val_f1 = val_thresholds.loc[val_thresholds["f1"].idxmax()].to_dict()
    best_val_f2 = val_thresholds.loc[val_thresholds["f2"].idxmax()].to_dict()
    metrics["best_f1_threshold"] = {k: float(v) for k, v in best_f1.items()}
    metrics["best_f2_threshold"] = {k: float(v) for k, v in best_f2.items()}
    metrics["best_val_f1_threshold"] = {
        k: float(v) for k, v in best_val_f1.items()
    }
    metrics["best_val_f2_threshold"] = {
        k: float(v) for k, v in best_val_f2.items()
    }
    metrics["test_at_best_val_f1_threshold"] = metrics_at_threshold(
        test_labels, test_probs, float(best_val_f1["threshold"])
    )
    metrics["test_at_best_val_f2_threshold"] = metrics_at_threshold(
        test_labels, test_probs, float(best_val_f2["threshold"])
    )
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": metrics,
        },
        model_path,
    )

    print("Saved model:", model_path)
    print("Saved predictions:", prediction_path)
    print("Saved history:", history_path)
    print("Saved thresholds:", threshold_path)
    print("Saved val thresholds:", val_threshold_path)
    print("Saved metrics:", metrics_path)


if __name__ == "__main__":
    main()
