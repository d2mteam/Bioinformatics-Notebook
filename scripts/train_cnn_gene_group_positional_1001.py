#!/usr/bin/env python3
"""Train a gene-group CNN with positional encoding on the 1001 bp dataset."""

from __future__ import annotations

import argparse
import hashlib
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
    parser.add_argument(
        "--dataset-tag",
        default="",
        help=(
            "Optional suffix after length for dataset files, e.g. fixed_refseq "
            "loads X_ref_alt_marker_601_fixed_refseq.npy."
        ),
    )
    parser.add_argument(
        "--label-mode",
        choices=["all", "definitive_only"],
        default="all",
        help=(
            "all keeps Benign/Likely benign and Pathogenic/Likely pathogenic labels. "
            "definitive_only keeps only exact Benign and Pathogenic labels."
        ),
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--hard-negative-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hard-negative-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--split-mode",
        choices=["gene_group", "purged_gene_group", "chromosome", "genome_block"],
        default="gene_group",
        help=(
            "Split strategy. purged_gene_group removes val/test variants near "
            "train regions; chromosome holds out whole chromosomes; genome_block "
            "splits fixed genomic coordinate blocks and purges nearby boundaries."
        ),
    )
    parser.add_argument(
        "--genome-block-size-bp",
        type=int,
        default=1_000_000,
        help="Block size for split-mode=genome_block.",
    )
    parser.add_argument(
        "--purge-distance-bp",
        type=int,
        default=5000,
        help="Minimum allowed distance from train/selection variants for purged_gene_group.",
    )
    parser.add_argument(
        "--sequence-purge-mode",
        choices=["none", "exact_ref", "exact_tensor"],
        default="none",
        help=(
            "Remove val/test rows whose sequence context is already present in "
            "earlier splits. exact_ref hashes REF one-hot channels only; "
            "exact_tensor hashes all input channels."
        ),
    )
    parser.add_argument(
        "--scheduler",
        choices=["none", "cosine", "onecycle"],
        default="cosine",
        help="Per-batch learning-rate scheduler for the main training phase.",
    )
    parser.add_argument("--warmup-epochs", type=float, default=1.0)
    parser.add_argument("--min-lr-ratio", type=float, default=0.05)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
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


def make_chromosome_groups(metadata: pd.DataFrame) -> np.ndarray:
    groups = (
        metadata["Chromosome"]
        .fillna("unknown")
        .astype(str)
        .str.replace("chr", "", case=False, regex=False)
        .to_numpy(copy=True)
    )
    unknown = (groups == "") | (groups == "unknown") | (groups == "nan")
    row_ids = np.arange(len(groups)).astype(str)
    groups[unknown] = np.char.add("unknown_chr_row_", row_ids[unknown])
    return groups


def make_genome_block_groups(
    metadata: pd.DataFrame, chromosome_groups: np.ndarray, block_size_bp: int
) -> np.ndarray:
    if block_size_bp <= 0:
        raise ValueError("genome block size must be positive")
    positions = pd.to_numeric(metadata["PositionVCF"], errors="coerce")
    blocks = ((positions - 1) // block_size_bp).astype("Int64")
    groups = pd.Series(chromosome_groups, index=metadata.index).astype(str)
    block_groups = groups + ":block_" + blocks.astype(str)
    invalid = positions.isna() | groups.isin(["", "unknown", "nan"]) | blocks.isna()
    if invalid.any():
        row_ids = pd.Series(np.arange(len(metadata)), index=metadata.index).astype(str)
        block_groups.loc[invalid] = "unknown_block_row_" + row_ids.loc[invalid]
    return block_groups.to_numpy(dtype=str, copy=True)


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


def select_label_indices(metadata: pd.DataFrame, label_mode: str) -> np.ndarray:
    if label_mode == "all":
        return np.arange(len(metadata), dtype=np.int64)
    if label_mode == "definitive_only":
        clinical = metadata["ClinicalSignificance"].fillna("").astype(str).str.strip()
        mask = clinical.isin(["Benign", "Pathogenic"]).to_numpy()
        return np.flatnonzero(mask).astype(np.int64)
    raise ValueError(f"Unsupported label_mode: {label_mode}")


def make_group_split(
    y: np.ndarray, groups: np.ndarray, random_state: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_idx = np.arange(len(y))
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=random_state)
    train_idx, temp_idx = next(gss1.split(all_idx, y, groups=groups))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=random_state)
    val_rel, test_rel = next(gss2.split(temp_idx, y[temp_idx], groups=groups[temp_idx]))
    val_idx = temp_idx[val_rel]
    test_idx = temp_idx[test_rel]
    return train_idx, val_idx, test_idx


def make_gene_group_split(
    y: np.ndarray, groups: np.ndarray, random_state: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return make_group_split(y, groups, random_state)


def make_chromosome_split(
    y: np.ndarray, chromosome_groups: np.ndarray, random_state: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return make_group_split(y, chromosome_groups, random_state)


def make_genome_block_split(
    y: np.ndarray, genome_block_groups: np.ndarray, random_state: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return make_group_split(y, genome_block_groups, random_state)


def nearest_distance_to_reference(
    metadata: pd.DataFrame, query_idx: np.ndarray, reference_idx: np.ndarray
) -> np.ndarray:
    train_pos_by_chr: dict[str, np.ndarray] = {}
    ref_df = metadata.iloc[reference_idx]
    for chrom, sub in ref_df.groupby("Chromosome", sort=False):
        positions = pd.to_numeric(sub["PositionVCF"], errors="coerce").dropna()
        train_pos_by_chr[str(chrom)] = np.sort(positions.to_numpy(dtype=np.int64))

    query_df = metadata.iloc[query_idx]
    query_chr = query_df["Chromosome"].astype(str).to_numpy()
    query_pos = pd.to_numeric(query_df["PositionVCF"], errors="coerce").to_numpy()
    distances = np.full(len(query_idx), np.inf, dtype=np.float64)
    for i, (chrom, pos) in enumerate(zip(query_chr, query_pos)):
        if pd.isna(pos):
            continue
        ref_positions = train_pos_by_chr.get(str(chrom))
        if ref_positions is None or len(ref_positions) == 0:
            continue
        pos_int = int(pos)
        insert_at = np.searchsorted(ref_positions, pos_int)
        best = np.inf
        if insert_at > 0:
            best = min(best, abs(pos_int - int(ref_positions[insert_at - 1])))
        if insert_at < len(ref_positions):
            best = min(best, abs(pos_int - int(ref_positions[insert_at])))
        distances[i] = best
    return distances


def distance_summary(distances: np.ndarray) -> dict[str, float]:
    finite = distances[np.isfinite(distances)]
    if len(finite) == 0:
        return {
            "min": float("inf"),
            "p5": float("inf"),
            "median": float("inf"),
            "p95": float("inf"),
        }
    return {
        "min": float(np.min(finite)),
        "p5": float(np.percentile(finite, 5)),
        "median": float(np.median(finite)),
        "p95": float(np.percentile(finite, 95)),
    }


def purge_nearby_splits(
    metadata: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    purge_distance_bp: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    val_dist = nearest_distance_to_reference(metadata, val_idx, train_idx)
    keep_val = val_dist >= purge_distance_bp
    kept_val_idx = val_idx[keep_val]
    purged_val_idx = val_idx[~keep_val]

    test_reference_idx = np.concatenate([train_idx, kept_val_idx])
    test_dist = nearest_distance_to_reference(metadata, test_idx, test_reference_idx)
    keep_test = test_dist >= purge_distance_bp
    kept_test_idx = test_idx[keep_test]
    purged_test_idx = test_idx[~keep_test]

    audit = {
        "purge_distance_bp": int(purge_distance_bp),
        "val_rows_before_purge": int(len(val_idx)),
        "val_rows_after_purge": int(len(kept_val_idx)),
        "val_rows_purged": int(len(purged_val_idx)),
        "test_rows_before_purge": int(len(test_idx)),
        "test_rows_after_purge": int(len(kept_test_idx)),
        "test_rows_purged": int(len(purged_test_idx)),
        "val_nearest_train_distance_bp": distance_summary(
            nearest_distance_to_reference(metadata, kept_val_idx, train_idx)
        ),
        "test_nearest_train_val_distance_bp": distance_summary(
            nearest_distance_to_reference(metadata, kept_test_idx, test_reference_idx)
        ),
    }
    return kept_val_idx, kept_test_idx, purged_val_idx, purged_test_idx, audit


def context_hash(x: np.ndarray, index: int, mode: str) -> bytes:
    if mode == "exact_ref":
        arr = x[index, :, :4]
    elif mode == "exact_tensor":
        arr = x[index]
    else:
        raise ValueError(f"Unknown sequence purge mode: {mode}")
    arr = np.ascontiguousarray(arr).view(np.uint8)
    return hashlib.blake2b(arr, digest_size=16).digest()


def hash_contexts(
    x: np.ndarray, indices: np.ndarray, mode: str, label: str
) -> set[bytes]:
    hashes: set[bytes] = set()
    for idx in tqdm(indices, desc=label, leave=False):
        hashes.add(context_hash(x, int(idx), mode))
    return hashes


def filter_sequence_context_duplicates(
    x: np.ndarray,
    query_idx: np.ndarray,
    reference_hashes: set[bytes],
    mode: str,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    keep = np.ones(len(query_idx), dtype=bool)
    for i, idx in enumerate(tqdm(query_idx, desc=label, leave=False)):
        if context_hash(x, int(idx), mode) in reference_hashes:
            keep[i] = False
    return query_idx[keep], query_idx[~keep]


def purge_sequence_context_splits(
    x: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    if mode == "none":
        return (
            val_idx,
            test_idx,
            np.asarray([], dtype=np.int64),
            np.asarray([], dtype=np.int64),
            {"sequence_purge_mode": mode},
        )

    train_hashes = hash_contexts(
        x, train_idx, mode, f"hash train contexts ({mode})"
    )
    kept_val_idx, purged_val_idx = filter_sequence_context_duplicates(
        x,
        val_idx,
        train_hashes,
        mode,
        f"purge val sequence duplicates ({mode})",
    )
    kept_val_hashes = hash_contexts(
        x, kept_val_idx, mode, f"hash kept val contexts ({mode})"
    )
    reference_hashes = train_hashes | kept_val_hashes
    kept_test_idx, purged_test_idx = filter_sequence_context_duplicates(
        x,
        test_idx,
        reference_hashes,
        mode,
        f"purge test sequence duplicates ({mode})",
    )
    audit = {
        "sequence_purge_mode": mode,
        "val_rows_before_sequence_purge": int(len(val_idx)),
        "val_rows_after_sequence_purge": int(len(kept_val_idx)),
        "val_rows_purged_by_sequence": int(len(purged_val_idx)),
        "test_rows_before_sequence_purge": int(len(test_idx)),
        "test_rows_after_sequence_purge": int(len(kept_test_idx)),
        "test_rows_purged_by_sequence": int(len(purged_test_idx)),
        "reference_context_hashes_after_val": int(len(reference_hashes)),
    }
    return kept_val_idx, kept_test_idx, purged_val_idx, purged_test_idx, audit


def label_count_dict(y: np.ndarray, indices: np.ndarray) -> dict[int, int]:
    labels, counts = np.unique(y[indices], return_counts=True)
    return {int(label): int(count) for label, count in zip(labels, counts)}


def make_split_audit(
    metadata: pd.DataFrame,
    y: np.ndarray,
    gene_groups: np.ndarray,
    chromosome_groups: np.ndarray,
    genome_block_groups: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    split_mode: str,
    purge_distance_bp: int,
    genome_block_size_bp: int,
) -> dict[str, object]:
    train_genes = set(gene_groups[train_idx])
    val_genes = set(gene_groups[val_idx])
    test_genes = set(gene_groups[test_idx])
    train_chromosomes = set(chromosome_groups[train_idx])
    val_chromosomes = set(chromosome_groups[val_idx])
    test_chromosomes = set(chromosome_groups[test_idx])
    train_blocks = set(genome_block_groups[train_idx])
    val_blocks = set(genome_block_groups[val_idx])
    test_blocks = set(genome_block_groups[test_idx])
    audit: dict[str, object] = {
        "split_mode": split_mode,
        "purge_distance_bp": int(purge_distance_bp),
        "genome_block_size_bp": int(genome_block_size_bp),
        "train_rows": int(len(train_idx)),
        "val_rows": int(len(val_idx)),
        "test_rows": int(len(test_idx)),
        "train_label_counts": label_count_dict(y, train_idx),
        "val_label_counts": label_count_dict(y, val_idx),
        "test_label_counts": label_count_dict(y, test_idx),
        "gene_overlap_train_val": int(len(train_genes & val_genes)),
        "gene_overlap_train_test": int(len(train_genes & test_genes)),
        "gene_overlap_val_test": int(len(val_genes & test_genes)),
        "chromosome_overlap_train_val": int(len(train_chromosomes & val_chromosomes)),
        "chromosome_overlap_train_test": int(len(train_chromosomes & test_chromosomes)),
        "chromosome_overlap_val_test": int(len(val_chromosomes & test_chromosomes)),
        "train_chromosomes": sorted(train_chromosomes),
        "val_chromosomes": sorted(val_chromosomes),
        "test_chromosomes": sorted(test_chromosomes),
        "genome_block_overlap_train_val": int(len(train_blocks & val_blocks)),
        "genome_block_overlap_train_test": int(len(train_blocks & test_blocks)),
        "genome_block_overlap_val_test": int(len(val_blocks & test_blocks)),
        "train_genome_blocks": int(len(train_blocks)),
        "val_genome_blocks": int(len(val_blocks)),
        "test_genome_blocks": int(len(test_blocks)),
        "val_nearest_train_distance_bp": distance_summary(
            nearest_distance_to_reference(metadata, val_idx, train_idx)
        ),
        "test_nearest_train_val_distance_bp": distance_summary(
            nearest_distance_to_reference(
                metadata, test_idx, np.concatenate([train_idx, val_idx])
            )
        ),
    }
    if split_mode == "purged_gene_group":
        val_min = audit["val_nearest_train_distance_bp"]["min"]  # type: ignore[index]
        test_min = audit["test_nearest_train_val_distance_bp"]["min"]  # type: ignore[index]
        if val_min < purge_distance_bp or test_min < purge_distance_bp:
            raise RuntimeError(
                "Purged split audit failed: nearest-distance minimum is below "
                f"{purge_distance_bp} bp."
            )
    if split_mode == "chromosome":
        if (
            audit["chromosome_overlap_train_val"] > 0
            or audit["chromosome_overlap_train_test"] > 0
            or audit["chromosome_overlap_val_test"] > 0
        ):
            raise RuntimeError("Chromosome split audit failed: chromosome overlap detected.")
    if split_mode == "genome_block":
        if (
            audit["genome_block_overlap_train_val"] > 0
            or audit["genome_block_overlap_train_test"] > 0
            or audit["genome_block_overlap_val_test"] > 0
        ):
            raise RuntimeError("Genome-block split audit failed: block overlap detected.")
        val_min = audit["val_nearest_train_distance_bp"]["min"]  # type: ignore[index]
        test_min = audit["test_nearest_train_val_distance_bp"]["min"]  # type: ignore[index]
        if val_min < purge_distance_bp or test_min < purge_distance_bp:
            raise RuntimeError(
                "Genome-block split audit failed: nearest-distance minimum is below "
                f"{purge_distance_bp} bp."
            )
    return audit


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
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    grad_clip_norm: float | None = None,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    train = optimizer is not None
    model.train(train)
    losses: list[float] = []
    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    last_lr = float(optimizer.param_groups[0]["lr"]) if optimizer is not None else float("nan")
    last_grad_norm = float("nan")

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
                if grad_clip_norm is not None and grad_clip_norm > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), grad_clip_norm
                    )
                    last_grad_norm = float(grad_norm.detach().cpu())
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                last_lr = float(optimizer.param_groups[0]["lr"])

        probs = torch.sigmoid(logits).detach().cpu().numpy()
        labels = y_batch.detach().cpu().numpy()
        losses.append(float(loss.item()) * len(labels))
        all_probs.append(probs)
        all_labels.append(labels)

    y_true = np.concatenate(all_labels).astype(int)
    y_prob = np.concatenate(all_probs)
    metrics = metrics_from_probs(y_true, y_prob, float(np.sum(losses) / len(y_true)))
    if train:
        metrics["lr"] = last_lr
        metrics["grad_norm"] = last_grad_norm
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


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
    max_lr: float,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    if scheduler_name == "none" or total_steps <= 0:
        return None
    warmup_steps = max(0, min(warmup_steps, total_steps - 1))
    min_lr_ratio = max(0.0, min(1.0, min_lr_ratio))

    if scheduler_name == "cosine":

        def lr_lambda(step: int) -> float:
            if warmup_steps > 0 and step < warmup_steps:
                return max((step + 1) / warmup_steps, min_lr_ratio)
            decay_steps = max(1, total_steps - warmup_steps)
            progress = min(1.0, max(0.0, (step - warmup_steps) / decay_steps))
            cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    if scheduler_name == "onecycle":
        pct_start = min(0.5, max(0.05, warmup_steps / max(total_steps, 1)))
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=pct_start,
            div_factor=25.0,
            final_div_factor=max(1.0, 25.0 / max(min_lr_ratio, 1e-6)),
        )

    raise ValueError(f"Unknown scheduler: {scheduler_name}")


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
    dataset_suffix = str(length)
    if args.dataset_tag:
        dataset_suffix = f"{dataset_suffix}_{args.dataset_tag}"
    positional_name = {
        "none": "no_positional_encoding",
        "sinusoidal": "positional_encoding",
        "relative": "relative_positional_encoding",
    }[args.positional_encoding]
    safe_name = f"cnn_gene_group_{positional_name}_{dataset_suffix}"
    if args.architecture != "baseline":
        safe_name = f"{safe_name}_{args.architecture}"
    if args.rc_augment:
        safe_name = f"{safe_name}_rcaug"
    if args.rc_tta:
        safe_name = f"{safe_name}_rctta"
    if args.split_mode == "purged_gene_group":
        safe_name = f"{safe_name}_purged{args.purge_distance_bp}"
    if args.split_mode == "chromosome":
        safe_name = f"{safe_name}_chromosome_split"
    if args.split_mode == "genome_block":
        safe_name = (
            f"{safe_name}_genome_block{args.genome_block_size_bp}"
            f"_purged{args.purge_distance_bp}"
        )
    if args.sequence_purge_mode != "none":
        safe_name = f"{safe_name}_seqpurge_{args.sequence_purge_mode}"
    if args.label_mode != "all":
        safe_name = f"{safe_name}_{args.label_mode}"
    if args.random_state != 42:
        safe_name = f"{safe_name}_seed{args.random_state}"
    x_path = processed_dir / f"X_ref_alt_marker_{dataset_suffix}.npy"
    y_path = processed_dir / f"y_{dataset_suffix}.npy"
    metadata_path = processed_dir / f"clinvar_training_metadata_{dataset_suffix}.parquet"
    model_path = model_dir / f"clinvar_{safe_name}_pytorch.pt"
    prediction_path = processed_dir / f"{safe_name}_test_predictions_pytorch.parquet"
    history_path = processed_dir / f"{safe_name}_training_history_pytorch.parquet"
    threshold_path = processed_dir / f"{safe_name}_threshold_table_pytorch.parquet"
    val_threshold_path = processed_dir / f"{safe_name}_val_threshold_table_pytorch.parquet"
    metrics_path = processed_dir / f"{safe_name}_metrics.json"
    split_indices_path = processed_dir / f"{safe_name}_split_indices.npz"

    outputs = [
        model_path,
        prediction_path,
        history_path,
        threshold_path,
        val_threshold_path,
        metrics_path,
        split_indices_path,
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

    selected_idx = select_label_indices(metadata, args.label_mode)
    metadata_for_split = metadata.iloc[selected_idx].reset_index(drop=True)
    y_for_split = np.asarray(y[selected_idx], dtype=np.int8)
    print("label_mode:", args.label_mode)
    print("selected rows:", f"{len(selected_idx):,}", "/", f"{len(metadata):,}")
    print(
        "selected label counts:",
        dict(zip(*np.unique(y_for_split, return_counts=True))),
    )

    gene_groups = make_groups(metadata_for_split)
    chromosome_groups = make_chromosome_groups(metadata_for_split)
    genome_block_groups = make_genome_block_groups(
        metadata_for_split, chromosome_groups, args.genome_block_size_bp
    )
    if args.split_mode == "chromosome":
        train_idx, val_idx, test_idx = make_chromosome_split(
            y_for_split, chromosome_groups, args.random_state
        )
    elif args.split_mode == "genome_block":
        train_idx, val_idx, test_idx = make_genome_block_split(
            y_for_split, genome_block_groups, args.random_state
        )
    else:
        train_idx, val_idx, test_idx = make_gene_group_split(
            y_for_split, gene_groups, args.random_state
        )
    purged_val_idx = np.asarray([], dtype=np.int64)
    purged_test_idx = np.asarray([], dtype=np.int64)
    sequence_purged_val_idx = np.asarray([], dtype=np.int64)
    sequence_purged_test_idx = np.asarray([], dtype=np.int64)
    purge_audit: dict[str, object] = {}
    sequence_purge_audit: dict[str, object] = {
        "sequence_purge_mode": args.sequence_purge_mode
    }
    debug_mode = any(
        value is not None
        for value in [
            args.debug_max_train_rows,
            args.debug_max_val_rows,
            args.debug_max_test_rows,
        ]
    )
    if args.split_mode in {"purged_gene_group", "genome_block"}:
        val_idx, test_idx, purged_val_idx, purged_test_idx, purge_audit = (
            purge_nearby_splits(
                metadata_for_split,
                train_idx,
                val_idx,
                test_idx,
                args.purge_distance_bp,
            )
        )
        if not debug_mode:
            if len(val_idx) < 10_000 or len(test_idx) < 10_000:
                raise RuntimeError(
                    f"{args.split_mode} split leaves too few validation/test rows: "
                    f"val={len(val_idx):,}, test={len(test_idx):,}."
                )
            if int(y_for_split[val_idx].sum()) < 1_000 or int(y_for_split[test_idx].sum()) < 1_000:
                raise RuntimeError(
                    f"{args.split_mode} split leaves too few pathogenic rows: "
                    f"val_pos={int(y_for_split[val_idx].sum()):,}, "
                    f"test_pos={int(y_for_split[test_idx].sum()):,}."
                )

    train_idx = maybe_subsample(
        train_idx, args.debug_max_train_rows, y_for_split, args.random_state
    )
    val_idx = maybe_subsample(
        val_idx, args.debug_max_val_rows, y_for_split, args.random_state + 1
    )
    test_idx = maybe_subsample(
        test_idx, args.debug_max_test_rows, y_for_split, args.random_state + 2
    )
    if args.sequence_purge_mode != "none":
        full_to_selected = np.full(len(metadata), -1, dtype=np.int64)
        full_to_selected[selected_idx] = np.arange(len(selected_idx), dtype=np.int64)
        (
            val_idx_full,
            test_idx_full,
            sequence_purged_val_idx_full,
            sequence_purged_test_idx_full,
            sequence_purge_audit,
        ) = purge_sequence_context_splits(
            x,
            selected_idx[train_idx],
            selected_idx[val_idx],
            selected_idx[test_idx],
            args.sequence_purge_mode,
        )
        val_idx = full_to_selected[val_idx_full]
        test_idx = full_to_selected[test_idx_full]
        sequence_purged_val_idx = sequence_purged_val_idx_full
        sequence_purged_test_idx = sequence_purged_test_idx_full
        if not debug_mode:
            if len(val_idx) < 10_000 or len(test_idx) < 10_000:
                raise RuntimeError(
                    "Sequence purge leaves too few validation/test rows: "
                    f"val={len(val_idx):,}, test={len(test_idx):,}."
                )
            if int(y_for_split[val_idx].sum()) < 1_000 or int(y_for_split[test_idx].sum()) < 1_000:
                raise RuntimeError(
                    "Sequence purge leaves too few pathogenic rows: "
                    f"val_pos={int(y_for_split[val_idx].sum()):,}, "
                    f"test_pos={int(y_for_split[test_idx].sum()):,}."
                )
    split_audit = make_split_audit(
        metadata_for_split,
        y_for_split,
        gene_groups,
        chromosome_groups,
        genome_block_groups,
        train_idx,
        val_idx,
        test_idx,
        args.split_mode,
        args.purge_distance_bp,
        args.genome_block_size_bp,
    )

    train_idx_local = train_idx
    val_idx_local = val_idx
    test_idx_local = test_idx
    purged_val_idx = selected_idx[purged_val_idx]
    purged_test_idx = selected_idx[purged_test_idx]
    train_idx = selected_idx[train_idx_local]
    val_idx = selected_idx[val_idx_local]
    test_idx = selected_idx[test_idx_local]

    print(
        f"train={len(train_idx):,}, val={len(val_idx):,}, test={len(test_idx):,}"
    )
    print("train labels:", dict(zip(*np.unique(y[train_idx], return_counts=True))))
    print("val labels:", dict(zip(*np.unique(y[val_idx], return_counts=True))))
    print("test labels:", dict(zip(*np.unique(y[test_idx], return_counts=True))))
    print("split_mode:", args.split_mode)
    print("purge_distance_bp:", args.purge_distance_bp)
    print("sequence_purge_mode:", args.sequence_purge_mode)
    print("sequence purge audit:", sequence_purge_audit)
    print("gene overlap train/val:", split_audit["gene_overlap_train_val"])
    print("gene overlap train/test:", split_audit["gene_overlap_train_test"])
    print("chromosome overlap train/val:", split_audit["chromosome_overlap_train_val"])
    print("chromosome overlap train/test:", split_audit["chromosome_overlap_train_test"])
    print(
        "genome block overlap train/val:",
        split_audit["genome_block_overlap_train_val"],
    )
    print(
        "genome block overlap train/test:",
        split_audit["genome_block_overlap_train_test"],
    )
    print("genome_block_size_bp:", args.genome_block_size_bp)
    print("train chromosomes:", split_audit["train_chromosomes"])
    print("val chromosomes:", split_audit["val_chromosomes"])
    print("test chromosomes:", split_audit["test_chromosomes"])
    print(
        "val nearest train distance:",
        split_audit["val_nearest_train_distance_bp"],
    )
    print(
        "test nearest train/val distance:",
        split_audit["test_nearest_train_val_distance_bp"],
    )

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
    print("scheduler:", args.scheduler)
    print("warmup_epochs:", args.warmup_epochs)
    print("min_lr_ratio:", args.min_lr_ratio)
    print("grad_clip_norm:", args.grad_clip_norm)

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
    main_total_steps = max(1, len(train_loader) * args.epochs)
    main_warmup_steps = int(round(args.warmup_epochs * len(train_loader)))
    scheduler = make_scheduler(
        optimizer,
        args.scheduler,
        main_total_steps,
        main_warmup_steps,
        args.min_lr_ratio,
        args.learning_rate,
    )

    history: list[dict[str, float | int | str]] = []
    best_val_pr_auc = -np.inf
    best_state: dict[str, torch.Tensor] | None = None
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        train_metrics, _, _ = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            f"train {epoch}",
            scheduler=scheduler,
            grad_clip_norm=args.grad_clip_norm,
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
        row = {
            "phase": "main",
            "epoch": epoch,
            "scheduler": args.scheduler,
            "warmup_epochs": args.warmup_epochs,
            "grad_clip_norm": args.grad_clip_norm,
        }
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
                model,
                hard_loader,
                criterion,
                optimizer,
                device,
                f"hard train {epoch}",
                scheduler=None,
                grad_clip_norm=args.grad_clip_norm,
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
            row = {
                "phase": "hard_negative",
                "epoch": epoch,
                "scheduler": "none",
                "warmup_epochs": 0,
                "grad_clip_norm": args.grad_clip_norm,
            }
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
    np.savez_compressed(
        split_indices_path,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        purged_val_idx=purged_val_idx,
        purged_test_idx=purged_test_idx,
        sequence_purged_val_idx=sequence_purged_val_idx,
        sequence_purged_test_idx=sequence_purged_test_idx,
    )

    metrics = {
        "model_name": safe_name,
        "length": length,
        "dataset_tag": args.dataset_tag,
        "dataset_suffix": dataset_suffix,
        "label_mode": args.label_mode,
        "selected_rows": int(len(selected_idx)),
        "input_channels": input_channels,
        "positional_encoding": args.positional_encoding,
        "positional_dim": args.positional_dim,
        "architecture": args.architecture,
        "rc_augment": args.rc_augment,
        "rc_tta": args.rc_tta,
        "split": args.split_mode,
        "purge_distance_bp": args.purge_distance_bp,
        "genome_block_size_bp": args.genome_block_size_bp,
        "sequence_purge_mode": args.sequence_purge_mode,
        "split_audit": split_audit,
        "purge_audit": purge_audit,
        "sequence_purge_audit": sequence_purge_audit,
        "scheduler": args.scheduler,
        "warmup_epochs": args.warmup_epochs,
        "min_lr_ratio": args.min_lr_ratio,
        "grad_clip_norm": args.grad_clip_norm,
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
    print("Saved split indices:", split_indices_path)


if __name__ == "__main__":
    main()
