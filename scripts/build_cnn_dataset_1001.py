#!/usr/bin/env python3
"""Build a 1001 bp ref/alt/marker CNN dataset from filtered ClinVar SNVs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from pyfaidx import Fasta
from tqdm.auto import tqdm


BASE_TO_INDEX = np.full(256, 255, dtype=np.uint8)
for _idx, _base in enumerate(b"ACGT"):
    BASE_TO_INDEX[_base] = _idx


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(
        description="Create a 1001 bp 9-channel ClinVar CNN dataset."
    )
    parser.add_argument("--project-dir", type=Path, default=root)
    parser.add_argument("--flank", type=int, default=500)
    parser.add_argument(
        "--output-tag",
        default="",
        help=(
            "Optional suffix added after context length, e.g. fixed_refseq writes "
            "X_ref_alt_marker_601_fixed_refseq.npy."
        ),
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=root / "processed" / "clinvar_filtered_step8.parquet",
        help="Filtered ClinVar parquet from notebook 01 before context extraction.",
    )
    parser.add_argument(
        "--fasta",
        type=Path,
        default=root
        / "Data"
        / "ncbi_dataset"
        / "ncbi_dataset"
        / "data"
        / "GCF_000001405.26"
        / "GCF_000001405.26_GRCh38_genomic.fna",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def clean_base(value: object) -> str:
    return str(value).strip().upper()


def encode_sequence(sequence: str) -> np.ndarray:
    encoded = BASE_TO_INDEX[np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)]
    if np.any(encoded == 255):
        raise ValueError("sequence contains non-ACGT bases")
    return encoded


def main() -> None:
    args = parse_args()
    project_dir = args.project_dir.resolve()
    processed_dir = project_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    flank = args.flank
    context_length = flank * 2 + 1
    suffix = str(context_length)
    if args.output_tag:
        suffix = f"{suffix}_{args.output_tag}"

    context_path = processed_dir / f"clinvar_context_{suffix}.parquet"
    metadata_path = processed_dir / f"clinvar_training_metadata_{suffix}.parquet"
    x_path = processed_dir / f"X_ref_alt_marker_{suffix}.npy"
    y_path = processed_dir / f"y_{suffix}.npy"

    outputs = [context_path, metadata_path, x_path, y_path]
    if any(path.exists() for path in outputs) and not args.force:
        existing = "\n".join(str(path) for path in outputs if path.exists())
        raise FileExistsError(
            "Output already exists. Use --force to overwrite:\n" + existing
        )

    print("Project:", project_dir)
    print("Input:", args.input)
    print("FASTA:", args.fasta)
    print("Context length:", context_length)

    required_cols = [
        "Type",
        "GeneSymbol",
        "ClinicalSignificance",
        "LastEvaluated",
        "Assembly",
        "Chromosome",
        "ReviewStatus",
        "PositionVCF",
        "ReferenceAlleleVCF",
        "AlternateAlleleVCF",
        "Y",
        "ChromosomeFASTA",
        "ReferenceBaseGRCh38",
        "ReferenceAlleleMatchesGRCh38",
    ]
    df = pd.read_parquet(args.input, columns=required_cols).copy()
    print("Input rows:", f"{len(df):,}")

    genome = Fasta(str(args.fasta), as_raw=True, sequence_always_upper=True)
    positions = np.arange(context_length)

    metadata_rows: list[dict[str, object]] = []
    ref_indices: list[np.ndarray] = []
    alt_indices: list[np.ndarray] = []
    y_values: list[int] = []
    skip_counts = {
        "bad_allele": 0,
        "chrom_missing": 0,
        "edge_or_short": 0,
        "non_acgt_context": 0,
        "ref_mismatch": 0,
    }

    iterator = df.itertuples(index=False)
    for row in tqdm(iterator, total=len(df), desc=f"Extract {context_length} bp"):
        ref = clean_base(row.ReferenceAlleleVCF)
        alt = clean_base(row.AlternateAlleleVCF)
        if len(ref) != 1 or len(alt) != 1 or ref not in "ACGT" or alt not in "ACGT":
            skip_counts["bad_allele"] += 1
            continue

        chrom = str(row.ChromosomeFASTA)
        if chrom not in genome:
            skip_counts["chrom_missing"] += 1
            continue

        pos = int(row.PositionVCF)
        start_1based = pos - flank
        end_1based = pos + flank
        if start_1based < 1:
            skip_counts["edge_or_short"] += 1
            continue

        seq = str(genome[chrom][start_1based - 1 : end_1based])
        if len(seq) != context_length:
            skip_counts["edge_or_short"] += 1
            continue

        if seq[flank] != ref:
            skip_counts["ref_mismatch"] += 1
            continue

        try:
            ref_encoded = encode_sequence(seq)
        except ValueError:
            skip_counts["non_acgt_context"] += 1
            continue

        alt_seq = seq[:flank] + alt + seq[flank + 1 :]
        alt_encoded = ref_encoded.copy()
        alt_encoded[flank] = BASE_TO_INDEX[ord(alt)]

        record = row._asdict()
        record["ContextStart1Based"] = start_1based
        record["ContextEnd1Based"] = end_1based
        metadata_rows.append(record)
        ref_indices.append(ref_encoded)
        alt_indices.append(alt_encoded)
        y_values.append(int(row.Y))

    metadata = pd.DataFrame(metadata_rows)
    y = np.asarray(y_values, dtype=np.int8)
    n_rows = len(metadata)
    print("Kept rows:", f"{n_rows:,}")
    print("Skipped:", skip_counts)
    print("Label counts:", dict(zip(*np.unique(y, return_counts=True))))

    metadata.to_parquet(metadata_path, index=False)
    context_cols = [
        "Chromosome",
        "PositionVCF",
        "ReferenceAlleleVCF",
        "AlternateAlleleVCF",
        "ChromosomeFASTA",
        "ContextStart1Based",
        "ContextEnd1Based",
        "Y",
    ]
    metadata[context_cols].to_parquet(context_path, index=False)
    np.save(y_path, y)

    print("Writing tensor:", x_path)
    x = np.lib.format.open_memmap(
        x_path, mode="w+", dtype=np.uint8, shape=(n_rows, context_length, 9)
    )
    for i in tqdm(range(n_rows), desc="Write one-hot tensor"):
        x[i, :, :] = 0
        x[i, positions, ref_indices[i]] = 1
        x[i, positions, alt_indices[i] + 4] = 1
        x[i, flank, 8] = 1
    x.flush()

    x_check = np.load(x_path, mmap_mode="r")
    print("Saved metadata:", metadata_path)
    print("Saved context index:", context_path)
    print("Saved X:", x_path, x_check.shape, x_check.dtype)
    print("Saved y:", y_path, y.shape, y.dtype)


if __name__ == "__main__":
    main()
