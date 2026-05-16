#!/usr/bin/env python3
"""Build a compact ClinVar reference-sequence Parquet file for CNN datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
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
        description=(
            "Create a compact Parquet dataset with one reference sequence string "
            "per ClinVar SNV. The CNN one-hot tensor can be built on the fly."
        )
    )
    parser.add_argument("--project-dir", type=Path, default=root)
    parser.add_argument("--flank", type=int, default=4000)
    parser.add_argument(
        "--input",
        type=Path,
        default=root / "processed" / "clinvar_filtered_step8.parquet",
        help="Filtered ClinVar parquet from notebook 01.",
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
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output Parquet path. Defaults to processed/clinvar_refseq_<length>.parquet.",
    )
    parser.add_argument("--chunk-size", type=int, default=10_000)
    parser.add_argument("--compression", default="zstd")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def clean_base(value: object) -> str:
    return str(value).strip().upper()


def is_acgt_sequence(sequence: str) -> bool:
    encoded = BASE_TO_INDEX[np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)]
    return not np.any(encoded == 255)


def main() -> None:
    args = parse_args()
    project_dir = args.project_dir.resolve()
    processed_dir = project_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    flank = args.flank
    context_length = flank * 2 + 1
    output_path = args.output
    if output_path is None:
        output_path = processed_dir / f"clinvar_refseq_{context_length}.parquet"
    output_path = output_path.resolve()

    if output_path.exists() and not args.force:
        raise FileExistsError(f"Output already exists. Use --force: {output_path}")

    required_cols = [
        "Type",
        "GeneSymbol",
        "ClinicalSignificance",
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
    output_cols = [
        "SourceRow",
        *required_cols,
        "ContextStart1Based",
        "ContextEnd1Based",
        "ContextLength",
        "CenterIndex",
        "ref_seq",
    ]

    print("Project:", project_dir)
    print("Input:", args.input)
    print("FASTA:", args.fasta)
    print("Output:", output_path)
    print("Context length:", context_length)

    df = pd.read_parquet(args.input, columns=required_cols).copy()
    print("Input rows:", f"{len(df):,}")

    genome = Fasta(str(args.fasta), as_raw=True, sequence_always_upper=True)
    skip_counts = {
        "bad_allele": 0,
        "chrom_missing": 0,
        "bad_position": 0,
        "edge_or_short": 0,
        "non_acgt_context": 0,
        "ref_mismatch": 0,
    }
    label_counts = {0: 0, 1: 0}
    kept_rows = 0
    buffer: list[dict[str, object]] = []
    writer: pq.ParquetWriter | None = None

    def flush() -> None:
        nonlocal buffer, writer
        if not buffer:
            return
        table = pa.Table.from_pandas(pd.DataFrame(buffer, columns=output_cols), preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(
                str(output_path),
                table.schema,
                compression=args.compression,
                use_dictionary=True,
            )
        writer.write_table(table)
        buffer = []

    try:
        iterator = df.itertuples(index=False)
        for source_row, row in tqdm(
            enumerate(iterator),
            total=len(df),
            desc=f"Extract ref_seq {context_length} bp",
        ):
            ref = clean_base(row.ReferenceAlleleVCF)
            alt = clean_base(row.AlternateAlleleVCF)
            if (
                len(ref) != 1
                or len(alt) != 1
                or ref not in "ACGT"
                or alt not in "ACGT"
            ):
                skip_counts["bad_allele"] += 1
                continue

            chrom = str(row.ChromosomeFASTA)
            if chrom not in genome:
                skip_counts["chrom_missing"] += 1
                continue

            try:
                pos = int(row.PositionVCF)
            except (TypeError, ValueError):
                skip_counts["bad_position"] += 1
                continue

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
            if not is_acgt_sequence(seq):
                skip_counts["non_acgt_context"] += 1
                continue

            record = row._asdict()
            y_value = int(record["Y"])
            label_counts[y_value] = label_counts.get(y_value, 0) + 1
            kept_rows += 1
            buffer.append(
                {
                    "SourceRow": source_row,
                    **record,
                    "ContextStart1Based": start_1based,
                    "ContextEnd1Based": end_1based,
                    "ContextLength": context_length,
                    "CenterIndex": flank,
                    "ref_seq": seq,
                }
            )
            if len(buffer) >= args.chunk_size:
                flush()
        flush()
    finally:
        if writer is not None:
            writer.close()

    size_bytes = output_path.stat().st_size if output_path.exists() else 0
    print("Kept rows:", f"{kept_rows:,}")
    print("Skipped:", skip_counts)
    print("Label counts:", label_counts)
    print(
        "Saved:",
        output_path,
        f"{size_bytes / 1_000_000_000:.3f} GB",
        f"({size_bytes / 1024**3:.3f} GiB)",
    )


if __name__ == "__main__":
    main()
