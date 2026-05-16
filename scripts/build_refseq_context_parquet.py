#!/usr/bin/env python3
"""Build a compact ref-sequence context Parquet dataset for CNN training."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pyfaidx import Fasta
from tqdm.auto import tqdm


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(
        description=(
            "Create a compact ClinVar context dataset with ref_seq strings. "
            "ALT one-hot can be generated on the fly during CNN training."
        )
    )
    parser.add_argument("--project-dir", type=Path, default=root)
    parser.add_argument("--flank", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=8192)
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
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--compression", default="zstd")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def clean_base(value: object) -> str:
    return str(value).strip().upper()


def flush_batch(
    writer: pq.ParquetWriter | None,
    rows: list[dict[str, object]],
    output_path: Path,
    compression: str,
) -> pq.ParquetWriter:
    table = pa.Table.from_pylist(rows)
    if writer is None:
        writer = pq.ParquetWriter(output_path, table.schema, compression=compression)
    writer.write_table(table)
    rows.clear()
    return writer


def main() -> None:
    args = parse_args()
    project_dir = args.project_dir.resolve()
    processed_dir = project_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    flank = args.flank
    context_length = flank * 2 + 1
    output_path = args.output
    if output_path is None:
        output_path = processed_dir / f"clinvar_refseq_context_{context_length}.parquet"
    output_path = output_path.resolve()

    if output_path.exists() and not args.force:
        raise FileExistsError(f"Output exists. Use --force to overwrite: {output_path}")
    if output_path.exists():
        output_path.unlink()

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
    df = pd.read_parquet(args.input, columns=required_cols)
    genome = Fasta(str(args.fasta), as_raw=True, sequence_always_upper=True)

    print("Project:", project_dir)
    print("Input:", args.input)
    print("FASTA:", args.fasta)
    print("Output:", output_path)
    print("Context length:", context_length)
    print("Input rows:", f"{len(df):,}")

    rows: list[dict[str, object]] = []
    writer: pq.ParquetWriter | None = None
    kept = 0
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
        if any(base not in "ACGT" for base in seq):
            skip_counts["non_acgt_context"] += 1
            continue

        rows.append(
            {
                "Type": row.Type,
                "GeneSymbol": row.GeneSymbol,
                "ClinicalSignificance": row.ClinicalSignificance,
                "Assembly": row.Assembly,
                "Chromosome": row.Chromosome,
                "ReviewStatus": row.ReviewStatus,
                "PositionVCF": int(row.PositionVCF),
                "ReferenceAlleleVCF": ref,
                "AlternateAlleleVCF": alt,
                "Y": int(row.Y),
                "ChromosomeFASTA": chrom,
                "ContextStart1Based": int(start_1based),
                "ContextEnd1Based": int(end_1based),
                "ref_seq": seq,
            }
        )
        kept += 1

        if len(rows) >= args.batch_size:
            writer = flush_batch(writer, rows, output_path, args.compression)

    if rows:
        writer = flush_batch(writer, rows, output_path, args.compression)
    if writer is not None:
        writer.close()

    print("Kept rows:", f"{kept:,}")
    print("Skipped:", skip_counts)
    print("Saved:", output_path)
    print("Output size MB:", f"{output_path.stat().st_size / 1e6:.2f}")


if __name__ == "__main__":
    main()
