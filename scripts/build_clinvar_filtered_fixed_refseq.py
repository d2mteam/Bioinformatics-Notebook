#!/usr/bin/env python3
"""Build filtered ClinVar SNV parquet with corrected GRCh38 RefSeq accessions."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from pyfaidx import Fasta
from tqdm.auto import tqdm


CHROMOSOME_TO_REFSEQ = {
    "1": "NC_000001.11",
    "2": "NC_000002.12",
    "3": "NC_000003.12",
    "4": "NC_000004.12",
    "5": "NC_000005.10",
    "6": "NC_000006.12",
    "7": "NC_000007.14",
    "8": "NC_000008.11",
    "9": "NC_000009.12",
    "10": "NC_000010.11",
    "11": "NC_000011.10",
    "12": "NC_000012.12",
    "13": "NC_000013.11",
    "14": "NC_000014.9",
    "15": "NC_000015.10",
    "16": "NC_000016.10",
    "17": "NC_000017.11",
    "18": "NC_000018.10",
    "19": "NC_000019.10",
    "20": "NC_000020.11",
    "21": "NC_000021.9",
    "22": "NC_000022.11",
    "X": "NC_000023.11",
    "Y": "NC_000024.10",
    "MT": "NC_012920.1",
    "M": "NC_012920.1",
}

POSITIVE_LABELS = {"Pathogenic", "Likely pathogenic"}
NEGATIVE_LABELS = {"Benign", "Likely benign"}
TRUSTED_REVIEW_PATTERN = "criteria provided|reviewed by expert panel|practice guideline"
LOW_CONFIDENCE_REVIEW = "no assertion criteria provided"


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(
        description="Filter ClinVar variant_summary.txt with fixed GRCh38 accession mapping."
    )
    parser.add_argument("--project-dir", type=Path, default=root)
    parser.add_argument(
        "--variant-summary",
        type=Path,
        default=root / "Data" / "variant_summary.txt",
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
        default=root / "processed" / "clinvar_filtered_step8_fixed_refseq.parquet",
    )
    parser.add_argument("--chunksize", type=int, default=500_000)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def split_clinical_significance(value: object) -> list[str]:
    if pd.isna(value):
        return []
    return [part.strip() for part in str(value).split("/") if part.strip()]


def map_clean_label(value: object) -> int | None:
    labels = split_clinical_significance(value)
    label_set = set(labels)
    if not labels:
        return None
    if label_set <= POSITIVE_LABELS:
        return 1
    if label_set <= NEGATIVE_LABELS:
        return 0
    return None


def normalize_chromosome(chromosome: object) -> str:
    chrom = str(chromosome).strip()
    if chrom.lower().startswith("chr"):
        chrom = chrom[3:]
    if chrom in {"M", "MT", "m", "mt"}:
        return "MT"
    return chrom.upper()


def clean_base(value: object) -> str:
    return str(value).strip().upper()


def choose_chromosome_fasta(row: pd.Series, fasta_names: set[str]) -> str:
    chrom = normalize_chromosome(row["Chromosome"])
    accession = str(row.get("ChromosomeAccession", "")).strip()
    candidates = [
        accession,
        str(row["Chromosome"]).strip(),
        chrom,
        f"chr{chrom}",
        CHROMOSOME_TO_REFSEQ.get(chrom, ""),
    ]
    for candidate in candidates:
        if candidate and candidate in fasta_names:
            return candidate
    return ""


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.force:
        raise FileExistsError(f"Output exists. Use --force to overwrite: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print("variant_summary:", args.variant_summary)
    print("FASTA:", args.fasta)
    print("output:", args.output)

    genome = Fasta(str(args.fasta), as_raw=True, sequence_always_upper=True)
    fasta_names = set(genome.keys())
    missing_primary = {
        chrom: accession
        for chrom, accession in CHROMOSOME_TO_REFSEQ.items()
        if accession not in fasta_names
    }
    if missing_primary:
        raise RuntimeError(f"Missing primary RefSeq accessions in FASTA: {missing_primary}")

    usecols = [
        "Type",
        "GeneSymbol",
        "ClinicalSignificance",
        "Assembly",
        "ChromosomeAccession",
        "Chromosome",
        "ReviewStatus",
        "PositionVCF",
        "ReferenceAlleleVCF",
        "AlternateAlleleVCF",
    ]
    counters: Counter[str] = Counter()
    kept_chunks: list[pd.DataFrame] = []
    reader = pd.read_csv(
        args.variant_summary,
        sep="\t",
        usecols=usecols,
        chunksize=args.chunksize,
        low_memory=False,
    )

    for chunk in tqdm(reader, desc="Filter variant_summary chunks"):
        counters["raw_rows"] += len(chunk)
        chunk = chunk.copy()
        chunk["Y"] = chunk["ClinicalSignificance"].map(map_clean_label)
        mask = chunk["Y"].notna()
        counters["labeled_rows"] += int(mask.sum())

        mask &= chunk["Assembly"].eq("GRCh38")
        counters["grch38_labeled_rows"] += int(mask.sum())

        mask &= chunk["Type"].eq("single nucleotide variant")
        counters["snv_rows"] += int(mask.sum())

        review_status = chunk["ReviewStatus"].fillna("").astype(str)
        trusted_review = review_status.str.contains(
            TRUSTED_REVIEW_PATTERN, case=False, regex=True
        )
        low_confidence_review = review_status.str.fullmatch(
            LOW_CONFIDENCE_REVIEW, case=False
        )
        mask &= trusted_review & ~low_confidence_review
        counters["trusted_rows"] += int(mask.sum())

        ref = chunk["ReferenceAlleleVCF"].map(clean_base)
        alt = chunk["AlternateAlleleVCF"].map(clean_base)
        position = pd.to_numeric(chunk["PositionVCF"], errors="coerce")
        mask &= ref.str.fullmatch("[ACGT]") & alt.str.fullmatch("[ACGT]")
        mask &= position.notna()
        counters["basic_allele_position_rows"] += int(mask.sum())

        out = chunk.loc[mask].copy()
        if out.empty:
            continue
        out["ReferenceAlleleVCF"] = ref.loc[mask].to_numpy()
        out["AlternateAlleleVCF"] = alt.loc[mask].to_numpy()
        out["PositionVCF"] = position.loc[mask].astype(np.int64).to_numpy()
        out["Y"] = out["Y"].astype(np.int8)
        out["Chromosome"] = out["Chromosome"].map(normalize_chromosome)
        out["ChromosomeFASTA"] = out.apply(
            choose_chromosome_fasta, axis=1, fasta_names=fasta_names
        )
        out = out[out["ChromosomeFASTA"].isin(fasta_names)].copy()
        counters["chromosome_fasta_rows"] += len(out)
        if out.empty:
            continue

        reference_bases = []
        matches = []
        for row in out.itertuples(index=False):
            base = str(genome[str(row.ChromosomeFASTA)][int(row.PositionVCF) - 1 : int(row.PositionVCF)])
            base = base.upper()
            reference_bases.append(base)
            matches.append(base == row.ReferenceAlleleVCF)
        out["ReferenceBaseGRCh38"] = reference_bases
        out["ReferenceAlleleMatchesGRCh38"] = matches
        out = out[out["ReferenceAlleleMatchesGRCh38"]].copy()
        counters["ref_match_rows"] += len(out)
        if out.empty:
            continue

        ordered_cols = [
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
        kept_chunks.append(out[ordered_cols])

    if not kept_chunks:
        raise RuntimeError("No rows passed filtering.")

    filtered = pd.concat(kept_chunks, ignore_index=True)
    filtered.to_parquet(args.output, index=False)
    print("Counters:", dict(counters))
    print("Saved:", args.output)
    print("Shape:", filtered.shape)
    print("Label counts:", dict(zip(*np.unique(filtered["Y"], return_counts=True))))
    print("Chromosome counts:")
    print(filtered["Chromosome"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
