# merge_eqtl_and_qtl.py
# Step 1: combine eQTL best-per-gene (liver + hypothalamus)
# Step 2: merge with QTL (HW2) by SNP ID

import pandas as pd
from pathlib import Path

# ============== CONFIG ==============
BASE = Path("data")  

# eQTL best-per-gene files (HW3)
EQTL_LIVER_CANDIDATES = [
    BASE / "eqtl_best_per_gene_liver.csv",
]
EQTL_HYPO_CANDIDATES = [
    BASE / "eqtl_best_per_gene_hypothalamus.csv",
]

# QTL results (HW2)
QTL_FILE = BASE / "snp_regression_results.csv" 

# Outputs
OUT_DIR = BASE / "final-merge"
OUT_EQTL_COMBINED = OUT_DIR / "eqtl_best_per_gene_combined.csv"
OUT_QTL_EQTL_MERGED = OUT_DIR / "qtl_eqtl_merged.csv"
# ===================================


def load_first_existing(paths):
    """Return the first existing CSV as DataFrame and its Path."""
    for p in paths:
        if Path(p).exists():
            return pd.read_csv(p), Path(p)
    raise FileNotFoundError(f"No file found in: {paths}")


def normalize_eqtl_columns(df: pd.DataFrame, tissue: str) -> pd.DataFrame:
    """
    Normalize eQTL columns to a consistent schema:
    snp_id, gene, p_value, q_value, cis_trans, tissue
    and keep useful pass-through columns if present (chromosome, position).
    """
    d = df.copy()

    # SNP ID can be under 'Locus' or 'snp_id'
    if "Locus" in d.columns:
        d["snp_id"] = d["Locus"].astype(str)
    elif "snp_id" in d.columns:
        d["snp_id"] = d["snp_id"].astype(str)
    else:
        raise KeyError("eQTL file must contain 'Locus' or 'snp_id' column.")

    # Gene symbol can be under 'Gene' or 'gene_symbol'
    if "Gene" in d.columns:
        d["gene"] = d["Gene"].astype(str)
    elif "gene_symbol" in d.columns:
        d["gene"] = d["gene_symbol"].astype(str)
    else:
        # allow missing gene but strongly recommended
        d["gene"] = pd.NA

    # Optional stats
    d["p_value"] = d["p_value"] if "p_value" in d.columns else pd.NA
    d["q_value"] = d["q_value"] if "q_value" in d.columns else pd.NA

    # cis/trans label if exists
    d["cis_trans"] = d["cis_trans"] if "cis_trans" in d.columns else pd.NA

    # pass-through genomic columns if present
    for src, dst in [("Chromosome", "eqtl_chr"), ("Position", "eqtl_pos")]:
        d[dst] = d[src] if src in d.columns else pd.NA

    d["tissue"] = tissue
    # Keep only a clean set plus any extra columns you want to preserve
    keep_cols = ["snp_id", "gene", "p_value", "q_value", "cis_trans", "tissue", "eqtl_chr", "eqtl_pos"]
    # Add any existing columns that start with 'n_' or other useful metadata if you wish
    return d[[c for c in keep_cols if c in d.columns]]


def load_qtl(qtl_path: Path) -> pd.DataFrame:
    """
    Normalize QTL columns to: snp_id, qtl_chr, qtl_pos, minus_log10_p
    If your QTL file has other names, adapt them here.
    """
    df = pd.read_csv(qtl_path).copy()

    # SNP column commonly 'SNP' or 'snp_id'
    if "SNP" in df.columns:
        df["snp_id"] = df["SNP"].astype(str)
    elif "snp_id" in df.columns:
        df["snp_id"] = df["snp_id"].astype(str)
    else:
        raise KeyError("QTL file must contain 'SNP' or 'snp_id' column.")

    # Chromosome / Position
    if "Chromosome" in df.columns:
        df["qtl_chr"] = df["Chromosome"]
    elif "chr" in df.columns:
        df["qtl_chr"] = df["chr"]
    else:
        df["qtl_chr"] = pd.NA

    if "Position" in df.columns:
        df["qtl_pos"] = df["Position"]
    elif "pos" in df.columns:
        df["qtl_pos"] = df["pos"]
    else:
        df["qtl_pos"] = pd.NA

    # Significance measure
    if "-log10(P)" in df.columns:
        df["minus_log10_p"] = df["-log10(P)"]
    elif "minus_log10_p" in df.columns:
        pass  # already present
    else:
        # If only 'p_value' exists, convert to -log10(p)
        if "p_value" in df.columns:
            import numpy as np
            df["minus_log10_p"] = -np.log10(df["p_value"].astype(float).clip(lower=1e-300))
        else:
            df["minus_log10_p"] = pd.NA

    return df[["snp_id", "qtl_chr", "qtl_pos", "minus_log10_p"]]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Step 1: combine eQTL (liver + hypothalamus) ---
    liver_df_raw, liver_path = load_first_existing(EQTL_LIVER_CANDIDATES)
    hypo_df_raw, hypo_path = load_first_existing(EQTL_HYPO_CANDIDATES)

    liver_df = normalize_eqtl_columns(liver_df_raw, "Liver")
    hypo_df  = normalize_eqtl_columns(hypo_df_raw,  "Hypothalamus")

    eqtl_combined = pd.concat([liver_df, hypo_df], ignore_index=True).drop_duplicates(subset=["snp_id", "gene", "tissue"])
    eqtl_combined.to_csv(OUT_EQTL_COMBINED, index=False)
    print(f"[OK] eQTL combined saved: {OUT_EQTL_COMBINED}  (rows={len(eqtl_combined)})")
    print(f"     sources: {liver_path.name}, {hypo_path.name}")

    # --- Step 2: merge with QTL by SNP ID ---
    qtl_df = load_qtl(QTL_FILE)
    qtl_eqtl_merged = eqtl_combined.merge(qtl_df, on="snp_id", how="inner")
    qtl_eqtl_merged.to_csv(OUT_QTL_EQTL_MERGED, index=False)
    print(f"[OK] QTL–eQTL merged saved: {OUT_QTL_EQTL_MERGED}  (rows={len(qtl_eqtl_merged)})")

    # quick sanity summaries
    by_tissue = qtl_eqtl_merged.groupby("tissue")["snp_id"].nunique().reset_index(name="overlapping_snp_ids")
    print("\nOverlap (unique SNP IDs) by tissue:")
    print(by_tissue.to_string(index=False))

    top_example = qtl_eqtl_merged.head(5)
    print("\nPreview:")
    print(top_example.to_string(index=False))


if __name__ == "__main__":
    main()
