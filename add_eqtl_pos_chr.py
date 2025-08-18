import pandas as pd
from typing import List

def _norm_chr(x):
    x = str(x).strip()
    return x[3:] if x.lower().startswith("chr") else x

def _prepare_ref(geno: pd.DataFrame, locus_col: str,
                 chr_col: str = "Chr_Build37", pos_col: str = "Build37_position") -> pd.DataFrame:
    """Return a minimal ref table: snp_id, eqtl_chr_new, eqtl_pos_new from a chosen Locus column."""
    if locus_col not in geno.columns:
        raise ValueError(f"'{locus_col}' not found in genotype table.")
    if chr_col not in geno.columns or pos_col not in geno.columns:
        raise ValueError(f"Genotype file missing required columns '{chr_col}' and/or '{pos_col}'.")

    ref = geno[[locus_col, chr_col, pos_col]].copy()
    ref = ref.rename(columns={locus_col: "snp_id", chr_col: "eqtl_chr_new", pos_col: "eqtl_pos_new"})
    ref["snp_id"] = ref["snp_id"].astype(str).str.strip()
    ref["eqtl_chr_new"] = ref["eqtl_chr_new"].map(_norm_chr)
    ref["eqtl_pos_new"] = pd.to_numeric(ref["eqtl_pos_new"], errors="coerce")
    ref = ref.drop_duplicates(subset=["snp_id"])
    return ref

def _merge_fill(eqtl: pd.DataFrame, ref: pd.DataFrame, tag: str) -> pd.DataFrame:
    """
    Merge eqtl with ref on snp_id and fill only missing eqtl_chr/eqtl_pos.
    'tag' is used to distinguish temporary columns for reporting.
    """
    merged = eqtl.merge(ref, on="snp_id", how="left")

    # Ensure target columns exist and correct dtypes
    if "eqtl_chr" not in merged.columns:
        merged["eqtl_chr"] = pd.NA
    if "eqtl_pos" not in merged.columns:
        merged["eqtl_pos"] = pd.NA

    # Count before
    before_chr = merged["eqtl_chr"].notna().sum()
    before_pos = pd.to_numeric(merged["eqtl_pos"], errors="coerce").notna().sum()

    # Fill only missing
    merged["eqtl_chr"] = merged["eqtl_chr"].astype(object)
    merged["eqtl_chr"] = merged["eqtl_chr"].where(merged["eqtl_chr"].notna(), merged["eqtl_chr_new"])
    merged["eqtl_pos"] = pd.to_numeric(merged["eqtl_pos"], errors="coerce")
    merged["eqtl_pos"] = merged["eqtl_pos"].where(merged["eqtl_pos"].notna(), merged["eqtl_pos_new"])

    # Count after and report
    after_chr = merged["eqtl_chr"].notna().sum()
    after_pos = merged["eqtl_pos"].notna().sum()
    print(f"[{tag}] eqtl_chr filled: {before_chr} -> {after_chr} (+{after_chr - before_chr})")
    print(f"[{tag}] eqtl_pos filled: {before_pos} -> {after_pos} (+{after_pos - before_pos})")

    # Drop temp cols
    merged = merged.drop(columns=["eqtl_chr_new", "eqtl_pos_new"], errors="ignore")
    return merged

def add_eqtl_coords_dual(
    eqtl_path: str,
    geno_path: str,
    out_path: str = "qtl_eqtl_with_eqtl_coords.csv",
    chr_col: str = "Chr_Build37",
    pos_col: str = "Build37_position",
    locus_prefix: str = "Locus"
):
    """
    Fill eqtl_chr/eqtl_pos in the merged QTL+eQTL table using genotype coordinates.
    Strategy:
      - Detect all columns starting with `locus_prefix` (e.g., 'Locus', 'Locus.1').
      - Try first Locus column; fill only missing.
      - Try second Locus column for still-missing rows.
    Inputs:
      eqtl_path: CSV with at least 'snp_id' (and optional eqtl_chr/eqtl_pos to be filled).
      geno_path: genotype matrix CSV containing Locus*, Chr_Build37, Build37_position.
    Output:
      out_path: CSV with the same columns as input plus completed eqtl_chr/eqtl_pos.
    """
    eqtl = pd.read_csv(eqtl_path)
    geno = pd.read_csv(geno_path)

    # Normalize snp_id
    if "snp_id" not in eqtl.columns:
        raise ValueError("eqtl table must contain 'snp_id'.")
    eqtl["snp_id"] = eqtl["snp_id"].astype(str).str.strip()

    # Identify Locus-like columns in genotype file
    locus_cols: List[str] = [c for c in geno.columns if c.startswith(locus_prefix)]
    if not locus_cols:
        raise ValueError(f"No '{locus_prefix}*' columns found in genotype file.")
    print(f"Detected Locus columns: {locus_cols}")

    # Pass 1: try the first Locus column
    ref1 = _prepare_ref(geno, locus_cols[0], chr_col=chr_col, pos_col=pos_col)
    merged = _merge_fill(eqtl, ref1, tag=f"match:{locus_cols[0]}")

    # If still missing, and there is a second Locus column, try it
    still_missing = merged["eqtl_chr"].isna() | pd.to_numeric(merged["eqtl_pos"], errors="coerce").isna()
    if len(locus_cols) > 1 and still_missing.any():
        ref2 = _prepare_ref(geno, locus_cols[1], chr_col=chr_col, pos_col=pos_col)
        # Only attempt to fill rows that remain missing to minimize overhead
        eqtl_missing = merged.loc[still_missing].copy()
        eqtl_filled = _merge_fill(eqtl_missing, ref2, tag=f"fallback:{locus_cols[1]}")

        # Combine back the filled subset
        merged.loc[eqtl_filled.index, ["eqtl_chr", "eqtl_pos"]] = eqtl_filled[["eqtl_chr", "eqtl_pos"]].values

    # Final report
    total = len(merged)
    final_chr = merged["eqtl_chr"].notna().sum()
    final_pos = merged["eqtl_pos"].notna().sum()
    print(f"[final] rows: {total} | eqtl_chr present: {final_chr} ({final_chr/total:.1%}) | "
          f"eqtl_pos present: {final_pos} ({final_pos/total:.1%})")

    # Save
    merged.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

# Example usage:
add_eqtl_coords_dual(
    eqtl_path="data/final-merge/qtl_eqtl_merged.csv",
    geno_path="data/filtered_genotype_matrix_hypothalamus.csv",
    out_path="data/final-mergeqtl_eqtl_with_eqtl_coords.csv"
)


# # Sanity checks for eqtl_chr / eqtl_pos completion
# # Run as-is or import and call validate_eqtl_coords(...)

# import pandas as pd
# import numpy as np

# def validate_eqtl_coords(
#     merged_path: str = "data/final-mergeqtl_eqtl_with_eqtl_coords.csv",
#     geno_path: str | None = None,         # e.g., "filtered_genotype_matrix_hypothalamus.csv"
#     locus_cols_prefix: str = "Locus",     # detects ["Locus", "Locus.1", ...] in geno file
#     chr_col: str = "Chr_Build37",
#     pos_col: str = "Build37_position",
#     sample_n: int = 10
# ) -> None:
#     df = pd.read_csv(merged_path)
#     print(f"Loaded merged file: {merged_path} | rows={len(df)} | cols={len(df.columns)}")

#     # --- Basic presence and NA checks ---
#     for c in ["eqtl_chr", "eqtl_pos"]:
#         if c not in df.columns:
#             raise ValueError(f"Missing column '{c}' in merged table.")
#     na_chr = df["eqtl_chr"].isna().sum()
#     na_pos = df["eqtl_pos"].isna().sum()
#     print(f"NA counts -> eqtl_chr: {na_chr}, eqtl_pos: {na_pos}")

#     # --- Dtypes & coercion checks (only prints; does not modify file) ---
#     eqtl_chr_num = pd.to_numeric(df["eqtl_chr"], errors="coerce")
#     eqtl_pos_num = pd.to_numeric(df["eqtl_pos"], errors="coerce")
#     print("Type coercion -> eqtl_chr numeric NAs:", eqtl_chr_num.isna().sum(),
#           "| eqtl_pos numeric NAs:", eqtl_pos_num.isna().sum())

#     # --- Range checks ---
#     pos_min, pos_max = eqtl_pos_num.min(), eqtl_pos_num.max()
#     print(f"Position range -> min: {pos_min}, max: {pos_max}")
#     if pos_min is not None and (pos_min < 0):
#         print("WARNING: Negative positions detected.")
#     if pos_max is not None and (pos_max < 1000):
#         print("WARNING: Max position is very small; check genome build/units.")

#     # --- Chromosome values snapshot ---
#     uniq_chr = pd.Series(eqtl_chr_num.dropna().astype(int)).unique()
#     uniq_chr_sorted = np.sort(uniq_chr) if len(uniq_chr) else uniq_chr
#     print("Unique numeric chromosomes (first 30 shown):", uniq_chr_sorted[:30])

#     # --- Quick sample preview ---
#     print("\nSample rows:")
#     cols_show = [c for c in ["snp_id", "Locus", "Locus.1", "eqtl_chr", "eqtl_pos"] if c in df.columns]
#     print(df[cols_show].sample(min(sample_n, len(df)), random_state=0))

#     # --- Optional: Cross-check against genotype file ---
#     if geno_path:
#         geno = pd.read_csv(geno_path)
#         locus_cols = [c for c in geno.columns if c.startswith(locus_cols_prefix)]
#         if not locus_cols:
#             print(f"\nNo '{locus_cols_prefix}*' columns in geno file; skip cross-check.")
#             return

#         # Build minimal reference from first available Locus column
#         ref = geno[[locus_cols[0], chr_col, pos_col]].copy()
#         ref = ref.rename(columns={locus_cols[0]: "snp_id_ref", chr_col: "chr_ref", pos_col: "pos_ref"})
#         ref["snp_id_ref"] = ref["snp_id_ref"].astype(str).str.strip()

#         # Try match by 'snp_id' first
#         tmp = df[["snp_id", "eqtl_chr", "eqtl_pos"]].copy()
#         tmp["snp_id"] = tmp["snp_id"].astype(str).str.strip()
#         chk = tmp.merge(ref, left_on="snp_id", right_on="snp_id_ref", how="left")

#         # Normalize for comparison
#         def _norm_chr(x):
#             x = str(x).strip()
#             return x[3:] if x.lower().startswith("chr") else x

#         chk["chr_ref"] = chk["chr_ref"].map(_norm_chr)
#         chk["eqtl_chr_num"] = pd.to_numeric(chk["eqtl_chr"], errors="coerce")
#         chk["eqtl_pos_num"] = pd.to_numeric(chk["eqtl_pos"], errors="coerce")
#         chk["chr_ref_num"] = pd.to_numeric(chk["chr_ref"], errors="coerce")
#         chk["pos_ref_num"] = pd.to_numeric(chk["pos_ref"], errors="coerce")

#         # Mismatch flags
#         chr_mismatch = (chk["eqtl_chr_num"].notna() & chk["chr_ref_num"].notna()
#                         & (chk["eqtl_chr_num"] != chk["chr_ref_num"]))
#         pos_mismatch = (chk["eqtl_pos_num"].notna() & chk["pos_ref_num"].notna()
#                         & (chk["eqtl_pos_num"] != chk["pos_ref_num"]))

#         n_chr_mis = int(chr_mismatch.sum())
#         n_pos_mis = int(pos_mismatch.sum())

#         print(f"\nCross-check vs geno ({geno_path}):")
#         print(f"Chromosome mismatches: {n_chr_mis}")
#         print(f"Position mismatches:   {n_pos_mis}")

#         if n_chr_mis or n_pos_mis:
#             print("\nExamples of mismatches:")
#             bad = chk[chr_mismatch | pos_mismatch].head(10)
#             print(bad[["snp_id", "eqtl_chr", "eqtl_pos", "chr_ref", "pos_ref"]])

# # --- Run (edit paths if needed) ---
# validate_eqtl_coords(
#     merged_path="data/final-mergeqtl_eqtl_with_eqtl_coords.csv",
#     geno_path="data/filtered_genotype_matrix_hypothalamus.csv"  # or liver file
# )
