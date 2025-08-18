import pandas as pd
from pathlib import Path

def select_top10_triplets_for_single_pheno(
    qtl_eqtl_merged_path: str = "qtl_eqtl_merged.csv",
    output_path: str = "top10_triplets_787.csv",
    prefer_cis: bool = True,
    prefer_same_chr: bool = True,
    n_triplets: int = 10,          # how many to RETURN at the end (usually 10)
    n_candidates: int = 15,        # how many to consider BEFORE SNP de-dup (your request)
    min_logp: float | None = None  # optional: e.g., 1.3 (~p<=0.05)
):
    """
    Build ranked triplets and return top-10 with unique SNPs.
    Steps:
      1) Rank by cis -> same_chr -> -log10P desc -> distance asc
      2) Take top n_candidates (e.g., 15)
      3) Drop duplicates by SNP (keep first within those candidates)
      4) Take top n_triplets (e.g., 10) and save
    Assumes the merged file is for ONE phenotype (e.g., 787).
    """

    # 1) Load & validate
    df = pd.read_csv(qtl_eqtl_merged_path)
    required_cols = [
        "snp_id", "gene", "cis_trans", "tissue",
        "eqtl_chr", "eqtl_pos", "qtl_chr", "qtl_pos",
        "minus_log10_p"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {qtl_eqtl_merged_path}: {missing}")

    # 2) Cleaning & features
    for c in ["eqtl_pos", "qtl_pos", "minus_log10_p"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # distance (absolute)
    df["distance_bp"] = (df["eqtl_pos"] - df["qtl_pos"]).abs()

    # same chromosome (numeric comparison to avoid '11' vs '11.0' issues)
    eqtl_chr_num = pd.to_numeric(df["eqtl_chr"], errors="coerce")
    qtl_chr_num  = pd.to_numeric(df["qtl_chr"],  errors="coerce")
    df["same_chr"] = (eqtl_chr_num == qtl_chr_num)

    # cis flag
    df["cis_flag"] = df["cis_trans"].astype(str).str.lower().str.contains("cis")

    # optional threshold on QTL strength
    if min_logp is not None:
        df = df[df["minus_log10_p"] >= float(min_logp)].copy()

    # core numeric fields must exist
    df = df.dropna(subset=["minus_log10_p", "eqtl_pos", "qtl_pos"]).copy()

    # Avoid exact duplicates SNP–gene (optional)
    df = df.drop_duplicates(subset=["snp_id", "gene"]).copy()

    # 3) Ranking
    sort_cols, ascending = [], []
    if prefer_cis:
        sort_cols.append("cis_flag");   ascending.append(False)  # cis first
    if prefer_same_chr:
        sort_cols.append("same_chr");   ascending.append(False)  # same chr first
    sort_cols += ["minus_log10_p", "distance_bp"]
    ascending += [False, True]  # higher -log10P first, then shorter distance

    df_sorted = df.sort_values(sort_cols, ascending=ascending, kind="mergesort").reset_index(drop=True)

    # 4) Take top n_candidates, then de-dup by SNP, then take top n_triplets
    candidates = df_sorted.head(n_candidates).copy()
    candidates_unique_snp = candidates.drop_duplicates(subset=["snp_id"], keep="first")
    topN = candidates_unique_snp.head(n_triplets).copy()

    # 5) Save only the useful columns
    out_cols = [
        "snp_id", "gene", "tissue", "cis_trans",
        "eqtl_chr", "eqtl_pos", "qtl_chr", "qtl_pos",
        "same_chr", "distance_bp", "minus_log10_p", "cis_flag"
    ]
    topN = topN[out_cols]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    topN.to_csv(output_path, index=False)
    print(f"[OK] Wrote {len(topN)} triplets to: {output_path}")
    return topN

# ---------- example run ----------
if __name__ == "__main__":
    candidate_triplets = select_top10_triplets_for_single_pheno(
        qtl_eqtl_merged_path="data/final-merge/final-mergeqtl_eqtl_with_eqtl_coords.csv",
        output_path="data/top10_triplets_787.csv",
        prefer_cis=True,
        prefer_same_chr=True,
        n_triplets=10,      # final 10
        n_candidates=15,    # first 15, then unique SNPs, then 10
        min_logp=None       # or 1.3 for ~p<=0.05
    )
