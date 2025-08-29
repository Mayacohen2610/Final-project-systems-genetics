import pandas as pd
import numpy as np
from pathlib import Path
import re
from pathlib import Path
# ------------------------------
# Helpers
# ------------------------------

def detect_strain_columns(df: pd.DataFrame) -> list:
    """
    Detect strain columns (BXD*) by regex. Falls back to columns that start with 'BXD'.
    """
    patt = re.compile(r"^BXD\d+", re.IGNORECASE)
    strain_cols = [c for c in df.columns if patt.match(str(c))]
    if not strain_cols:
        strain_cols = [c for c in df.columns if str(c).upper().startswith("BXD")]
    return strain_cols

def map_genotypes_to_numeric(s: pd.Series) -> pd.Series:
    """
    Map genotype codes to numeric: B->0, D->1, H->0.5, U/others->NaN.
    If already numeric-like, keep as-is.
    """
    # Try numeric first
    s_num = pd.to_numeric(s, errors="coerce")
    if s_num.notna().sum() >= s.notna().sum() * 0.8:
        return s_num

    mapper = {"B": 0.0, "D": 1.0, "H": 0.5, "U": np.nan}
    return s.astype(str).str.upper().map(mapper)

def find_phenotype_row(df_pheno: pd.DataFrame, pheno_id=787) -> pd.Series:
    """
    Find the phenotype row for 'pheno_id' by common ID columns.
    Falls back to exact match in any numeric-looking column if needed.
    """
    candidates = ["ID_FOR_CHECK", "ID", "Phenotype_ID", "phenotype_id"]
    for col in candidates:
        if col in df_pheno.columns:
            hit = df_pheno[df_pheno[col].astype(str) == str(pheno_id)]
            if len(hit) == 1:
                return hit.squeeze()

    # Fallback: try any column that equals 787 exactly
    for col in df_pheno.columns:
        if pd.to_numeric(df_pheno[col], errors="coerce").eq(pheno_id).any():
            hit = df_pheno[pd.to_numeric(df_pheno[col], errors="coerce").eq(pheno_id)]
            if len(hit) == 1:
                return hit.squeeze()

    raise ValueError(f"Phenotype row with ID {pheno_id} not found in phenotypes.csv")

def get_expression_vector(expr_df: pd.DataFrame, gene_symbol: str) -> pd.Series:
    """
    Extract expression vector for a given gene by 'Gene Symbol' column.
    """
    gene_col = None
    for cand in ["Gene Symbol", "gene", "Gene", "Symbol"]:
        if cand in expr_df.columns:
            gene_col = cand
            break
    if gene_col is None:
        raise ValueError("Could not find a gene symbol column in expression matrix")

    hit = expr_df[expr_df[gene_col].astype(str) == str(gene_symbol)]
    if hit.empty:
        raise KeyError(f"Gene '{gene_symbol}' not found in expression matrix")
    strains = detect_strain_columns(expr_df)
    return hit.iloc[0][strains].astype(float)

def get_genotype_vector(geno_df: pd.DataFrame, snp_id: str) -> pd.Series:
    """
    Extract genotype vector for a given SNP id. Row key can be 'snp_id' or 'SNP' or first column.
    """
    snp_col = None
    for cand in ["snp_id", "SNP", "snp", "marker", "Marker"]:
        if cand in geno_df.columns:
            snp_col = cand
            break
    if snp_col is None:
        # assume first column is SNP id
        snp_col = geno_df.columns[0]

    hit = geno_df[geno_df[snp_col].astype(str) == str(snp_id)]
    if hit.empty:
        raise KeyError(f"SNP '{snp_id}' not found in genotype matrix")
    strains = detect_strain_columns(geno_df)
    return map_genotypes_to_numeric(hit.iloc[0][strains])

def align_LRC(L: pd.Series, R: pd.Series, C: pd.Series) -> pd.DataFrame:
    """
    Align L,R,C on common strains and drop rows with any NaN.
    """
    df = pd.DataFrame({"L": L, "R": R, "C": C})
    df.index.name = "strain"
    df = df.dropna(axis=0, how="any").copy()
    return df

# ------------------------------
# Main extraction
# ------------------------------

def extract_triplet_vectors(
    top_triplets_csv: str = "top10_triplets_787.csv",
    expr_liver_csv: str = "filtered_expression_matrix_liver.csv",
    expr_hypo_csv: str = "filtered_expression_matrix_hypothalamus.csv",
    geno_liver_csv: str = "filtered_genotype_matrix_liver.csv",
    geno_hypo_csv: str = "filtered_genotype_matrix_hypothalamus.csv",
    phenotypes_csv: str = "phenotypes.csv",
    pheno_id: int = 787,
    out_dir: str = "triplet_vectors",
    summary_csv: str = "triplets_vector_summary.csv",
):
    """
    For each triplet (snp_id, gene, tissue) in 'top_triplets_csv':
      - L: genotype vector from the matching tissue-specific genotype matrix
      - R: expression vector from the matching tissue-specific expression matrix
      - C: phenotype vector from phenotypes.csv for pheno_id
    Align by strains (BXD...), drop missing, save per-triplet CSVs + summary.
    """

    # Load inputs
    triplets = pd.read_csv(top_triplets_csv)
    expr_liver = pd.read_csv(expr_liver_csv)
    expr_hypo  = pd.read_csv(expr_hypo_csv)
    geno_liver = pd.read_csv(geno_liver_csv)
    geno_hypo  = pd.read_csv(geno_hypo_csv)
    phenos     = pd.read_csv(phenotypes_csv)

    # Prepare phenotype vector (C)
    pheno_row = find_phenotype_row(phenos, pheno_id=pheno_id)
    pheno_strains = detect_strain_columns(phenos)
    C_full = pheno_row[pheno_strains].astype(float)
    C_full.index = [str(i) for i in C_full.index]  # normalize index to string

    # Output setup
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    summary = []

    # Iterate triplets
    for idx, row in triplets.iterrows():
        snp_id = str(row["snp_id"])
        gene   = str(row["gene"])
        tissue = str(row["tissue"]).strip().lower()

        # Pick tissue-specific matrices
        if tissue == "liver":
            expr_df = expr_liver
            geno_df = geno_liver
        elif tissue == "hypothalamus":
            expr_df = expr_hypo
            geno_df = geno_hypo
        else:
            summary.append({
                "rank": idx+1,
                "snp_id": snp_id,
                "gene": gene,
                "tissue": row["tissue"],
                "n_strains_after_align": 0,
                "status": f"Unsupported tissue: {row['tissue']}"
            })
            continue

        # Extract vectors
        try:
            R = get_expression_vector(expr_df, gene)
            R.index = [str(i) for i in R.index]
        except Exception as e:
            summary.append({
                "rank": idx+1, "snp_id": snp_id, "gene": gene, "tissue": row["tissue"],
                "n_strains_after_align": 0, "status": f"Expression error: {e}"
            })
            continue

        try:
            L = get_genotype_vector(geno_df, snp_id)
            L.index = [str(i) for i in L.index]
        except Exception as e:
            summary.append({
                "rank": idx+1, "snp_id": snp_id, "gene": gene, "tissue": row["tissue"],
                "n_strains_after_align": 0, "status": f"Genotype error: {e}"
            })
            continue

        # Align and drop NaNs
        try:
            df_aligned = align_LRC(L, R, C_full)
        except Exception as e:
            summary.append({
                "rank": idx+1, "snp_id": snp_id, "gene": gene, "tissue": row["tissue"],
                "n_strains_after_align": 0, "status": f"Align error: {e}"
            })
            continue

        n_strains = len(df_aligned)
        status = "OK" if n_strains > 0 else "Empty after NaN drop"

        # Add id columns
        df_aligned = df_aligned.reset_index()  # strain as a column
        df_aligned["snp_id"] = snp_id
        df_aligned["gene"] = gene
        df_aligned["tissue"] = row["tissue"]

        # Save per-triplet
        safe_snp = re.sub(r"[^A-Za-z0-9_.-]+", "_", snp_id)
        safe_gene = re.sub(r"[^A-Za-z0-9_.-]+", "_", gene)
        out_file = out_path / f"triplet_{idx+1:02d}_{safe_snp}_{safe_gene}_{tissue}.csv"
        df_aligned.to_csv(out_file, index=False)

        summary.append({
            "rank": idx+1,
            "snp_id": snp_id,
            "gene": gene,
            "tissue": row["tissue"],
            "n_strains_after_align": n_strains,
            "status": status,
            "output_file": str(out_file)
        })

    # Save summary
    pd.DataFrame(summary).to_csv(summary_csv, index=False)
    print(f"[OK] Wrote per-triplet vectors to '{out_dir}/' and summary to '{summary_csv}'")


# ---------- example run ----------
BASE = Path("data")  
if __name__ == "__main__":
    extract_triplet_vectors(
        top_triplets_csv= BASE / "top10_triplets_787.csv",
        expr_liver_csv=BASE / "filtered_expression_matrix_liver.csv",
        expr_hypo_csv=BASE / "filtered_expression_matrix_hypothalamus.csv",
        geno_liver_csv=BASE / "filtered_genotype_matrix_liver.csv",
        geno_hypo_csv=BASE / "filtered_genotype_matrix_hypothalamus.csv",
        phenotypes_csv=BASE / "phenotypes.csv",
        pheno_id=787,
        out_dir=BASE / "triplet_vectors",
        summary_csv=BASE / "triplets_vector_summary.csv",
    )
