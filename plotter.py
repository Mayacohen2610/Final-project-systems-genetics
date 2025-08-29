import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def plot_likelihoods(df_summary, outdir="plots"):
    os.makedirs(outdir, exist_ok=True)
    for i, row in df_summary.iterrows():
        triplet_name = f"{row['Gene']} ({row['SNP']}, {row['Tissue']})"
        values = [row["LL_M1"], row["LL_M2"], row["LL_M3"]]
        labels = ["M1", "M2", "M3"]

        plt.figure(figsize=(5,4))
        plt.bar(labels, values, color=["skyblue","lightcoral","lightgreen"])
        plt.title(f"Triplet {i+1}: {triplet_name}\nBest: {row['Best_Model']}")
        plt.ylabel("Log-likelihood")
        plt.tight_layout()
        fname = os.path.join(outdir, f"triplet_{i+1}_likelihoods.png")
        plt.savefig(fname, dpi=300)
        plt.show()
        plt.close()

def plot_pvalue_heatmap(df_summary, outdir="plots"):
    os.makedirs(outdir, exist_ok=True)
    pvals = df_summary[["M1_vs_M2_pval","M1_vs_M3_pval"]]
    pvals.index = df_summary["Gene"] + "_" + df_summary["Tissue"]

    plt.figure(figsize=(8,6))
    sns.heatmap(pvals, annot=True, cmap="viridis", cbar_kws={"label": "p-value"}, fmt=".3f")
    plt.title("Permutation test p-values")
    plt.ylabel("Triplet (Gene_Tissue)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "pvalue_heatmap.png"), dpi=300)
    plt.show()
    plt.close()

def plot_model_preference(df_summary, outdir="plots"):
    os.makedirs(outdir, exist_ok=True)
    counts = df_summary["Best_Model"].value_counts()
    plt.figure(figsize=(5,5))
    plt.pie(counts, labels=counts.index, autopct='%1.1f%%', colors=["skyblue","lightcoral","lightgreen"])
    plt.title("Best Model Distribution Across Triplets")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "best_model_distribution.png"), dpi=300)
    plt.show()
    plt.close()

def plot_volcano(df_summary, outdir="plots"):
    os.makedirs(outdir, exist_ok=True)
    plt.figure(figsize=(7,6))
    plt.scatter(df_summary["M1_vs_M2_diff"], -np.log10(df_summary["M1_vs_M2_pval"]+1e-10),
                color="blue", label="M1 vs M2")
    plt.scatter(df_summary["M1_vs_M3_diff"], -np.log10(df_summary["M1_vs_M3_pval"]+1e-10),
                color="green", label="M1 vs M3")
    plt.axhline(-np.log10(0.05), color="red", linestyle="--", label="p=0.05")
    plt.xlabel("Log-likelihood difference")
    plt.ylabel("-log10(p-value)")
    plt.title("Permutation Test Results")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "volcano_plot.png"), dpi=300)
    plt.show()
    plt.close()
