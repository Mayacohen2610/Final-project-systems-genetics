import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# --- helper functions (same as before) ---
def fit_linear(x, y):
    x = x.reshape(-1, 1)
    model = LinearRegression().fit(x, y)
    y_pred = model.predict(x)
    residuals = y - y_pred
    sigma2 = np.var(residuals, ddof=1)
    n = len(y)
    loglik = -0.5 * n * (np.log(2 * np.pi * sigma2) + 1)
    return loglik, model.coef_[0], model.intercept_, sigma2

def causality_test(L, R, C):
    ll_R_L, _, _, _ = fit_linear(L, R)
    ll_C_R, _, _, _ = fit_linear(R, C)
    ll_M1 = ll_R_L + ll_C_R

    ll_C_L, _, _, _ = fit_linear(L, C)
    ll_R_C, _, _, _ = fit_linear(C, R)
    ll_M2 = ll_C_L + ll_R_C

    ll_R_L2, _, _, _ = fit_linear(L, R)
    ll_C_L2, _, _, _ = fit_linear(L, C)
    ll_M3 = ll_R_L2 + ll_C_L2

    return {"M1": ll_M1, "M2": ll_M2, "M3": ll_M3}

def permutation_test(L, R, C, n_perm=1000):
    observed = causality_test(L, R, C)
    diff12_obs = observed["M1"] - observed["M2"]
    diff13_obs = observed["M1"] - observed["M3"]

    diffs12, diffs13 = [], []
    for _ in range(n_perm):
        L_perm = np.random.permutation(L)
        res = causality_test(L_perm, R, C)
        diffs12.append(res["M1"] - res["M2"])
        diffs13.append(res["M1"] - res["M3"])

    pval12 = np.mean(np.array(diffs12) >= diff12_obs)
    pval13 = np.mean(np.array(diffs13) >= diff13_obs)

    return {
        "observed": observed,
        "M1_vs_M2": {"diff": diff12_obs, "pval": pval12},
        "M1_vs_M3": {"diff": diff13_obs, "pval": pval13}
    }

# --- batch processing function ---
def analyze_folder(base, folder="triplet_vectors", n_perm=1000, savefile="causality_summary.csv"):
    rows = []
    for fname in os.listdir(base / folder):
        if fname.endswith(".csv"):
            path = os.path.join(base / folder, fname)
            df = pd.read_csv(path)
            
            L = df["L"].values
            R = df["R"].values
            C = df["C"].values
            snp_id = df["snp_id"].iloc[0]
            gene = df["gene"].iloc[0]
            tissue = df["tissue"].iloc[0]

            res = permutation_test(L, R, C, n_perm=n_perm)
            obs = res["observed"]
            best_model = max(obs, key=obs.get)

            row = {
                "File": fname,
                "SNP": snp_id,
                "Gene": gene,
                "Tissue": tissue,
                "LL_M1": obs["M1"],
                "LL_M2": obs["M2"],
                "LL_M3": obs["M3"],
                "M1_vs_M2_diff": res["M1_vs_M2"]["diff"],
                "M1_vs_M2_pval": res["M1_vs_M2"]["pval"],
                "M1_vs_M3_diff": res["M1_vs_M3"]["diff"],
                "M1_vs_M3_pval": res["M1_vs_M3"]["pval"],
                "Best_Model": best_model
            }
            rows.append(row)

    df_out = pd.DataFrame(rows)
    df_out.to_csv(base / savefile, index=False)
    print(f"✅ Summary saved to {base / savefile}")
    return df_out
