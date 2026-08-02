from datetime import datetime
import os
import argparse
import pandas as pd
import numpy as np
import glob
from scipy.stats import mannwhitneyu, fisher_exact, chi2_contingency
from statsmodels.stats.multitest import fdrcorrection

def get_args():
    parser = argparse.ArgumentParser(description='Compute metadata statistics per cohort or aggregated across all cohorts')
    parser.add_argument('--aggregate', action='store_true', default=False,
                        help='Aggregate all cohorts before computing statistics (default: compute per cohort)')
    parser.add_argument("--mode", dest="mode", type=str)
    parser.add_argument("--MF", action='store_true', default=False,help='Compute stats for MF')
    return parser.parse_args()


def chi2_monte_carlo(table, n_sim=10000, random_state=0):
    """
    Chi-square Monte Carlo with FIXED marginals (matches R's simulate.p.value=TRUE).
    Uses repeated Fisher-Yates style shuffling to preserve row/col totals.
    """
    table = np.asarray(table, dtype=int)
    chi2_obs, _, _, _ = chi2_contingency(table, correction=False)

    row_sums = table.sum(axis=1)
    col_sums = table.sum(axis=0)
    n = table.sum()
    n_rows, n_cols = table.shape

    # Build the flat pool of observations that gives exactly the right marginals
    # E.g. col 0 appears col_sums[0] times, col 1 appears col_sums[1] times, etc.
    pool = np.repeat(np.arange(n_cols), col_sums)  # length n

    rng = np.random.default_rng(random_state)
    extreme = 0

    for _ in range(n_sim):
        rng.shuffle(pool)
        # Assign first row_sums[0] draws to row 0, next row_sums[1] to row 1, etc.
        sim = np.zeros((n_rows, n_cols), dtype=int)
        idx = 0
        for i, rs in enumerate(row_sums):
            chunk = pool[idx: idx + rs]
            for j in range(n_cols):
                sim[i, j] = (chunk == j).sum()
            idx += rs

        if (sim.sum(axis=0) == 0).any() or (sim.sum(axis=1) == 0).any():
            continue

        chi2_sim, _, _, _ = chi2_contingency(sim, correction=False)
        if chi2_sim >= chi2_obs:
            extreme += 1

    return (extreme + 1) / (n_sim + 1)


def run_post_hoc_fisher(mf_vals, ec_vals):
    """Performs one-vs-all Fisher tests for categorical data."""
    categories = pd.concat([mf_vals, ec_vals]).unique()
    results = {}

    for cat in categories:
        a = (mf_vals == cat).sum()
        b = (mf_vals != cat).sum()
        c = (ec_vals == cat).sum()
        d = (ec_vals != cat).sum()

        _, p = fisher_exact([[a, b], [c, d]])
        results[cat] = {
            'p': p,
            'mf_pct': a / len(mf_vals),
            'ec_pct': c / len(ec_vals)
        }

    # Apply BH correction to subcategories
    cats = list(results.keys())
    _, padjs = fdrcorrection([results[c]['p'] for c in cats])
    for i, cat in enumerate(cats):
        results[cat]['p_adj'] = padjs[i]

    return results



def compute_stats(data: pd.DataFrame, mode="train"):
    data.dropna(subset=["wahre Diagnose"],inplace=True)
    sample_counts = 0
    stats_dict = {}
    stats_dict["mode"] = mode
    stats_dict["Cohort"] = pd.unique(data["Zentrum"])[0]
    if mode == "train":
        data = data.loc[data["train_test"] == "train", :]
        data = data.loc[data["wahre Diagnose"].str.startswith("SS_", na=False), :]
        data["wahre Diagnose"] = data["wahre Diagnose"].str.replace("SS_", "")
    elif mode == "test":
        data = data.loc[data["train_test"] == "test", :]
        data = data.loc[data["wahre Diagnose"].str.startswith("TT_", na=False), :]
        data["wahre Diagnose"] = data["wahre Diagnose"].str.replace("TT_", "")
    elif (mode == "discovery") | (mode == "aggregated"):
        pass
    else:
        raise ValueError(f"Unknown train_test value")
    total_patients = len(pd.unique(data["Manuscript Patient ID"]))
    total_samples = len(pd.unique(data["Manuscript Sample ID"]))
    stats_dict["total_patients"] = total_patients
    stats_dict["total_samples"] = total_samples
    print(f"Total number of patients: {total_patients} and samples: {total_samples}")
    for diag in pd.unique(data["wahre Diagnose"]):
        # Stats on sample-level
        print(f" {diag}")
        data_diag = data.loc[data["wahre Diagnose"] == diag, :]
        n_samples = len(pd.unique(data_diag["sampleID"]))
        stats_dict[f"{diag}_samples"] = n_samples
        age_median = data_diag["Age"].median()
        age_q1 = data_diag["Age"].quantile(0.25)
        age_q3 = data_diag["Age"].quantile(0.75)
        stats_dict[f"{diag}_age_median"] = age_median
        stats_dict[f"{diag}_age_q1"] = age_q1
        stats_dict[f"{diag}_age_q3"] = age_q3
        sex_values =  data_diag["Sex"].dropna()
        stats_dict[f"{diag}_sex_male_pct"] = (sex_values == "male").mean()
        # Compute the new columsn
        localization_counts = data_diag.loc[:, "Localization"].value_counts(normalize=True, dropna=False)
        duration_mean = np.mean(data_diag.loc[:, "Duration of disease"])
        duration_std = np.std(data_diag.loc[:, "Duration of disease"])
        if diag == "MF":
            staging  = data_diag.loc[:, "MF Stage"].value_counts(normalize=True, dropna=False)
        else:
            staging = data_diag.loc[:, "PGA"].value_counts(normalize=True, dropna=False)
        tcr_data = data_diag.loc[:, "TCR receptor analysis"].value_counts(normalize=True, dropna=False)
        for category, count in localization_counts.items():
            stats_dict[f'{diag}_localization_{category}'] = count
        # Add TCR counts with prefixed keys
        for category, count in tcr_data.items():
            stats_dict[f'{diag}_tcr_{category}'] = count
        for stage, count in staging.items():
            stats_dict[f'{diag}_staging_{stage}'] = count
        stats_dict[f'{diag}_duration_mean'] = duration_mean
        stats_dict[f'{diag}_duration_std'] = duration_std
        sample_counts += n_samples
    assert sample_counts == total_samples
    return pd.DataFrame(stats_dict, index=[0])


def main():
    args = get_args()
    rebuttal_data_dir = "./Cohort_Data"
    today = datetime.today().strftime('%Y_%m_%d')
    if args.mode == "stats":
        if args.aggregate:
            # Aggregate mode: preprocess each cohort according to its mode, then merge and compute stats
            all_data = []

            for file in glob.glob(os.path.join(rebuttal_data_dir, "*.xlsx")):
                _, tail = os.path.split(file)
                if "processed" not in file:
                    continue
                metadata = pd.read_excel(file)
                metadata.dropna(subset=["wahre Diagnose"], inplace=True)

                # Preprocess each file according to its mode
                if tail.startswith("VD"):
                    metadata = metadata.loc[metadata["train_test"] == "test", :]
                    metadata = metadata.loc[metadata["wahre Diagnose"].str.startswith("TT_"), :]
                    metadata["wahre Diagnose"] = metadata["wahre Diagnose"].str.replace("TT_", "")
                    all_data.append(metadata)
                elif tail.startswith("DE"):
                    metadata = metadata.loc[metadata["train_test"] == "train", :]
                    metadata = metadata.loc[metadata["wahre Diagnose"].str.startswith("SS_"), :]
                    metadata["wahre Diagnose"] = metadata["wahre Diagnose"].str.replace("SS_", "")
                    all_data.append(metadata)
                elif tail.startswith("Discovery"):
                    all_data.append(metadata)
                else:
                    raise ValueError("Unknown File")
            # Concatenate all preprocessed cohorts
            merged_data = pd.concat(all_data, ignore_index=True)
            merged_data.loc[merged_data["Sex"] == "w", "Sex"] = "f"
            if args.MF:
                cols = ["Manuscript Patient ID", "Manuscript Sample ID", "Sex", "Age", "wahre Diagnose", "Localization",
                        "Duration of disease", "TCR receptor analysis", "MF Stage", ]
                merged_data = merged_data.loc[merged_data["wahre Diagnose"] == "MF", cols].copy()
                mask = merged_data["Manuscript Patient ID"].notna() & merged_data["Manuscript Patient ID"].apply(lambda x: isinstance(x, str))
                merged_data.loc[mask, "Manuscript Patient ID"] = merged_data.loc[mask, "Manuscript Patient ID"].str.replace("Kempf", "Zürich")
                prefix_order = ["DI", "DE1", "DE2"] + [f"VD{i}" for i in range(1,9)]
                split_ids = merged_data["Manuscript Sample ID"].str.split("_", expand=True)
                split_ids.columns=["_prefix", "_patient_num", "_sample_num"]
                merged_data["_prefix"] = pd.Categorical(split_ids["_prefix"], categories = prefix_order, ordered=True)
                merged_data["_patient_num"] = pd.to_numeric(split_ids["_patient_num"])
                merged_data["_sample_num"] = pd.to_numeric(split_ids["_sample_num"])
                merged_data.sort_values(by=["_prefix", "_patient_num", "_sample_num"], inplace=True, na_position="last")
                merged_data.to_excel(
                    os.path.join(rebuttal_data_dir, f"merged_MF_{today}_stats.xlsx"), index=False)
            else:
                cols = [ "Manuscript Patient ID", "Manuscript Sample ID", "Sex", "Age", "wahre Diagnose", "Localization", "Duration of disease", "TCR receptor analysis"]
                merged_data.loc[:, cols].to_excel(os.path.join(rebuttal_data_dir, f"merged_raw_data_{today}_stats.xlsx"), index=False)
            exit(0)
        else:
            # Original mode: compute stats per cohort
            df = []
            for file in glob.glob(os.path.join(rebuttal_data_dir, "*.xlsx")):
                _, tail = os.path.split(file)
                print(f"File: {tail}")
                if "processed" not in tail:
                    continue
                metadata = pd.read_excel(file)
                if tail.startswith("VD"):
                    stats = compute_stats(metadata, mode="test")
                elif tail.startswith("DE"):
                    stats = compute_stats(metadata, mode="train")
                elif tail.startswith("Discovery"):
                    stats = compute_stats(metadata, mode="discovery")
                else:
                    raise ValueError("Unknown file")
                df.append(stats)
            df = pd.concat(df)
        output_suffix = "_aggregated" if args.aggregate else "_all"
        column_order = ["mode", "Cohort", "total_patients", "total_samples", "MF_samples", "MF_age_median", "MF_age_q1", "MF_age_q3",
                        "MF_sex_male_pct", "MF_staging_nan", "MF_staging_IA", "MF_staging_IA/IB", "MF_staging_IB", "MF_staging_IIA", "MF_staging_IIB", "MF_staging_IIIA",
                        "MF_staging_IIIB", "MF_staging_IVa", "Eczema_Pso_samples", "Eczema_Pso_age_median", "Eczema_Pso_age_q1", "Eczema_Pso_age_q3",
                        "Eczema_Pso_sex_male_pct", "Eczema_Pso_staging_nan", "Eczema_Pso_staging_1.0", "Eczema_Pso_staging_2.0", "Eczema_Pso_staging_3.0",
                        "Eczema_Pso_staging_4.0", "Parapsoriasis_samples", "Parapsoriasis_age_median", "Parapsoriasis_age_q1", "Parapsoriasis_age_q3", "Parapsoriasis_sex_male_pct"]
        df.loc[:, column_order].to_excel(f"./rebuttal_stats{output_suffix}_{today}_subset.xlsx", index=False)
    elif args.mode == "testing":
        data = pd.read_excel(os.path.join("/Users/martin.meinel/Desktop/Projects/Eyerich Projects/Natalie/Classifier/04_Rebuttal_data", "merged_raw_data_2026_07_14_stats.xlsx"))
        mf_data = data.loc[data["wahre Diagnose"] == "MF", :].copy()
        pso_ec_data = data.loc[data["wahre Diagnose"] == "Eczema_Pso", :].copy()
        total_patients = len(pd.unique(data["Manuscript Patient ID"]))
        pvals = {}
        print(f"Total number of patients: {total_patients}")
        for col in ["Manuscript Patient ID", "Manuscript Sample ID", "Age", "Sex", "Duration of disease", "Localization", "TCR receptor analysis"]:
            if col != "TCR receptor analysis":
                col_values_mf = mf_data[col].dropna(inplace=False).reset_index(drop=True)
                col_values_eczema = pso_ec_data[col].dropna(inplace=False).reset_index(drop=True)
            else:
                col_values_mf = mf_data[col].fillna("Not tested")
                col_values_eczema = pso_ec_data[col].fillna("Not tested")
            if col == "Manuscript Patient ID":
                mf_patients = len(pd.unique(col_values_mf))
                ec_patients = len(pd.unique(col_values_eczema))
                print(f"MF patients: {mf_patients}, EC patients: {ec_patients}")
            elif col == "Manuscript Sample ID":
                mf_samples = len(pd.unique(col_values_mf))
                ec_samples = len(pd.unique(col_values_eczema))
                print(f"MF samples: {mf_samples}, EC samples: {ec_samples}")
            elif col in ["Age", "Duration of disease"]:
                mf_median = col_values_mf.median()
                ec_median = col_values_eczema.median()

                # Calculate IQR (25th and 75th percentiles)
                mf_q1, mf_q3 = col_values_mf.quantile([0.25, 0.75])
                ec_q1, ec_q3 = col_values_eczema.quantile([0.25, 0.75])

                print(f"--- {col} Statistics ---")
                print(f"MF: Median={mf_median:.1f}, IQR=[{mf_q1:.1f} - {mf_q3:.1f}]")
                print(f"EC: Median={ec_median:.1f}, IQR=[{ec_q1:.1f} - {ec_q3:.1f}]")

                _, pv = mannwhitneyu(col_values_mf, col_values_eczema, alternative="two-sided")
                print(f"Mann-Whitney U test for {col}: {pv}")
                pvals[col] = pv
            elif col == "Sex":
                eczema_sex_data = col_values_eczema.value_counts(normalize=False)
                eczema_m, eczema_w = eczema_sex_data.get("male", 0), eczema_sex_data.get("female", 0)
                eczema_fraction = eczema_m / (eczema_m + eczema_w)
                mf_sex_data = col_values_mf.value_counts(normalize=False)
                mf_m, mf_w = mf_sex_data.get("male", 0), mf_sex_data.get("female", 0)
                mf_fraction = mf_m / (mf_m + mf_w)
                table = np.array([[mf_m, mf_w], [eczema_m, eczema_w]])
                _, p = fisher_exact(table=table, alternative="two-sided")
                print(f"Fraction of men in MF: {mf_fraction}, Fraction of men in Eczema: {eczema_fraction}")
                print(f"Fisher exact test for {col}: {p}")
                pvals[col] = p
            elif col in ["Localization", "TCR receptor analysis"]:
                contingency = pd.crosstab(pd.concat([col_values_mf, col_values_eczema]),
                                          ["MF"] * len(col_values_mf) + ["EC"] * len(col_values_eczema))
                chi2, pvals[col], _, expected = chi2_contingency(contingency)
                if (expected < 5).any():
                    print(f" Warning: Low expected counts in {col}")
                    print("Running Monte Carlo simulations")
                    pvals[col] = chi2_monte_carlo(contingency.values, n_sim=10000)
                    print(f"\n{col} Monte Carlo p: {pvals[col]}")
                else:
                    print(f"\n{col} Omnibus p: {pvals[col]}")
                if pvals[col] < 0.05:
                    post_hoc = run_post_hoc_fisher(col_values_mf, col_values_eczema)
                    for cat, res in post_hoc.items():
                        print(f"  {cat}: MF={res['mf_pct']:.1%}, EC={res['ec_pct']:.1%}, p={res['p']}, p_adj={res['p_adj']}")
            else:
                raise ValueError(f"Unknown column: {col}")
        keys = list(pvals.keys())
        _, padjs = fdrcorrection([pvals[k] for k in keys])
        print("\n--- Top-level BH-corrected p-values ---")
        for k, padj in zip(keys, padjs):
            print(f"  {k}: {padj}")
    else: raise ValueError(f"Unknown mode: {args.mode}")

if __name__ == "__main__":
    main()