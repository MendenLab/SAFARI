import glob
import itertools
import logging
import os.path
import pdb
import re
import warnings
from sys import platform
from typing import Tuple

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from omegaconf import DictConfig
from scipy.stats.mstats import gmean
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score, average_precision_score, balanced_accuracy_score
from sklearn.model_selection import (StratifiedKFold, GridSearchCV, TunedThresholdClassifierCV)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from pyutils import compute_specificity, compute_sensitivity, compute_false_discovery_rate, get_scorer, createModel, \
    get_qpcr_parameter_space, sort_data

warnings.filterwarnings("ignore", category=ConvergenceWarning)

sns.set_style("white")
custom_palette = {"cutaneous lymphoma": "#ff5b57",
                  "cutaneous_lymphoma_NL": "#eea488",
                  "psoriasis": "#39476e",
                  "psoriasis_NL": "#dee7ef",
                  "eczema": "#39ac5e",
                  "eczema_NL": "#b2d7a8",
                  "non-lesional": "#ac8a66",
                  "non_lesional_others": "#a69e9b",
                  "pso_ec": "#397A66"}

class CompositeStratifiedKFold(StratifiedKFold):
    def __init__(self, n_splits=5, shuffle=True, random_state=None):
        super().__init__(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    def _create_composite_key(self, X, y):
        """
        Creates a composite key for stratified k-folds using the index of the dataframe and the label of the sample.
        Parameters
        ----------
        y

        Returns
        -------

        """
        if hasattr(X, "index"):
            cohort_labels = pd.Series(X.index.str.startswith("NL35")).astype(int).astype(str)
        else:
            cohort_labels = pd.Series(np.zeros(X.shape[0], dtype=int)).astype(str)
            print("Warning no index found, assuming all samples are from the same cohort. Falling back to StratifiedKFold.")

        y_array = np.asarray(y).ravel()
        y_str = y_array.astype(str)
        composite_key = cohort_labels + y_str
        if len(np.unique(composite_key)) != 4:
            print(np.unique(composite_key))
            raise ValueError("Composite key must have 4 unique values")
        return composite_key

    # Adjusted split to stratify cohort and disease
    def split(self, X: pd.DataFrame, y, groups=None):
        composite_key = self._create_composite_key(X, y)
        return super().split(X, composite_key, groups)


def compute_wilson_interval(y_true, y_pred, metric) -> Tuple[float, float]:
    """
    Computes the wilson interval for sensitivity or specificity depending on the given metric.
    Parameters
    ----------
    y_true
    y_pred
    metric

    Returns
    -------

    """
    c = 1.96
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    if metric == "sensitivity":
        n = tp + fn
        k = tp
    elif metric == "specificity":
        n = tn + fp
        k = tn
    else:
        return np.inf, np.inf
    mean_nominator = k + 0.5 * (c ** 2)
    mean_denominator = n + c ** 2
    mean = mean_nominator / mean_denominator
    root = np.sqrt((k / n) * (1 - (k / n)) + (c ** 2) / (4 * n))
    std_nominator = c * np.sqrt(n)
    std_denominator = n + c ** 2
    std = (std_nominator / std_denominator) * root
    return mean, std


def read_features(feature_file):
    features = pd.read_excel(feature_file)
    features = features.loc[features["Corrected_p_values"] < 0.01, "Gene_name"].values
    return features



def match_columns(column_name, several_runs=False):
    if several_runs:
        return column_name in ['Sample Name', 'Sample', 'Target Name', 'CT']
    return column_name in ['Sample Name', 'Sample', 'Target Name', 'Ct Mean']

def compute_geometric_mean(data, genes) -> pd.DataFrame:
    gmean_data = []
    for gene in genes:
        run1 = pd.to_numeric(data[f"{gene}_run1"], errors="coerce")
        run2 = pd.to_numeric(data[f"{gene}_run2"], errors="coerce")
        gmean_data.append(gmean([run1, run2], axis=0))
    gmean_data = pd.DataFrame(np.array(gmean_data).T,index=data.sampleID, columns=genes)
    return gmean_data


def quality_control_qpcr(data: pd.DataFrame, genes, mode, skip_ct_mask=False)-> list:
    """
    Executes quality control by ensuring that the STD of the measurements across runs is not bigger than 0.5 if both
    measurements are below 30. Otherwise, checks for STD is not bigger than 1.
    Parameters: containing the data with its genes
    ----------
    data:
    genes:
    mode:

    Returns:
    -------

    """
    # Compute the average across the two runs
    drop_samples = set()
    if mode == "train":
        for gene in genes:
            run1 = pd.to_numeric(data[f"{gene}_run1"], errors="coerce")
            run2 = pd.to_numeric(data[f"{gene}_run2"], errors="coerce")
            std = np.std([run1, run2], axis=0)
            assert len(std) ==  len(run1)
            std_mask = std > 0.5
            ct_mask = (run1 >= 36) | (run2 >= 36)
            if gene in ["SDHAF_2", "TBP_2"]:
                nan_mask = (np.isnan(run1) | np.isnan(run2)) & (data.sampleID.str.startswith(("NL5", "NL10")))
            else:
                nan_mask = np.isnan(run1) | np.isnan(run2)
            overall_mask = std_mask | ct_mask | nan_mask
            drop_samples.update(data.loc[overall_mask].sampleID)
    elif mode == "test":
        gene_data = data.copy()
        gene_data = gene_data.apply(pd.to_numeric, errors='coerce')
        if skip_ct_mask:
            ct_mask = np.full(gene_data.shape[0], False, dtype=bool)
        else:
            ct_mask = (gene_data >= 36).any(axis=1)
        nan_mask = np.isnan(gene_data).any(axis=1)
        nan_ct_mask = nan_mask | ct_mask
        drop_samples.update(gene_data.loc[nan_ct_mask].index)
        if not skip_ct_mask:
            std_agg = gene_data.groupby(gene_data.index).std(ddof=0)
            std_mask = (std_agg > 0.5).any(axis=1)
            drop_samples.update(std_agg.loc[std_mask].index)
    else:
        raise ValueError("mode must be either 'train' or 'test'")
    drop_samples = sorted(list(drop_samples))
    return drop_samples


def read_train_data(path: str, features, binary_labels = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    usecols = ["sampleID", "train_test", "Zentrum", "wahre Diagnose", "Histo Diagnose_Code", "HOMER1_run1", "HOMER1_run2",
               "LCK_run1", "LCK_run2", "TBP_run1", "TBP_run2", "SDHAF_run1", "SDHAF_run2", "SERBP1_run1", "SERBP1_run2"]
    old_samples = ["NL10-021", "NL10-025", "NL10-040", "NL10-064", "NL10-069", "NL35-090 Wdh", "NL35-122", "NL35-125"]
    data = pd.read_excel(path, usecols=usecols)
    train_data = data.loc[data["train_test"] == "train", :]
    pilsen = train_data.loc[train_data["Zentrum"] == "Pilsen", :]
    ukf = train_data.loc[train_data["Zentrum"] == "UKF", :]
    assert all(pilsen.sampleID.str.startswith(("NL5","NL10")))
    assert all(ukf.sampleID.str.startswith("NL35"))
    faulty_samples = quality_control_qpcr(train_data, genes=features, mode="train")
    print(f"Samples not surpassing QC: {faulty_samples}")
    gmean_data = compute_geometric_mean(train_data, features)
    gmean_data.drop(index=old_samples + faulty_samples, inplace=True)
    train_data["wahre Diagnose"] = train_data["wahre Diagnose"].str.replace("SS_", "")
    if binary_labels:
        labels = train_data[["sampleID", "wahre Diagnose"]]
        labels.rename(columns={"wahre Diagnose": "diag"}, inplace=True)
        labels.loc[:, "diag"] = np.where(labels["diag"] == "MF", 1, 0)
    else:
        labels = train_data[["sampleID", "Histo Diagnose_Code"]]
        labels.rename(columns={"Histo Diagnose_Code": "diag"}, inplace=True)
        labels["diag"] = labels["diag"].replace("Psoriasis/Eczema", "Eczema_Pso")
    labels.set_index(labels["sampleID"], inplace=True)
    labels.drop(columns=["sampleID"], inplace=True)
    labels = labels.loc[gmean_data.index].copy()
    assert all(gmean_data.index == labels.index)
    assert sum(gmean_data.index.str.startswith("AKF") == 0)
    assert gmean_data.shape[0] == labels.shape[0] == 166
    return gmean_data, labels


def read_raw_data(path: str, reference_gene, include_several_runs=False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reads the raw data
    Parameters
    ----------
    path:
    reference_gene: potentially several ones
    include_several_runs: includes all CT runs if True, otherwise only the mean

    Returns
    -------

    """
    final_counts = pd.DataFrame()
    new_data = pd.DataFrame()
    for file in glob.glob(os.path.join(path, "*.xlsx")):
        if include_several_runs:
            if "AKF1" in file:
                raw_counts = (pd.read_excel(file, sheet_name="Results", header=4, usecols=lambda x: match_columns(x, several_runs=True))).rename(
                    columns={"Target Name": "Gene", "CT": "Count"})
            elif "AKF2" in file:
                raw_counts = (pd.read_excel(file, sheet_name="Results", header=6, usecols=lambda x: match_columns(x, several_runs=True))).rename(
                    columns={"Target Name": "Gene", "CT": "Count"})
            elif "AKF3" in file:
                raw_counts1 = (pd.read_excel(file, sheet_name="AKF3_1-12", header=16, usecols=lambda x: match_columns(x, several_runs=True))).rename(
                    columns={"Target Name": "Gene", "CT": "Count"}
                )
                raw_counts2 = (pd.read_excel(file, sheet_name="AKF3_13-20", header=16, usecols=lambda x: match_columns(x, several_runs=True))).rename(
                    columns={"Target Name": "Gene", "CT": "Count"}
                )
                raw_counts2.dropna(inplace=True)
                raw_counts = pd.concat([raw_counts1, raw_counts2], axis=0)
            elif "AKF4" in file:
                raw_counts = (pd.read_excel(file, sheet_name="Results", header=16, usecols=lambda x: match_columns(x, several_runs=True))).rename(
                    columns={"Target Name": "Gene", "CT": "Count"})
            elif "AKF5"  in file:
                raw_counts1 = (pd.read_excel(file, sheet_name="Results AKF7, AKF5, AKF6", header=16, usecols=lambda x: match_columns(x, several_runs=True))).rename(
                    columns={"Target Name": "Gene", "CT": "Count"}
                )
                raw_counts2 = (pd.read_excel(file, sheet_name="Results AKF6 rest, AKF3 Whd", header=16, usecols=lambda x: match_columns(x, several_runs=True))).rename(
                    columns={"Target Name": "Gene", "CT": "Count"}
                )
                raw_counts2.loc[raw_counts2["Gene"] == "Homer", "Gene"] = "HOMER"
                nan_rows = raw_counts2.isna().all(axis=1)
                if nan_rows.any():
                    raw_counts2 = raw_counts2.loc[:nan_rows.idxmax() - 1]
                raw_counts = pd.concat([raw_counts1, raw_counts2], axis=0)
            elif "AKF9" in file:
                raw_counts1 = (pd.read_excel(file, sheet_name="Results_plate 1", header=19, usecols=lambda x: match_columns(x, several_runs=True))).rename(
                    columns={"Target Name": "Gene", "CT": "Count"})
                raw_counts2 = (pd.read_excel(file, sheet_name="Results_plate 2", header=19, usecols=lambda x: match_columns(x, several_runs=True))).rename(
                    columns={"Target Name": "Gene", "CT": "Count"}
                )
                raw_counts1.loc[raw_counts1["Gene"] == "Lck", "Gene"] = "LCK"
                raw_counts = pd.concat([raw_counts1, raw_counts2], axis=0)
            else:
                raw_counts = (pd.read_excel(file, sheet_name="Results", header=47, usecols=lambda x: match_columns(x, several_runs=True))).rename(
                    columns={"Target Name": "Gene", "CT": "Count"})
        else:
            raw_counts = (pd.read_excel(file, sheet_name="Results", header=47, usecols=match_columns)).rename(
                columns={"Target Name": "Gene", "Ct Mean": "Count"})
        if "Sample Name" in raw_counts.columns:
            raw_counts.rename(columns={"Sample Name": "Sample"}, inplace=True)
        nan_rows = raw_counts.isna().all(axis=1)
        if nan_rows.any():
            raw_counts = raw_counts.loc[:nan_rows.idxmax() - 1]
        if include_several_runs:
            raw_counts["run"] = raw_counts.groupby(["Sample", "Gene"]).cumcount() + 1
        raw_counts.drop_duplicates(inplace=True)
        raw_counts.dropna(inplace=True)
        if "AKF1" in file:
            raw_counts.Sample = "AKF1-" + raw_counts.Sample.astype(str)
        elif "AKF2" in file or "AKF3" in file or "AKF4" in file or "AKF5" in file or "AKF6" in file or "AKF7" in file:
            raw_counts.loc[raw_counts["Sample"] == "AK2_18", "Sample"] = "AKF2_18"
            raw_counts.Sample = raw_counts.Sample.map(
                lambda x: re.sub(r'(AKF\d+)_(\d+)', lambda m: f'{m.group(1)}-{int(m.group(2)):03d}', str(x)))
        elif "AKF9" in file:
            raw_counts.loc[raw_counts["Sample"].str.startswith("Whd"), "Sample"] = \
            raw_counts.loc[raw_counts["Sample"].str.startswith("Whd"), "Sample"].str.replace("^Whd ", "", regex=True) + " Whd"
            raw_counts.Sample = raw_counts.Sample.map(
                lambda x: re.sub(r'(AKF\d+)_(\d+)', lambda m: f'{m.group(1)}-{int(m.group(2)):03d}', str(x)))
        elif "Pilsen" in file:
            raw_counts.Sample = raw_counts.Sample.map(lambda x: re.sub(r'(NL\d+)-(\d+)', lambda m: f'{m.group(1)}-{int(m.group(2)):03d}', str(x)))
        else:
            raw_counts.Sample = raw_counts.Sample.map(lambda x: re.sub(r'(NL\d+)-(\d+)', lambda m: f'{m.group(1)}-{int(m.group(2)):03d}', str(x)))
            raw_counts.Sample = raw_counts.Sample.map(lambda x: str.split(x, "_")[0])
        if include_several_runs:
            raw_counts = raw_counts.pivot(index=["Sample", "run"], columns="Gene", values="Count")
        else:
            raw_counts = raw_counts.pivot(index="Sample", columns="Gene", values="Count")
        # Rename all the genes
        if "2024" in file:
            raw_counts.rename(columns={"18S short": "18S", "NLRC5_2": "NLRC5", "FLRT3_1": "FLRT3", "LCK_2": "LCK",
                                       "ZC3H12": "ZC3H12D"}, inplace=True)
        else:
            raw_counts.rename(columns={"NLRC5_2": "NLRC5", "LCK_2": "LCK", "ZC3H12": "ZC3H12D"}, inplace=True)
        if "FLTR3_2" in raw_counts.columns:
            raw_counts.drop(columns=["FLTR3_2"], inplace=True)
        if "FLRT3_2" in raw_counts.columns:
            raw_counts.drop(columns=["FLRT3_2"], inplace=True)
        if "CCL27" in raw_counts.columns:
            raw_counts.drop(columns=["CCL27"], inplace=True)
        if "NOS2" in raw_counts.columns:
            raw_counts.drop(columns=["NOS2"], inplace=True)
        head, tail = os.path.split(file)
        if (tail.startswith("2024-12")) | (tail.startswith("2025")):
            if "Pilsen" in head:
                if "SERPB1" in raw_counts.columns:
                    raw_counts.rename(columns={"SERPB1": "SERBP1"}, inplace=True)
                if "TBP" in raw_counts.columns:
                    raw_counts.rename(columns={"TBP": "TBP_2"}, inplace=True)
                if "SDHAF" in raw_counts.columns:
                    raw_counts.rename(columns={"SDHAF": "SDHAF_2"}, inplace=True)
                if "HOMER1" in raw_counts.columns:
                    raw_counts.drop(columns=["HOMER1"], inplace=True)
            new_data = pd.concat([new_data, raw_counts], axis=0)
        else:
            final_counts = pd.concat([final_counts, raw_counts], axis=0)
    # Filter individuals
    # Drop the NL23 individuals as they are part of the test set
    final_counts.rename(index={"NL1024": "NL10-024", "BL10-62": "BL10-062"}, inplace=True)
    new_data.rename(index={"NL549": "NL5-049", "NL10-062": "BL10-062"}, inplace=True)
    if include_several_runs:
        indizes = new_data.index.get_level_values(0).str.contains("NL23")
    else:
        indizes = new_data.index.str.contains("NL23")
    new_data.drop(index=new_data.index[indizes], inplace=True)
    if final_counts.shape[0] > 0:
        if "reference" in final_counts.index:
            final_counts.drop(index=["reference"], inplace=True)
        if "Refernce" in final_counts.index:
            final_counts.drop(index=["Refernce"], inplace=True)
        if "Referenz" in final_counts.index:
            final_counts.drop(index=["Referenz"], inplace=True)
    # Merge new_data with final_counts
    final_counts = pd.concat([final_counts, new_data], axis=1)
    # Remove samples for which 18S was not assessed and remove reference genes which are not used
    if "18S" in reference_gene:
        final_counts.dropna(subset=["18S"], inplace=True)
    if "18S" not in reference_gene:
        if "18S" in final_counts.columns:
            final_counts.drop(columns=["18S"], inplace=True)
    if "SDHAF" not in reference_gene:
        final_counts.drop(columns=["SDHAF", "SDHAF_2"], inplace=True)
    if "TBP" not in reference_gene:
        final_counts.drop(columns=["TBP", "TBP_2"], inplace=True)
    if include_several_runs:
        labels = pd.DataFrame(data=final_counts.index.get_level_values(0).str.startswith("NL5"), index=final_counts.index, columns=["labels"])
    else:
        labels = pd.DataFrame(data=final_counts.index.str.startswith("NL5"), index=final_counts.index, columns=["labels"])
    if "AKF" in file:
        final_counts.rename(columns={"BTNA3": "BTN3A1", "Lck": "LCK", "HOMER": "HOMER1",
                                     "ZC3H": "ZC3H12D", "SERPB1": "SERBP1"}, inplace=True)
    if "AKF2" in file:
        final_counts.rename(columns={"SDHAF2": "SDHAF"}, inplace=True)
    if "AKF3" in file:
        final_counts.rename(columns={"SDHAF2": "SDHAF", "HOMER": "HOMER1", "ZC3H12": "ZC3H12D", "Lck":"LCK",
                                     "BTNA3":"BTN3A1", "SerpB1": "SERBP1"}, inplace=True)
    if "AKF4" in file:
        final_counts.rename(columns={"SDHAF2": "SDHAF", "HOMER": "HOMER1", "Lck":"LCK", "SERPB1" : "SERBP1"}, inplace=True)
    if "AKF5" in file or "AKF6" in file or "AKF7" in file:
        final_counts.rename(columns={"SDHAF2": "SDHAF", "HOMER": "HOMER1", "Lck":"LCK"}, inplace=True)
    if "AKF9" in file:
        final_counts.rename(columns={"HOMER": "HOMER1"}, inplace=True)
    return final_counts, labels



def augment_data(X, reference_gene, augmentation_method, augmentation_strength, repetition):
    """
    This method adds noise coming from a Gaussian with mean 0 and varaince of permutation strength. It is important to note that for one sample the noise is the same for all reference genes.
    Parameters
    ----------
    X: data frame with samples and features
    reference_gene: list with reference genes
    augmentation_method: noise or shift, if noise sample from Gaussian with mean 0 and variance permutation strength, if shift add fixed constant offset.
    augmentation_strength: variance of the Gaussian or constant shift
    repetition: for seeding

    Returns: Dataframe where each sample is kept with its original and a noise copy is added. Shape afterwards is (2*samples, features)
    -------

    """
    if ("BTN3A1" in X.columns) | ("SERBP1" in X.columns):
        combined_reference_genes = [gene for housekeeper in reference_gene for gene in [housekeeper, f"{housekeeper}_2"]]
    else:
        combined_reference_genes = reference_gene
    augmented_data = X.copy()
    reference_mask = X.columns.isin(combined_reference_genes)
    np.random.seed(repetition)
    # Sample for one housekeeper (TBP) and copy it over to TBP2
    if augmentation_method == "noise": # Adds Gaussian noise
        noise = np.random.normal(loc=0, scale=augmentation_strength, size=(X.shape[0], 1))
    else: #Adds shift
        raise ValueError("shift is not a valid augmentation method")
        signs = np.random.choice([-1, 1], size=(X.shape[0], 1))
        noise = np.ones((X.shape[0], 1)) * augmentation_strength * np.sign(signs)
    # As we want to simulate the same shifted noise within a sample for the worst case,
    # we apply fixed noise as often as we have reference genes (2 times or 4 times)
    duplicated_noise = np.repeat(noise, len(combined_reference_genes)).reshape(X.shape[0], -1)
    augmented_data.loc[:, reference_mask] += duplicated_noise
    # This is only for validation purposes that SDHAF and its counterpart SDHAF_2
    # are shifted by the same amount in the same sample
    check = augmented_data - X
    if ("SDHAF" in X.columns) & ("SDHAF_2" in X.columns):
        mask = pd.notna(X["SDHAF_2"])
        check["SDHAF"] =  pd.to_numeric(check["SDHAF"], errors='coerce')
        check["SDHAF_2"] =  pd.to_numeric(check["SDHAF_2"], errors='coerce')
        assert np.allclose(check.loc[mask,"SDHAF"].values, check.loc[mask, "SDHAF_2"].values)
    if ("TBP" in X.columns) & ("TBP_2" in X.columns):
        mask = pd.notna(X["TBP_2"])
        check["TBP"] =  pd.to_numeric(check["TBP"], errors='coerce')
        check["TBP_2"] =  pd.to_numeric(check["TBP_2"], errors='coerce')
        assert np.allclose(check.loc[mask, "TBP"].values, check.loc[mask, "TBP_2"].values)
    augmented_data.index = augmented_data.index.astype(str) + "_augmented"
    return pd.concat([X, augmented_data], axis=0)


def train_on_augmented_data(data: pd.DataFrame, labels, model: str, kernel: str, reference_gene: list,
                            augmentation_method: str, augmentation_strength: float):
    """
    Trains on augmented data and predicts on the real test set.
    Parameters
    ----------
    augmentation_strength
    augmentation_method
    data
    labels
    model
    kernel
    reference_gene

    Returns
    -------

    """
    if platform == "darwin":
        base_path = os.path.join("/Users/martin.meinel/Desktop/Projects/Eyerich Projects/Natalie/Classifier/Validation data/Augmentations", augmentation_method)
    else:
        base_path = os.path.join("/lustre/groups/cbm01/datasets/martin.meinel/Safari/Augmentations", augmentation_method)
    os.makedirs(base_path, exist_ok=True)
    # Create model
    repetitions = 100
    # Metrics
    roc_aucs = []
    accuracies = []
    sensitivity = []
    specificity = []
    f1_scores = []
    f1_scores_weighted = []
    fdrs = []
    for rep in tqdm(range(repetitions)):
        outer_split = CompositeStratifiedKFold(n_splits=4, random_state=rep, shuffle=True)
        ground_truth_over_all_folds = []
        predictions_over_all_folds = []
        pos_probabilities_over_all_folds = []
        for train_idx, test_idx in outer_split.split(X=data, y=labels):
            train_data, test_data = data.iloc[train_idx], data.iloc[test_idx]
            train_labels, test_labels = labels.iloc[train_idx].to_numpy().ravel(), labels.iloc[
                test_idx].to_numpy().ravel()
            # Normalize training data, Permute test data and normalize test data
            if augmentation_strength == 0.0:
                augmented_train_data = train_data
                augmented_train_labels = train_labels
            else:
                augmented_train_data = augment_data(train_data,
                                                    reference_gene=reference_gene,
                                                    augmentation_method=augmentation_method,
                                                    augmentation_strength=augmentation_strength,
                                                    repetition=rep)
                augmented_train_labels = np.concatenate([train_labels, train_labels])
            normalized_augmented_train = normalize_raw_counts(augmented_train_data, reference_genes=reference_gene)
            normalized_test_data = normalize_raw_counts(test_data, reference_genes=reference_gene)
            # Create model here to ensure a new model in each run
            weighting_factor = (sum(train_labels == 0) / sum(train_labels == 1))
            estimator = createModel(model, kernel=kernel, scale_pos_weight=weighting_factor)
            pipeline = Pipeline(steps=[('scaler', StandardScaler()), ("estimator", estimator)])
            pipeline.fit(X=normalized_augmented_train, y=augmented_train_labels)
            model_name = pipeline['estimator'].__class__.__name__
            # Store here the coefficients per run and then create a boxplot to understand the parameters better
            y_hat = pipeline.predict(X=normalized_test_data)
            ground_truth_over_all_folds.extend(test_labels)
            predictions_over_all_folds.extend(y_hat)
            if model_name in ["LogisticRegression", "RandomForestClassifier", "XGBClassifier",
                              "BalancedRandomForestClassifier"]:
                probs = pipeline.predict_proba(X=normalized_test_data)
                probs_pos = probs[:, 1]
            else:
                probs_pos = pipeline.decision_function(X=normalized_test_data)
            pos_probabilities_over_all_folds.extend(probs_pos)
            # Compute metrics here for the specific repetition over all folds
        fdrs.append(compute_false_discovery_rate(y_true=np.array(ground_truth_over_all_folds), y_pred=np.array(predictions_over_all_folds)))
        accuracies.append(balanced_accuracy_score(y_true=ground_truth_over_all_folds,
                                                    y_pred=predictions_over_all_folds))
        sensitivity.append(compute_sensitivity(y_true=ground_truth_over_all_folds, y_pred=predictions_over_all_folds))
        specificity.append(compute_specificity(y_true=ground_truth_over_all_folds, y_pred=predictions_over_all_folds))
        f1_scores.append(f1_score(y_true=ground_truth_over_all_folds, y_pred=predictions_over_all_folds))
        roc_aucs.append(roc_auc_score(y_true=ground_truth_over_all_folds, y_score=pos_probabilities_over_all_folds))
        f1_scores_weighted.append(
            f1_score(y_true=ground_truth_over_all_folds, y_pred=predictions_over_all_folds, average="weighted"))
    # Create order of genes
    features = data.columns.tolist()
    batch_references = [gene for housekeeper in reference_gene for gene in [housekeeper, f"{housekeeper}_2"]]
    features = set(features).difference(batch_references)
    features = sorted(features)
    features = "_".join(features)
    if len(reference_gene) > 1:
        reference_gene_str = "both"
    else:
        reference_gene_str = reference_gene[0]
    # Store the data in files as before
    df = {"accuracy": accuracies, "sensitivity": sensitivity, "specificity": specificity, "f1_score": f1_scores,
          "ROC AUC": roc_aucs, "f1_score_weighted": f1_scores_weighted, "fdr": fdrs}
    df = pd.DataFrame(df)
    path = os.path.join(base_path,
                        f"{model}_{kernel}_{reference_gene_str}_{features}_{augmentation_strength}.h5")
    df.to_hdf(path, key="results", mode="w")



def filter_features(data: pd.DataFrame, labels: pd.DataFrame, features: list, reference_gene) -> Tuple[
    pd.DataFrame, pd.DataFrame]:
    """
    Gets the entire dataset and labels and filters for the features which are of interest for the given model.
    Additionally, we omit individuals which have for certain genes too high standard deviations between the two wells.
    Parameters
    ----------
    reference_gene:
    data: Data set in shape NxD
    features: list of features k
    labels:
    Returns Nxk or in case of features with unreliable measurement N' x k
    -------

    """
    # if PNLIPRP3 was chosen we omit all the new samples as no PNLIPRP3 and FLRt3 was measured
    # This is a list of samples where for the respective gene the two measurements have exhibit a standard deviation over 0.5 or 1 respectively.
    data = data.copy()
    rnf_samples = ["NL10-018", "NL5-040", "NL5-020", "NL35-011", "NL35-087"]

    # NL5-45 does not have an data for a second qPCR round (BTN3A1, SERBP1 and TBP2 and SDHAF< NL10-23 and NL10-18) have unreliable measurements for TBP_2
    tbp_samples = ["NL10-038", "NL5-045", "NL10-023", "NL10-018", "NL35-016 Wdh"]

    zc3h12d_samples = ["NL5-040", "NL5-005", "NL35-010", "NL35-011", "NL35-098"]

    btn3a1_samples = ["NL5-020", "NL5-039", "NL5-040", "NL5-045", "NL35-013", "NL35-022", "NL35-090", "NL35-103"]

    serbp1_samples = ["NL10-069", "NL5-026", "NL5-045"]

    homer1_samples = ["NL35-107"]

    # for the new samples with btn3a1 and serbp1 the tbp gene was measured unreliably in two samples NL10-23 and NL10-18
    samples_to_drop = []
    if "TBP" in reference_gene:
        samples_to_drop.extend(tbp_samples)
    if "RNF213" in features:
        samples_to_drop.extend(rnf_samples)
    if "ZC3H12D" in features:
        samples_to_drop.extend(zc3h12d_samples)
    if "BTN3A1" in features:
        samples_to_drop.extend(btn3a1_samples)
    if "SERBP1" in features:
        samples_to_drop.extend(serbp1_samples)
    if "HOMER1" in features:
        samples_to_drop.extend(homer1_samples)
    overall_samples_to_drop = list(set(samples_to_drop))
    data.drop(index=overall_samples_to_drop, inplace=True)
    labels.drop(index=overall_samples_to_drop, inplace=True)
    if "BL10-062" in data.index:
        data.rename(index={'BL10-062': 'NL10-062'}, inplace=True)
        labels.rename(index={'BL10-062': 'NL10-062'}, inplace=True)
    assert data.shape[0] == labels.shape[0]
    assert all(data.index == labels.index)
    return data.loc[:, features], labels


def normalize_raw_counts(X: pd.DataFrame, reference_genes: list) -> pd.DataFrame:
    """
    Normalizes the raw counts according to delta or div method regarding a given reference gene
    Parameters
    ----------
    X: Data including the reference gene
    reference_genes: list of housekeeper genes for delta normalization with at least one gene

    Returns normalized pandas dataframe without the reference gene included
    -------

    """
    assert len(reference_genes) > 0
    # NL5 and NL10 Pilsen training, NL35 UKF training, all others AKF for testing
    assert all(X.index.str.startswith(("NL", "AKF")))
    X = X.astype(float)
    type2_housekeepers = [f"{r}_2" for r in reference_genes]
    # Group of samples which have only TBP and SDHAF, for the other group TBP_2 and SDHAF_2 is used for BTN3A1 and SERBP1 normalization
    pilsen_samples = X.index.str.startswith(("NL5", "NL10"))

    group1_genes = ["LCK", "HOMER1", "RNF213", "ZC3H12D"]
    group2_genes = ["BTN3A1", "SERBP1"]

    combined_reference_genes = reference_genes + type2_housekeepers
    target_genes = [col for col in X.columns if col not in combined_reference_genes]

    normalized_data = []

    group1_genes_used = [gene for gene in target_genes if gene in group1_genes]
    if group1_genes_used:
        for hk in reference_genes:
            group1_normalized = X.loc[:, group1_genes_used].sub(X.loc[:, hk], axis=0)
            group1_normalized.columns = [f"{gene}_{hk}" for gene in group1_genes_used]
            normalized_data.append(group1_normalized)

    group2_genes_used = [gene for gene in target_genes if gene in group2_genes]
    if group2_genes_used:
        for hk in reference_genes:
            group2_normalized_freiburg_wuerzburg = X.loc[~pilsen_samples, group2_genes_used].sub(X.loc[~pilsen_samples, hk], axis=0)
            if any(pilsen_samples):
                group2_normalized_pilsen = X.loc[pilsen_samples, group2_genes_used].sub(X.loc[pilsen_samples, f"{hk}_2"], axis=0)
            else:
                group2_normalized_pilsen = pd.DataFrame(index=[], columns=group2_genes_used)
            group2_normalized_overall = pd.concat([group2_normalized_freiburg_wuerzburg, group2_normalized_pilsen], axis=0)
            group2_normalized_overall.columns = [f"{gene}_{hk}" for gene in group2_genes_used]
            normalized_data.append(group2_normalized_overall)
    all_genes_normalized = pd.concat(normalized_data, axis=1)
    all_genes_normalized = all_genes_normalized.loc[X.index]
    return all_genes_normalized


def permute_raw_data(X, permutation_method, permutation_strength, reference_gene, repetition) -> pd.DataFrame:
    """
    This method is used to permute the test data. Supported modes are overall, relative and noise.
    Parameters
    ----------
    X: Data containing target and housekeepers
    permutation_method: Overall shifts all genes by a given constant, relative shifts housekeepers by constant, noise adds noise to all genes
    permutation_strength: Gives the constant for overall and relative shifts, for noise the sd of the Gaussian with mean zero.
    reference_gene: Housekeepers that are shifted
    repetition:

    Returns: Shifted dataframe X \tilde
    -------

    """
    # FROM 13.06 we shift the housekeepers not the target genes
    if ("BTN3A1" in X.columns) | ("SERBP1" in X.columns):
        combined_reference_genes = [gene for housekeeper in reference_gene for gene in [housekeeper, f"{housekeeper}_2"]]
    else:
        combined_reference_genes = reference_gene
    if permutation_method == "overall":
        # Shifts all genes by a certain constant
        assert abs(permutation_strength) <= 3
        X = X + permutation_strength
    elif permutation_method == "relative":
        # Shifts reference_genes
        assert abs(permutation_strength) <= 1
        reference_mask = X.columns.isin(combined_reference_genes)
        X.loc[:, reference_mask] = X.loc[:, reference_mask] + permutation_strength
    elif permutation_method == "noise":
        np.random.seed(repetition)
        X = X + np.random.normal(loc=0.0, scale=permutation_strength, size=(X.shape[0], X.shape[1]))
    return X


def robustness_evaluation_with_cv(data: pd.DataFrame, labels: pd.DataFrame, model, kernel, reference_gene,
                                  permutation_method, permutation_strength, scorer_name,
                                  train_augmentation, train_augmentation_strength=0,
                                  debug=False, mode="baseline", additional_labels: pd.DataFrame=None, disease_subset=None):
    """
    Trains and evaluates a given model with a given feature set for a given reference gene with Repeated Stacked Stratified
    Cross-validation.
    Parameters
    ----------
    data: Contains the count matrix with NxD (including the reference gene)
    labels: Binary labels (0=eczema|Pso, 1=MF) Nx1
    model:Specifies the model which is used for training and evaluation
    kernel: specifies kernel for SVM
    reference_gene: Name of reference gene for normalization
    permutation_method: Either relative or overall. Determines the permutation on the test set
    permutation_strength: If relative between [-1, 1] if overall between [-3, 3], if noise [0.5, 1.0] specifying the sd of the Gaussian noise
    train_augmentation: if specified determines how training data is augmented for training to see its influence on perturbed test data
    debug: binary flag for debugging
    additional_labels: providing more specific labels like Psoraisis and Eczema to be able to test how well the classifier works on the single subtypes
    disease_subset: defines whether the analysis is only run for psoriasis or eczema alone instead of aggregating them to one label
    mode: baseline, gridsearch_cv or threshold_tuning
    train_augmentation_strength: Defines the standard deviation of the Gaussian if train_augmentation is noise,
     otherwise if train_augmentation is relative then it determines the shift
    Returns
    -------

    """
    if platform == "darwin":
        base_path = os.path.join(
            "/Users/martin.meinel/Desktop/Projects/Eyerich Projects/Natalie/Classifier/Validation data/Robustness_study",
            permutation_method)
    else:
        base_path = os.path.join("/lustre/groups/cbm01/datasets/martin.meinel/Safari/Robustness_study", permutation_method,
                             "delta")
    os.makedirs(base_path, exist_ok=True)
    coefficients = []
    # Create model
    repetitions = 100
    # Metrics
    roc_aucs = []
    pr_aucs = []
    accuracies = []
    sensitivity = []
    specificity = []
    f1_scores = []
    fdrs = []
    f1_scores_weighted = []
    if debug:
        pdb.set_trace()
    for rep in tqdm(range(repetitions)):
        # outer_split = StratifiedKFold(n_splits=4, random_state=rep, shuffle=True)
        # Adjustment to Composite Stratification
        outer_split = CompositeStratifiedKFold(n_splits=4, random_state=rep, shuffle=True)
        ground_truth_over_all_folds = []
        predictions_over_all_folds = []
        pos_probabilities_over_all_folds = []
        for train_idx, test_idx in outer_split.split(X=data, y=labels):
            train_data, test_data = data.iloc[train_idx].copy(), data.iloc[test_idx].copy()
            train_labels, test_labels = labels.iloc[train_idx].to_numpy().ravel(), labels.iloc[
                test_idx].to_numpy().ravel()
            # Subset data to psoraisis or eczema only if desired
            if disease_subset is not None:
                assert all(labels.index == additional_labels.index)
                disease_test_labels = additional_labels.iloc[test_idx]
                mf_disease_mask = disease_test_labels.isin(["MF", disease_subset]).values.squeeze()
                test_data = test_data.loc[mf_disease_mask]
                test_labels = test_labels[mf_disease_mask]
                assert len(test_labels) == test_data.shape[0]
            # Normalize training data, Permute test data and normalize test data
            if train_augmentation is not None:
                assert train_augmentation_strength > 0
                train_data = augment_data(train_data, reference_gene=reference_gene,
                                          augmentation_method=train_augmentation,
                                          augmentation_strength=train_augmentation_strength, repetition=rep)
                train_labels = np.concatenate([train_labels, train_labels])
            normalized_train = normalize_raw_counts(train_data, reference_genes=reference_gene).copy()
            permuted_test = permute_raw_data(test_data, permutation_method=permutation_method,
                                             permutation_strength=permutation_strength,
                                             reference_gene=reference_gene,
                                             repetition=rep)
            permuted_normalized_test = normalize_raw_counts(X=permuted_test, reference_genes=reference_gene)
            # Create model here to ensure a new model in each run
            weighting_factor = (sum(train_labels == 0) / sum(train_labels == 1))
            estimator = createModel(model, kernel=kernel, scale_pos_weight=weighting_factor)
            pipeline = Pipeline(steps=[('scaler', StandardScaler()), ("estimator", estimator)])
            model_params = get_qpcr_parameter_space()
            ind_scorer = get_scorer(scorer_name=scorer_name)
            # Baseline is just using Pipeline [Scaling + Model] and without any tuning
            # Gridsearch for hyperparameter tuning
            # Threshold for GridSearch with Hyperparameter tuning
            if mode == "baseline":
                pipeline.fit(X=normalized_train, y=train_labels)
                y_hat = pipeline.predict(X=permuted_normalized_test)
                coefficients.append(pipeline["estimator"].coef_.ravel())
                probs_pos = pipeline.predict_proba(X=permuted_normalized_test)[:, 1]
            elif mode == "gridsearch_cv":
                inner_cv = CompositeStratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                search = GridSearchCV(pipeline, param_grid=model_params,
                                      scoring=ind_scorer, cv=inner_cv, n_jobs=-1,
                                      error_score="raise")
                search.fit(X=normalized_train, y=train_labels)
                y_hat = search.predict(permuted_normalized_test)
                coefficients.append(search.best_estimator_['estimator'].coef_.ravel())
                probs_pos = search.predict_proba(permuted_normalized_test)[:, 1]
            elif mode == "threshold_tuning":
                middle_cv = CompositeStratifiedKFold(n_splits=5, shuffle=True, random_state=41)
                inner_cv = CompositeStratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                search = GridSearchCV(pipeline, param_grid=model_params, scoring=ind_scorer, cv=inner_cv, n_jobs=-1)
                threshold_tuner = TunedThresholdClassifierCV(estimator=search, scoring=ind_scorer, cv=middle_cv, n_jobs=-1)
                threshold_tuner.fit(X=normalized_train, y=train_labels)
                y_hat = threshold_tuner.predict(permuted_normalized_test)
                probs_pos = threshold_tuner.predict_proba(permuted_normalized_test)[:, 1]
                coefficients.append(threshold_tuner.estimator_.best_estimator_['estimator'].coef_.ravel())
            else:
                raise ValueError("mode must be either 'baseline', 'gridsearch_cv' or 'threshold_tuning'")
            ground_truth_over_all_folds.extend(test_labels)
            predictions_over_all_folds.extend(y_hat)
            pos_probabilities_over_all_folds.extend(probs_pos)
            # Compute metrics here for the specific repetition over all folds
        fdrs.append(compute_false_discovery_rate(y_true=np.array(ground_truth_over_all_folds), y_pred=np.array(predictions_over_all_folds)))
        accuracies.append(balanced_accuracy_score(y_true=ground_truth_over_all_folds,
                                                    y_pred=predictions_over_all_folds))
        sensitivity.append(compute_sensitivity(y_true=ground_truth_over_all_folds, y_pred=predictions_over_all_folds))
        specificity.append(compute_specificity(y_true=ground_truth_over_all_folds, y_pred=predictions_over_all_folds))
        f1_scores.append(f1_score(y_true=ground_truth_over_all_folds, y_pred=predictions_over_all_folds))
        roc_aucs.append(roc_auc_score(y_true=ground_truth_over_all_folds, y_score=pos_probabilities_over_all_folds))
        f1_scores_weighted.append(
            f1_score(y_true=ground_truth_over_all_folds, y_pred=predictions_over_all_folds, average="weighted"))
        pr_aucs.append(average_precision_score(y_true=ground_truth_over_all_folds, y_score=pos_probabilities_over_all_folds))
    coefficients = np.array(coefficients)
    coefficients = pd.DataFrame(coefficients, columns=normalized_train.columns)
    # Create order of genes
    features = data.columns.tolist()
    batch_references = [gene for housekeeper in reference_gene for gene in [housekeeper, f"{housekeeper}_2"]]
    features = set(features).difference(batch_references)
    features = sorted(features)
    features = "_".join(features)
    if len(reference_gene) > 1:
        reference_gene_str = "both"
    else:
        reference_gene_str = reference_gene[0]
    # Store the data in files as before
    df = {"accuracy": accuracies, "sensitivity": sensitivity, "specificity": specificity, "f1_score": f1_scores,
          "ROC AUC": roc_aucs, "f1_score_weighted": f1_scores_weighted, "fdr": fdrs, "PR AUC": pr_aucs}
    df = pd.DataFrame(df)
    if train_augmentation is not None:
        base_path = os.path.join(base_path, f"augmentation_{train_augmentation}")
        if disease_subset is not None:
            base_path = os.path.join(base_path, disease_subset)
        base_path = os.path.join(base_path, "mixed_data")
        base_path = os.path.join(base_path, mode)
        os.makedirs(base_path, exist_ok=True)
        path = os.path.join(base_path, f"{model}_{kernel}_{reference_gene_str}_{features}_{permutation_strength}_{train_augmentation_strength}_{scorer_name}.h5")
    else:
        base_path = os.path.join(base_path, "mixed_data")
        base_path = os.path.join(base_path, mode)
        os.makedirs(base_path, exist_ok=True)
        path = os.path.join(base_path,
                        f"{model}_{kernel}_{reference_gene_str}_{features}_{permutation_strength}_{scorer_name}.h5")
    df.to_hdf(path, key="results", mode="w")
    # Sufficient to plot the coefficients once per run
    if permutation_strength == 1:
        boxplot_dir = "/lustre/groups/cbm01/workspace/martin.meinel/Safari/Classifier/figures/coef_plots"
        if train_augmentation is not None:
            boxplot_dir = os.path.join(boxplot_dir, f"augmentation_{train_augmentation}")
        os.makedirs(boxplot_dir, exist_ok=True)
        coefficients_long = coefficients.melt(var_name="Gene", value_name="Coefficient")
        plt.figure(figsize=(6, 6))
        sns.boxplot(x="Gene", y="Coefficient", data=coefficients_long, palette="husl", linewidth=0.5)
        plt.xticks(rotation=45, fontsize=8)
        plt.yticks(fontsize=8)
        plt.tight_layout()
        plt.savefig(
            os.path.join(boxplot_dir, f"{model}_{kernel}_{reference_gene_str}_{features}_{permutation_strength}.jpg")
            , bbox_inches="tight", dpi=300)
        plt.close()


@hydra.main(version_base=None, config_path="conf", config_name="validation")
def main(cfg: DictConfig):
    logger = logging.getLogger(__name__)
    mode_shifts_combinations = np.array(list(itertools.product(cfg.mode, cfg.permutation_strengths)))
    array_id = int(os.getenv("SLURM_ARRAY_TASK_ID", "0"))
    mode, perm = mode_shifts_combinations[array_id]
    perm = int(perm)
    logger.info(f"Running mode {mode} with permutation strength {perm}")
    logger.info(f"Config: {cfg}")
    reference_genes = cfg.reference_gene
    if cfg.debug:
        pdb.set_trace()
    if platform == "darwin":
        main_path = "/Users/martin.meinel/Desktop/Projects/Eyerich Projects/Natalie/Classifier/Validation data/Test data"
    else:
        main_path = "/lustre/groups/cbm01/datasets/martin.meinel/Safari"
    if os.path.isfile(cfg.featureFile):
        features = read_features(cfg.featureFile)
        # In case not all features are used, we use the raw counts we have to include the reference gene here
        logger.info(f"Reference genes: {cfg.reference_gene}")
        features = np.append(features, cfg.reference_gene)
        # We always use both housekeepers TBP and tBP2
        if ("SERBP1" in features) | ("BTN3A1" in features):
            for r in reference_genes:
                features = np.append(features, r + "_2")
    else:
        raise ValueError("Invalid feature file")
    # Load training data
    train_data, binary_labels = read_train_data(cfg.trainingSet, features=features, binary_labels=True)
    train_data = sort_data(train_data)
    binary_labels = binary_labels.loc[train_data.index]
    if cfg.disease_subset is not None:
        logger.info(f"Validation for MF and {cfg.disease_subset} only")
        _, disease_labels = read_train_data(os.path.join(main_path,"train_update.xlsx"), features=features, binary_labels=False)
        assert all(disease_labels.index == binary_labels.index)
    # This evaluation here
    # Filter features
    if cfg.debug:
        pdb.set_trace()
    if cfg.disease_subset is not None:
        logger.info("Filtering the disease labels according to binary labels.")
        disease_labels = disease_labels.loc[binary_labels.index]
    else:
        disease_labels = None
    features_wo2 = [f for f in features if f.endswith("_2") is False]
    assert train_data.loc[:, features_wo2].isna().sum().sum() == 0
    # Fill nas here due to problems later in cv splitting
    train_data.fillna(value=-100, inplace=True)
    print(f"Number of samples: {train_data.shape[0]}")
    logger.info("Evaluate qPCR data")
    if cfg.augmentation: # This is used for finding the best augmentation strength
        logger.info("Training on augmented data and testing on original data")
        if cfg.debug:
            pdb.set_trace()
        training_permutation_strengths = [0.5, 0.75, 1, 1.25, 1.5]
        for training_permutation_strength in training_permutation_strengths:
            logger.info(f"Augmentation strength: {training_permutation_strength}")
            train_on_augmented_data(train_data, binary_labels, cfg.model, cfg.kernel, cfg.reference_gene,
                                    augmentation_method=cfg.train_augmentation,
                                    augmentation_strength=training_permutation_strength)
    else:
        if cfg.debug:
            pdb.set_trace()
        if cfg.train_augmentation is not None:
            # Focus on minimal augmentation
            train_augmentation_strengths = [0.5]
            for train_augmentation_strength in train_augmentation_strengths:
                # for permutation_strength in permutation_strengths:
                logger.info(f"Training on augmented data with augmentation {train_augmentation_strength} and permutated test data {perm}")
                robustness_evaluation_with_cv(train_data, binary_labels, cfg.model, cfg.kernel,
                                              cfg.reference_gene, permutation_method=cfg.permutation,
                                              permutation_strength=perm, scorer_name=cfg.scorer,
                                              train_augmentation=cfg.train_augmentation,
                                              train_augmentation_strength=train_augmentation_strength,
                                              debug=cfg.debug, mode=mode, additional_labels=disease_labels,
                                              disease_subset=cfg.disease_subset)
        else:
            # for permutation_strength in permutation_strengths:
            logger.info(f"Training on original data and testing with permutation: {perm}")
            robustness_evaluation_with_cv(train_data, binary_labels, cfg.model, cfg.kernel,
                                          cfg.reference_gene,
                                          permutation_method=cfg.permutation,
                                          permutation_strength=perm,
                                          scorer_name=cfg.scorer,
                                          train_augmentation=cfg.train_augmentation,
                                          debug=cfg.debug,
                                          mode=mode, additional_labels=disease_labels,
                                          disease_subset=cfg.disease_subset)


if __name__ == '__main__':
    main()
