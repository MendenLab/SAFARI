import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.calibration import calibration_curve

from pyutils import compute_sensitivity, compute_specificity
import seaborn as sns
sns.set_style("white")

def create_calibration_histogram(data: pd.DataFrame, mode="baseline"):
    if mode == "baseline":
        predictions =  "HOMER1_TBP_LCK_TBP_HOMER1_SDHAF_LCK_SDHAF_baseline_predictions"
        probabilities =  "HOMER1_TBP_LCK_TBP_HOMER1_SDHAF_LCK_SDHAF_baseline_probabilities"
    elif mode == "gridsearch_cv":
        predictions = "HOMER1_TBP_LCK_TBP_HOMER1_SDHAF_LCK_SDHAF_gridsearch_cv_predictions"
        probabilities = "HOMER1_TBP_LCK_TBP_HOMER1_SDHAF_LCK_SDHAF_gridsearch_cv_probabilities"
    elif mode == "baseline_calibrated":
        predictions =  "HOMER1_TBP_LCK_TBP_HOMER1_SDHAF_LCK_SDHAF_baseline_predictions_calibrated"
        probabilities = "HOMER1_TBP_LCK_TBP_HOMER1_SDHAF_LCK_SDHAF_baseline_probabilities_calibrated"
    elif mode == "gridsearch_cv_calibrated":
        predictions = "HOMER1_TBP_LCK_TBP_HOMER1_SDHAF_LCK_SDHAF_gridsearch_cv_predictions_calibrated"
        probabilities = "HOMER1_TBP_LCK_TBP_HOMER1_SDHAF_LCK_SDHAF_gridsearch_cv_probabilities_calibrated"
    else: raise ValueError("Unknown mode")
    fig, (ax1, ax2) = plt.subplots(2,1, sharex=True, figsize=(4,3))
    data["correct"] = data["wahre Diagnose"] == data[predictions]
    correct_probs = data.loc[data["correct"], probabilities]

    incorrect_probs = data.loc[~data["correct"], probabilities]
    bin_edges = np.linspace(0,1,20)
    ax1.hist(correct_probs, bins=bin_edges, alpha=0.5, label="Correct", color="tab:blue")
    ax1.axvline(x=0.5, color='gray', linestyle='--')
    ax1.legend(fontsize=6)
    ax1.set_ylabel("Number of correct predictions", fontsize=6)
    ax1.tick_params(axis='both', labelsize=6)
    sns.despine(ax=ax1)

    # Wrong predictions
    ax2.hist(incorrect_probs, bins=bin_edges, alpha=0.5, label="Incorrect", color="tab:orange")
    ax2.set_xlabel("Predicted Probability", fontsize=6)
    ax2.set_ylabel("Number of wrong predictions", fontsize=6)
    xticks = np.arange(0, 1.05, 0.05)
    ax2.set_xticks(xticks)
    plt.setp(ax2.get_xticklabels(), rotation=-45, fontsize=6)
    ax2.tick_params(axis='both', labelsize=6)
    ax2.axvline(x=0.5, color='gray', linestyle='--')
    ax2.legend(fontsize=6)
    fig.suptitle(f"Calibration plot for model: {mode}", fontsize=8)
    sns.despine(ax=ax2)
    plt.tight_layout()
    plt.savefig(f"/Figures/Misc/calibration_plot_{mode}.svg")
    plt.close()

    binary_labels = data["wahre Diagnose"] == "MF"
    ece = expected_calibration_error(binary_labels, data[probabilities].values, n_bins=10)
    print(f"ECE for {mode}: {ece}")



def expected_calibration_error(y_true, y_prob, n_bins=20):
    """
    Compute the Expected Calibration Error (ECE).

    Parameters:
    - y_true: array-like of true binary labels (0 or 1)
    - y_prob: array-like of predicted probabilities for the positive class
    - n_bins: number of equal-width bins to partition [0,1]

    Returns:
    - ece: float, the expected calibration error
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins) - 1  # bin index for each predicted prob

    ece = 0.0
    total_samples = len(y_prob)

    for i in range(n_bins):
        bin_mask = bin_indices == i
        if np.any(bin_mask):
            bin_confidence = np.mean(y_prob[bin_mask])
            bin_accuracy = np.mean(y_true[bin_mask])
            bin_size = np.sum(bin_mask)
            ece += (bin_size / total_samples) * np.abs(bin_accuracy - bin_confidence)

    return float(ece)


def create_rac_curves(data, mode, symmetric=True):
    if mode == "baseline_calibrated":
        predictions = "HOMER1_TBP_LCK_TBP_HOMER1_SDHAF_LCK_SDHAF_baseline_predictions_calibrated"
        probabilities = "HOMER1_TBP_LCK_TBP_HOMER1_SDHAF_LCK_SDHAF_baseline_probabilities_calibrated"
    elif mode == "gridsearch_cv_calibrated":
        predictions = "HOMER1_TBP_LCK_TBP_HOMER1_SDHAF_LCK_SDHAF_gridsearch_cv_predictions_calibrated"
        probabilities = "HOMER1_TBP_LCK_TBP_HOMER1_SDHAF_LCK_SDHAF_gridsearch_cv_probabilities_calibrated"
    else: raise ValueError("Unknown mode")
    data = data.loc[:, ["wahre Diagnose", predictions, probabilities]]
    data["wahre Diagnose"] = np.where(data["wahre Diagnose"] == "MF", 1, 0)
    data[predictions] = np.where(data[predictions] == "MF", 1, 0)
    total_samples = len(data)
    if symmetric:
        thresholds = np.arange(0.0, 0.81, 0.1)
        rejected_samples = []
        rejected_percentages = []
        sensitivity_values = []
        specificity_values = []
        for limit in thresholds:
            data["rejected"] = (data[probabilities] > 0.5 - limit/2) & (data[probabilities] < 0.5 + limit/2)
            rejected_samples.append(data.rejected.sum())
            rejected_percentages.append((data.rejected.sum() / total_samples) * 100)
            unrejected_samples = data.loc[~data["rejected"], :].copy()
            sensitivity = compute_sensitivity(unrejected_samples["wahre Diagnose"], unrejected_samples[predictions])
            specificity = compute_specificity(unrejected_samples["wahre Diagnose"], unrejected_samples[predictions])
            sensitivity_values.append(sensitivity)
            specificity_values.append(specificity)

        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Plot sensitivity and specificity on the primary y-axis
        sns.lineplot(x=thresholds, y=sensitivity_values, ax=ax1, label="Sensitivity", color="tab:blue", marker="o")
        sns.lineplot(x=thresholds, y=specificity_values, ax=ax1, label="Specificity", color="tab:orange", marker="s")
        ax1.set_xlabel("Rejection Interval Width")
        ax1.set_ylabel("Sensitivity / Specificity")
        ax1.set_ylim(0, 1.1)
        ax1.set_yticks(np.arange(0.1, 1.1, 0.1))
        # Create a secondary y-axis to plot rejected sample counts

        # Annotate sensitivity values
        for x, y in zip(thresholds, sensitivity_values):
            ax1.text(x, y - 0.04, f"{y:.2f}", color="tab:blue", ha='center', fontsize=8)

        # Annotate specificity values
        for x, y in zip(thresholds, specificity_values):
            ax1.text(x, y + 0.02, f"{y:.2f}", color="tab:orange", ha='center', fontsize=8)

        ax2 = ax1.twinx()
        sns.lineplot(x=thresholds, y=rejected_percentages, ax=ax2, label="Rejected % of data", color="tab:gray", linestyle="--", marker="d")

        for x, y in zip(thresholds, rejected_percentages):
            ax2.text(x,y+0.5, f"{str(int(y))}%", color="tab:gray", ha='center', fontsize=8)

        ax2.set_ylabel("Percentage of rejected sammples")
        ax2.get_legend().remove()

        # Combine legends from both axes
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="lower right")

        plt.title(f"Sensitivity, Specificity, and Percentage of Rejected Samples vs Rejection Interval Width for model: {mode}")
        plt.tight_layout()
        plt.savefig(f"/Figures/Misc/RAC_curve_{mode}_symmetric.svg")
        plt.close()
    else:
        sensitivity_threshold = np.arange(0.0, 0.5, 0.05)
        specificity_threshold = np.arange(0.0, 0.5, 0.05)
        rejected_sensitivity_samples = []
        sensitivity_values_sens_reject = []
        specificity_values_sens_reject = []

        rejected_specificity_samples = []
        sensitivity_values_spec_reject = []
        specificity_values_spec_reject = []
        for threshold in sensitivity_threshold:
            data["rejected"] = (data[probabilities] < 0.5) & (data[probabilities] >  0.5 - threshold)
            rejected_sensitivity_samples.append(data.rejected.sum())
            unrejected_samples = data.loc[~data["rejected"], :].copy()
            if len(unrejected_samples) > 0:
                sensitivity = compute_sensitivity(unrejected_samples["wahre Diagnose"], unrejected_samples[predictions])
                specificity = compute_specificity(unrejected_samples["wahre Diagnose"], unrejected_samples[predictions])
            else:
                sensitivity = np.nan
                specificity = np.nan
            sensitivity_values_sens_reject.append(sensitivity)
            specificity_values_sens_reject.append(specificity)
        for threshold in specificity_threshold:
            data["rejected"] = (data[probabilities] > 0.5) & (data[probabilities] < 0.5 + threshold)
            rejected_specificity_samples.append(data.rejected.sum())
            unrejected_samples = data.loc[~data["rejected"], :].copy()

            if len(unrejected_samples) > 0:
                sensitivity = compute_sensitivity(unrejected_samples["wahre Diagnose"], unrejected_samples[predictions])
                specificity = compute_specificity(unrejected_samples["wahre Diagnose"], unrejected_samples[predictions])
            else:
                sensitivity = np.nan
                specificity = np.nan
            sensitivity_values_spec_reject.append(sensitivity)
            specificity_values_spec_reject.append(specificity)

        fig,(ax1, ax2) = plt.subplots(1,2, sharey=True, figsize=(8,6))
        sns.lineplot(x=0.5 - sensitivity_threshold, y=sensitivity_values_sens_reject, ax=ax1, color="tab:blue",
                     label="Sensitivity", marker="o")
        sns.lineplot(x=0.5 - sensitivity_threshold, y=specificity_values_sens_reject, ax=ax1, color="tab:orange",
                     label="Specificity", marker="s")
        ax1_twin = ax1.twinx()
        sns.lineplot(x=0.5 - sensitivity_threshold, y=rejected_sensitivity_samples, ax=ax1_twin, color="tab:gray",
                     label="Num Rejected", marker="d")
        sns.lineplot(x=0.5 + specificity_threshold, y=sensitivity_values_spec_reject, ax=ax2, color="tab:blue",
                     label="Sensitivity", marker="o")
        sns.lineplot(x=0.5 + specificity_threshold, y=specificity_values_spec_reject, ax=ax2, color="tab:orange",
                     label="Specificity", marker="s")
        ax2_twin = ax2.twinx()
        sns.lineplot(x=0.5 + specificity_threshold, y=rejected_specificity_samples, ax=ax2_twin, color="tab:gray",
                     label="Num Rejected", marker="d")
        # Add legends, labels, titles as appropriate
        ax1.set_title(f"Sensitivity_{mode}")
        ax2.set_title(f"Specificity_{mode}")

        for x, y in zip(0.5 - sensitivity_threshold, sensitivity_values_sens_reject):
            ax1.text(x, y + 0.003, f"{y:.2f}", color="tab:blue", ha='center', fontsize=8)
        for x, y in zip(0.5 - sensitivity_threshold, specificity_values_sens_reject):
            ax1.text(x, y + 0.003, f"{y:.2f}", color="tab:orange", ha='center', fontsize=8)
        for x, y in zip(0.5 - sensitivity_threshold, rejected_sensitivity_samples):
            ax1_twin.text(x, y + 0.05 * max(rejected_sensitivity_samples), str(int(y)), color="tab:gray", ha='center',
                          fontsize=8)

        for x, y in zip(0.5 + specificity_threshold, sensitivity_values_spec_reject):
            ax2.text(x, y + 0.003, f"{y:.2f}", color="tab:blue", ha='center', fontsize=8)
        for x, y in zip(0.5 + specificity_threshold, specificity_values_spec_reject):
            ax2.text(x, y + 0.003, f"{y:.2f}", color="tab:orange", ha='center', fontsize=8)
        for x, y in zip(0.5 + specificity_threshold, rejected_specificity_samples):
            ax2_twin.text(x, y + 0.05 * max(rejected_specificity_samples), str(int(y)), color="tab:gray", ha='center',
                          fontsize=8)
        ax1_twin.get_legend().remove()
        ax2_twin.get_legend().remove()

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines_twin1, labels_twin1 = ax1_twin.get_legend_handles_labels()
        ax1.legend(lines1 + lines_twin1, labels1 + labels_twin1, loc="upper right")

        lines2, labels2 = ax2.get_legend_handles_labels()
        lines_twin2, labels_twin2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines2 + lines_twin2, labels2 + labels_twin2, loc="upper left")
        plt.tight_layout()
        plt.savefig(f"/Figures/Misc/RAC_curve_{mode}_asymmetric.svg")
        plt.close()


def create_reliability_curve(data):
    n_bins = 8
    data["wahre Diagnose"] = np.where(data["wahre Diagnose"] == "MF", 1, 0)
    ground_truth = data["wahre Diagnose"].values
    # Raw baseline
    #Calibrated baseline
    prob_baseline_calibrated = data["HOMER1_TBP_LCK_TBP_HOMER1_SDHAF_LCK_SDHAF_baseline_probabilities_calibrated"]
    ece_baseline_calibrated = expected_calibration_error(ground_truth, prob_baseline_calibrated.values, n_bins=n_bins)
    prob_true_baseline_calibrated, prob_pred_baseline_calibrated = calibration_curve(ground_truth, prob_baseline_calibrated, n_bins=n_bins)
    fig, ax1 = plt.subplots(figsize=(4, 4))
    sns.lineplot(x=prob_pred_baseline_calibrated, y=prob_true_baseline_calibrated, ax=ax1, label="Baseline calibrated: ECE {:.3f}".format(ece_baseline_calibrated), color="tab:orange", marker="s", linewidth=1.5, markersize=5)
    # Add the Perfect Calibration Line
    ax1.plot([0, 1], [0, 1], "k--", label="Perfect Calibration", linewidth=1.0)  # Use "k--" for a dashed black line
    ax1.set_xlabel("Mean predicted probability (Confidence)", fontsize=6)
    ax1.set_ylabel("Fraction of positives (Accuracy)", fontsize=6)
    ax1.set_ylim(0, 1.1)
    ax1.set_yticks(np.arange(0, 1.1, 0.1))
    ax1.set_xticks(np.arange(0, 1.1, 0.1))
    ax1.tick_params(axis='both', which='major', labelsize=6)
    ax1.get_legend().remove()
    sns.despine(ax=ax1)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig("/Figures/Misc/reliability_curve_no_legend.svg")
    plt.close()


def main():
    path = "/Test data"
    file = os.path.join(path, "Paper_sheet.xlsx")
    file = pd.read_excel(file, usecols=["sampleID", "train_test", "Zentrum", "wahre Diagnose",
                                        "HOMER1_TBP_LCK_TBP_HOMER1_SDHAF_LCK_SDHAF_baseline_predictions",
                                        "HOMER1_TBP_LCK_TBP_HOMER1_SDHAF_LCK_SDHAF_baseline_probabilities",
                                         "HOMER1_TBP_LCK_TBP_HOMER1_SDHAF_LCK_SDHAF_gridsearch_cv_predictions",
                                        "HOMER1_TBP_LCK_TBP_HOMER1_SDHAF_LCK_SDHAF_gridsearch_cv_probabilities",
                                        "HOMER1_TBP_LCK_TBP_HOMER1_SDHAF_LCK_SDHAF_baseline_predictions_calibrated",
                                        "HOMER1_TBP_LCK_TBP_HOMER1_SDHAF_LCK_SDHAF_baseline_probabilities_calibrated",
                                        "HOMER1_TBP_LCK_TBP_HOMER1_SDHAF_LCK_SDHAF_gridsearch_cv_predictions_calibrated",
                                        "HOMER1_TBP_LCK_TBP_HOMER1_SDHAF_LCK_SDHAF_gridsearch_cv_probabilities_calibrated"])
    file.dropna(subset=["train_test", "HOMER1_TBP_LCK_TBP_HOMER1_SDHAF_LCK_SDHAF_baseline_predictions"], inplace=True)
    train = file.loc[file["train_test"] == "train", :]
    train.dropna(inplace=True)
    train["wahre Diagnose"] = train["wahre Diagnose"].str.replace("SS_", "")
    train["wahre Diagnose"] = train["wahre Diagnose"].str.replace("Eczema_Pso", "Eczema|Psoriasis")
    print(train.shape)
    assert train.shape[0] == 166
    create_reliability_curve(train)
    create_rac_curves(train, "baseline_calibrated", symmetric=True)
    create_rac_curves(train, "baseline_calibrated", symmetric=False)

if __name__ == "__main__":
    main()