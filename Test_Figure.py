import os
from collections import Counter
from adjustText import adjust_text
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.pyplot import subplot
from scipy.special import logit
from sklearn.metrics import f1_score, balanced_accuracy_score

from Validation_run import compute_geometric_mean, quality_control_qpcr, normalize_raw_counts
from pyutils import compute_sensitivity, compute_specificity

sns.set_style("white")
from Figure3 import load_qpcr_data, load_params


def plot_pie_chart(data: pd.DataFrame):
    data = data.loc[data["wahre Diagnose"] != "Parapsoriasis"]
    custom_palette = {"MF": "#ff5b57",
                      "Eczema_Pso": "#8F8071",
                      "unclear": "black"}
    c = Counter(data["wahre Diagnose"])
    print(c)
    occurences = c.values()
    colors = [custom_palette[diag] for diag in c.keys()]
    plt.figure(figsize=(2, 2))
    plt.pie(occurences, colors=colors, autopct=None, radius=2)
    center_circle = plt.Circle((0, 0), 1.4, color='black', fc='white', linewidth=0)
    fig = plt.gcf()
    fig.gca().add_artist(center_circle)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig("/Figures/Figure4/pie_chart_overall_test.svg", bbox_inches="tight")
    plt.close()


def plot_test_metrics(test_data):
    probabilities = test_data["HOMER1_TBP_LCK_TBP_HOMER1_SDHAF_LCK_SDHAF_baseline_probabilities_calibrated"].values
    predictions = test_data["HOMER1_TBP_LCK_TBP_HOMER1_SDHAF_LCK_SDHAF_baseline_predictions_calibrated"].map(
        lambda x: 1 if x == "MF" else 0).values
    labels = test_data["wahre Diagnose"].map(lambda x: 1 if x == "MF" else 0).values
    keep = np.max(np.column_stack((probabilities, 1 - probabilities)), axis=1) > 0.6
    labels = labels[keep]
    preds = predictions[keep]
    f1 = f1_score(labels, preds)
    sensitivity = compute_sensitivity(labels, preds)
    specificity = compute_specificity(labels, preds)
    balanced_acc = balanced_accuracy_score(labels, preds)
    print(f"F1 score: {f1:.2f}")
    print(f"Sensitivity: {sensitivity:.2f}")
    print(f"Specificity: {specificity:.2f}")
    print(f"Balanced accuracy: {balanced_acc:.2f}")


def main():
    subplot = "b"
    test_data = load_qpcr_data()
    # test_data = test_data.loc[test_data["wahre Diagnose"] != "unclear", :]
    test_data_annotations = pd.read_excel(
        "/2025-09-25_Data Figures.xlsx",
    usecols=["sampleID", "wahre Diagnose"])
    test_data_annotations = test_data_annotations.iloc[169:311]
    test_data.drop(columns = "wahre Diagnose", inplace=True)
    test_data = pd.merge(test_data, test_data_annotations, on="sampleID", how="right")
    test_data["wahre Diagnose"] = test_data["wahre Diagnose"].str.replace("TT_", "")
    test_data.dropna(subset="wahre Diagnose", inplace=True)
    test_data = test_data.loc[test_data["sampleID"] != "NL35-117", :]
    if subplot == "a":
        plot_pie_chart(test_data)
    elif subplot == "b":
        test_data = test_data.loc[test_data["wahre Diagnose"] != "unclear", :]
        test_data = test_data.loc[test_data["Zentrum"] != "Athen", :]
        plot_test_metrics(test_data)
    else:
        cohort = ["Mainz"]
        cohort = ["Göttingen"]
        cohort = ["UKF"]
        cohort = ["Kempf", "Kempf_Röllchen"]
        cohort = ["Würzburg"]
        cohort = ["Dubai"]
        cohort = ["Athens"]
        if "Kempf" in cohort:
            highlight_samples = ["AKF2-006", "AKF2-019", "AKF2-018", "AKF7-011"]
        elif "Dubai" in cohort:
            highlight_samples = ["AKF5-016"]
        elif "Athens" in cohort:
            highlight_samples = ["AKF6-039 Whd", "AKF6-010", "AKF6-035", "AKF6-017", "AKF6-027",
                                 "AKF6-021", "AKF6-026", "AKF6-029", "AKF6-022 Whd", "AKF6-013",
                                 "AKF6-015", "AKF6-012", "AKF6-016"]
        elif "UKF" in cohort:
            highlight_samples = []
        else:
            highlight_samples = []
        # cohort = "Würzburg"
        cols = ["sampleID", "wahre Diagnose","Zentrum", "train_test", "LCK_run1", "LCK_run2", "HOMER1_run1", "HOMER1_run2",
                "SDHAF_run1", "SDHAF_run2", "TBP_run1", "TBP_run2", "HOMER1_TBP_LCK_TBP_HOMER1_SDHAF_LCK_SDHAF_baseline_probabilities_calibrated"]
        params = load_params(
            "/params/HOMER1_TBP_LCK_TBP_HOMER1_SDHAF_LCK_SDHAF_baseline__calibrated_params.json")
        qpcr_data = load_qpcr_data()
        housekeeper= "TBP"
        training_data = qpcr_data.iloc[1:167]
        training_data = training_data.loc[:, cols]
        test_data = test_data.loc[:, cols]
        training_data["wahre Diagnose"] = training_data["wahre Diagnose"].str.replace("SS_", "")
        faulty_train_samples = quality_control_qpcr(training_data, genes=["HOMER1", "LCK", "TBP", "SDHAF"], mode="train")
        faulty_test_samples = quality_control_qpcr(test_data, genes=["HOMER1", "LCK", "TBP", "SDHAF"], mode="train")
        aggregated_training_data = compute_geometric_mean(training_data, genes=["HOMER1", "LCK", "TBP", "SDHAF"])
        aggregated_test_data = compute_geometric_mean(test_data, genes=["HOMER1", "LCK", "TBP", "SDHAF"])
        aggregated_training_data = aggregated_training_data.drop(faulty_train_samples, axis=0)
        aggregated_test_data = aggregated_test_data.drop(faulty_test_samples, axis=0)
        normalized_train_data = normalize_raw_counts(aggregated_training_data, reference_genes=[housekeeper])
        normalized_test_data = normalize_raw_counts(aggregated_test_data, reference_genes=[housekeeper])
        all_train_data = pd.merge(normalized_train_data, training_data.loc[:, ["wahre Diagnose","Zentrum", "sampleID", "train_test"]], left_index=True, right_on="sampleID", how="inner")
        all_test_data = pd.merge(normalized_test_data, test_data.loc[:, ["wahre Diagnose","Zentrum", "sampleID", "train_test"]], left_index=True, right_on="sampleID", how="inner")
        # Without NL35-117
        all_test_data = all_test_data.loc[all_test_data.sampleID != "NL35-117", :]
        # Filter for the right cohort
        if "Athens" in cohort:
            test_cohort = ["Athen"]
        else:
            test_cohort = cohort
        test_data = test_data.loc[test_data["Zentrum"].isin(test_cohort), :]
        fig, axes = plt.subplots(nrows=1, ncols=3, gridspec_kw={'width_ratios': [1., 2, 3]}, figsize=(4.5, 2))
        test_data = test_data.loc[test_data["wahre Diagnose"] != "Parapsoriasis"]
        custom_palette = {"MF": "#ff5b57",
                          "Eczema_Pso": "#8F8071",
                          "unclear": "black"}
        c = Counter(test_data["wahre Diagnose"])
        print(cohort)
        print(c)
        occurences = c.values()
        colors = [custom_palette[diag] for diag in c.keys()]
        axes[0].pie(occurences, colors=colors, autopct=None, radius=1)
        center_circle = plt.Circle((0, 0), 0.6, color='black', fc='white', linewidth=0)
        axes[0].add_artist(center_circle)
        axes[0].set_aspect('equal')
        # Probabilities
        sns.stripplot(x="wahre Diagnose", y="HOMER1_TBP_LCK_TBP_HOMER1_SDHAF_LCK_SDHAF_baseline_probabilities_calibrated",
                      hue="wahre Diagnose", order=["Eczema_Pso", "MF", "unclear"],
                      jitter=True, data=test_data, ax=axes[1], palette=custom_palette, size=2)
        category_order = ["Eczema_Pso", "MF", "unclear"]
        cat_to_num = {cat: i for i, cat in enumerate(category_order)}
        label_data = test_data.loc[test_data["sampleID"].isin(highlight_samples), :]
        texts=[]
        for i, row in label_data.iterrows():
            x_num = cat_to_num[row['wahre Diagnose']]
            y_pos = row["HOMER1_TBP_LCK_TBP_HOMER1_SDHAF_LCK_SDHAF_baseline_probabilities_calibrated"]
            txt = axes[1].text(x_num, y_pos, row['sampleID'], fontsize=5, ha='center', va='bottom')
            texts.append(txt)

        adjust_text(texts, ax=axes[1], expand_points=(1.2, 1.2), arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

        sns.despine(ax=axes[1])
        axes[1].axhspan(0.4, 0.6, alpha=0.35, color="lightgrey", label="Rejection area")
        axes[1].grid(False)

        axes[1].set_xlabel("Ground truth label", fontsize=6)
        axes[1].set_ylabel("Probability for MF", fontsize=6)
        axes[1].tick_params(axis='both', labelsize=6)

        # Scatter plot
        all_test_data = all_test_data.loc[all_test_data["Zentrum"].isin(test_cohort), :]

        assert all_test_data.shape[0] == test_data.shape[0]
        all_data = pd.concat([all_train_data, all_test_data], axis=0)

        feature_cols = all_data.columns.str.endswith(housekeeper)
        xcol, ycol = all_data.columns[feature_cols].tolist()[0], all_data.columns[feature_cols].tolist()[1]

        def scale_point(x, y):
            x_scaled = (x - params["scaler_mean"][xcol]) / params["scaler_scale"][xcol]
            y_scaled = (y - params["scaler_mean"][ycol]) / params["scaler_scale"][ycol]
            return x_scaled, y_scaled

        def get_decision_boundary_y(x, threshold=0.5):
            x_scaled, _ = scale_point(x, 0)
            if "a" in params.keys() & "b" in params.keys():
                a, b = params["a"], params["b"]
            else:
                a, b = 1, 0
            if "decision_threshold" in params.keys():
                threshold = params["decision_threshold"]
            offset = logit(threshold)
            # Solve a * (b0 + b1*x + b2*y) + b = offset
            # => y_scaled = (1/b2) * ((offset - b)/a - b0 - b1x1)
            y_scaled = (1 / params['coefficients'][ycol]) * (
                        ((offset - b) / a) - params['intercept'] - params['coefficients'][xcol] * x_scaled)
            y = y_scaled * params['scaler_scale'][ycol] + params['scaler_mean'][ycol]
            return y

        y_min, y_max = all_data[ycol].min() - 1, all_data[ycol].max() + 1
        x_min, x_max = all_data[xcol].min(), all_data[xcol].max()
        x_plot = np.linspace(x_min, x_max, 100)
        y_plot = [get_decision_boundary_y(x) for x in x_plot]
        y_plot_40 = [get_decision_boundary_y(x, threshold=0.4) for x in x_plot]
        y_plot_60 = [get_decision_boundary_y(x, threshold=0.6) for x in x_plot]


        grey_palette = {"MF": "#888888", "Eczema_Pso": "#888888"}
        train_data = all_data.loc[all_data["train_test"] == "train", :]
        test_data = all_data.loc[all_data["train_test"] == "test", :]
        sns.scatterplot(data=train_data, x=xcol, y=ycol, hue="wahre Diagnose", ax=axes[2], s=5,
                        alpha=0.3, palette=grey_palette)
        sns.scatterplot(data=test_data, x=xcol, y=ycol, hue="wahre Diagnose", ax=axes[2], s=7.5,
                        palette=custom_palette)

        # Labelling
        label_data = test_data.loc[test_data["sampleID"].isin(highlight_samples), :]
        texts = []
        for i, row in label_data.iterrows():
            x_num = row[xcol]
            y_pos = row[ycol]
            txt = axes[2].text(x_num, y_pos, row['sampleID'], fontsize=5, ha='center', va='bottom')
            texts.append(txt)

        adjust_text(texts, ax=axes[2], expand_points=(1.2, 1.2), arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

        axes[2].plot(x_plot, y_plot, '--', color='black', alpha=0.35, label="Calibrated boundary 0.5", linewidth=1.)
        axes[2].fill_between(x_plot, y_plot_40, y_plot_60, color='lightgrey', alpha=0.35, label="Rejection area 0.4-0.6")
        axes[2].set_ylim(y_min, y_max)
        axes[2].tick_params(axis='both', labelsize=6)
        axes[2].grid(False)
        axes[2].set_xlabel(xcol, fontsize=6)
        axes[2].set_ylabel(ycol, fontsize=6)
        axes[2].get_legend().remove()
        sns.despine(ax=axes[2])
        plt.tight_layout()
        feature1 = xcol.split("_")[0]
        feature2 = ycol.split("_")[0]
        output_path = "/Figures/Figure4"
        plt.savefig(os.path.join(output_path,
                                 f"{feature1}_{feature2}_{cohort[0]}_test_plot.svg"),
                    dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()


if __name__ == "__main__":
    main()