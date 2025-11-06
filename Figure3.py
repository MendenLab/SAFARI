import json
import os
from collections import Counter
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from Classifier.pyutils import compute_sensitivity, compute_specificity
from matplotlib.colors import to_hex, to_rgb
from scipy.special import logit
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_curve, roc_auc_score, f1_score, balanced_accuracy_score
from statannotations.Annotator import Annotator
from statsmodels.stats.multitest import fdrcorrection

from Validation_run import compute_geometric_mean, normalize_raw_counts, quality_control_qpcr

sns.set_style("white")


def create_site_variants(base_color, lightness_factors=[0.9, 1.1]):
    """Create lighter and darker variants of a base color"""
    rgb = np.array(to_rgb(base_color))

    variants = []
    for factor in lightness_factors:
        if factor > 1:  # Lighter
            variant = rgb + (1 - rgb) * (factor - 1) / 2
        else:  # Darker
            variant = rgb * factor

        # Clip to valid RGB range
        variant = np.clip(variant, 0, 1)
        variants.append(to_hex(variant))

    return variants

def create_color_palette():
    diagnostic_colors = {}
    # base_colors = ["#ff5b57", "#39476e", "#39ac5e", "#8f8071"]
    base_colors = ["#ff5b57", "#39476e", "#39ac5e", "#00858a"]
    diagnoses = ["MF", "Psoriasis", "Eczema", "Eczema_Pso"]
    medical_sites = ["UKF", "Pilsen"]
    for i, base_colors in enumerate(base_colors):
        site_colors = create_site_variants(base_colors)
        diagnostic_colors[f"{diagnoses[i]}_{medical_sites[0]}"] = site_colors[0]
        diagnostic_colors[f"{diagnoses[i]}_{medical_sites[1]}"] = site_colors[1]
    return diagnostic_colors



def plot_piechart(data: pd.DataFrame, main):
    if main:
        counter = Counter(data["Histo Diagnose_Code"])
        print(counter)
        occurences = list(counter.values())
        labels = list(counter.keys())
        custom_palette = {"MF": "#ff5b57",
                          "Psoriasis": "#39476e",
                          "Eczema": "#39ac5e",
                          "Eczema_Pso": "#00858a"}
        colors = [custom_palette[disease] for disease in counter.keys()]
        path = "/Figures/Figure3/pie_chart_qpcr_train_main.svg"
        # Adjustment to get the brown ring
        mf_mask = [label == "MF" for label in labels]
        non_mf_total = sum(occ for i, occ in enumerate(occurences) if not mf_mask[i])
        if any(mf_mask):
            mf_size = occurences[mf_mask.index(True)]
            outer_sizes = [mf_size, non_mf_total]
            outer_colors = ["white", "#8f8071"]
        else:
            outer_sizes = [non_mf_total]
            outer_colors = ["#8f8071"]
    else:
        combination = data["Histo Diagnose_Code"].str.cat(data["Zentrum"], sep="_")
        counter = Counter(combination)
        print(counter)
        custom_palette = create_color_palette()
        def get_diagnosis_for_sorting(label):
            # Split by underscore and take all parts except the last (which is the site)
            parts = label.split('_')
            return '_'.join(parts[:-1])

        sorted_items = sorted(counter.items(), key=lambda x: get_diagnosis_for_sorting(x[0]))
        occurences = [item[1] for item in sorted_items]
        labels = [item[0] for item in sorted_items]
        colors = [custom_palette[disease] for disease in labels]
        path = "/Figures/Figure3/pie_chart_qpcr_train_supplements.pdf"

    plt.figure(figsize=(2, 2))
    if main:
        plt.pie(outer_sizes, colors=outer_colors, radius=2.6,
                wedgeprops=dict(width=0.6, edgecolor=None),
                startangle=90
                )
    plt.pie(occurences, colors=colors, autopct=None, radius=2.01,
            wedgeprops=dict(width=0.6, edgecolor=None),
            startangle=90)

    center_circle = plt.Circle((0, 0), 1.4, color='black', fc='white', linewidth=0)
    fig = plt.gcf()
    fig.gca().add_artist(center_circle)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()



def apply_mannwhitneyutest(data:pd.DataFrame, pairs: list)-> Tuple[list, list]:
    p_values = []
    for pair in pairs:
        group1, group2 = pair
        gene1, diag1 = group1
        gene2, diag2 = group2
        group1_data = data.loc[(data["gene"] == gene1) & (data["diag"] == diag1), "expression"].dropna().values
        group2_data = data.loc[(data["gene"] == gene2) & (data["diag"] == diag2), "expression"].dropna().values
        _, pval = mannwhitneyu(group1_data, group2_data, alternative="two-sided")
        p_values.append(pval)
    padjs = fdrcorrection(p_values)[1]
    return p_values, padjs

def create_expression_boxplot(data, supplements, reference_gene: str = "TBP"):
    if supplements:
        cols = ["Histo Diagnose_Code","Zentrum", "sampleID", "HOMER1_run1", "HOMER1_run2", "LCK_run1", "LCK_run2",
                "RNF213_run1", "RNF213_run2", "SERBP1_run1", "SERBP1_run2", "ZC3H12D_run1", "ZC3H12D_run2", "BTN3A1_run1", "BTN3A1_run2",
                "TBP_run1", "TBP_run2", "SDHAF_run1", "SDHAF_run2", "TBP_2_run1", "TBP_2_run2", "SDHAF_2_run1", "SDHAF_2_run2"]
        data_filtered = data.loc[:, cols]
        if reference_gene == "TBP":
            faulty_samples = quality_control_qpcr(data_filtered, genes=["HOMER1", "LCK", "RNF213", "SERBP1", "ZC3H12D", "BTN3A1", "TBP", "TBP_2"], mode="train")
            count_data = compute_geometric_mean(data_filtered, genes=["HOMER1", "LCK", "RNF213", "SERBP1", "ZC3H12D", "BTN3A1", "TBP", "TBP_2"])
        elif reference_gene == "SDHAF":
            faulty_samples = quality_control_qpcr(data_filtered, genes=["HOMER1", "LCK", "RNF213", "SERBP1", "ZC3H12D", "BTN3A1", "SDHAF", "SDHAF_2"], mode="train")
            count_data = compute_geometric_mean(data_filtered, genes=["HOMER1", "LCK", "RNF213", "SERBP1", "ZC3H12D", "BTN3A1", "SDHAF", "SDHAF_2"])
        count_data = count_data.drop(faulty_samples, axis=0)
        normalized_data = normalize_raw_counts(count_data, reference_genes=[reference_gene])
        diag = data_filtered.loc[:, ["Histo Diagnose_Code", "Zentrum", "sampleID"]]
        diag["combination"] = diag["Histo Diagnose_Code"].str.cat(diag["Zentrum"], sep="_")
        diag.set_index("sampleID", inplace=True)
        diag.drop(columns=["Histo Diagnose_Code", "Zentrum"], inplace=True)
        diag.rename(columns={"combination": "diag"}, inplace=True)
        data_merged = pd.merge(normalized_data, diag, left_index=True, right_index=True, how="inner")
        data_merged.columns = data_merged.columns.str.split('_').str[0]
        print(f"Data shape: {data_merged.shape}")
        order = ["LCK", "HOMER1", "ZC3H12D", "SERBP1", "BTN3A1", "RNF213"]
        hue_order = ["MF_Pilsen", "MF_UKF", "Eczema_Pilsen", "Eczema_UKF", "Psoriasis_Pilsen",
                     "Psoriasis_UKF", "Eczema_Pso_Pilsen", "Eczema_Pso_UKF", ]
        custom_palette = create_color_palette()
        melted = data_merged.melt(id_vars=["diag"], var_name="gene", value_name="expression")
        melted["expression"] = pd.to_numeric(melted["expression"], errors='coerce')
        melted["expression"] = (-1) * melted["expression"]
        size = (3.5, 3)
        fig, ax = plt.subplots(figsize=size)
        sns.boxplot(ax=ax, data=melted, linewidth=0.5, fliersize=1.5, x="gene", y="expression", hue="diag", order=order, hue_order=hue_order,
                    palette=custom_palette)
        ax.get_legend().remove()
    else:
        cols = ["wahre Diagnose", "sampleID", "HOMER1_run1", "HOMER1_run2", "LCK_run1", "LCK_run2",
                "RNF213_run1", "RNF213_run2", "SERBP1_run1", "SERBP1_run2", "ZC3H12D_run1", "ZC3H12D_run2", "BTN3A1_run1", "BTN3A1_run2",
                "TBP_run1", "TBP_run2", "SDHAF_run1", "SDHAF_run2", "TBP_2_run1", "TBP_2_run2", "SDHAF_2_run1", "SDHAF_2_run2"]
        data_filtered = data.loc[:, cols]
        if reference_gene == "TBP":
            faulty_samples = quality_control_qpcr(data_filtered, genes=["HOMER1", "LCK", "RNF213", "SERBP1", "ZC3H12D", "BTN3A1", "TBP", "TBP_2"], mode="train")
            count_data = compute_geometric_mean(data_filtered, genes=["HOMER1", "LCK", "RNF213", "SERBP1", "ZC3H12D", "BTN3A1", "TBP", "TBP_2"])
        elif reference_gene == "SDHAF":
            faulty_samples = quality_control_qpcr(data_filtered, genes=["HOMER1", "LCK", "RNF213", "SERBP1", "ZC3H12D", "BTN3A1", "SDHAF", "SDHAF_2"], mode="train")
            count_data = compute_geometric_mean(data_filtered, genes=["HOMER1", "LCK", "RNF213", "SERBP1", "ZC3H12D", "BTN3A1", "SDHAF", "SDHAF_2"])
        count_data = count_data.drop(faulty_samples, axis=0)
        normalized_data = normalize_raw_counts(count_data, reference_genes=[reference_gene])
        diag = data_filtered.loc[:, ["wahre Diagnose", "sampleID"]]
        diag["wahre Diagnose"] = diag["wahre Diagnose"].str.replace("SS_", "")
        diag.set_index("sampleID", inplace=True)
        diag.rename(columns={"wahre Diagnose": "diag"}, inplace=True)
        data_merged = pd.merge(normalized_data, diag, left_index=True, right_index=True, how="inner")
        data_merged.columns = data_merged.columns.str.split('_').str[0]
        print(f"Data shape: {data_merged.shape}")
        # assert data_merged.shape[0] == 166
        custom_palette = {"MF": "#ff5b57",
                          "Eczema_Pso": "#8f8071"}
        melted = data_merged.melt(id_vars=["diag"], var_name="gene", value_name="expression")
        melted["expression"] = pd.to_numeric(melted["expression"], errors='coerce')
        melted["expression"] = (-1) * melted["expression"]
        size = (3., 2.5)
        hue_order = ["Eczema_Pso", "MF"]
        pairs = [
            [("LCK", "Eczema_Pso"), ("LCK", "MF")],
            [("HOMER1", "Eczema_Pso"), ("HOMER1", "MF")],
            [("SERBP1", "Eczema_Pso"), ("SERBP1", "MF")],
            [("BTN3A1", "Eczema_Pso"), ("BTN3A1", "MF")],
            [("ZC3H12D", "Eczema_Pso"), ("ZC3H12D", "MF")],
            [("RNF213", "Eczema_Pso"), ("RNF213", "MF")],
        ]
        pvals, padjs = apply_mannwhitneyutest(melted, pairs)
        # Create a list of (gene, padj) tuples and sort by padj
        sorted_indizes = np.argsort(padjs)
        sorted_pairs = [pairs[i] for i in sorted_indizes]
        sorted_padjs = np.array(padjs)[sorted_indizes]
        sorted_order = [pair[0][0] for pair in sorted_pairs]
        print(f"Sorted padjs: {sorted_padjs}")
        fig, ax = plt.subplots(figsize=size)
        annotator_params = dict(order=sorted_order,
                                hue_order=hue_order,
                                data=melted,
                                linewidth=0.5,
                                fliersize=1.5,
                                x="gene",
                                y="expression",
                                hue="diag",
                                palette=custom_palette)
        sns.boxplot(ax=ax, **annotator_params)
        # annotator = Annotator(ax, sorted_pairs, **annotator_params)
        # annotator.configure(test=None, fontsize=6)
        # annotator.set_pvalues_and_annotate(sorted_padjs)
        ax.get_legend().remove()
    plt.xticks(rotation=45, fontsize=6)
    plt.yticks(fontsize=6)
    plt.ylabel(rf'-$\Delta$C$_T$ ({reference_gene})', fontsize=6)
    plt.xlabel("Gene", fontsize=6)
    sns.despine()
    plt.grid(False)
    plt.tight_layout()
    if supplements:
        plt.savefig(
            f"/Figures/Figure3/Expression_boxplot_{reference_gene}_supplements.svg",
            bbox_inches="tight")
    else:
        plt.savefig(
            f"/Figures/Figure3/Expression_boxplot_{reference_gene}.svg",
            bbox_inches="tight")
    plt.close()


def load_qpcr_data() -> pd.DataFrame:
    path = "/Users/martin.meinel/Desktop/Projects/Eyerich Projects/Natalie/Classifier/Validation data/Test data/Paper_sheet.xlsx"
    return pd.read_excel(path)

def scatter_plot_training(data: pd.DataFrame, housekeeper, params, highlight=False):
    feature_cols = data.columns.str.endswith(housekeeper)
    xcol, ycol = data.columns[feature_cols].tolist()[0], data.columns[feature_cols].tolist()[1]

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
        y_scaled = (1/params['coefficients'][ycol]) * (((offset - b)/a) - params['intercept'] - params['coefficients'][xcol] * x_scaled)
        y = y_scaled * params['scaler_scale'][ycol] + params['scaler_mean'][ycol]
        return y

    def calculate_distance_to_boundary(x, y):
        """Calculate perpendicular distance from point to decision boundary"""
        x_scaled, y_scaled = scale_point(x, y)

        # Get coefficients
        if "a" in params.keys() & "b" in params.keys():
            a, b = params["a"], params["b"]
        else:
            a, b = 1, 0

        threshold = params.get("decision_threshold", 0.5)
        offset = logit(threshold)

        # Decision boundary: a * (b0 + b1*x + b2*y) + b = offset
        # Rearranged: (a*b1)*x + (a*b2)*y + (a*b0 + b - offset) = 0
        A = a * params['coefficients'][xcol]
        B = a * params['coefficients'][ycol]
        C = a * params['intercept'] + b - offset

        # Distance = |Ax + By + C| / sqrt(A² + B²)
        distance = abs(A * x_scaled + B * y_scaled + C) / np.sqrt(A ** 2 + B ** 2)

        # Determine which side of boundary (positive = MF side, negative = Eczema_Pso side)
        side = np.sign(A * x_scaled + B * y_scaled + C)

        return distance, side
    outlier_samples =[]
    if highlight:
        data_copy = data.copy()
        distances = []
        sides = []
        for idx, row in data_copy.iterrows():
            dist, side = calculate_distance_to_boundary(row[xcol], row[ycol])
            distances.append(dist)
            sides.append(side)
        data_copy["distance"] = distances
        data_copy["side"] = sides

        misclassified = (
                ((data_copy['wahre Diagnose'] == 'MF') & (data_copy['side'] < 0)) |
                ((data_copy['wahre Diagnose'] == 'Eczema_Pso') & (data_copy['side'] > 0))
        )

        # Define "far away" threshold (e.g., top quartile of distances among misclassified)
        distance_threshold = data_copy[misclassified]['distance'].quantile(0.75)

        # Select samples that are both misclassified and far from boundary
        outlier_samples = data_copy[misclassified & (data_copy['distance'] >= distance_threshold)]

    y_min, y_max = data[ycol].min() - 1, data[ycol].max() + 1
    x_min, x_max = data[xcol].min(), data[xcol].max()
    x_plot = np.linspace(x_min, x_max, 100)
    y_plot = [get_decision_boundary_y(x) for x in x_plot]
    y_plot_40 = [get_decision_boundary_y(x, threshold=0.4) for x in x_plot]
    y_plot_60 = [get_decision_boundary_y(x, threshold=0.6) for x in x_plot]

    plt.figure(figsize=(2, 2))
    sns.scatterplot(data=data, x=xcol, y=ycol, hue="wahre Diagnose", s=7.5, palette={"MF":"#ff5b57",
                                                                                     "Eczema_Pso":"#ac8a66"})
    if len(outlier_samples) > 0:
        plt.scatter(outlier_samples[xcol], outlier_samples[ycol],
                    s=30, facecolors='none', edgecolors='black', linewidths=1.5,
                    label=f'Misclassified outliers (n={len(outlier_samples)})')

        # Add sample IDs as text annotations
        for idx, row in outlier_samples.iterrows():
            plt.annotate(row['sampleID'],
                         (row[xcol], row[ycol]),
                         xytext=(5, 5), textcoords='offset points',
                         fontsize=4, ha='left', va='bottom',
                         bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))

    plt.plot(x_plot, y_plot, '--', color='black', alpha=0.5, label="Calibrated boundary 0.5")
    plt.fill_between(x_plot, y_plot_40, y_plot_60, color='lightgrey', alpha=0.35, label="Rejection area 0.4-0.6")
    plt.ylim(y_min, y_max)
    plt.tick_params(axis='both', labelsize=6)
    plt.xlabel(plt.gca().get_xlabel(), fontsize=6)
    plt.ylabel(plt.gca().get_ylabel(), fontsize=6)
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=6, frameon=False)
    plt.legend().remove()
    plt.tight_layout()
    sns.despine()
    plt.grid(False)
    feature1 = xcol.split("_")[0]
    feature2 = ycol.split("_")[0]
    output_path = "/Figures/Figure3"
    if highlight:
        plt.savefig(os.path.join(output_path,
                                 f"{feature1}_{feature2}_{housekeeper}_highlighted_scatter_plot.svg"),
                    dpi=300, bbox_inches='tight')
    else:
        plt.savefig(os.path.join(output_path,
                                 f"{feature1}_{feature2}_{housekeeper}_scatter_plot.svg"),
                    dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


def load_params(path: str):
    with open(path, "r") as f:
        params = json.load(f)
    return params


def plot_roc_curve(training_data):
    training_data["wahre Diagnose"] = training_data["wahre Diagnose"].str.replace("SS_", "")
    probabilities = training_data["HOMER1_TBP_LCK_TBP_HOMER1_SDHAF_LCK_SDHAF_baseline_probabilities_calibrated"].values
    labels = training_data["wahre Diagnose"].map(lambda x: 1 if x == "MF" else 0).values
    keep = np.max(np.column_stack((probabilities, 1 - probabilities)), axis=1) > 0.6
    probs = probabilities[keep]
    labels = labels[keep]
    plt.figure(figsize=(2, 2))
    fpr, tpr, thresholds = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)
    plt.plot(fpr, tpr, color='blue', alpha=0.7, label=f"AUC = {auc:.2f}")
    plt.plot([0, 1], [0, 1], color='grey', linestyle='--', label="Random")
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=6)
    plt.ylabel('True Positive Rate', fontsize=6)
    plt.tick_params(axis='both', labelsize=6)
    plt.legend(fontsize=6)
    plt.tight_layout()
    sns.despine()
    plt.grid(False)
    plt.savefig(
        "/Figures/Figure3/Training_data_roc.svg",
        dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


def create_probability_plot(training_data, supplements=False):
    # Probabilities
    training_data["wahre Diagnose"] = training_data["wahre Diagnose"].str.replace("SS_", "")
    if supplements:
        plt.figure(figsize=(2, 2))
        sns.stripplot(x="Histo Diagnose_Code",y="HOMER1_TBP_LCK_TBP_HOMER1_SDHAF_LCK_SDHAF_baseline_probabilities_calibrated",
                      hue="Histo Diagnose_Code", jitter=True, data=training_data,
                      order=["Eczema_Pso","Psoriasis", "Eczema", "MF"], size=3,
                      palette={"MF": "#ff5b57",
                          "Psoriasis": "#39476e",
                          "Eczema": "#39ac5e",
                          "Eczema_Pso": "#00858a"})
        sup="supplements"
    else:
        plt.figure(figsize=(1.5, 2))
        sns.violinplot(data=training_data, x="wahre Diagnose", y="HOMER1_TBP_LCK_TBP_HOMER1_SDHAF_LCK_SDHAF_baseline_probabilities_calibrated",
                       hue="wahre Diagnose", order=["Eczema_Pso", "MF"], fill=True, alpha=0.3,
                       linewidth=0.5, cut=0,
                       palette={"MF": "#ff5b57", "Eczema_Pso": "#ac8a66"})
        sns.stripplot(x="wahre Diagnose",
                      y="HOMER1_TBP_LCK_TBP_HOMER1_SDHAF_LCK_SDHAF_baseline_probabilities_calibrated",
                      hue="wahre Diagnose", order=["Eczema_Pso", "MF"],
                      jitter=True, data=training_data, palette={"MF": "#ff5b57", "Eczema_Pso": "#ac8a66"}, size=2)
        sup="main"
    plt.axhspan(0.4, 0.6, alpha=0.35, color="lightgrey", label="Rejection area")
    sns.despine()
    plt.grid(False)
    plt.xlabel("Ground truth label", fontsize=6)
    plt.ylabel("Probability for MF", fontsize=6)
    plt.tick_params(axis='both', labelsize=6)
    plt.tight_layout()
    plt.savefig(f"/Figures/Figure3/Probability_plot_{sup}.svg",
                bbox_inches='tight')
    plt.show()
    plt.close()


def create_predictions_pie(training_data):
    plt.figure(figsize=(1.5, 1.5))
    predicted_probabilities = training_data.loc[:,"HOMER1_TBP_LCK_TBP_HOMER1_SDHAF_LCK_SDHAF_baseline_probabilities_calibrated"]
    mf_predictions = (predicted_probabilities > 0.6).sum()
    eczema_predictions = (predicted_probabilities < 0.4).sum()
    rejected = ((predicted_probabilities > 0.4) & (predicted_probabilities < 0.6)).sum()
    sizes = [mf_predictions, eczema_predictions, rejected]
    print(mf_predictions, eczema_predictions, rejected)
    colors = ["#ff5b57", "#ac8a66", "lightgrey"]
    plt.pie(sizes, colors=colors, startangle=90)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(
        "/Figures/Figure3/Predictions_pie.svg",
    )
    plt.close()


def create_metrics_table(training_data):
    training_data["wahre Diagnose"] = training_data["wahre Diagnose"].str.replace("SS_", "")
    probabilities = training_data["HOMER1_TBP_LCK_TBP_HOMER1_SDHAF_LCK_SDHAF_baseline_probabilities_calibrated"].values
    predictions = training_data["HOMER1_TBP_LCK_TBP_HOMER1_SDHAF_LCK_SDHAF_baseline_predictions_calibrated"].map(lambda x: 1 if x == "MF" else 0).values
    labels = training_data["wahre Diagnose"].map(lambda x: 1 if x == "MF" else 0).values
    wrong_predictions = (predictions != labels).sum()
    keep = np.max(np.column_stack((probabilities, 1 - probabilities)), axis=1) > 0.6
    rejected = ~keep
    rejected_wrong_predictions = (predictions[rejected] != labels[rejected]).sum()
    print(f"Wrong predictions: {wrong_predictions}")
    print(f"Rejected wrong predictions")
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
    all_qpcr_data = load_qpcr_data()
    training_data = all_qpcr_data.iloc[1:167]
    training_data.loc[training_data["sampleID"].str.contains("=", regex=False), "sampleID"] = \
    training_data.loc[training_data["sampleID"].str.contains("=", regex=False), "sampleID"].str.split("=").str[1]
    training_data.loc[training_data["Zentrum"] == "Freiburg", "Zentrum"] = "UKF"
    training_data.loc[training_data["Histo Diagnose_Code"] == "Psoriasis/Eczema", "Histo Diagnose_Code"] = "Eczema_Pso"
    if training_data.shape[0] != 166:
        raise ValueError("Training data has wrong number of rows")
    subplot = "g"
    if subplot == "a":
        plot_piechart(training_data, main=True)
    elif subplot == "b":
        create_expression_boxplot(training_data, supplements=False, reference_gene="TBP")
        create_expression_boxplot(training_data, supplements=False, reference_gene="SDHAF")
    elif subplot == "c":
        housekeeper = "TBP"
        cols = ["sampleID", "wahre Diagnose", "LCK_run1", "LCK_run2", "HOMER1_run1", "HOMER1_run2",
                "SDHAF_run1", "SDHAF_run2", "TBP_run1", "TBP_run2"]
        training_data = training_data.loc[:, cols]
        training_data["wahre Diagnose"] = training_data["wahre Diagnose"].str.replace("SS_", "")
        # Remove faulty samples
        faulty_samples = quality_control_qpcr(training_data, genes=["HOMER1", "LCK", "SDHAF", "TBP"], mode="train")
        # Aggregate gene expression
        aggregated_data = compute_geometric_mean(training_data, genes=["HOMER1", "LCK", "SDHAF", "TBP"])
        aggregated_data = aggregated_data.drop(faulty_samples, axis=0)
        # normalize raw counts
        normalized_data  = normalize_raw_counts(aggregated_data, reference_genes=[housekeeper])
        all_data = pd.merge(normalized_data, training_data.loc[:, ["wahre Diagnose", "sampleID"]], left_index=True, right_on="sampleID", how="inner")
        params = load_params("/params/HOMER1_TBP_LCK_TBP_HOMER1_SDHAF_LCK_SDHAF_baseline_calibrated_params.json")
        scatter_plot_training(all_data, housekeeper=housekeeper,params=params, highlight=False)
    elif subplot == "d":
        plot_roc_curve(training_data)
    elif subplot == "e":
        create_probability_plot(training_data, supplements=False)
    elif subplot == "f":
        create_predictions_pie(training_data)
    elif subplot == "g":
        create_metrics_table(training_data)
    else:
        raise ValueError("Invalid subplot")

    # Create
    # Load qPCR training data from Freiburg and Pilsen
    # plot piechart
    # b qPCR boxplot
    # Turkis 00858a
    # Grey 8f8071
    # create_robustness_plot()
    # create_augmentation_plot()

if __name__ == "__main__":
    main()