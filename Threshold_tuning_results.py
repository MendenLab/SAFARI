import logging
import os
import pdb
import hydra
from omegaconf import DictConfig
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from Validation_run import read_features

sns.set_style("white")


def collect_data(genes: str, augmented_training, scorer_name, modes, disease=None, reference_gene = "both"):
    if disease is not None:
        base_path = "/lustre/groups/cbm01/datasets/martin.meinel/Safari/Robustness_study/relative/delta/augmentation_noise"
        base_path = os.path.join(base_path, disease, "mixed_data")
    else:
        if augmented_training is not None:
            base_path = "/lustre/groups/cbm01/datasets/martin.meinel/Safari/Robustness_study/relative/delta/augmentation_noise/mixed_data"
        else:
            base_path = "/lustre/groups/cbm01/datasets/martin.meinel/Safari/Robustness_study/relative/delta/mixed_data"
    augmentation_strengths = [0.5]
    relative_shifts = [-1, 0, 1]
    data = []
    for mode in modes:
        for augmentation_strength in augmentation_strengths:
            for shift in relative_shifts:
                dir = os.path.join(base_path, mode)
                if augmented_training is not None:
                    file = os.path.join(dir, f"logistic_linear_{reference_gene}_{genes}_{shift}_{augmentation_strength}_{scorer_name}.h5")
                else:
                    file = os.path.join(dir, f"logistic_linear_{reference_gene}_{genes}_{shift}_{scorer_name}.h5")
                f = pd.read_hdf(file, key="results")
                f["model"] = f"{mode}_{shift}_{augmentation_strength}"
                data.append(f)
    results = pd.concat(data)
    return results

def create_latex_table(data: pd.DataFrame, genes: str, scorer_name, disease, augmented_training, reference_gene = "both"):
    genes = genes.replace("_", " ")
    data = data.melt(id_vars=['model'], var_name='Metric', value_name='Value')
    metrics = data.Metric.unique()
    initialize_table_string = (
                    " \\begin{sidewaystable} \n\\centering \n\\resizebox{\\textwidth}{!}{\n\\begin{tabular}{@{}lcc" + "c" * len(
                metrics) + "@{}} \n\\toprule \n Training & CT shift & Augmentation strength ")

    for metric in metrics:
        if metric == "f1_score":
            metric = "F1-score"
        if metric == "f1_score_weighted":
            metric = "Weighted F1-score"
        initialize_table_string += f" & {metric} "
    initialize_table_string += (" \\\\ \n\\midrule \n")

    latex_string = initialize_table_string

    grouped = data.groupby(['model', 'Metric'])
    means = grouped['Value'].mean().unstack()
    stds = grouped['Value'].std().unstack()

    max_values = {metric: means[metric].max() for metric in metrics if metric != "fdr"}
    min_values = {metric: means[metric].min() for metric in metrics if metric == "fdr"}

    for model in pd.unique(data['model']):
        if model.startswith("baseline"):
            model_parts = model.split('_')
            training = model_parts[0]
            ct_shift = model_parts[1]
            augmentation = model_parts[2]
        else:
            model_parts = model.split('_')
            training = " ".join(model_parts[:2])
            ct_shift = model_parts[2]
            augmentation = model_parts[3]

        latex_string += f"{training} & {ct_shift} & {augmentation}"
        for metric in metrics:
            mean = means.loc[model, metric] * 100
            std = stds.loc[model, metric] * 100
            formatted_value = f"{mean:.1f} $\\pm$ {std:.1f}\\%"

            if metric == "fdr":
                if np.isclose(mean, min_values[metric] * 100):
                    formatted_value = f"\\textbf{{{formatted_value}}}"
            else:
                # Check if this is the max value in the column
                if np.isclose(mean, max_values[metric] * 100):
                    formatted_value = f"\\textbf{{{formatted_value}}}"

            latex_string += f" & {formatted_value}"

        latex_string += " \\\\ \n"
    if disease is not None:
        end_string = f"\\bottomrule \n\\end{{tabular}}}}  \\caption{{Results for {genes} with augmented training and optimization {scorer_name} using Mf and {disease}}}  \n\\end{{sidewaystable}}"
    else:
        if augmented_training is not None:
            end_string = f"\\bottomrule \n\\end{{tabular}}}}  \\caption{{Results for {genes} with augmented training and optimization {scorer_name} with {reference_gene}}}  \n\\end{{sidewaystable}}"
        else:
            end_string = f"\\bottomrule \n\\end{{tabular}}}}  \\caption{{Results for {genes} with {reference_gene}}}  \n\\end{{sidewaystable}}"
    latex_string += end_string
    print(latex_string)


def create_boxplot(data: pd.DataFrame, genes: str, augmented_training):
    data_melted = data.melt(id_vars=["model"], var_name='Metric', value_name='Value')
    data_melted = data_melted.loc[data_melted['Metric'] != "fdr", :]
    plt.figure(figsize=(6, 3))
    ax = sns.boxplot(data=data_melted, x="Metric", y="Value", hue="model", linewidth=0.5, fliersize=1.5)
    y_min = data_melted["Value"].min()
    sns.despine(left=True, bottom=True)
    # sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.legend(loc='upper left', bbox_to_anchor=(1.2, 1.2), fontsize=4)
    plt.ylim(y_min - 0.1, 1.0)
    plt.xticks(rotation=45, ha='right')
    ax.tick_params(axis='x', labelsize=6)  # x-ticks font size
    ax.tick_params(axis='y', labelsize=6)  # y-ticks font size
    # Set font sizes for labels
    ax.set_xlabel('Metric', fontsize=6)  # x-label font size
    ax.set_ylabel('Performance', fontsize=6)  # y-label font size
    plt.grid(False)
    plt.box(False)
    plt.tight_layout()
    if augmented_training is not None:
        plt.savefig(f"/lustre/groups/cbm01/workspace/martin.meinel/Safari/Classifier/figures/Training_procedures/Mixed_data_augmented_training_threshold_tuning_boxplot_{genes}_f1.svg", bbox_inches="tight")
    else:
        plt.savefig(f"/lustre/groups/cbm01/workspace/martin.meinel/Safari/Classifier/figures/Training_procedures/Mixed_data_original_training_threshold_tuning_boxplot_{genes}.svg", bbox_inches="tight")
    plt.close()

@hydra.main(version_base=None, config_path="conf", config_name="validation")
def main(cfg: DictConfig):
    log = logging.getLogger(__name__)
    genes = read_features(cfg.paths.featureFile)
    genes = sorted(genes)
    genes = "_".join(genes)
    if cfg.debug:
        pdb.set_trace()
    log.info(f"Collect results for genes: {genes}")
    if cfg.disease_subset is not None:
        log.info(f"Disease comparison for: {cfg.disease_subset}")
    log.info(f"Results for reference genes: {cfg.reference_gene}")
    if len(cfg.reference_gene) > 1:
        reference_gene = "both"
    else:
        reference_gene = cfg.reference_gene[0]
    log.info(f"Tested modes: {cfg.mode}")
    data = collect_data(genes=genes, augmented_training=cfg.train_augmentation, modes=cfg.mode, scorer_name=cfg.scorer, disease=cfg.disease_subset, reference_gene = reference_gene)
    create_boxplot(data, genes=genes, augmented_training = cfg.train_augmentation)
    create_latex_table(data, genes=genes, scorer_name=cfg.scorer, disease=cfg.disease_subset, augmented_training=cfg.train_augmentation, reference_gene = reference_gene)

if __name__ == "__main__":
    main()