import argparse
import os.path
from collections import Counter
from sys import platform
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
import rpy2.robjects as robjects
import pandas as pd
import plotly.express as px

from R_helpers import RToPandas, edgeRNormalization
import seaborn as sns
import umap

sns.set_style("white")
edgeR = importr("edgeR")
base = importr("base")
pandas2ri.activate()
# OVERALL colors
# https://huemint.com/illustration-2/#palette=dee7ef-38466e-4270a7-ff5a56-ac8a66-39ac5e-b2d7a8-eea488
custom_palette = {"cutaneous lymphoma": "#ff5b57",
                  "cutaneous_lymphoma_NL": "#eea488",
                  "psoriasis": "#39476e",
                  "psoriasis_NL": "#dee7ef",
                  "eczema": "#39ac5e",
                  "eczema_NL": "#b2d7a8",
                  "non-lesional": "#ac8a66"}


# Mixed eczema Psoriasis color: #397A66

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding', dest='embedding', type=str, default="umap")
    return parser.parse_args()


def process_colData(colData: pd.DataFrame) -> pd.DataFrame:
    colData.batchID = colData.batch.str.extract('(\d+)')
    return colData


def create_anndata() -> sc.AnnData:
    readRDS = robjects.r['readRDS']
    summarizedExperiment = importr("SummarizedExperiment")
    base = importr("base")
    # Read count Matrix files -------------------------------------------------------
    if platform == "darwin":
        countData = readRDS(
            "/Users/martin.meinel/Desktop/Projects/Eyerich Projects/Therapy "
            "Response/dds_highQual_Sexandbatchcorrected_v04.rds")
    else:
        countData = readRDS(os.path.join("/lustre/groups/cbm01/datasets/martin.meinel/Safari",
                                         "dds_highQual_Sexandbatchcorrected_v04.rds"))
    counts_R = summarizedExperiment.assay(countData)
    rowData_r = summarizedExperiment.rowData(countData)
    colData_r = summarizedExperiment.colData(countData)
    rowData_r = base.data_frame(rowData_r)
    colData_r = base.data_frame(colData_r)
    # Transformation to python
    colData = RToPandas(colData_r)
    colData = process_colData(colData)
    rowData = RToPandas(rowData_r)
    counts = RToPandas(counts_R)
    counts = pd.DataFrame(counts, columns=colData["sampleID"])
    # use ensembl_ids to not have duplicated indices
    counts = counts.set_axis(rowData["ensembl_id"], axis="index")
    anndata = sc.AnnData(X=counts.T, var=rowData, obs=colData, dtype=np.dtype(int))
    return anndata


def normalize_data(data: sc.AnnData, mode="scanpy") -> sc.AnnData:
    """
    Normalizes the data either with best practice single-cell normalization or TMM normalization

    Parameters
    ----------
    data: AnnData object
    mode: scanpy for single-cell normalization, otherwise TMM

    Returns: AnnData Object with new normalized counts in X.
    -------

    """
    if mode == "scanpy":
        # single-cell normalization as here https://www.biorxiv.org/content/10.1101/2022.05.06.490859v1
        sc.pp.log1p(data)
    else:
        # TMM
        counts = data.to_df()
        normalized = edgeRNormalization(counts=counts.T, log=True)
        data.X = normalized.T
    return data


def filter_samples(data: sc.AnnData, diag) -> sc.AnnData:
    """
    Filter samples
    Parameters
    ----------
    data
    diag

    Returns
    -------

    """
    adata = data.copy()
    adata = adata[adata.obs.diag.isin(diag)]
    return adata


def filter_genes(data: sc.AnnData, genes) -> sc.AnnData:
    """
    Filter data for genes
    Parameters
    ----------
    data: containing
    genes: list of genes

    Returns: Anndata object with only selected genes
    -------

    """
    data = data[:, data.var["Gene_name"].isin(genes)]
    return data


def correct_batches(data: sc.AnnData) -> sc.AnnData:
    sc.pp.combat(data, key="batchID")
    sc.pp.combat(data, key="Sex.x")
    return data


def plot_embedding(data: sc.AnnData, mode: str, label: str, interactive=False):
    sc.pp.highly_variable_genes(data, flavor="cell_ranger", n_top_genes=5000)
    sc.tl.pca(data, use_highly_variable=True)
    if mode == "pca":
        sc.pl.pca(data, color="diag", size=100, frameon=False, palette=custom_palette)
    else:
        sc.pp.neighbors(data, n_pcs=30)
        sc.tl.umap(data)
        umap_data = data.obsm["X_umap"]
        if interactive:
            df = pd.DataFrame(data=dict(umap1=umap_data[:, 0], umap2=umap_data[:, 1], diag=data.obs.diag, samples=data.obs.sampleID))
            fig = px.scatter(df, x="umap1", y="umap2", color="diag", hover_data=["samples"],
                                 color_discrete_map=custom_palette)
            fig.show()
        else:
            plt.figure()
            g = sns.jointplot(x=umap_data[:, 0], y=umap_data[:, 1], hue=data.obs.diag,
                              palette=custom_palette, legend=False,
                              marginal_kws={"common_norm": False})
            g.ax_joint.set_axis_off()
            # plt.axis('off')
            if platform == "darwin":
                base_path = "/Users/martin.meinel/Desktop/Projects/Eyerich Projects/Natalie/Classifier/Final_results/Figures/Figure1"
                path = os.path.join(base_path, f"{mode}_tmm_4000_{label}_jointplot.pdf")
                # plt.savefig(path, format="pdf", bbox_inches="tight")
                plt.show()
            else:
                plt.savefig(
                    f"/lustre/groups/cbm01/workspace/martin.meinel/Safari/Classifier/figures/Figure1/{mode}_{label}.pdf",
                    format="pdf", bbox_inches="tight")


def split_non_lesionals(data: sc.AnnData) -> sc.AnnData:
    obs = data.obs.copy()
    obs['diag'] = obs['diag'].astype('category')
    # Add new categories
    new_categories = ["eczema_NL", "psoriasis_NL", "cutaneous_lymphoma_NL"]
    obs['diag'] = obs['diag'].cat.add_categories(new_categories)
    obs.loc[(obs["diag"] == "non-lesional") & (obs["healthysamp_diag"] == "eczema"), "diag"] = "eczema_NL"
    obs.loc[(obs["diag"] == "non-lesional") & (obs["healthysamp_diag"] == "psoriasis"), "diag"] = "psoriasis_NL"
    obs.loc[(obs["diag"] == "non-lesional") & (
            obs["healthysamp_diag"] == "cutaneous lymphoma"), "diag"] = "cutaneous_lymphoma_NL"
    data.obs = obs
    return data


def load_clinical_attributes():
    clinical_attributes = pd.read_excel(
        "/Users/martin.meinel/Desktop/Projects/Eyerich Projects/Natalie/Classifier/Normed_encoded_imputed_Clinical_attributes.xlsx")
    clinical_attributes.set_axis(clinical_attributes["Helmholtz_identifyer"].values.tolist(), axis="index",
                                 inplace=True)
    diag = pd.Series(data=clinical_attributes["diag"].values.tolist(),
                     index=clinical_attributes["Helmholtz_identifyer"].values.tolist())
    clinical_attributes.drop(["diag", "Helmholtz_identifyer"], axis=1, inplace=True)
    return clinical_attributes, diag


def compute_umap(data: pd.DataFrame):
    reducer = umap.UMAP(random_state=1)
    embedding = reducer.fit_transform(data)
    return embedding


def plot_clinical_attributes(umap_data, diag, samples, interactive=False):
    if interactive:
        df = pd.DataFrame(data=dict(umap1=umap_data[:, 0], umap2=umap_data[:, 1], diag=diag, samples=samples))
        fig = px.scatter(df, x="umap1", y="umap2", color="diag", hover_data=["samples"],
                         color_discrete_map=custom_palette)
        fig.show()
    else:
        plt.figure()
        g = sns.jointplot(x=umap_data[:, 0], y=umap_data[:, 1], hue=diag, palette=custom_palette, legend=False,
                          marginal_kws={"common_norm": False})
        g.ax_joint.set_axis_off()
        plt.axis('off')
        plt.savefig(
            "/Users/martin.meinel/Desktop/Projects/Eyerich Projects/Natalie/Classifier/Final_results/Figures/Figure1/clinical_attributes_umap.pdf",
            format="pdf", bbox_inches="tight")
        plt.show()


def plot_barchart(orientation="vertical"):
    data = {
        "metric": ["Balanced Accuracy", "F1-score", "Sensitivity", "Specificity", "ROC AUC Score", "PR AUC Score"] * 2,
        "features": ["Gene expression"] * 6 + ["Clinical attributes"] * 6,
        "mean": [91.8, 66.9, 92.5, 91.1, 97.3, 75.1, 75.2, 44.0, 89.5, 60.9, 81.4, 44.4],
        "std": [2.2, 3.2, 0.9, 4.4, 1.0, 5.7, 3.5, 5.4, 3.6, 8.7, 3.9, 6.7]}
    df = pd.DataFrame(data)
    if orientation == "vertical":
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(x='metric', y='mean', hue='features', data=df, palette="pastel", ci=None)
        for i in range(len(df)):
            ax.errorbar(x=i % 6 + (i // 6) * 0.4 - 0.2,
                        y=df['mean'][i],
                        yerr=df['std'][i],
                        fmt='none',
                        capsize=5,
                        color='black')
        plt.xlabel("Metric")
        plt.yticks(np.arange(0, 110, 10))
        plt.box(False)
        plt.legend()
        plt.ylabel("Mean and standard deviation in % over 100 repetitions")
    else:
        plt.figure(figsize=(8, 8))
        ax = sns.barplot(x='mean', y='metric', hue='features', data=df,
                         palette='pastel', ci=None, orient='h')

        # Add error bars manually using the standard deviation
        for i, bar in enumerate(ax.patches):
            # Get the x and y position of each bar
            x = bar.get_width()
            y = bar.get_y() + bar.get_height() / 2
            # Get the corresponding standard deviation
            std_dev = df['std'][i]
            # Add error bars
            ax.errorbar(x=x, y=y, xerr=std_dev, fmt='none', capsize=5, color='black')
        plt.box(False)
        plt.xticks(np.arange(0, 110, 10))
    ax.get_legend().set_visible(False)
    plt.tight_layout()
    plt.savefig(
        f"/Users/martin.meinel/Desktop/Projects/Eyerich Projects/Natalie/Classifier/Final_results/Figures/Figure3/clincal_attributes_performance_barchart_{orientation}.pdf",
        bbox_inches="tight")
    plt.show()
    plt.close()

def plot_piechart(adata: sc.AnnData):
    colData = adata.obs
    counter = Counter(colData.diag)
    print(counter)
    occurences = list(counter.values())
    labels = list(counter.keys())
    colors = [custom_palette[disease] for disease in labels]
    # Adjustment to get the brown ring
    mf_mask = [label == "cutaneous lymphoma" for label in labels]
    non_mf_total = sum(occ for i, occ in enumerate(occurences) if not mf_mask[i])
    if any(mf_mask):
        mf_size = occurences[mf_mask.index(True)]
        outer_sizes = [non_mf_total, mf_size]
        outer_colors= ["#8f8071", "white"]
    else:
        outer_sizes = [non_mf_total]
        outer_colors = ["#8f8071"]

    plt.figure(figsize=(8, 8))
    plt.pie(outer_sizes, colors=outer_colors, radius=3.2,
            wedgeprops=dict(width=0.6, edgecolor='white', linewidth=2),
            startangle=90
            )
    plt.pie(occurences, colors=colors, autopct=None, radius=2.6,
            wedgeprops=dict(width=0.66, edgecolor='white', linewidth=1),
            startangle=90)
    center_circle = plt.Circle((0, 0), 2.0, color='black', fc='white', linewidth=0)
    fig = plt.gcf()
    fig.gca().add_artist(center_circle)
    fig.set_size_inches(12, 12)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(
        "/Users/martin.meinel/Desktop/Projects/Eyerich Projects/Natalie/Classifier/Final_results/Figures/Figure1/pie_chart_lesional.pdf",
        bbox_inches="tight")
    plt.close()


def plot_radar_plot(df_patients, categories, save_folder, diag):
    N = len(categories)
    # We are going to plot the first line of the data frame.
    # But we need to repeat the first value to close the circular graph:
    letter = ['E', 'D', 'F']
    colors = [custom_palette[d] for d in diag]
    for ind, patient in enumerate(df_patients['Helmholtz_identifyer']):  # values
        values = df_patients.loc[ind].drop('Helmholtz_identifyer').values.flatten().tolist()
        values += values[:1]
        # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        # Initialise the spider plot
        ax = plt.subplot(111, polar=True)
        # Draw one axe per variable + add labels
        # plt.xticks(angles[:-1], categories, color='k', size=10)
        ax.set_xticks(angles[:-1])
        # Remove the labels
        ax.set_xticklabels([''] * N)
        # Draw ylabels
        ax.set_rlabel_position(0)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8])
        ax.set_yticklabels([" "] * 4)
        ax.set_ylim(0, 1)
        # Plot data
        ax.plot(angles, values, linewidth=2, linestyle='solid', color=colors[ind])
        # Fill area
        ax.fill(angles, values, 'grey', alpha=0.4)
        plt.savefig(os.path.join(save_folder, 'Figure_1{}_Radarplot_patient_{}.pdf'.format(letter[ind], patient)),
                    bbox_inches='tight')
        plt.close()


def compute_spiderplot_categories(df_patients):
    age = df_patients["age"]
    sex = df_patients["Sex_x_M"]
    histology = df_patients.loc[:, df_patients.columns.str.startswith("Hist_")].mean(axis=1)
    comorbidity = df_patients.loc[:, df_patients.columns.str.startswith("Com_")].mean(axis=1)
    history = df_patients.loc[:, df_patients.columns.str.startswith("Hty_")].mean(axis=1)
    morphology = df_patients.loc[:, df_patients.columns.str.startswith("Morph_")].mean(axis=1)
    lab = df_patients.loc[:, df_patients.columns.str.startswith("Lab_")].mean(axis=1)
    return pd.DataFrame(
        dict(Age=age, Sex=sex, Histology=histology, Comorbidity=comorbidity, History=history, Morphology=morphology,
             Lab=lab, Helmholtz_identifyer=df_patients.index.values.tolist())).reset_index(drop=True)

def compute_state(adata_lesional):
    adata_lesional_mf = adata_lesional[adata_lesional.obs.diag.isin(["cutaneous lymphoma"])]
    adata_lesional_eczema_pso = adata_lesional[adata_lesional.obs.diag.isin(["eczema", "psoriasis"])]

    for adata in [adata_lesional_mf, adata_lesional_eczema_pso]:
        print(adata.obs.diag)
        n_samples = adata.n_obs
        n_patients = adata.obs["PatientID"].nunique()
        # Drop duplicates
        # Cutaneous lymphoma
        unique_patients = adata.obs.drop_duplicates(subset=["PatientID"])
        sex = unique_patients["Sex.x"].values
        age_mean = np.mean(unique_patients.age)
        age_std = np.std(unique_patients.age)
        sex_counts = unique_patients["Sex.x"].value_counts(normalize=True)
        print(f"Number of samples: {n_samples}")
        print(f"Number of patients: {n_patients}")
        print(f"Sex percentage: {sex_counts}")
        print(f"Age mean: {age_mean}")
        print(f"Age std: {age_std}")
    exit(0)
    n_samples  = adata_lesional.n_obs
    n_patients = adata_lesional.obs["PatientID"].nunique()
    print(adata_lesional.obs.columns)
    # Drop duplicates
    # Cutaneous lymphoma
    unique_patients = adata_lesional.obs.drop_duplicates(subset=["PatientID"])
    sex = unique_patients["Sex.x"].values
    age_mean = np.mean(unique_patients.age)
    age_std = np.std(unique_patients.age)
    sex_counts = unique_patients["Sex.x"].value_counts(normalize=True)
    print(f"Number of samples: {n_samples}")
    print(f"Number of patients: {n_patients}")
    print(f"Sex percentage: {sex_counts}")
    print(f"Age mean: {age_mean}")
    print(f"Age std: {age_std}")


if __name__ == "__main__":
    args = get_args()
    # plot_barchart(orientation="horizontal")
    adata = create_anndata()
    adata_lesional = filter_samples(adata, diag=["eczema", "cutaneous lymphoma", "psoriasis"])
    compute_state(adata_lesional)
    # adata_with_nl = filter_samples(adata, diag=["eczema", "cutaneous lymphoma", "psoriasis", "non-lesional"])
    # print(adata_with_nl.n_vars)
    # print(adata_with_nl.n_obs)
    # print(Counter(adata_with_nl.obs.diag))
    exit()
    plot_piechart(adata_lesional)
    # plot_piechart(adata_with_nl)
    # # Plot with NL
    # adata_with_nl = normalize_data(adata_with_nl, mode="tmm")
    # adata_with_nl = correct_batches(adata_with_nl)
    # plot_embedding(adata_with_nl, mode=args.embedding, label="with_NL", interactive=True)
    # # Lesional Plot
    adata_lesional = normalize_data(adata_lesional, mode="tmm")
    adata_lesional = correct_batches(adata_lesional)
    plot_embedding(adata_lesional, mode=args.embedding, label="lesional")
    exit(0)
    # # Clinical attributes plot
    clinical_attributes, diags = load_clinical_attributes()
    sampleIds = clinical_attributes.index.values.tolist()
    clinical_attributes_embedding = compute_umap(clinical_attributes)
    plot_clinical_attributes(clinical_attributes_embedding, diags, sampleIds, interactive=True)
    # Create spiderplots
    # This is for MF, Eczema and Psoriasis
    spiderplot_samples = ["MUC20709", "MUC3729", "MUC4319"]
    df_patients = clinical_attributes.loc[spiderplot_samples, :]
    diag_spiderplot = diags.loc[spiderplot_samples]
    df_patients_spiderplot = compute_spiderplot_categories(df_patients)
    categories = df_patients_spiderplot.columns.values.tolist()
    categories = [category for category in categories if category != "Helmholtz_identifyer"]
    save_folder = "/Users/martin.meinel/Desktop/Projects/Eyerich Projects/Natalie/Classifier/Final_results/Figures/Figure1"
    plot_radar_plot(df_patients=df_patients_spiderplot, categories=categories, save_folder=save_folder,
                    diag=diag_spiderplot)
