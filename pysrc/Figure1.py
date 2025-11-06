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
            "/dds_highQual_Sexandbatchcorrected_v04.rds")
    else:
        countData = readRDS(os.path.join("/Safari",
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


def normalize_data(data: sc.AnnData) -> sc.AnnData:
    """
    Normalizes the data either with TMM CPM normalization

    Parameters
    ----------
    data: AnnData object

    Returns: AnnData Object with new normalized counts in X.
    -------

    """
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
                base_path = "/Figures/Figure1"
                path = os.path.join(base_path, f"{mode}_tmm_4000_{label}_jointplot.pdf")
                plt.savefig(path, format="pdf", bbox_inches="tight")
                plt.show()
            else:
                plt.savefig(
                    f"/figures/Figure1/{mode}_{label}.pdf",
                    format="pdf", bbox_inches="tight")



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
        age_mean = np.mean(unique_patients.age)
        age_std = np.std(unique_patients.age)
        sex_counts = unique_patients["Sex.x"].value_counts(normalize=True)
        print(f"Number of samples: {n_samples}")
        print(f"Number of patients: {n_patients}")
        print(f"Sex percentage: {sex_counts}")
        print(f"Age mean: {age_mean}")
        print(f"Age std: {age_std}")


if __name__ == "__main__":
    adata = create_anndata()
    adata_lesional = filter_samples(adata, diag=["eczema", "cutaneous lymphoma", "psoriasis"])
    compute_state(adata_lesional)
    plot_piechart(adata_lesional)
    # # Lesional Plot
    adata_lesional = normalize_data(adata_lesional)
    adata_lesional = correct_batches(adata_lesional)
    plot_embedding(adata_lesional, mode="umap", label="lesional")
