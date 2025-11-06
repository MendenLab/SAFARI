from typing import Tuple

from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import fdrcorrection

from Figure1 import create_anndata, normalize_data
import pandas as pd
import seaborn as sns
sns.set_style("white")
import matplotlib.pyplot as plt
from statannotations.Annotator import Annotator

custom_palette = {"cutaneous lymphoma": "#ff5b57",
                  "cutaneous_lymphoma_NL": "#eea488",
                  "psoriasis": "#39476e",
                  "psoriasis_NL": "#dee7ef",
                  "eczema": "#39ac5e",
                  "eczema_NL": "#b2d7a8",
                  "non-lesional": "#ac8a66",
                  "non_lesional_others": "#a69e9b",
                  # "pso_ec": "#ac8a66",
                  "pso_ec": "#8F8071"}


def applyWhitneyUTest(data: pd.DataFrame, pairs: list) -> Tuple[list, list]:
    """
    Applies two-sided WhitneyU test checking whether two non-normally distributed distributions are equal.
    The tests are executed between all different distributions
    :param data: contain all the different distributions
    :param pairs: contains all the pairs between a test is executed
    :return: p-values and p-adjusted values with BH correction
    """
    p_values = []
    for pair in pairs:
        p1, p2 = pair
        e1, dt1 = p1
        e2, dt2 = p2
        x = data.loc[(data["gene"] == e1) & (data["diag"] == dt1), "expression"]
        x = x.dropna()
        y = data.loc[(data["gene"] == e2) & (data["diag"] == dt2), "expression"]
        y = y.dropna()
        p_values.append(mannwhitneyu(x, y).pvalue)
    p_adjusted = fdrcorrection(p_values)[1]
    return p_values, p_adjusted




def create_boxplot_with_stats(df: pd.DataFrame, file_name, annotate):
    melted = pd.melt(df, id_vars='diag', var_name='gene', value_name='expression')
    melted["expression"] = pd.to_numeric(melted["expression"], errors='coerce')
    size= (3.5, 3)
    if annotate:
        order = ["LCK", "HOMER1", "SERBP1", "BTN3A1", "ZC3H12D", "RNF213", "PNLIPRP3"]
        hue_order = ["pso_ec", "cutaneous lymphoma"]
        pairs = [
            [("LCK", "pso_ec"), ("LCK", "cutaneous lymphoma")],
            [("HOMER1", "pso_ec"), ("HOMER1", "cutaneous lymphoma")],
            [("SERBP1", "pso_ec"), ("SERBP1", "cutaneous lymphoma")],
            [("BTN3A1", "pso_ec"), ("BTN3A1", "cutaneous lymphoma")],
            [("ZC3H12D", "pso_ec"), ("ZC3H12D", "cutaneous lymphoma")],
            [("RNF213", "pso_ec"), ("RNF213", "cutaneous lymphoma")],
            [("PNLIPRP3", "pso_ec"), ("PNLIPRP3", "cutaneous lymphoma")]
        ]
        fig, ax = plt.subplots(figsize=size)
        annotator_params = dict(order = order,
                                hue_order = hue_order,
                      data = melted,
                    linewidth = 0.5,
                    fliersize=1.5,
                    x = "gene",
                    y="expression",
                    hue="diag",
                    palette=custom_palette)
        sns.boxplot(ax=ax, **annotator_params)
        annotator = Annotator(ax, pairs, **annotator_params)
        annotator.configure(test="Mann-Whitney", text_format="full", comparisons_correction="BH",
                            correction_format="replace", show_test_name=False, verbose=True, fontsize=6)
        annotator.apply_and_annotate()
        ax.get_legend().remove()
    else:
        fig, _ = plt.subplots(figsize=size)
        hue_order = ["pso_ec", "cutaneous lymphoma"]
        pairs = [
            [("HOMER1", "pso_ec"), ("HOMER1", "cutaneous lymphoma")],
            [("ZC3H12D", "pso_ec"), ("ZC3H12D", "cutaneous lymphoma")],
            [("BTN3A1", "pso_ec"), ("BTN3A1", "cutaneous lymphoma")],
            [("RNF213", "pso_ec"), ("RNF213", "cutaneous lymphoma")],
            [("LCK", "pso_ec"), ("LCK", "cutaneous lymphoma")],
            [("SERBP1", "pso_ec"), ("SERBP1", "cutaneous lymphoma")],
            [("PNLIPRP3", "pso_ec"), ("PNLIPRP3", "cutaneous lymphoma")]
        ]
        # hue_order = ["psoriasis", "eczema","cutaneous lymphoma"]
        order = [ "HOMER1","ZC3H12D",  "BTN3A1",  "RNF213","LCK","SERBP1", "PNLIPRP3"]
        pvals, padjs = applyWhitneyUTest(melted, pairs)
        print(pvals)
        print(padjs)
        ax = sns.boxplot(data=melted, x="gene", y="expression",hue="diag", linewidth=0.5, fliersize=1.5,
                         order=order,hue_order=hue_order, palette=custom_palette)
        ax.get_legend().remove()
    plt.xticks(rotation=45, fontsize=6)
    plt.yticks(fontsize=6)
    if file_name == "RNA":
        plt.ylabel("Log CPM Expression", fontsize=6)
    else:
        plt.ylabel(r'-$\Delta$C$_T$ (SDHAF)', fontsize=6)
    plt.xlabel("Gene", fontsize=6)
    sns.despine()
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(
            f"/Figures/Figure2/{file_name}_boxplot.svg",bbox_inches="tight")
    plt.show()
    plt.close()


def create_baseline_boxplot_with_stats(df, baseline="1"):
    if baseline == "1":
        order = ["HOMER1", "GPR68", "LMOD1", "CDK14", "PHKB", "FAH"]
    else:
        order = ["HOMER1", "GPR68", "LMOD1", "CDK14", "SS18", "PEX7"]
    melted = pd.melt(df, id_vars='diag', var_name='gene', value_name='expression')
    melted["expression"] = pd.to_numeric(melted["expression"], errors='coerce')
    size = (3.5, 3)
    fig, _ = plt.subplots(figsize=size)
    hue_order = ["pso_ec", "cutaneous lymphoma"]
    if baseline == "1":
        pairs = [
            [("HOMER1", "pso_ec"), ("HOMER1", "cutaneous lymphoma")],
            [("GPR68", "pso_ec"), ("GPR68", "cutaneous lymphoma")],
            [("LMOD1", "pso_ec"), ("LMOD1", "cutaneous lymphoma")],
            [("CDK14", "pso_ec"), ("CDK14", "cutaneous lymphoma")],
            [("PHKB", "pso_ec"), ("PHKB", "cutaneous lymphoma")],
            [("FAH", "pso_ec"), ("FAH", "cutaneous lymphoma")],
        ]
    else:
        pairs = [
            [("HOMER1", "pso_ec"), ("HOMER1", "cutaneous lymphoma")],
            [("GPR68", "pso_ec"), ("GPR68", "cutaneous lymphoma")],
            [("LMOD1", "pso_ec"), ("LMOD1", "cutaneous lymphoma")],
            [("CDK14", "pso_ec"), ("CDK14", "cutaneous lymphoma")],
            [("SS18", "pso_ec"), ("SS18", "cutaneous lymphoma")],
            [("PEX7", "pso_ec"), ("PEX7", "cutaneous lymphoma")],
        ]
    pvals, padjs = applyWhitneyUTest(melted, pairs)
    print(pvals)
    print(padjs)
    ax = sns.boxplot(data=melted, x="gene", y="expression", hue="diag", linewidth=0.5, fliersize=1.5,
                     order=order, hue_order=hue_order, palette=custom_palette)
    ax.get_legend().remove()
    plt.xticks(rotation=45, fontsize=6)
    plt.yticks(fontsize=6)
    plt.ylabel("Log CPM Expression", fontsize=6)
    plt.xlabel("Gene", fontsize=6)
    sns.despine()
    plt.grid(False)
    plt.tight_layout()
    if baseline == "1":
        plt.savefig(
            f"/Figures/Baseline/ffs_all_genes_boxplot.svg",
            bbox_inches="tight")
    else:
        plt.savefig(
            f"/Figures/Baseline/ffs_lesional_genes_boxplot.svg",
            bbox_inches="tight")
    plt.show()
    plt.close()



def main():
    plot= "a"
    genes = ["BTN3A1", "HOMER1", "LCK", "RNF213", "SERBP1", "ZC3H12D", "PNLIPRP3"]
    if plot == "a":
        adata = create_anndata()
        adata = adata[adata.obs.diag.isin(["cutaneous lymphoma", "psoriasis", "eczema"])].copy()
        adata = normalize_data(adata)
        adata = adata[:, adata.var.Gene_name.isin(genes)]
        df = pd.DataFrame(data=adata.X, columns=adata.var.Gene_name, index=adata.obs_names)
        diag = pd.Series([x if x == "cutaneous lymphoma" else "pso_ec" for x in adata.obs.diag], name="diag")
        # diag = pd.Series(adata.obs.diag, name="diag")
        df = pd.concat([df, diag.set_axis(df.index.values)], axis=1)
        create_boxplot_with_stats(df, file_name="RNA", annotate=False)
    if plot == "b":
        baseline ="2"
        if baseline == "1":
            genes = ["HOMER1", "GPR68", "LMOD1", "CDK14", "PHKB", "FAH"]
        else:
            genes = ["HOMER1", "GPR68", "LMOD1", "CDK14", "SS18", "PEX7"]
        adata = create_anndata()
        adata = adata[adata.obs.diag.isin(["cutaneous lymphoma", "psoriasis", "eczema"])].copy()
        adata = normalize_data(adata)
        adata = adata[:, adata.var.Gene_name.isin(genes)]
        df = pd.DataFrame(data=adata.X, columns=adata.var.Gene_name, index=adata.obs_names)
        diag = pd.Series([x if x == "cutaneous lymphoma" else "pso_ec" for x in adata.obs.diag], name="diag")
        # diag = pd.Series(adata.obs.diag, name="diag")
        df = pd.concat([df, diag.set_axis(df.index.values)], axis=1)
        create_baseline_boxplot_with_stats(df, baseline=baseline)

if __name__ == '__main__':
    main()