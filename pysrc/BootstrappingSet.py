import pandas as pd
from rpy2.robjects.packages import importr
from rpy2.robjects import  pandas2ri
from R_helpers import pandasToR, RToPandas

pandas2ri.activate()
base = importr("base")
edgeR = importr("edgeR")
stats = importr("stats")
limma = importr("limma")


class BootstrappingSet:
    def __init__(self, counts: pd.DataFrame, colData: pd.DataFrame):
        """
        Creates BootstrappingSet which is used for DESeq2 Analysis
        Parameters
        ----------
        counts: containing the raw countMatrix for the DESeq2Dataset
        colData: containing the colData information for the DESeq2 analysis
        """
        self.counts: pd.DataFrame = counts
        self.colData: pd.DataFrame = colData
        # contains the results of the DGE analysis between the diseases
        self.results: dict = {}
        # contains the intersection of the significant genes of the paired non-lesional analysis and the genes of the
        # inter-disease analysis
        self.significantGenes = None
        self.edgeRGenes = None


    def __edgeRAnalysis(self, fdr) -> set:
        """
        Applies Differential Gene Expression Analysis for Eczema versus cutaneous lymphoma
         in edgeR and stores its result in the result table.
        Returns the significant genes according to fdr
        Parameters
        ----------
        fdr

        Returns:
        -------

        """
        assert self.counts.columns.equals(self.colData.index)
        self.colData.loc[:, "diag"] = self.colData.loc[:, "diag"].astype(str)
        counts_r = pandasToR(self.counts)
        colData_r = pandasToR(self.colData)
        # Set up the design
        diag = colData_r.rx2("diag")
        design = pd.get_dummies(diag)
        design = design.set_axis(self.counts.columns, axis="index")
        design = pandasToR(design)
        y = edgeR.DGEList(counts_r)
        keep = edgeR.filterByExpr(y, design=design)
        # Filters genes and recomputes the library sizes after filtering (keep.lib.sizes=False)
        y = y.rx(keep, True, False)
        y = edgeR.calcNormFactors_DGEList(y)
        y = edgeR.estimateDisp_DGEList(y, design, robust=True)
        fit = edgeR.glmFit_DGEList(y, design)
        sig_genes = []
        # Specify contrast here
        diagnoses = self.colData["diag"].unique()
        assert len(diagnoses) == 2
        comparisons = [f"{diagnoses[0]}-{diagnoses[1]}"]
        contrasts = limma.makeContrasts(f"{diagnoses[0]}-{diagnoses[1]}", levels=base.colnames(design))
        for i in range(len(comparisons)):
            lrt = edgeR.glmLRT(fit, contrast=contrasts[:, i])
            result = lrt.rx2("table")
            padj = stats.p_adjust(result.rx2("PValue"), "BH")
            pd_result = RToPandas(result)
            pd_result["padj"] = padj
            pd_result.rename(columns={"logFC": "log2FoldChange"}, inplace=True)
            self.results[comparisons[i]] = pd_result
            sig_genes_comparison = pd.Series(pd_result.loc[pd_result["padj"] < fdr, :].index).values.tolist()
            if len(comparisons) == 1:
                sig_genes = sig_genes_comparison
            else:
                sig_genes += sig_genes_comparison
        return set(sig_genes)

    def selectOverallSignificantGenes(self, fdr: float, paired_significant_genes):
        """
        Executes a DGE analysis in between all the different diagnoses for the sampled 3 vs 3 patients
        Stores the result in results and filters the significant genes according the given fdr. These genes from the
        in-between disease comparisons are then intersected with the genes which are significantly expressed between
        the diseases and the healthy controls.
        Returns the genes which are consistently differentially expressed (in between diseases and healthy controls)
        Parameters
        ----------
        DGE: determines whether edgeR or DESeq2 is used for differential Gene expression Analysis
        paired_significant_genes: contains the genes which are at least of one of the paired DGE analysis differentially
                                    expressed
        fdr : determines the cutoff for which the genes are considered to be significantly expressed

        Returns: genes which are consistently differentially expressed in the paired non-lesional analysis such as
        the analysis between the diseases
        -------

        """
        # Applies the DGE analysis between the samples
        sig_genes = self.__edgeRAnalysis(fdr=fdr)
        sig_genes = set(sig_genes)
        self.edgeRGenes = sig_genes
        paired_significant_genes = set(paired_significant_genes)
        self.significantGenes = sig_genes.intersection(paired_significant_genes)
