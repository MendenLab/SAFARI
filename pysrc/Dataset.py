from collections import Counter
from typing import Tuple
import pandas as pd
from TrainingSet import TrainingSet
from BootstrappingSet import BootstrappingSet


class Dataset:

    def __init__(self, x_train: pd.DataFrame, x_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame,
                 colData: pd.DataFrame, colData_test: pd.DataFrame, rowData: pd.DataFrame, eczema_healthy: pd.DataFrame,
                 mf_healthy: pd.DataFrame, binary_target: bool = False, fdr: float = 0.05):
        """
        Constructor of the dataset class containing all the important information for training and testing procedure
        Parameters
        ----------
        x_train: raw count matrix of the training set (100 samples 84/16) which
        x_test: raw count matrix of the test set (24 samples 21/3)
        y_train: labels of the training set
        y_test: labels of the test set
        colData : contains the colData corresponding to the counts of x_train
        colData_test : contains the colData corresponding to the counts of x_test
        rowData: contains the gene information as entrezgene_id, ensembl_id and hugo gene name
        eczema_healthy: Result table of the unpaired DESeq2 analysis  (eczema versus non-lesional)
        mf_healthy: Result table of the unpaired DESeq2 analysis (mf versus non-lesional)
        binary_target: states whether the given labels are already binary or have to be transformed
        fdr: determines the cutoff for which the genes are filtered (only significant)
        """
        self.x_train: pd.DataFrame = x_train
        self.x_test: pd.DataFrame = x_test
        self.y_train: pd.DataFrame = y_train
        self.y_test: pd.DataFrame = y_test
        self.colData: pd.DataFrame = colData
        self.colData_test = colData_test
        self.rowData = rowData
        self.eczema_healthy: pd.DataFrame = eczema_healthy
        self.mf_healthy: pd.DataFrame = mf_healthy
        self.binary_target: bool = binary_target
        self.fdr = fdr
        # Patient ids of eczema and mf to sample directly for the bootstrapping
        self.eczema_patients: pd.Series = None
        self.mf_patients: pd.Series = None
        # significant_genes contains the significant genes of mf and eczema vs. non-lesional (union) in an
        # unpaired fashion
        # filtered for fdr
        self.significant_genes: set = None
        self.__getPatientsByDiagnose()
        self.__getSignificantGenes()
        if not binary_target:
            self.__transformTargetsToBinary()

    def __getSignificantGenes(self):
        """
        Computes the significant genes of the paired DGE Analysis of paired mf and eczema versus non-lesional for a
        certain fdr.
        Returns: set of significant genes to remove duplicates
        -------

        """
        significant_mf_genes = pd.Series(self.mf_healthy.loc[self.mf_healthy["padj"] < self.fdr, :].index).values.tolist()
        significant_eczema_genes = pd.Series(self.eczema_healthy.loc[self.eczema_healthy["padj"] < self.fdr, :].index). \
            values.tolist()
        assert len(set(significant_eczema_genes).intersection(set(self.rowData["ensembl_id"].values.tolist()))) == len(significant_eczema_genes)
        assert len(set(significant_mf_genes).intersection(set(self.rowData["ensembl_id"].values.tolist()))) == len(significant_mf_genes)
        significant_genes = significant_eczema_genes + significant_mf_genes
        self.significant_genes = set(significant_genes)

    def __transformTargetsToBinary(self):
        """
        Transforms the labels into binary labels from text labels, where the underrepresented class is encoded with 1.
        Returns
        -------

        """
        c = Counter(self.y_train["diag"])
        minority_class = min(c.items(), key=lambda x: x[1])[0]
        self.y_train["diag"] = self.y_train["diag"] == minority_class
        self.y_test["diag"] = self.y_test["diag"] == minority_class
        self.binary_target = True


    def __getPatientsByDiagnose(self):
        """
        Extracts the Ids of Eczema and Mf of the training set, so that the bootstrapping is easier later on
        Returns
        -------

        """
        diagnoses = self.y_train["diag"].unique()
        self.mf_patients = pd.Series(self.x_train.columns[self.y_train["diag"] == diagnoses[0]].tolist())
        self.eczema_patients = pd.Series(self.x_train.columns[self.y_train["diag"] == diagnoses[1]].tolist())

    def samplePatients(self, seed: int, sample: int = 3) -> Tuple[BootstrappingSet, TrainingSet]:
        """
        Samples from both diseases the same number of samples (sample) which are used for DESeq2 analysis.
        The other samples are used for training and evaluating the classifier
        Parameters:
        ----------
        sample: determines the number of samples which are used for each disease group for the following DESeq2 analysis
        random_state: used for reproducing the results
        seed: used for ensuring deterministic results

        Returns: BootstrappingSet containing the counts and the metadata of the sampled patients
        -----

        """
        # Sample n patients from both diseases for edgeR analysis
        eczema_samples = self.eczema_patients.sample(n=sample, random_state=seed).values.tolist()
        mf_samples = self.mf_patients.sample(n=sample, random_state=seed).values.tolist()
        bootstrap_samples = eczema_samples + mf_samples
        # Counts and colData matched with sampled patients for EdgeR analysis
        counts = self.x_train.loc[:, bootstrap_samples]
        colData = self.colData.loc[bootstrap_samples, :]
        # Create bootstrapping set with the sampled patients
        bs = BootstrappingSet(counts=counts, colData=colData)
        # Create the training set with the leftover samples
        ts = TrainingSet(countData=self.x_train.loc[:, ~self.x_train.columns.isin(bootstrap_samples)],
                         colData=self.colData.loc[~self.colData.index.isin(bootstrap_samples), :],
                         labels=self.y_train.loc[~self.y_train["sampleID"].isin(bootstrap_samples), "diag"].to_numpy())
        return bs, ts
