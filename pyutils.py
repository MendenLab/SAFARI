import xgboost
from scipy.stats import uniform, randint
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
import xgboost as xgb
from collections import Counter
from Dataset import Dataset
import pandas as pd
import numpy as np
import scanpy as sc
from R_helpers import RToPandas, edgeRNormalization
from sklearn.metrics import f1_score, recall_score, fbeta_score, balanced_accuracy_score, make_scorer
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
base = importr("base")

pandas2ri.activate()


def get_qpcr_parameter_space():
    """
    This is just to store the parameters at one location and load and adjust them centrally.
    Returns
    -------

    """
    class_weights = ["balanced", {0: 1, 1: 2}, {0: 1, 1: 3}, {0: 1, 1: 4}, {0: 1, 1: 5}, {0: 1, 1: 6}, {0: 1, 1: 7}]
    model_params = [
        {"estimator__C": np.logspace(-3, 0, 8), "estimator__penalty": ["l1"],
         "estimator__class_weight": class_weights},
        {"estimator__C": np.logspace(-3, 0, 8), "estimator__penalty": ["l2"],
         "estimator__class_weight": class_weights},
        {"estimator__C": np.logspace(-3, 0, 8), "estimator__penalty": ["elasticnet"],
         "estimator__class_weight": class_weights,
         "estimator__l1_ratio": np.linspace(0, 1, 10)},
    ]
    return model_params

def get_scorer(scorer_name):
    if scorer_name == "f1":
        return "f1"
    elif scorer_name == "roc_auc":
        return "roc_auc"
    elif scorer_name == "f2":
        return make_scorer(fbeta_score, beta=2, pos_label=1, greater_is_better=True)
    elif scorer_name =="balanced_accuracy":
        return "balanced_accuracy"
    else: raise ValueError("scorer_name must be either f1, roc_auc, f2, or balanced_accuracy")


def compute_sensitivity(y_true, y_pred):
    return recall_score(y_true, y_pred)


def compute_specificity(y_true, y_pred):
    return recall_score(y_true, y_pred, pos_label=0)

def compute_false_discovery_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    FP = np.sum(np.logical_and(y_pred == 1, y_true == 0))
    TP = np.sum(np.logical_and(y_pred == 1, y_true == 1))
    return FP / (FP + TP)

def calculate_vif(df):
    """Calculate the VIF for each variable in the DataFrame, including a constant."""
    # Add a constant (intercept) column
    df_with_const = sm.add_constant(df)
    vif_data = pd.DataFrame()
    vif_data['Feature'] = df_with_const.columns
    vif_data['VIF'] = [variance_inflation_factor(df_with_const.to_numpy(), i) for i in range(df_with_const.shape[1])]
    return vif_data

def createLogisticRegressionClassifier():
    """
    Creates a logistic Regression model and returns it with its hyperparameters to optimize for and the strategy for
     it.
    Returns
    -------

    """
    model = LogisticRegression(solver="saga", class_weight="balanced", max_iter=500, random_state=0)
    C_2d_range = [1e-2, 1e-1, 1, 1e1, 1e2]
    class_weight = ["balanced", {0: 1, 1: 12}, {0: 1, 1: 11}, {0: 1, 1: 10}, {0:1, 1: 9}, {0: 1, 1: 8}]
    regularizations = ["l1", "l2"]
    params = {"estimator__C": C_2d_range, "estimator__class_weight": class_weight,
              "estimator__fit_intercept": [True, False], "estimator__penalty": regularizations}
    optimizer = "grid"
    return model, params, optimizer



def createXGBoost(scale_pos_weight):
    # Before scale_pos_weight was 0.2
    model = xgb.XGBClassifier(objective="binary:logistic", scale_pos_weight=scale_pos_weight, random_state=0)
    params = {
        "estimator__gamma": uniform(0, 0.5),
        "estimator__learning_rate": uniform(0.05, 0.5),  # default 0.1
        "estimator__max_depth": randint(2, 6),  # default 3
        "estimator__lambda": [1e-2, 1e-1, 1, 1e1, 1e2],
        "estimator__scale_pos_weight": [scale_pos_weight, 8, 9, 10, 11, 12]
    }
    optimizer = "random"
    return model, params, optimizer


def createSVM(kernel="linear"):
    """
    Creates a SVM model with the specified kernel and returns it with the corresponding hyperparameters to optimize for
    and a strategy to find them.
    Parameters
    ----------
    kernel: specifies the kernel of the SVM (linear, polynomial, sigmoid or rbf)

    Returns: SVM model with params and optimizer
    -------

    """
    gamma_range = np.logspace(-9, 3, 13)
    coef0 = np.arange(-100, 100, 5)
    class_weight = ["balanced", {0: 1, 1: 12}, {0: 1, 1: 11}, {0: 1, 1: 10}, {0:1, 1: 9}, {0: 1, 1: 8}]
    C_2d_range = [1e-2, 1e-1, 1, 1e1, 1e2]
    regularizations = ["l1", "l2"]
    degree = [3, 4, 5, 6, 7]
    if kernel == "poly":
        # Train with default values of gamma and degree
        params = dict(estimator__C=C_2d_range, estimator__coef0=coef0, estimator__degree=degree)
        model = SVC(kernel=kernel, cache_size=260000, class_weight="balanced", random_state=0)
    elif kernel == "rbf":
        params = dict(estimator__C=C_2d_range, estimator__gamma=gamma_range, estimator__class_weight=class_weight)
        model = SVC(kernel=kernel, cache_size=260000, random_state=0)
    elif kernel == "sigmoid":
        params = dict(estimator__C=C_2d_range, estimator__coef0=coef0, estimator__gamma=gamma_range,
                      estimator__class_weight=class_weight)
        model = SVC(kernel=kernel, cache_size=260000, random_state=0)
    else:
        params = dict(estimator__C=C_2d_range, estimator__penalty=regularizations,
                      estimator__class_weight=class_weight)
        model = LinearSVC(dual=False, random_state=0)
    optimizer = "random"
    return model, params, optimizer


def select_genes_based_on_vif(counts: pd.DataFrame, genes, qpcr=False, threshold_vif=5):
    """

    Parameters
    ----------
    counts: unnormalized counts in case of rna_seq otherwise delta CT values
    genes: filtering for these genes after
    qpcr: boolean indicator whether rna-seq or qpcr counts are fed into
    threshold_vif

    Returns
    -------

    """
    if qpcr:
        df_reduced = counts
    else:
        normalized_counts = edgeRNormalization(counts, log=True)
        filtered_counts = normalized_counts.loc[genes, :]
        df_reduced = filtered_counts.T
    stop = False
    while not stop:
        vif_data = calculate_vif(df_reduced)
        # Exclude the constant from consideration for removal
        vif_data = vif_data.loc[vif_data['Feature'] != 'const', :]
        max_vif = vif_data['VIF'].max()

        if max_vif < threshold_vif:
            stop = True
        else:
            # Drop the feature with the highest VIF
            max_vif_feature = vif_data.loc[vif_data['VIF'].idxmax(), 'Feature']
            df_reduced.drop(columns=[max_vif_feature], inplace=True)
    final_counts = df_reduced.T
    ensembl_genes = final_counts.index.values.tolist()
    return ensembl_genes


def filterCorrelatedGenes(dataset: Dataset, genes: list, threshold_vif: int = 5) -> list:
    """
    Applies Variance inflation factor filtering for correlated genes
    Parameters
    ----------
    dataset: containing the training and normalized training set to apply gene filtering
    genes: Filtering already before the vif filtering to only these genes (ensembl ids)
    threshold_vif: determines until which correlation structure genes are removed (5 corresponds to R^2 of 0.8)

    Returns: List of ensembl ids making the cut-off
    -------

    """
    # use normalized counts
    normalized_counts = edgeRNormalization(dataset.x_train, log=True)
    # Filter for genes
    filtered_counts = normalized_counts.loc[genes, :]
    # Start VIF
    df_reduced = filtered_counts.T
    stop = False
    while not stop:
        df_reduced_corr = df_reduced.corr()
        vif = pd.DataFrame(np.linalg.inv(df_reduced_corr.values), index=df_reduced_corr.index,
                           columns=df_reduced_corr.columns)
        vifs = np.diag(vif)
        if np.max(vifs) < threshold_vif:
            break
        else:
            column_index = np.argmax(vifs)
            column = df_reduced.columns[column_index]
            df_reduced.drop(columns=[column], inplace=True)
    final_counts = df_reduced.T
    ensembl_genes = final_counts.index.values.tolist()
    return ensembl_genes

def createModel(model: str, kernel, scale_pos_weight):
    # Do not overfit, so take default parameters first and later on train, test split tune hyperparameters
    if model == "logistic": # Before liblinear was here
        model = LogisticRegression(penalty="l2", solver="saga", class_weight="balanced", random_state=0)
    elif model == "xgboost":
        # https://machinelearningmastery.com/xgboost-for-imbalanced-classification/
        model = xgboost.XGBClassifier(objective="binary:logistic", scale_pos_weight=scale_pos_weight, random_state=0)
    elif model == "svm":
        if kernel == "linear":
            model = svm.LinearSVC(class_weight="balanced", random_state=0)
        elif kernel == "rbf":
            model = svm.SVC(kernel="rbf", class_weight="balanced", random_state=0)
        elif kernel == "poly":
            model = svm.SVC(kernel="poly", class_weight="balanced", random_state=0)
        elif kernel == "sigmoid":
            model = svm.SVC(kernel="sigmoid", class_weight="balanced", random_state=0)
        else:
            print("Invalid kernel type")
            exit(-1)
    else:
        print("Invalid model")
        exit(-1)
    return model


def fixBatchIDs(colData: pd.DataFrame) -> pd.DataFrame:
    colData.batchID = colData.batch.str.extract('(\d+)')
    return colData


def readData(path: str) -> sc.AnnData:
    """
    Reads in the R Summarized Experiment and maps it to an AnnData object without any filtering
    Parameters
    ----------
    path: to the R Summarized Experiment
    Return: AnnData Object corresponding to the R Summarized Experiment
    -------

    """
    readRDS = robjects.r['readRDS']
    summarizedExperiment = importr("SummarizedExperiment")
    base = importr("base")
    # Read count Matrix files -------------------------------------------------------
    countData = readRDS(path)
    counts_R = summarizedExperiment.assay(countData)
    rowData_r = summarizedExperiment.rowData(countData)
    colData_r = summarizedExperiment.colData(countData)
    rowData_r = base.data_frame(rowData_r)
    colData_r = base.data_frame(colData_r)
    # Transformation to python
    colData = RToPandas(colData_r)
    colData = fixBatchIDs(colData)
    rowData = RToPandas(rowData_r)
    counts = RToPandas(counts_R)
    counts = pd.DataFrame(counts, columns=colData["sampleID"])
    # Create AnnData object
    adata = sc.AnnData(counts.T, obs=colData, var=rowData, dtype=np.float32)
    return adata


def createDataFromFiles(eczema_path: str, mf_path: str, count_matrix_path: str, split_path: str,
                        fdr=0.05) -> Dataset:
    """
    Reads in the count matrix with its row and column data, the results of the paired analysis of eczema and mf versus
     non-lesional and the split between train and test set to return a single Data object
    Parameters
    ----------
    fdr: determines the fdr for the Dataset object which is created here to see which genes are significant
    eczema_path: path to the results file of the paired comparison vs. non-lesional (.rds file)
    mf_path: path to the results file of the paired comparison vs. non-lesional (rds.file)
    count_matrix_path: path to the file containing count matrix, rowData and colData
    split_path: path to file determining how to split train and test

    Returns Data object with all the important information
    -------

    """
    # R setup
    readRDS = robjects.r['readRDS']
    summarizedExperiment = importr("SummarizedExperiment")
    base = importr("base")
    # Read count Matrix files -------------------------------------------------------
    countData = readRDS(count_matrix_path)
    counts_R = summarizedExperiment.assay(countData)
    rowData_r = summarizedExperiment.rowData(countData)
    colData_r = summarizedExperiment.colData(countData)
    rowData_r = base.data_frame(rowData_r)
    colData_r = base.data_frame(colData_r)
    # Transformation to python
    colData = RToPandas(colData_r)
    rowData = RToPandas(rowData_r)
    counts = RToPandas(counts_R)
    counts = pd.DataFrame(counts, columns=colData["sampleID"])
    # use ensembl_ids to not have duplicated indices
    counts = counts.set_axis(rowData["ensembl_id"], axis="index")
    assert counts.index.is_unique
    # Read train / test split -------------------------------------------------------
    feature_file = pd.read_excel(split_path, sheet_name=None)
    keep = ["sampleID", "diag"]
    train_labels = feature_file["train"].loc[:, keep]
    test_labels = feature_file["test"].loc[:, keep]
    train_labels = train_labels.set_axis(train_labels["sampleID"], axis="index")
    test_labels = test_labels.set_axis(test_labels["sampleID"], axis="index")
    # Split in train and test set
    colData_train = colData.loc[train_labels["sampleID"], :]
    colData_test = colData.loc[test_labels["sampleID"], :]
    counts_train = counts.loc[:, train_labels["sampleID"]]
    counts_test = counts.loc[:, test_labels["sampleID"]]
    # Read paired results of eczema and mf -------------------------------------------
    mf_results = readRDS(mf_path)
    mf_results = base.data_frame(mf_results)
    mf_results = pandas2ri.rpy2py_dataframe(mf_results)
    mf_results = pd.DataFrame(mf_results)
    eczema_results = readRDS(eczema_path)
    eczema_results = base.data_frame(eczema_results)
    eczema_results = pandas2ri.rpy2py_dataframe(eczema_results)
    eczema_results = pd.DataFrame(eczema_results)
    assert (set(colData["diag"].unique()) == set(train_labels["diag"].unique()))
    assert all(eczema_results.index.str.startswith("ENSG"))
    assert all(mf_results.index.str.startswith("ENSG"))
    # --------------------------------------------------------------------------------
    # initialize Data class here
    return Dataset(x_train=counts_train, x_test=counts_test, y_train=train_labels, y_test=test_labels,
                   colData=colData_train, colData_test=colData_test, rowData=rowData,
                   eczema_healthy=eczema_results, mf_healthy=mf_results, binary_target=False, fdr=fdr)


def addSignInformation(bootstrapping_list: list, ranked_features: pd.DataFrame, rowData: pd.DataFrame) -> pd.DataFrame:
    """
    adds the sign of the log2 fold change for the genes to identify eczema and mf genes where a positive number 
    indicates an eczema gene and a negative number indicates a cutaneous lymphoma gene. Besides, it adds how often a 
    gene was found to be consistently differentially expressed.
    Parameters
    ----------
    bootstrapping_list: lists of the single bootstrapping containing the information of the eczema vs. mf comparisons
    ranked_features: contains the robust rank aggregated genes with the adjusted p_value
    rowData: contains the information to map from ensemble_ids to Hugo Gene name

    Returns: returns the ranked_feature dataframe with an additional column for the sign
    -------

    """""
    # ranked_features["Eczema_Gene"] = 0
    ranked_features["Sig_count"] = 0
    for bootstrap in bootstrapping_list:
        # extract the genes and see whether they are up-regulated in eczema or mf This would only work with twi diseases
        # genes = ranked_features["Name"].values
        # signs = np.sign(bootstrap.result.loc[genes, "log2FoldChange"])
        # ranked_features["Eczema_Gene"] += signs
        # check which genes are upregulated how often
        sig_genes = bootstrap.significantGenes
        ranked_features.loc[ranked_features.index.intersection(sig_genes), "Sig_count"] += 1
    # Map the gene names
    ranked_features = pd.merge(left=ranked_features, right=rowData[["ensembl_id", "Gene_name"]], left_on="Name",
                               right_on="ensembl_id")
    return ranked_features


def sort_data(data)->pd.DataFrame:
    """
    Sorts the data by indexes in the order NL5, NL10, NL35, other. It ensures reproducability for
    augmentation and calibration.
    Parameters
    ----------
    data with indexes as sampleIDs:

    Returns Sorted dataframe
    -------

    """
    data_sorted = data.copy()
    nl5_indices = [idx for idx in data.index if idx.startswith('NL5-')]
    nl10_indices = [idx for idx in data.index if idx.startswith('NL10-')]
    nl35_indices = [idx for idx in data.index if idx.startswith('NL35-')]
    other_indices = [idx for idx in data.index if
                     not any(idx.startswith(prefix) for prefix in ['NL5-', 'NL10-', 'NL35-'])]
    # Sort each group individually
    nl5_indices.sort()
    nl10_indices.sort()
    nl35_indices.sort()
    other_indices.sort()
    # Combine in desired order
    sorted_indices = nl5_indices + nl10_indices + nl35_indices + other_indices
    # Reindex the dataframe
    return data_sorted.reindex(sorted_indices)
