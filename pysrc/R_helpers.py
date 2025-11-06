import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri, Formula
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr

DESeq2 = importr("DESeq2")
base = importr("base")
edgeR = importr("edgeR")
summarizedExperiment = importr("SummarizedExperiment")
ra = importr("RobustRankAggreg")
stats = importr("stats")


def robustRankAggregation(feature_lists):
    """
    Applies RobustRankAggregation on the ranked features lists of the bootstrapped models
    Parameters
    ----------
    feature_lists: contains for each training set of the bootstrapping the number 

    Returns: features ordered by their significance and their occurrence in the lists
    -------

    """
    # Create nested vector of string vectors with the single feature lists from python list in shape
    res = ro.ListVector.from_length(len(feature_lists))
    for i, x in enumerate(feature_lists):
        res[i] = ro.StrVector(x)
    aggregated_features = ra.aggregateRanks(res)
    scores = aggregated_features.rx2["Score"]
    # adjust for multiple testing by multiplying with number of lists and by adjusting for features in the lists
    scores = scores * len(feature_lists)
    scores = stats.p_adjust(scores, "BH")
    aggregated_features_df = RToPandas(aggregated_features)
    aggregated_features_df["Corrected_p_values"] = scores
    return aggregated_features_df


def vst(counts: pd.DataFrame) -> pd.DataFrame:
    """
    Apples variance stabilizing transformation to the given dataset
    Parameters
    ----------
    counts: count matrix
    Returns: vst transformed count matrix
    -------

    """
    patients = counts.shape[1]
    static = pd.Series([1] * patients)
    colData = pd.DataFrame({"col1": static})
    colData.index = counts.columns
    counts_r = pandasToR(counts)
    colData_r = pandasToR(colData)
    # use design 1 to not make use of any design information
    dds = DESeq2.DESeqDataSetFromMatrix(countData=counts_r, colData=colData_r,
                                        design=Formula("~1"))
    # Adjustment so that colData is not taken into account anymore
    dds.sizeFactor = None
    dds = DESeq2.estimateSizeFactors_DESeqDataSet(dds)
    dds = DESeq2.estimateDispersions_DESeqDataSet(dds)
    normalized_counts = DESeq2.varianceStabilizingTransformation(dds, blind=False)
    normalized_counts = summarizedExperiment.assay(normalized_counts)
    pd_normalized_counts = pd.DataFrame(normalized_counts, columns=counts.columns)
    pd_normalized_counts = pd_normalized_counts.set_axis(counts.index, axis="index")
    return pd_normalized_counts


def edgeRNormalization(counts: pd.DataFrame, log=True) -> pd.DataFrame:
    """
    Applies CPM normalization from EdgeR on the data.
    Parameters
    ----------
    log: boolean flag whether the counts are log-transformed as well
    counts: count Matrix with raw counts

    Returns: count Matrix with normalized counts
    -------

    """
    counts_r = pandasToR(counts)
    y = edgeR.DGEList(counts_r)
    # not sure whether calcNormFactors should be used
    y = edgeR.calcNormFactors(y, method="TMM")
    cpms = edgeR.cpm(y, log=log)
    pd_cmps = pd.DataFrame(cpms, columns=counts.columns)
    pd_cmps = pd_cmps.set_axis(labels=counts.index, axis="index")
    return pd_cmps


def pandasToR(dataframe: pd.DataFrame):
    """
    Transforms a pandas counts to R
    Parameters
    ----------
    dataframe: Pandas counts

    Returns: R counts corresponding to given pandas Dataframe
    -------

    """
    with localconverter(ro.default_converter + pandas2ri.converter):
        return ro.conversion.py2rpy(dataframe)


def RToPandas(dataframe) -> pd.DataFrame:
    """
    Transforms a counts in R to a counts in pandas
    Parameters
    ----------
    dataframe: R counts

    Returns: counts transformed to pandas
    -------

    """
    with localconverter(ro.default_converter + pandas2ri.converter):
        return ro.conversion.rpy2py(dataframe)
