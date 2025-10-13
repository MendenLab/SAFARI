import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from R_helpers import edgeRNormalization

warnings.filterwarnings("ignore", category=ConvergenceWarning)


class ClassificationResult:
    """Used for storing the result for each training set"""

    def __init__(self, coefficients, features):
        self.coefficients = coefficients
        self.features = features


class TrainingSet:
    def __init__(self, countData: pd.DataFrame, colData: pd.DataFrame, labels: np.ndarray):
        # raw counts and corresponding colData and labels of 81 eczema and 13 mf samples
        self.countData: pd.DataFrame = countData
        self.colData: pd.DataFrame = colData
        self.labels: np.ndarray = labels
        # contains vst normalized counts and reduced gene counts
        self.normalizedCounts: pd.DataFrame = None  # in the form dxN
        self.__applyTMMTransformation()
        self.trainingResult: ClassificationResult = None

    def __applyTMMTransformation(self):
        """
        Apply variance stabilizing transformation for the gene counts / apply standardization as well
        Returns
        -------

        """
        self.normalizedCounts = edgeRNormalization(self.countData)

    def filterSignificantGenes(self, significantGenes: list):
        """
        Removes the genes which are not consistently differentially expressed for all the DESeq2 analysis
        Parameters
        ----------
        significantGenes: genes which are consistently up- or down regulated in the paired analysis between the diseases
         and non-lesional samples but also between the two diseases

        Returns: countData and normalizedCountData containing only the significantGenes
        -------

        """
        self.countData = self.countData.loc[self.countData.index.isin(significantGenes), :]
        self.normalizedCounts = self.normalizedCounts.loc[self.normalizedCounts.index.isin(significantGenes), :]
        assert self.countData.index.equals(self.normalizedCounts.index)


    def getEstimator(self, estimator: str):
        """
        Given an estimator name we create an estimator with the default parameters but with parameters for the
         imbalanced dataset to use this estimator for the forward feature selection process in train
        Parameters
        ----------
        estimator: Name of estimator (SVM, logistic, xgboost)

        Returns: Model of desired estimator with default parameters, but set up for imbalanced data set
        -------

        """
        weighting_factor = (sum(self.labels == 0) / sum(self.labels == 1))
        if estimator == "logistic":
            estimator = LogisticRegression(class_weight="balanced", solver="saga", random_state=0)
        elif estimator == "xgboost":
            estimator = xgb.XGBClassifier(objective="binary:logistic", scale_pos_weight=weighting_factor, random_state=0)
        else:
            estimator = LinearSVC(class_weight="balanced", random_state=0)
        return estimator

    def train(self, method: str = "logistic", ffs_model="logistic"):
        """
        Trains a logistic regression model with different weights for MF and eczema (6:1)
        and stores the parameters and its features in ClassificationResult
        Parameters:
        ------------
        @method: uses the parameters of a logistic regression model to rank features accordingly if "logistic" was chosen
        otherwise uses shapley values of a Random Forest to estimate

        Returns: ClassificationResult
        -------

        """
        if len(np.unique(self.labels)) > 2:
            raise ValueError("Not implemented for more than two classes.")
        scl = StandardScaler()
        if method == "logistic":
            # Logistic Regression for two classes [use default values]
            model = LogisticRegression(solver="saga", max_iter=500, penalty="l1", class_weight="balanced",
                                       random_state=0)
            model = Pipeline(steps=[("scaler", scl), ("estimator", model)])
            model.fit(self.normalizedCounts.T.to_numpy(), self.labels)
            coefficients = np.zeros(self.normalizedCounts.shape[0])
            coefficients += np.squeeze(model[1].coef_)
            sorted_args_coefficients = np.argsort(np.abs(coefficients))[::-1]
            sorted_coefficients = coefficients[sorted_args_coefficients]
            sorted_features = self.normalizedCounts.index[sorted_args_coefficients].values
            # omit features who are not relevant for the model
            features = sorted_features[sorted_coefficients != 0]
            coefficients = sorted_coefficients[sorted_coefficients != 0]
        else:
            selected_features = []
            best_performance = 0.0
            tolerance = 0.005
            # Here the shape is D times N
            while len(selected_features) < self.normalizedCounts.shape[0]:
                remaining_features = sorted(list(set(self.normalizedCounts.index) - set(selected_features)))
                performances = []
                for feature in remaining_features:
                    current_features = selected_features + [feature]
                    X = self.normalizedCounts.loc[current_features, :].T  # X has shape N times D
                    cv = StratifiedKFold(n_splits=4)
                    estimator = self.getEstimator(ffs_model)
                    model = Pipeline(steps=[("scaler", StandardScaler()),
                                            ("estimator", estimator)])
                    predictions = []
                    ground_truth = []
                    for train_ix, test_ix in cv.split(X, self.labels):
                        X_train = X.iloc[train_ix, :]
                        X_test = X.iloc[test_ix, :]
                        y_train = self.labels[train_ix]
                        y_test = self.labels[test_ix]
                        model.fit(X_train, y_train)
                        y_hat = model.predict(X_test)
                        predictions.extend(y_hat)
                        ground_truth.extend(y_test)
                    performance = f1_score(y_true=ground_truth, y_pred=predictions)
                    performances.append(performance)

                best_feature = np.argmax(performances)
                current_best_performance = performances[best_feature]

                if current_best_performance > best_performance + tolerance:
                    best_performance = current_best_performance
                    # Store best feature of selection round
                    selected_features.append(remaining_features[best_feature])
                else:
                    break
            features = selected_features
            coefficients = [i for i in range(len(features))]
        self.trainingResult = ClassificationResult(coefficients=coefficients, features=features)