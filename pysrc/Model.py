import os
import pdb

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score, f1_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from Dataset import Dataset
from R_helpers import edgeRNormalization
from pyutils import (compute_sensitivity, compute_specificity,
                     createLogisticRegressionClassifier, createSVM, createXGBoost,  get_scorer, createModel)


class Model:
    def __init__(self, dataset: Dataset, hugo_gene_list: list):
        self.hugoGeneList = hugo_gene_list
        self.x_train: pd.DataFrame = dataset.x_train  # D x N
        self.colData = dataset.colData
        self.y_train = dataset.y_train["diag"]
        self.x_test: pd.DataFrame = dataset.x_test  # D x N
        self.colData_test = dataset.colData_test
        self.y_test: np.ndarray = dataset.y_test["diag"]
        self.rowData: pd.DataFrame = dataset.rowData
        self.model = None  # Stores the best model
        self.normalized_x_train: pd.DataFrame = None  # N x D
        self.normalized_x_test: pd.DataFrame = None  # N x D
        self.features = None
        self.__mapToEnsemblIds()
        self.__reduceDataset()

    def __mapToEnsemblIds(self):
        """
        Maps Hugo_Gene_names to
        Returns
        -------

        """
        features = self.rowData.loc[self.rowData["Gene_name"].isin(self.hugoGeneList), ["ensembl_id", "Gene_name"]]
        features.drop_duplicates(subset=["Gene_name"], inplace=True)
        features = features["ensembl_id"]
        assert len(features) == len(set(self.hugoGeneList)), f"Probably a wrong gene name was fed in"
        self.features = features

    def __reduceDataset(self):
        """
        Filter dataset for particular features and normalise dataset with TMM normalization
        Returns dataset with only the given features
        -------

        """
        # Apply vst on Training set
        self.normalized_x_train = edgeRNormalization(self.x_train, log=True)
        self.normalized_x_train = self.normalized_x_train.loc[self.normalized_x_train.index.isin(self.features), :].T
        self.normalized_x_test = edgeRNormalization(self.x_test, log=True)
        self.normalized_x_test = self.normalized_x_test.loc[self.normalized_x_test.index.isin(self.features), :].T


    def estimatePerformance(self, model_name, kernel, base_path: str, scorer, debug=False):
        """
        Used to estimate the performance of a given model using nested cross-validation. The inner loop is used for
         hyperparameter tuning while th outer one is used to evaluate the performance.
        Parameters
        ----------
        model_name: defines the model logistic, SVM or xgboost
        kernel: Specifies the kernel if an SVM model is chosen
        scorer: scorer for hyperparameter tuning
        -------

        """
        base_path = os.path.join(base_path, scorer)
        os.makedirs(base_path, exist_ok=True)
        scorer = get_scorer(scorer_name=scorer)
        repetitions = 100
        if debug:
            repetitions = 2
        roc_auc_scores = []
        pr_auc_scores = []
        balanced_acc_scores = []
        f1_scores = []
        specificity_scores = []
        sensitivity_scores = []
        weighted_f1_scores = []
        # Nested CV starts here
        for rep in tqdm(range(repetitions)):
            cv_outer = StratifiedKFold(n_splits=4, shuffle=True, random_state=rep)
            # Predictions, probabilities of 0 class, and ground truth for the entire shuffled training set
            predictions = []
            ground_truth = []
            # used for PR AUC
            probabilities = []
            # used for ROC auc
            pos_probabilities = []
            for train_ix, test_ix in cv_outer.split(self.normalized_x_train, self.y_train):
                if model_name == "logistic":
                    model, params, optimizer = createLogisticRegressionClassifier()
                elif model_name == "SVM":
                    model, params, optimizer = createSVM(kernel=kernel)
                elif model_name == "xgboost":
                    weighting_factor = (sum(self.y_train[train_ix] == 0) / sum(self.y_train[train_ix] == 1))
                    model, params, optimizer = createXGBoost(weighting_factor)
                else:
                    raise ValueError("Model name not recognized")
                model = Pipeline(steps=[("scaler", StandardScaler()), ("estimator", model)])
                model_name_sklearn = model['estimator'].__class__.__name__
                cv_inner = StratifiedKFold(n_splits=3)
                if optimizer == "grid":
                    search = GridSearchCV(model, params, scoring=scorer, cv=cv_inner)
                else:
                    search = RandomizedSearchCV(model, params, scoring=scorer, cv=cv_inner)
                X_train, X_test = self.normalized_x_train.iloc[train_ix, :].to_numpy(), self.normalized_x_train.iloc[
                                                                                        test_ix, :].to_numpy()
                y_train, y_test = self.y_train[train_ix], self.y_train[test_ix]
                result = search.fit(X_train, y_train)
                best_model = result.best_estimator_
                y_hat = best_model.predict(X_test)
                if model_name_sklearn in ["LogisticRegression", "RandomForestClassifier", "XGBClassifier",
                                  "BalancedRandomForestClassifier"]:
                    probs = best_model.predict_proba(X_test)
                    probs_pos = probs[:, 1]
                    probs = probs[:, 0]
                else:
                    probs = best_model.decision_function(X_test) * (-1)
                    probs_pos = best_model.decision_function(X_test)
                # PR AUC
                probabilities.extend(probs)
                # ROC AUC
                pos_probabilities.extend(probs_pos)
                # extract the samples which are wrongly classified
                predictions.extend(y_hat)
                ground_truth.extend(y_test)
            # Precision recall curve and AUC
            pr_auc = average_precision_score(ground_truth, probabilities, pos_label=0)
            pr_auc_scores.append(pr_auc)
            # Evaluate balanced accuracy scores
            acc_score = balanced_accuracy_score(y_true=ground_truth, y_pred=predictions)
            balanced_acc_scores.append(acc_score)
            # Evaluate f1 scores
            f1_sco = f1_score(y_true=ground_truth, y_pred=predictions)
            f1_scores.append(f1_sco)
            # Evaluate sensitivity
            specificity = compute_specificity(y_true=ground_truth, y_pred=predictions)
            specificity_scores.append(specificity)
            # Sensitivity TP/(TP+FN)
            sensitivity = compute_sensitivity(y_true=ground_truth, y_pred=predictions)
            sensitivity_scores.append(sensitivity)
            # ROC AUC score
            auc_score = roc_auc_score(y_true=ground_truth, y_score=pos_probabilities)
            roc_auc_scores.append(auc_score)
            # Weighted F1-score
            weighted_f1_scores.append(f1_score(y_true=ground_truth, y_pred=predictions, average="weighted"))
            # Print confusion matrix
            print(confusion_matrix(y_true=ground_truth, y_pred=predictions))
        if debug:
            pdb.set_trace()
        # Create file
        df = dict(accuracy = balanced_acc_scores, f1 = f1_scores, weighted_f1 = weighted_f1_scores,
                  sensitivity = sensitivity_scores, specificity = specificity_scores,
                  roc_auc = roc_auc_scores, pr_auc = pr_auc_scores)
        df = pd.DataFrame(data=df)
        features = self.normalized_x_train.columns.tolist()
        hugo_gene_names = self.rowData.loc[features, "Gene_name"].values.tolist()
        hugo_gene_names = sorted(hugo_gene_names)
        hugo_gene_names = "_".join(hugo_gene_names)
        path = os.path.join(base_path,
                            f"{model_name}_{hugo_gene_names}.h5")
        df.to_hdf(path, key="results", mode="w")


    def forward_selection(self, model_name="logistic"):
        """
        This is used for the baseline. We do Forward Selection for genes and optimize for the F1-score with (0 as positive label).
        We also use a tolerance of 0.005 to stop if we cannot optimize any further and not to overfit.
        Returns: Selected genes
        -------

        """
        selected_features = []
        best_performance = 0.0
        best_sensitivity = 0.0
        best_specificity = 0.0
        best_bal_accuracy = 0.0
        tolerance = 0.005

        while len(selected_features) < self.normalized_x_train.shape[1]:
            remaining_features = sorted(list(set(self.normalized_x_train.columns) - set(selected_features)))
            performances = []
            sensitivities = []
            specificities = []
            balanced_accuracies = []
            for feature in remaining_features:
                current_features = selected_features + [feature]
                X = self.normalized_x_train.loc[:, current_features]
                cv = StratifiedKFold(n_splits=5)
                predictions = []
                ground_truth = []
                for train_ix, test_ix in cv.split(X, self.y_train):
                    # Create model instance here
                    X_train = X.iloc[train_ix, :]
                    X_test = X.iloc[test_ix, :]
                    y_train = self.y_train[train_ix]
                    y_test = self.y_train[test_ix]
                    weighting_factor = (sum(y_train == 0) / sum(y_train == 1))
                    estimator = createModel(model=model_name, kernel="linear", scale_pos_weight=weighting_factor)
                    model = Pipeline(steps=[("scaler", StandardScaler()), ("estimator", estimator)])
                    model.fit(X_train, y_train)
                    y_hat = model.predict(X_test)
                    predictions.extend(y_hat)
                    ground_truth.extend(y_test)
                performance = f1_score(y_true=ground_truth, y_pred=predictions)
                performances.append(performance)
                sensitivity = compute_sensitivity(y_true=ground_truth, y_pred=predictions)
                sensitivities.append(sensitivity)
                specificity = compute_specificity(y_true=ground_truth, y_pred=predictions)
                specificities.append(specificity)
                balanced_acc = balanced_accuracy_score(y_true=ground_truth, y_pred=predictions)
                balanced_accuracies.append(balanced_acc)

            best_feature = np.argmax(performances)
            current_best_performance = performances[best_feature]

            if current_best_performance > best_performance + tolerance:
                best_performance = current_best_performance
                best_sensitivity = sensitivities[best_feature]
                best_specificity = specificities[best_feature]
                best_bal_accuracy = balanced_accuracies[best_feature]
                # Store best feature of selection round
                selected_features.append(remaining_features[best_feature])
            else:
                break
        # Mapping to hugo gene names
        gene_data = self.rowData.loc[selected_features, ["ensembl_id", "Gene_name"]]
        print(f"Best f1-score (pos_label = 0): {best_performance}")
        print(f"Best Specificity: {best_specificity}")
        print(f"Best Sensitivity: {best_sensitivity}")
        print(f"Best Balanced accuracy: {best_bal_accuracy}")
        return gene_data, best_performance
