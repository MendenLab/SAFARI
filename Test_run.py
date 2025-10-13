import json
import logging
import os
import pdb

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from scipy.stats import gmean
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, TunedThresholdClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from Validation_run import CompositeStratifiedKFold, augment_data, normalize_raw_counts, read_features, \
    createModel, get_scorer, read_train_data, quality_control_qpcr
from pyutils import get_qpcr_parameter_space, sort_data


def fit_predict(train_data, train_labels, test_data, scorer, mode="baseline", save_params=False, augmentation=True, calibration=False,
                output_path=None)->pd.DataFrame:
    """
    Trains Logistic Regression model on training data with augmentation and generates predictions for test data.
    Parameters
    ----------
    output_path
    calibration
    train_data
    train_labels
    test_data
    mode: baseline, gridsearch_Cvb or threshold_tuning
    save_params: Bool flag indicating whether the coefficients are stored or not

    Returns
    -------

    """
    param_grid = get_qpcr_parameter_space()
    scorer = get_scorer(scorer_name=scorer)
    output_dir = os.path.join(output_path, "params")
    # output_dir = "/lustre/groups/cbm01/datasets/martin.meinel/Safari/params"
    # output_dir = "/Users/martin.meinel/Desktop/Projects/Eyerich Projects/Natalie/Classifier/Validation data/Test data/Test_set_predictions_paper/local/params"
    os.makedirs(output_dir, exist_ok=True)
    sorted_features = "_".join(train_data.columns)
    if augmentation:
        if calibration:
            params_file = os.path.join(output_dir, f"{sorted_features}_{mode}_calibrated_params_sorted.json")
        else:
            params_file = os.path.join(output_dir, f"{sorted_features}_{mode}_params.json")
    else:
        if calibration:
            params_file = os.path.join(output_dir, f"{sorted_features}_{mode}_train_data_only_calibrated_params.json")
        else:
            params_file = os.path.join(output_dir, f"{sorted_features}_{mode}_train_data_only_params.json")
    # Align with test data
    test_data = test_data.loc[:, train_data.columns]
    pipe = Pipeline(steps=[('scaler', StandardScaler()),
                           ("estimator", LogisticRegression(random_state=0, max_iter=500, class_weight="balanced", solver="saga", penalty="l2", C=1))])
    if mode == "baseline":
        if calibration:
            middle_cv = CompositeStratifiedKFold(n_splits=5, shuffle=True, random_state=41)
            calibrated_classifier = CalibratedClassifierCV(pipe, method="sigmoid", cv=middle_cv, n_jobs=-1, ensemble=False)
            calibrated_classifier.fit(train_data, train_labels)
            y_hat = calibrated_classifier.predict(test_data)
            y_hat_probs = calibrated_classifier.predict_proba(test_data)[:, 1]
            # Store the parameters
            calibrator = calibrated_classifier.calibrated_classifiers_[0]
            a = calibrator.calibrators[0].a_
            b = calibrator.calibrators[0].b_
            # 3. Extract beta coefficients and standardization parameters
            pipeline = calibrator.estimator
            scaler = pipeline['scaler']
            estimator = pipeline['estimator']
            scaling_mean = scaler.mean_
            scaling_std = scaler.scale_
            coefficients = estimator.coef_[0]
            intercept = estimator.intercept_[0]
            model_params = {
                'intercept': intercept,
                'coefficients': dict(zip(train_data.columns, coefficients)),
                'scaler_mean': dict(zip(train_data.columns, scaling_mean)),
                'scaler_scale': dict(zip(train_data.columns, scaling_std)),
                'a': a,
                'b': b
            }
            json_params = {
                'intercept': float(model_params['intercept']),
                'coefficients': {k: float(v) for k, v in model_params['coefficients'].items()},
                'scaler_mean': {k: float(v) for k, v in model_params['scaler_mean'].items()},
                'scaler_scale': {k: float(v) for k, v in model_params['scaler_scale'].items()},
                'a': float(model_params['a']),
                'b': float(model_params['b'])

            }
        else:
            pipe.fit(train_data, train_labels)
            y_hat = pipe.predict(test_data)
            y_hat_probs = pipe.predict_proba(test_data)[:, 1]
            model_params = {
                'intercept': pipe["estimator"].intercept_[0],
                'coefficients': dict(zip(train_data.columns,
                                         pipe["estimator"].coef_[0])),
                'scaler_mean': dict(zip(train_data.columns,
                                        pipe["scaler"].mean_)),
                'scaler_scale': dict(zip(train_data.columns,
                                         pipe["scaler"].scale_))
            }
            json_params = {
                'intercept': float(model_params['intercept']),
                'coefficients': {k: float(v) for k, v in model_params['coefficients'].items()},
                'scaler_mean': {k: float(v) for k, v in model_params['scaler_mean'].items()},
                'scaler_scale': {k: float(v) for k, v in model_params['scaler_scale'].items()}
            }
    elif mode == "gridsearch_cv":
        sfk = CompositeStratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        search = GridSearchCV(pipe, param_grid=param_grid, scoring=scorer, cv=sfk, n_jobs=-1, error_score="raise")
        if calibration:
            middle_cv = CompositeStratifiedKFold(n_splits=5, shuffle=True, random_state=41)
            calibrated_classifier = CalibratedClassifierCV(search, method="sigmoid", cv=middle_cv, ensemble=False,
                                                           n_jobs=-1)
            calibrated_classifier.fit(train_data, train_labels)
            y_hat = calibrated_classifier.predict(test_data)
            y_hat_probs = calibrated_classifier.predict_proba(test_data)[:, 1]


            # Extract the calibration parameters
            calibrator = calibrated_classifier.calibrated_classifiers_[0]
            a = calibrator.calibrators[0].a_
            b = calibrator.calibrators[0].b_
            estimator = calibrator.estimator.best_estimator_
            scaler = estimator['scaler']
            coefficients = estimator['estimator'].coef_[0]
            intercept = estimator['estimator'].intercept_[0]
            model_params = {
                'intercept': intercept,
                'coefficients': dict(zip(train_data.columns, coefficients)),
                'scaler_mean': dict(zip(train_data.columns, scaler.mean_)),
                'scaler_scale': dict(zip(train_data.columns, scaler.scale_)),
                'a': a,
                'b': b
            }
            json_params = {
                'intercept': float(model_params['intercept']),
                'coefficients': {k: float(v) for k, v in model_params['coefficients'].items()},
                'scaler_mean': {k: float(v) for k, v in model_params['scaler_mean'].items()},
                'scaler_scale': {k: float(v) for k, v in model_params['scaler_scale'].items()},
                'a': float(model_params['a']),
                'b': float(model_params['b'])

            }
        else:
            search.fit(train_data, train_labels)
            y_hat = search.predict(test_data)
            y_hat_probs = search.predict_proba(test_data)[:, 1]
            model_params = {
                'intercept': search.best_estimator_[1].intercept_[0],
                'coefficients': dict(zip(train_data.columns, search.best_estimator_[1].coef_[0])),
                'scaler_mean': dict(zip(train_data.columns, search.best_estimator_[0].mean_)),
                'scaler_scale': dict(zip(train_data.columns, search.best_estimator_[0].scale_))
            }
            json_params = {
                'intercept': float(model_params['intercept']),
                'coefficients': {k: float(v) for k, v in model_params['coefficients'].items()},
                'scaler_mean': {k: float(v) for k, v in model_params['scaler_mean'].items()},
                'scaler_scale': {k: float(v) for k, v in model_params['scaler_scale'].items()}
            }
    elif mode == "threshold_tuning":
        outer_cv = CompositeStratifiedKFold(n_splits=5, shuffle=True, random_state=41)
        inner_cv = CompositeStratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        search = GridSearchCV(pipe, param_grid=param_grid, scoring=scorer, cv=inner_cv, n_jobs=-1, error_score="raise")
        threshold_tuner = TunedThresholdClassifierCV(estimator=search, scoring=scorer, cv=outer_cv, n_jobs=-1)
        threshold_tuner.fit(train_data, train_labels)
        y_hat = threshold_tuner.predict(test_data)
        y_hat_probs = threshold_tuner.predict_proba(test_data)[:, 1]
        model_params = {
            'intercept': threshold_tuner.estimator_.best_estimator_[1].intercept_[0],
            'coefficients': dict(zip(train_data.columns, threshold_tuner.estimator_.best_estimator_[1].coef_[0])),
            'scaler_mean': dict(zip(train_data.columns, threshold_tuner.estimator_.best_estimator_[0].mean_)),
            'scaler_scale': dict(zip(train_data.columns, threshold_tuner.estimator_.best_estimator_[0].scale_)),
            'decision_threshold': float(threshold_tuner.best_threshold_)
        }
        json_params = {
            'intercept': float(model_params['intercept']),
            'coefficients': {k: float(v) for k, v in model_params['coefficients'].items()},
            'scaler_mean': {k: float(v) for k, v in model_params['scaler_mean'].items()},
            'scaler_scale': {k: float(v) for k, v in model_params['scaler_scale'].items()},
            'decision_threshold': float(model_params['decision_threshold'])
        }
    else: raise ValueError("Invalid mode")
    y_hat = np.where(y_hat == 0, "Eczema|Psoriasis", "MF")
    if save_params:
        with open(params_file, 'w') as f:
            json.dump(json_params, f, indent=4)
        print(f"\nModel parameters saved to: {params_file}")
    if calibration:
        results = pd.DataFrame(data= {f"{sorted_features}_{mode}_predictions_calibrated": y_hat, f"{sorted_features}_{mode}_probabilities_calibrated": y_hat_probs},
                        index=test_data.index)
    else:
        results = pd.DataFrame(data= {f"{sorted_features}_{mode}_predictions": y_hat, f"{sorted_features}_{mode}_probabilities": y_hat_probs},
                        index=test_data.index)
    return results


def read_test_data(path: str, genes, debug, skip_ct_mask=False)-> pd.DataFrame:
    """
    Reads the entire test data from WB, Kempf, NL23, NL35 time series
    Parameters
    ----------
    path:
    genes:

    Returns
    -------

    """
    if debug:
        pdb.set_trace()
    data = pd.read_excel(path)
    data.set_index("sampleID", inplace=True)
    data = data.loc[:, genes]
    faulty_test_samples = quality_control_qpcr(data, genes=genes, mode="test", skip_ct_mask=skip_ct_mask)
    print(f"Faulty test samples: {faulty_test_samples}")
    data.drop(index=faulty_test_samples, inplace=True)
    data = data.apply(pd.to_numeric, errors='coerce')
    data = data.astype(float)
    data = data.groupby("sampleID").aggregate(gmean)
    akf4_training_samples = ["AKF4-001", "AKF4-002", "AKF4-004", "AKF4-005", "AKF4-006", "AKF4-008", "AKF4-009", "AKF4-011", "AKF4-012"]
    data.drop(index = akf4_training_samples, inplace=True)
    data.rename(index={"AKF4-003": "AKF1-16_Rep", "AKF4-007": "AKF1-14_Rep", "AKF4-013": "AKF1-12_Rep", "AKF4-010": "NL35-020_Rep"}, inplace=True)
    return data


@hydra.main(version_base=None, config_path="conf", config_name="test_predictions")
def main(cfg: DictConfig):
    log = logging.getLogger(__name__)
    run_config = getattr(cfg, cfg.run_type)
    log.info(fr"Run config for: {run_config}")
    if cfg.skip_ct_mask:
        log.warning("Skipping CT mask")
    today = pd.Timestamp.now().strftime("%Y-%m-%d")
    features = read_features(run_config.featureFile)
    if cfg.debug:
        pdb.set_trace()
    features = np.append(features, cfg.reference_gene)
    features_string = "_".join(sorted(features))
    if ("SERBP1" in features) | ("BTN3A1" in features):
        for r in cfg.reference_gene:
            features = np.append(features, r + "_2")
    # Load the training data
    data, train_labels = read_train_data(run_config.trainingSet, features, binary_labels=True)
    data = sort_data(data)
    train_labels = train_labels.loc[data.index]
    assert sum(data.index.str.startswith("AKF4") == 0)
    train_data = data.loc[:, features]
    features_wo2 = [f for f in features if f.endswith("_2") is False]
    assert train_data.loc[:, features_wo2].isna().sum().sum() == 0
    # Fill nas here due to problems later in cv splitting
    train_data.fillna(value=-100, inplace=True)
    log.info(f"Number of training samples: {train_data.shape[0]}")
    dataset_path = run_config.save_dir
    dataset_path = os.path.join(dataset_path, today)
    os.makedirs(dataset_path, exist_ok=True)
    if cfg.calibration:
        log.info("Calibration of probabilities")
    if cfg.debug:
        pdb.set_trace()
    if cfg.predict == "test":
        log.info("Predict Test data")
        test_data = read_test_data(run_config.testSet, genes=features, debug=cfg.debug, skip_ct_mask=cfg.skip_ct_mask)
        # Subset test data to respective housekeepers and
        log.info(f"Number of test samples: {test_data.shape[0]}")
        if cfg.debug:
            pdb.set_trace()
        if cfg.train_augmentation:
            train_data = augment_data(train_data, reference_gene=cfg.reference_gene,
                                                augmentation_method="noise", augmentation_strength=0.5,
                                      repetition=0)
            train_labels = np.concatenate([train_labels, train_labels]).ravel().astype('int64')
        else:
            train_labels = train_labels.values.ravel().astype('int64')
        normalized_train_data = normalize_raw_counts(train_data, reference_genes=cfg.reference_gene)
        # Normalization of test data
        normalized_test_data = normalize_raw_counts(test_data, reference_genes=cfg.reference_gene)
        if cfg.debug:
            pdb.set_trace()
        assert normalized_test_data.isna().sum().sum() == 0
        results = fit_predict(train_data=normalized_train_data, train_labels=train_labels, test_data=normalized_test_data,
                                 scorer=cfg.scorer,  mode=cfg.mode, save_params=True, augmentation=cfg.train_augmentation,
                              calibration=cfg.calibration, output_path=dataset_path)
        if cfg.train_augmentation:
            if cfg.calibration:
                output_file = os.path.join(dataset_path,
                                           f"augmented_mixed_data_test_calibrated_{features_string}_{cfg.mode}_{today}_{cfg.scorer}.csv")
            else:
                output_file = os.path.join(dataset_path,f"augmented_mixed_data_test_{features_string}_{cfg.mode}_{today}_{cfg.scorer}.csv")
        else:
            output_file = os.path.join(dataset_path,f"original_mixed_test_data_{features_string}_{cfg.mode}_{today}.csv")
        results.to_csv(output_file)
    else:
        log.info("Predict Training data")
        outer_split = CompositeStratifiedKFold(n_splits=4, random_state=0, shuffle=True)
        indizes_over_all_folds = []
        predictions_over_all_folds = []
        pos_probabilities_over_all_folds = []
        for train_idx, test_idx in outer_split.split(X=train_data, y=train_labels):
            trainfold_data, testfold_data = train_data.iloc[train_idx].copy(), train_data.iloc[test_idx].copy()
            trainfold_label, testfold_labels = train_labels.iloc[train_idx].to_numpy().ravel(), train_labels.iloc[
                test_idx].to_numpy().ravel()
            # Normalize training data, Permute test data and normalize test data
            if cfg.train_augmentation:
                train_data_augmented = augment_data(trainfold_data, reference_gene=cfg.reference_gene,
                                          augmentation_method="noise",
                                          augmentation_strength=0.5, repetition=0)
                train_labels_concatenated = np.concatenate([trainfold_label, trainfold_label]).ravel().astype('int64')
            else:
                train_data_augmented = trainfold_data.copy()
                train_labels_concatenated = trainfold_label.copy().ravel().astype('int64')
            normalized_train = normalize_raw_counts(train_data_augmented, reference_genes=cfg.reference_gene).copy()
            normalized_test = normalize_raw_counts(testfold_data, reference_genes=cfg.reference_gene)
            # Create model here to ensure a new model in each run
            weighting_factor = (sum(train_labels_concatenated == 0) / sum(train_labels_concatenated == 1))
            estimator = createModel(cfg.model, kernel=cfg.kernel, scale_pos_weight=weighting_factor)
            pipeline = Pipeline(steps=[('scaler', StandardScaler()), ("estimator", estimator)])
            param_grid = get_qpcr_parameter_space()
            ind_scorer = get_scorer(scorer_name=cfg.scorer)
            if cfg.mode == "baseline":
                if cfg.calibration:
                    middle_cv = CompositeStratifiedKFold(n_splits=5, shuffle=True, random_state=41)
                    calibrated_classifier = CalibratedClassifierCV(pipeline, method="sigmoid", cv=middle_cv, ensemble=False,
                                                                   n_jobs=-1)
                    calibrated_classifier.fit(X=normalized_train, y=train_labels_concatenated)
                    y_hat = calibrated_classifier.predict(normalized_test)
                    probs_pos = calibrated_classifier.predict_proba(normalized_test)[:, 1]
                else:
                    pipeline.fit(X=normalized_train, y=train_labels_concatenated)
                    y_hat = pipeline.predict(X=normalized_test)
                    probs_pos = pipeline.predict_proba(X=normalized_test)[:, 1]
            elif cfg.mode == "gridsearch_cv":
                inner_cv = CompositeStratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                search = GridSearchCV(pipeline, param_grid=param_grid,
                                      scoring=ind_scorer, cv=inner_cv, n_jobs=-1,
                                      error_score="raise")
                if cfg.calibration:
                    middle_cv = CompositeStratifiedKFold(n_splits=5, shuffle=True, random_state=41)
                    calibrated_classifier = CalibratedClassifierCV(search, method="sigmoid", cv=middle_cv, ensemble=False,
                                                                   n_jobs=-1)
                    calibrated_classifier.fit(X=normalized_train, y=train_labels_concatenated)
                    y_hat = calibrated_classifier.predict(normalized_test)
                    probs_pos = calibrated_classifier.predict_proba(normalized_test)[:, 1]
                else:
                    search.fit(X=normalized_train, y=train_labels_concatenated)
                    y_hat = search.predict(normalized_test)
                    probs_pos = search.predict_proba(normalized_test)[:, 1]
            elif cfg.mode == "threshold_tuning":
                middle_cv = CompositeStratifiedKFold(n_splits=5, shuffle=True, random_state=41)
                inner_cv = CompositeStratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                search = GridSearchCV(pipeline, param_grid=param_grid, scoring=ind_scorer, cv=inner_cv, n_jobs=-1)
                threshold_tuner = TunedThresholdClassifierCV(estimator=search, scoring=ind_scorer, cv=middle_cv, n_jobs=-1)
                threshold_tuner.fit(X=normalized_train, y=train_labels_concatenated)
                y_hat = threshold_tuner.predict(normalized_test)
                probs_pos = threshold_tuner.predict_proba(normalized_test)[:, 1]
            else:
                raise ValueError("Invalid mode")
            pos_probabilities_over_all_folds.extend(probs_pos)
            predictions_over_all_folds.extend(y_hat)
            indizes_over_all_folds.extend(normalized_test.index)
        if cfg.debug:
            pdb.set_trace()
        predictions_over_all_folds = np.where(np.array(predictions_over_all_folds) == 0, "Eczema|Psoriasis", "MF")
        sorted_features = "_".join(normalized_train.columns)
        if cfg.calibration:
            training_predictions = pd.DataFrame(data= {f"{sorted_features}_{cfg.mode}_predictions_calibrated": predictions_over_all_folds, f"{sorted_features}_{cfg.mode}_probabilities_calibrated": pos_probabilities_over_all_folds},
                            index=indizes_over_all_folds)
            if cfg.train_augmentation:
                output_file = os.path.join(dataset_path, f"augmented_mixed_data_train_calibrated_{features_string}_{cfg.mode}_{today}_{cfg.scorer}.csv")
            else:
                output_file = os.path.join(dataset_path, f"original_mixed_data_train_calibrated_{features_string}_{cfg.mode}_{today}_{cfg.scorer}.csv")
        else:
            training_predictions = pd.DataFrame(data= {f"{sorted_features}_{cfg.mode}_predictions": predictions_over_all_folds, f"{sorted_features}_{cfg.mode}_probabilities": pos_probabilities_over_all_folds},
                            index=indizes_over_all_folds)
            output_file = os.path.join(dataset_path, f"mixed_data_train_{features_string}_{cfg.mode}_{today}_{cfg.scorer}.csv")
        training_predictions.to_csv(output_file)

if __name__ == '__main__':
    main()