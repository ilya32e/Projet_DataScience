from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from retention_ai.config import CV_FOLDS, RANDOM_STATE
from retention_ai.features import BusinessFeatureEngineer


@dataclass(frozen=True)
class ModelSpec:
    name: str
    label: str
    family: str
    estimator: object
    use_sampler: bool = False


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, selector(dtype_include=np.number)),
            ("categorical", categorical_pipeline, selector(dtype_include=object)),
        ],
        sparse_threshold=0.0,
        verbose_feature_names_out=False,
    )


def get_model_specs() -> list[ModelSpec]:
    return [
        ModelSpec(
            name="logistic_regression",
            label="Logistic Regression",
            family="Baseline interpretable",
            estimator=LogisticRegression(
                class_weight="balanced",
                max_iter=1500,
                random_state=RANDOM_STATE,
                solver="liblinear",
            ),
        ),
        ModelSpec(
            name="random_forest",
            label="Random Forest",
            family="Ensemble arbres",
            estimator=RandomForestClassifier(
                class_weight="balanced_subsample",
                max_depth=None,
                min_samples_leaf=2,
                n_estimators=350,
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
        ),
        ModelSpec(
            name="gradient_boosting",
            label="Gradient Boosting",
            family="Boosting",
            estimator=GradientBoostingClassifier(
                learning_rate=0.05,
                n_estimators=250,
                random_state=RANDOM_STATE,
                subsample=0.85,
            ),
            use_sampler=True,
        ),
        ModelSpec(
            name="mlp_classifier",
            label="MLP Classifier",
            family="Deep Learning",
            estimator=MLPClassifier(
                activation="relu",
                alpha=5e-4,
                early_stopping=True,
                hidden_layer_sizes=(128, 64, 32),
                learning_rate_init=8e-4,
                max_iter=500,
                random_state=RANDOM_STATE,
            ),
            use_sampler=True,
        ),
    ]


def build_pipeline(spec: ModelSpec) -> ImbPipeline:
    steps: list[tuple[str, object]] = [
        ("features", BusinessFeatureEngineer()),
        ("preprocess", build_preprocessor()),
    ]
    if spec.use_sampler:
        steps.append(("sampler", RandomOverSampler(random_state=RANDOM_STATE)))
    steps.append(("model", spec.estimator))
    return ImbPipeline(steps=steps)


def get_cv_results(pipeline: ImbPipeline, X_train, y_train) -> dict[str, float]:
    splitter = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "roc_auc": "roc_auc",
        "average_precision": "average_precision",
        "f1": "f1",
        "recall": "recall",
        "precision": make_scorer(precision_score, zero_division=0),
    }
    cv_results = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=splitter,
        n_jobs=1,
        return_train_score=False,
        scoring=scoring,
    )

    return {
        "cv_fit_time_mean": float(np.mean(cv_results["fit_time"])),
        "cv_roc_auc_mean": float(np.mean(cv_results["test_roc_auc"])),
        "cv_roc_auc_std": float(np.std(cv_results["test_roc_auc"])),
        "cv_pr_auc_mean": float(np.mean(cv_results["test_average_precision"])),
        "cv_pr_auc_std": float(np.std(cv_results["test_average_precision"])),
        "cv_f1_mean": float(np.mean(cv_results["test_f1"])),
        "cv_f1_std": float(np.std(cv_results["test_f1"])),
        "cv_recall_mean": float(np.mean(cv_results["test_recall"])),
        "cv_precision_mean": float(np.mean(cv_results["test_precision"])),
    }


def choose_threshold(y_true, probabilities: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if len(thresholds) == 0:
        return 0.50

    f1_scores = (2 * precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-9)
    best_index = int(np.nanargmax(f1_scores))
    return float(thresholds[best_index])


def evaluate_classifier(y_true, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions).ravel()

    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def select_final_model(results_df):
    ordered = results_df.sort_values(
        by=["test_pr_auc", "test_f1", "test_recall", "cv_pr_auc_mean"],
        ascending=False,
    )
    return ordered.iloc[0]
