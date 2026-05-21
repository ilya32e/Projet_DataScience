from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate, train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from retention_ai.config import CV_FOLDS, ID_COLUMN, RANDOM_STATE, TARGET_COLUMN, TEST_SIZE

ENGAGEMENT_COLUMNS = [
    "monthly_logins",
    "weekly_active_days",
    "avg_session_time",
    "features_used",
    "usage_growth_rate",
    "last_login_days_ago",
]


def _plain_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                selector(dtype_include=np.number),
            ),
            (
                "categorical",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore")),
                ]),
                selector(dtype_include=object),
            ),
        ],
        sparse_threshold=0.0,
        verbose_feature_names_out=False,
    )


def _regression_specs() -> list[tuple[str, str, str, object]]:
    return [
        (
            "ridge",
            "Ridge",
            "Linéaire",
            Ridge(alpha=1.0),
        ),
        (
            "random_forest",
            "Random Forest",
            "Ensemble arbres",
            RandomForestRegressor(
                n_estimators=200,
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
        ),
        (
            "gradient_boosting",
            "Gradient Boosting",
            "Boosting",
            GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                subsample=0.85,
                random_state=RANDOM_STATE,
            ),
        ),
        (
            "mlp_regressor",
            "MLP Regressor",
            "Deep Learning",
            MLPRegressor(
                hidden_layer_sizes=(128, 64, 32),
                activation="relu",
                alpha=5e-4,
                early_stopping=True,
                learning_rate_init=8e-4,
                max_iter=500,
                random_state=RANDOM_STATE,
            ),
        ),
    ]


def _clf_specs() -> list[tuple[str, str, str, object]]:
    return [
        (
            "logistic_regression",
            "Logistic Regression",
            "Baseline interprétable",
            LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                random_state=RANDOM_STATE,
                solver="lbfgs",
            ),
        ),
        (
            "random_forest",
            "Random Forest",
            "Ensemble arbres",
            RandomForestClassifier(
                class_weight="balanced_subsample",
                n_estimators=200,
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
        ),
        (
            "gradient_boosting",
            "Gradient Boosting",
            "Boosting",
            GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                subsample=0.85,
                random_state=RANDOM_STATE,
            ),
        ),
        (
            "mlp_classifier",
            "MLP Classifier",
            "Deep Learning",
            MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                activation="relu",
                alpha=5e-4,
                early_stopping=False,
                learning_rate_init=8e-4,
                max_iter=500,
                random_state=RANDOM_STATE,
            ),
        ),
    ]


def _compute_engagement_score(df: pd.DataFrame) -> pd.Series:
    cols = ENGAGEMENT_COLUMNS
    min_vals = df[cols].min()
    range_vals = (df[cols].max() - min_vals).replace(0, 1e-9)
    norm = (df[cols] - min_vals) / range_vals
    return (
        0.25 * norm["monthly_logins"]
        + 0.20 * norm["weekly_active_days"]
        + 0.20 * norm["avg_session_time"]
        + 0.15 * norm["features_used"]
        + 0.10 * norm["usage_growth_rate"]
        + 0.10 * (1 - norm["last_login_days_ago"])
    )


def _train_regression(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    records: list[dict] = []
    best_pipeline = None
    best_r2 = -np.inf
    best_X_test = X_test
    best_y_test = y_test

    for name, label, family, estimator in _regression_specs():
        pipeline = Pipeline([("preprocess", _plain_preprocessor()), ("model", estimator)])
        cv = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=kf,
            scoring={"rmse": "neg_root_mean_squared_error", "r2": "r2"},
            n_jobs=1,
        )
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        test_r2 = float(r2_score(y_test, y_pred))

        records.append({
            "name": name,
            "label": label,
            "family": family,
            "test_rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "test_mae": float(mean_absolute_error(y_test, y_pred)),
            "test_r2": test_r2,
            "cv_rmse_mean": float(-np.mean(cv["test_rmse"])),
            "cv_rmse_std": float(np.std(cv["test_rmse"])),
            "cv_r2_mean": float(np.mean(cv["test_r2"])),
            "cv_r2_std": float(np.std(cv["test_r2"])),
        })

        if test_r2 > best_r2:
            best_r2 = test_r2
            best_pipeline = pipeline
            best_X_test = X_test
            best_y_test = y_test

    results_df = pd.DataFrame(records).sort_values("test_r2", ascending=False).reset_index(drop=True)

    perm = permutation_importance(
        best_pipeline,
        best_X_test,
        best_y_test,
        n_repeats=10,
        random_state=RANDOM_STATE,
        scoring="r2",
        n_jobs=-1,
    )
    importance_df = (
        pd.DataFrame({
            "feature": best_X_test.columns,
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std,
        })
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )

    return results_df, importance_df


def _train_classification(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    records: list[dict] = []

    for name, label, family, estimator in _clf_specs():
        pipeline = Pipeline([("preprocess", _plain_preprocessor()), ("model", estimator)])
        cv = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=skf,
            scoring={"f1_macro": "f1_macro", "accuracy": "accuracy"},
            n_jobs=1,
        )
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        records.append({
            "name": name,
            "label": label,
            "family": family,
            "test_accuracy": float(accuracy_score(y_test, y_pred)),
            "test_f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
            "cv_f1_macro_mean": float(np.mean(cv["test_f1_macro"])),
            "cv_f1_macro_std": float(np.std(cv["test_f1_macro"])),
            "cv_accuracy_mean": float(np.mean(cv["test_accuracy"])),
        })

    return pd.DataFrame(records).sort_values("test_f1_macro", ascending=False).reset_index(drop=True)


def run_all_secondary(
    dataset: pd.DataFrame, scored_customers: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    base_X = dataset.drop(columns=[TARGET_COLUMN, ID_COLUMN])

    # ── Tâche 1 : Revenu à Risque (Régression) ────────────────────────────────
    # Cible : expected_monthly_loss = monthly_fee × churn_probability
    # (dérivée du modèle de churn déjà entraîné, alignée ligne à ligne)
    rar_target = pd.Series(
        scored_customers["expected_monthly_loss"].values,
        name="expected_monthly_loss",
    )
    rar_results, rar_importance = _train_regression(base_X, rar_target)

    # ── Tâche 2 : CLV — Valeur Vie Client (Régression) ────────────────────────
    # Cible : total_revenue (déjà dans le dataset)
    # On retire total_revenue des features pour éviter la fuite d'info
    clv_X = base_X.drop(columns=["total_revenue"])
    clv_target = dataset["total_revenue"].reset_index(drop=True)
    clv_results, clv_importance = _train_regression(clv_X, clv_target)

    # ── Tâche 3 : Score d'Engagement (Régression) ─────────────────────────────
    # Cible : engagement_score (formule du sujet, calculée sur tout le dataset)
    # Features : profil + finance + support + marketing (sans les 6 colonnes d'usage)
    engagement_score = _compute_engagement_score(dataset).reset_index(drop=True)
    engagement_X = base_X.drop(columns=ENGAGEMENT_COLUMNS)
    eng_reg_results, eng_importance = _train_regression(engagement_X, engagement_score)

    # ── Tâche 4 : Catégorie d'Engagement (Classification) ─────────────────────
    # Cible : Faible / Moyen / Fort selon les terciles du score
    engagement_cat = pd.cut(
        engagement_score,
        bins=[-0.001, 0.33, 0.67, 1.001],
        labels=["Faible", "Moyen", "Fort"],
    ).astype(str)
    eng_clf_results = _train_classification(engagement_X, engagement_cat)

    return {
        "revenue_at_risk": rar_results,
        "rar_importance": rar_importance,
        "clv": clv_results,
        "clv_importance": clv_importance,
        "engagement_regression": eng_reg_results,
        "engagement_importance": eng_importance,
        "engagement_classification": eng_clf_results,
    }
