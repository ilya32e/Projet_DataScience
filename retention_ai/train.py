from __future__ import annotations

from datetime import datetime, timezone

import joblib
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from retention_ai.config import (
    CLV_IMPORTANCE_PATH,
    CLV_LEADERBOARD_PATH,
    ENG_CLF_LEADERBOARD_PATH,
    ENG_IMPORTANCE_PATH,
    ENG_REG_LEADERBOARD_PATH,
    FEATURE_IMPORTANCE_PATH,
    FIGURES_DIR,
    FINAL_MODEL_BUNDLE_PATH,
    ID_COLUMN,
    LEADERBOARD_PATH,
    MODELS_DIR,
    OVERVIEW_PATH,
    RAR_IMPORTANCE_PATH,
    RAR_LEADERBOARD_PATH,
    RANDOM_STATE,
    SCORED_CUSTOMERS_PATH,
    SCHEMA_PATH,
    SECONDARY_METRICS_DIR,
    TARGET_COLUMN,
    TEST_SIZE,
    VALIDATION_SIZE,
)
from retention_ai.data import build_schema, ensure_directories, load_dataset, save_json
from retention_ai.extra_tasks import run_all_secondary
from retention_ai.modeling import (
    build_pipeline,
    choose_threshold,
    evaluate_classifier,
    get_cv_results,
    get_model_specs,
    select_final_model,
)
from retention_ai.reporting import (
    plot_class_balance,
    plot_confusion_matrix,
    plot_contract_churn,
    plot_feature_importance,
    plot_model_comparison,
)


def train_all_models() -> dict[str, str]:
    ensure_directories()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    LEADERBOARD_PATH.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset()
    schema = build_schema(dataset)
    save_json(SCHEMA_PATH, schema)

    X = dataset.drop(columns=[TARGET_COLUMN, ID_COLUMN])
    y = dataset[TARGET_COLUMN].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    X_fit, X_valid, y_fit, y_valid = train_test_split(
        X_train,
        y_train,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )

    records: list[dict[str, object]] = []
    trained_bundles: dict[str, dict[str, object]] = {}

    for spec in get_model_specs():
        evaluation_pipeline = build_pipeline(spec)
        cv_summary = get_cv_results(evaluation_pipeline, X_train, y_train)

        threshold_pipeline = build_pipeline(spec)
        threshold_pipeline.fit(X_fit, y_fit)
        validation_probabilities = threshold_pipeline.predict_proba(X_valid)[:, 1]
        threshold = choose_threshold(y_valid, validation_probabilities)

        final_pipeline = build_pipeline(spec)
        final_pipeline.fit(X_train, y_train)
        test_probabilities = final_pipeline.predict_proba(X_test)[:, 1]
        test_metrics = evaluate_classifier(y_test, test_probabilities, threshold)

        bundle = {
            "name": spec.name,
            "label": spec.label,
            "family": spec.family,
            "threshold": threshold,
            "pipeline": final_pipeline,
            "metrics": test_metrics,
        }
        trained_bundles[spec.name] = bundle
        joblib.dump(bundle, MODELS_DIR / f"{spec.name}_bundle.joblib")

        records.append(
            {
                "name": spec.name,
                "label": spec.label,
                "family": spec.family,
                "threshold": threshold,
                **cv_summary,
                **{f"test_{metric}": value for metric, value in test_metrics.items()},
            }
        )

    results_df = pd.DataFrame(records).sort_values(by="test_pr_auc", ascending=False)
    results_df.to_csv(LEADERBOARD_PATH, index=False)

    best_row = select_final_model(results_df)
    final_bundle = trained_bundles[str(best_row["name"])]
    joblib.dump(final_bundle, FINAL_MODEL_BUNDLE_PATH)

    importance = permutation_importance(
        final_bundle["pipeline"],
        X_test,
        y_test,
        n_repeats=10,
        random_state=RANDOM_STATE,
        scoring="average_precision",
        n_jobs=-1,
    )
    feature_importance = (
        pd.DataFrame(
            {
                "feature": X_test.columns,
                "importance_mean": importance.importances_mean,
                "importance_std": importance.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )
    feature_importance.to_csv(FEATURE_IMPORTANCE_PATH, index=False)

    full_probabilities = final_bundle["pipeline"].predict_proba(X)[:, 1]
    scored_customers = dataset[
        [
            ID_COLUMN,
            "customer_segment",
            "contract_type",
            "monthly_fee",
            "total_revenue",
            "support_tickets",
            "payment_failures",
            "nps_score",
            TARGET_COLUMN,
        ]
    ].copy()
    scored_customers["churn_probability"] = full_probabilities
    scored_customers["predicted_churn"] = (
        scored_customers["churn_probability"] >= float(final_bundle["threshold"])
    ).astype(int)
    scored_customers["expected_monthly_loss"] = (
        scored_customers["monthly_fee"] * scored_customers["churn_probability"]
    )
    scored_customers["expected_revenue_at_risk"] = (
        scored_customers["total_revenue"] * scored_customers["churn_probability"]
    )
    scored_customers.to_csv(SCORED_CUSTOMERS_PATH, index=False)

    overview = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_rows": int(dataset.shape[0]),
        "dataset_columns": int(dataset.shape[1]),
        "train_rows": int(X_train.shape[0]),
        "test_rows": int(X_test.shape[0]),
        "positive_class_rate": float(y.mean()),
        "final_model_name": final_bundle["name"],
        "final_model_label": final_bundle["label"],
        "final_model_threshold": float(final_bundle["threshold"]),
        "final_model_metrics": final_bundle["metrics"],
        "portfolio_risk_summary": {
            "predicted_high_risk_customers": int(scored_customers["predicted_churn"].sum()),
            "expected_monthly_loss_total": float(scored_customers["expected_monthly_loss"].sum()),
            "expected_revenue_at_risk_total": float(
                scored_customers["expected_revenue_at_risk"].sum()
            ),
        },
    }
    save_json(OVERVIEW_PATH, overview)

    plot_class_balance(dataset, FIGURES_DIR / "class_balance.png")
    plot_contract_churn(dataset, FIGURES_DIR / "contract_churn_rate.png")
    plot_model_comparison(results_df, FIGURES_DIR / "model_comparison.png")
    plot_feature_importance(feature_importance, FIGURES_DIR / "feature_importance.png")
    plot_confusion_matrix(final_bundle["metrics"], FIGURES_DIR / "final_confusion_matrix.png")

    SECONDARY_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    secondary = run_all_secondary(dataset, scored_customers)
    secondary["revenue_at_risk"].to_csv(RAR_LEADERBOARD_PATH, index=False)
    secondary["rar_importance"].to_csv(RAR_IMPORTANCE_PATH, index=False)
    secondary["clv"].to_csv(CLV_LEADERBOARD_PATH, index=False)
    secondary["clv_importance"].to_csv(CLV_IMPORTANCE_PATH, index=False)
    secondary["engagement_regression"].to_csv(ENG_REG_LEADERBOARD_PATH, index=False)
    secondary["engagement_importance"].to_csv(ENG_IMPORTANCE_PATH, index=False)
    secondary["engagement_classification"].to_csv(ENG_CLF_LEADERBOARD_PATH, index=False)

    return {
        "leaderboard": str(LEADERBOARD_PATH),
        "final_model": str(FINAL_MODEL_BUNDLE_PATH),
        "feature_importance": str(FEATURE_IMPORTANCE_PATH),
        "scored_customers": str(SCORED_CUSTOMERS_PATH),
        "secondary_revenue_at_risk": str(RAR_LEADERBOARD_PATH),
        "secondary_clv": str(CLV_LEADERBOARD_PATH),
        "secondary_engagement_regression": str(ENG_REG_LEADERBOARD_PATH),
        "secondary_engagement_classification": str(ENG_CLF_LEADERBOARD_PATH),
    }


if __name__ == "__main__":
    outputs = train_all_models()
    for label, path in outputs.items():
        print(f"{label}: {path}")
