from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


@dataclass(frozen=True)
class EngagementWeights:
    monthly_logins: float = 0.25
    weekly_active_days: float = 0.20
    avg_session_time: float = 0.20
    features_used: float = 0.15
    usage_growth_rate: float = 0.10
    last_login_days_ago: float = 0.10


class BusinessFeatureEngineer(BaseEstimator, TransformerMixin):
    """Adds business-oriented features while keeping train/test separation clean."""

    engagement_columns = [
        "monthly_logins",
        "weekly_active_days",
        "avg_session_time",
        "features_used",
        "usage_growth_rate",
        "last_login_days_ago",
    ]

    def __init__(self, weights: EngagementWeights | None = None) -> None:
        self.weights = weights or EngagementWeights()

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "BusinessFeatureEngineer":
        frame = self._ensure_frame(X)
        self.feature_names_in_ = frame.columns.tolist()
        self.minimums_ = frame[self.engagement_columns].min()
        self.maximums_ = frame[self.engagement_columns].max()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, ["feature_names_in_", "minimums_", "maximums_"])
        frame = self._ensure_frame(X).copy()
        tenure = frame["tenure_months"].replace(0, 1)

        frame["support_ticket_rate"] = frame["support_tickets"] / tenure
        frame["revenue_per_month"] = frame["total_revenue"] / tenure
        frame["payment_risk_index"] = frame["payment_failures"] * frame["monthly_fee"]

        frame["engagement_score"] = (
            self.weights.monthly_logins * self._normalize(frame["monthly_logins"], "monthly_logins")
            + self.weights.weekly_active_days
            * self._normalize(frame["weekly_active_days"], "weekly_active_days")
            + self.weights.avg_session_time
            * self._normalize(frame["avg_session_time"], "avg_session_time")
            + self.weights.features_used * self._normalize(frame["features_used"], "features_used")
            + self.weights.usage_growth_rate
            * self._normalize(frame["usage_growth_rate"], "usage_growth_rate")
            + self.weights.last_login_days_ago
            * (1 - self._normalize(frame["last_login_days_ago"], "last_login_days_ago"))
        )

        return frame

    def _normalize(self, series: pd.Series, column: str) -> pd.Series:
        denominator = self.maximums_[column] - self.minimums_[column]
        if denominator == 0:
            return pd.Series(0.0, index=series.index)
        return (series - self.minimums_[column]) / denominator

    @staticmethod
    def _ensure_frame(X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("BusinessFeatureEngineer attend un pandas.DataFrame en entree.")
        return X
