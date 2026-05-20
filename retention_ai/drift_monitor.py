"""
Module pour détection de drift et validation temporelle.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, chi2_contingency
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score


class DataDriftDetector:
    """Détection de drift dans les données."""
    
    @staticmethod
    def kl_divergence(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-10) -> float:
        """KL divergence (mesure asymétrique)."""
        p = p + epsilon
        q = q + epsilon
        p = p / p.sum()
        q = q / q.sum()
        return (p * (np.log(p) - np.log(q))).sum()
    
    @staticmethod
    def wasserstein_distance(x1: np.ndarray, x2: np.ndarray) -> float:
        """Wasserstein distance (Earth Mover's Distance)."""
        from scipy.stats import wasserstein_distance
        return wasserstein_distance(x1, x2)
    
    def detect_drift_numeric(
        self,
        baseline: np.ndarray,
        current: np.ndarray,
        method: str = "ks",
        threshold: float = 0.05
    ) -> dict:
        """
        Détecter drift pour feature numérique.
        
        Args:
            baseline: Données de baseline (référence)
            current: Données courantes
            method: 'ks' (Kolmogorov-Smirnov) ou 'wasserstein'
            threshold: Seuil de p-value
            
        Returns:
            Dict avec détails du drift
        """
        if method == "ks":
            statistic, p_value = ks_2samp(baseline, current)
            drift_detected = p_value < threshold
        elif method == "wasserstein":
            statistic = self.wasserstein_distance(baseline, current)
            # Empirical threshold (peut être ajusté)
            drift_detected = statistic > 0.1
            p_value = None
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return {
            "method": method,
            "statistic": statistic,
            "p_value": p_value,
            "drift_detected": drift_detected,
            "threshold": threshold,
        }
    
    def detect_drift_categorical(
        self,
        baseline: np.ndarray,
        current: np.ndarray,
        threshold: float = 0.05
    ) -> dict:
        """
        Détecter drift pour feature catégorique (Chi-square test).
        
        Args:
            baseline: Données de baseline
            current: Données courantes
            threshold: Seuil de p-value
            
        Returns:
            Dict avec détails du drift
        """
        categories = np.unique(np.concatenate([baseline, current]))
        
        baseline_counts = pd.Series(baseline).value_counts().reindex(categories, fill_value=0).values
        current_counts = pd.Series(current).value_counts().reindex(categories, fill_value=0).values
        
        contingency_table = np.array([baseline_counts, current_counts])
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        
        drift_detected = p_value < threshold
        
        return {
            "method": "chi_square",
            "statistic": chi2,
            "p_value": p_value,
            "drift_detected": drift_detected,
            "threshold": threshold,
            "dof": dof,
        }


class ModelDriftMonitor:
    """Monitoring du drift de performance du modèle."""
    
    def __init__(self, baseline_metrics: dict):
        """
        Args:
            baseline_metrics: Dict {metric_name: value} de référence
        """
        self.baseline_metrics = baseline_metrics
        self.monitoring_history = []
    
    def compute_metrics(self, y_true, y_pred_proba, y_pred=None) -> dict:
        """Calculer les métriques."""
        if y_pred is None:
            y_pred = (y_pred_proba >= 0.5).astype(int)
        
        metrics = {
            "auc_roc": roc_auc_score(y_true, y_pred_proba),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "precision": precision_score(y_true, y_pred, zero_division=0),
        }
        
        return metrics
    
    def detect_performance_drift(
        self,
        current_metrics: dict,
        threshold_pct: float = 10.0
    ) -> dict:
        """
        Détecter drift de performance (>10% de dégradation).
        
        Args:
            current_metrics: Métriques courantes
            threshold_pct: Seuil de dégradation en %
            
        Returns:
            Dict avec drift détecté et métriques dégradées
        """
        drift_detected = False
        degraded_metrics = {}
        
        for metric_name, baseline_value in self.baseline_metrics.items():
            if metric_name not in current_metrics:
                continue
            
            current_value = current_metrics[metric_name]
            degradation_pct = ((baseline_value - current_value) / baseline_value) * 100
            
            if degradation_pct > threshold_pct:
                drift_detected = True
                degraded_metrics[metric_name] = {
                    "baseline": baseline_value,
                    "current": current_value,
                    "degradation_pct": degradation_pct,
                }
        
        return {
            "drift_detected": drift_detected,
            "degraded_metrics": degraded_metrics,
            "threshold_pct": threshold_pct,
        }
    
    def log_monitoring(self, timestamp: str, metrics: dict, drift_info: dict):
        """Enregistrer le monitoring."""
        record = {
            "timestamp": timestamp,
            **metrics,
            **{f"drift_{k}": v for k, v in drift_info.items()},
        }
        self.monitoring_history.append(record)
    
    def get_history_df(self) -> pd.DataFrame:
        """Retourner l'historique sous forme DataFrame."""
        return pd.DataFrame(self.monitoring_history)


class TemporalValidator:
    """Validation temporelle stratifiée."""
    
    @staticmethod
    def temporal_train_test_split(
        X: pd.DataFrame,
        y: pd.Series,
        time_col: str,
        test_size: float = 0.2,
        gap_days: int = 0
    ):
        """
        Split temporel avec respect de la chronologie.
        
        Args:
            X: Features avec colonne time_col
            y: Target
            time_col: Nom de la colonne temporelle (datetime)
            test_size: Fraction pour test
            gap_days: Jours d'écart entre train et test (éviter leakage)
            
        Yields:
            (X_train, X_test, y_train, y_test)
        """
        if time_col not in X.columns:
            raise ValueError(f"Column {time_col} not found in X")
        
        # Trier par temps
        sorted_indices = X[time_col].argsort()
        X_sorted = X.iloc[sorted_indices].reset_index(drop=True)
        y_sorted = y.iloc[sorted_indices].reset_index(drop=True)
        
        # Calcul du split point
        split_point = int(len(X_sorted) * (1 - test_size))
        
        # Ajouter gap si nécessaire
        if gap_days > 0:
            split_time = X_sorted.iloc[split_point][time_col]
            gap_time = split_time + pd.Timedelta(days=gap_days)
            actual_split = (X_sorted[time_col] <= gap_time).sum()
            split_point = max(split_point, actual_split)
        
        return (
            X_sorted.iloc[:split_point],
            X_sorted.iloc[split_point:],
            y_sorted.iloc[:split_point],
            y_sorted.iloc[split_point:],
        )
    
    @staticmethod
    def walk_forward_validation(
        X: pd.DataFrame,
        y: pd.Series,
        time_col: str,
        train_size_days: int = 180,
        test_size_days: int = 30,
        step_days: int = 15
    ):
        """
        Walk-forward validation (time series cross-validation).
        
        Args:
            X: Features avec colonne time_col
            y: Target
            time_col: Colonne temporelle
            train_size_days: Taille du training window
            test_size_days: Taille du test window
            step_days: Pas du glissement
            
        Yields:
            (X_train, X_test, y_train, y_test, fold)
        """
        if time_col not in X.columns:
            raise ValueError(f"Column {time_col} not found in X")
        
        X_sorted = X.sort_values(time_col).reset_index(drop=True)
        y_sorted = y.iloc[X_sorted.index].reset_index(drop=True)
        
        min_date = X_sorted[time_col].min()
        max_date = X_sorted[time_col].max()
        
        train_end = min_date + pd.Timedelta(days=train_size_days)
        test_end = train_end + pd.Timedelta(days=test_size_days)
        fold = 0
        
        while test_end <= max_date:
            # Train set
            train_mask = X_sorted[time_col] <= train_end
            X_train = X_sorted[train_mask]
            y_train = y_sorted[train_mask]
            
            # Test set
            test_mask = (X_sorted[time_col] > train_end) & (X_sorted[time_col] <= test_end)
            X_test = X_sorted[test_mask]
            y_test = y_sorted[test_mask]
            
            if len(X_test) > 0:  # Only yield if test set non-empty
                yield X_train, X_test, y_train, y_test, fold
            
            # Glisser les fenêtres
            train_end += pd.Timedelta(days=step_days)
            test_end += pd.Timedelta(days=step_days)
            fold += 1
