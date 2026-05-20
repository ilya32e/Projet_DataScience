"""
Module pour calibration probabiliste et optimisation du seuil de décision.
"""

from __future__ import annotations

import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, log_loss


class ProbabilityCalibrator:
    """Calibration des probabilités prédites."""
    
    def __init__(self, method: str = "sigmoid", cv: int = 5):
        """
        Args:
            method: 'sigmoid' (Platt) ou 'isotonic'
            cv: nombre de folds pour calibration
        """
        self.method = method
        self.cv = cv
        self.calibrator = None
    
    def fit(self, pipeline, X_train, y_train):
        """Calibrer le pipeline."""
        self.calibrator = CalibratedClassifierCV(
            pipeline,
            method=self.method,
            cv=self.cv,
            n_jobs=-1
        )
        self.calibrator.fit(X_train, y_train)
        return self
    
    def predict_proba(self, X):
        """Prédire avec probabilités calibrées."""
        return self.calibrator.predict_proba(X)
    
    def evaluate_calibration(self, y_true, y_proba_pred) -> dict:
        """Évaluer la qualité de calibration."""
        # Extraire probabilités pour la classe positive
        y_proba_positive = y_proba_pred[:, 1]
        
        # Calcul des métriques de calibration
        prob_true, prob_pred = calibration_curve(
            y_true, y_proba_positive, n_bins=10, strategy='uniform'
        )
        
        # Expected Calibration Error (ECE)
        ece = np.abs(prob_true - prob_pred).mean()
        
        # Maximum Calibration Error (MCE)
        mce = np.abs(prob_true - prob_pred).max()
        
        # Brier Score (plus bas = mieux)
        brier = brier_score_loss(y_true, y_proba_positive)
        
        # Log Loss
        log_loss_val = log_loss(y_true, y_proba_pred)
        
        return {
            "ece": ece,
            "mce": mce,
            "brier_score": brier,
            "log_loss": log_loss_val,
            "prob_true": prob_true,
            "prob_pred": prob_pred,
        }


class ThresholdOptimizer:
    """Optimisation du seuil de décision."""
    
    def __init__(self, metric: str = "f1"):
        """
        Args:
            metric: 'f1', 'precision_recall', 'roc', 'youden'
        """
        self.metric = metric
        self.optimal_threshold = 0.5
        self.threshold_history = {}
    
    def optimize(self, y_true, y_proba_pred, thresholds: np.ndarray = None) -> float:
        """
        Trouver le seuil optimal.
        
        Args:
            y_true: Labels vrais
            y_proba_pred: Probabilités prédites (classe positive)
            thresholds: Array de seuils à tester
            
        Returns:
            Seuil optimal
        """
        if thresholds is None:
            thresholds = np.linspace(0, 1, 101)
        
        from sklearn.metrics import f1_score, precision_recall_curve, roc_curve, auc
        
        best_score = -1
        best_threshold = 0.5
        
        for threshold in thresholds:
            y_pred = (y_proba_pred >= threshold).astype(int)
            
            if self.metric == "f1":
                score = f1_score(y_true, y_pred, zero_division=0)
            elif self.metric == "youden":
                # Youden's J = Sensitivity + Specificity - 1
                tn = ((y_pred == 0) & (y_true == 0)).sum()
                fp = ((y_pred == 1) & (y_true == 0)).sum()
                fn = ((y_pred == 0) & (y_true == 1)).sum()
                tp = ((y_pred == 1) & (y_true == 1)).sum()
                
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                score = sensitivity + specificity - 1
            
            self.threshold_history[threshold] = score
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        self.optimal_threshold = best_threshold
        return best_threshold
    
    def predict(self, y_proba_pred) -> np.ndarray:
        """Prédire avec le seuil optimal."""
        return (y_proba_pred >= self.optimal_threshold).astype(int)


class ThresholdAnalyzer:
    """Analyse détaillée des seuils."""
    
    @staticmethod
    def compute_metrics_by_threshold(y_true, y_proba_pred, thresholds=None):
        """Calculer métriques pour chaque seuil."""
        if thresholds is None:
            thresholds = np.linspace(0, 1, 101)
        
        from sklearn.metrics import (
            f1_score, precision_score, recall_score,
            confusion_matrix, roc_curve, auc
        )
        
        results = {
            "threshold": [],
            "precision": [],
            "recall": [],
            "f1": [],
            "specificity": [],
            "sensitivity": [],
            "tn": [],
            "fp": [],
            "fn": [],
            "tp": [],
        }
        
        for threshold in thresholds:
            y_pred = (y_proba_pred >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            
            results["threshold"].append(threshold)
            results["precision"].append(precision_score(y_true, y_pred, zero_division=0))
            results["recall"].append(recall_score(y_true, y_pred, zero_division=0))
            results["f1"].append(f1_score(y_true, y_pred, zero_division=0))
            results["specificity"].append(tn / (tn + fp) if (tn + fp) > 0 else 0)
            results["sensitivity"].append(tp / (tp + fn) if (tp + fn) > 0 else 0)
            results["tn"].append(tn)
            results["fp"].append(fp)
            results["fn"].append(fn)
            results["tp"].append(tp)
        
        return results
