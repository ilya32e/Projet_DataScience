"""
Diagnostic et analysis tools pour les problèmes identifiés.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, f1_score
)


class ModelDiagnostics:
    """Diagnostique des problèmes de modèle."""
    
    @staticmethod
    def diagnose_random_forest_cv_issue(y_train, y_pred_train, y_pred_test):
        """
        Diagnostiquer le problème de Random Forest (F1 nul en CV).
        
        Causes probables:
        1. Seuil de décision > 0.5 → beaucoup de 0 prédits
        2. Imbalance extrême
        3. Features manquantes ou faibles
        """
        print("=" * 60)
        print("DIAGNOSTIC: Random Forest CV Issue")
        print("=" * 60)
        
        # 1. Distribution des prédictions
        print("\n1. DISTRIBUTION DES PRÉDICTIONS")
        print(f"   Train - Classe 0: {(y_pred_train == 0).sum()}, Classe 1: {(y_pred_train == 1).sum()}")
        print(f"   Test  - Classe 0: {(y_pred_test == 0).sum()}, Classe 1: {(y_pred_test == 1).sum()}")
        
        # 2. Imbalance ratio
        print("\n2. IMBALANCE RATIO")
        train_ratio = (y_train == 1).sum() / len(y_train)
        print(f"   Ratio positifs en train: {train_ratio:.2%}")
        
        # 3. Test de seuil
        if (y_pred_train == 0).sum() == len(y_pred_train):
            print("\n3. ⚠️ PROBLÈME DÉTECTÉ: Le modèle prédit toujours 0!")
            print("   Causes possibles:")
            print("   - Seuil de décision mal calibré (> 0.5)")
            print("   - Features trop faibles")
            print("   - Problème de CV split (leakage ou data distribution)")
            return "threshold_or_weak_signal"
        
        # 4. F1 score
        from sklearn.metrics import f1_score
        f1 = f1_score(y_train, y_pred_train, zero_division=0)
        print(f"\n4. F1 SCORE: {f1:.3f}")
        
        if f1 < 0.1:
            print("   ⚠️ F1 très bas → problème de seuil ou signal faible")
        
        return "ok"
    
    @staticmethod
    def analyze_feature_strength(X_train, y_train, top_n: int = 10):
        """
        Analyser la force des features.
        
        Returns:
            DataFrame avec feature importance (univarié)
        """
        from sklearn.feature_selection import mutual_info_classif
        
        # Mutual information
        mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
        
        feature_names = X_train.columns if hasattr(X_train, 'columns') else [f"Feature_{i}" for i in range(X_train.shape[1])]
        
        df = pd.DataFrame({
            "feature": feature_names,
            "mi_score": mi_scores
        }).sort_values("mi_score", ascending=False)
        
        print("\n" + "=" * 60)
        print("TOP FEATURES (Mutual Information)")
        print("=" * 60)
        print(df.head(top_n).to_string(index=False))
        print(f"\nAverage MI score: {mi_scores.mean():.4f}")
        print(f"Median MI score: {np.median(mi_scores):.4f}")
        
        if mi_scores.mean() < 0.01:
            print("\n⚠️ SIGNAL TRÈS FAIBLE - MI moyen < 0.01")
        
        return df
    
    @staticmethod
    def analyze_class_balance(y_train) -> dict:
        """Analyser l'imbalance de classe."""
        unique, counts = np.unique(y_train, return_counts=True)
        
        analysis = {}
        for u, c in zip(unique, counts):
            analysis[f"class_{u}"] = {
                "count": c,
                "percentage": c / len(y_train) * 100
            }
        
        print("\n" + "=" * 60)
        print("CLASS BALANCE")
        print("=" * 60)
        for class_name, stats in analysis.items():
            print(f"{class_name}: {stats['count']:6d} ({stats['percentage']:6.2f}%)")
        
        ratio = counts[1] / counts[0]
        print(f"\nImbalance ratio: {ratio:.3f}")
        
        if ratio < 0.1:
            print("⚠️ IMBALANCE CRITIQUE (<10%)")
        elif ratio < 0.3:
            print("⚠️ IMBALANCE SÉVÈRE (<30%)")
        
        return analysis
    
    @staticmethod
    def debug_threshold_issue(y_true, y_proba_pred):
        """Déboguer les problèmes de seuil."""
        print("\n" + "=" * 60)
        print("THRESHOLD ANALYSIS")
        print("=" * 60)
        
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
        
        print(f"\n{'Threshold':<12} {'Positif':<12} {'F1':<12} {'Recall':<12} {'Precision':<12}")
        print("-" * 60)
        
        for t in thresholds:
            y_pred = (y_proba_pred >= t).astype(int)
            n_positive = (y_pred == 1).sum()
            f1 = f1_score(y_true, y_pred, zero_division=0)
            recall = np.sum((y_pred == 1) & (y_true == 1)) / np.sum(y_true == 1) if np.sum(y_true == 1) > 0 else 0
            precision = np.sum((y_pred == 1) & (y_true == 1)) / n_positive if n_positive > 0 else 0
            
            print(f"{t:<12.2f} {n_positive:<12} {f1:<12.3f} {recall:<12.3f} {precision:<12.3f}")
    
    @staticmethod
    def plot_reliability_diagram(y_true, y_proba_pred, n_bins: int = 10):
        """Tracer le diagramme de fiabilité (calibration)."""
        from sklearn.calibration import calibration_curve
        import matplotlib.pyplot as plt
        
        prob_true, prob_pred = calibration_curve(
            y_true, y_proba_pred, n_bins=n_bins, strategy='uniform'
        )
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
        
        # Actual calibration
        ax.plot(prob_pred, prob_true, 'o-', linewidth=2, markersize=8, label='Classifier')
        
        ax.set_xlabel('Mean predicted probability')
        ax.set_ylabel('Fraction of positives')
        ax.set_title('Calibration Diagram')
        ax.legend()
        ax.grid(alpha=0.3)
        
        return fig
    
    @staticmethod
    def plot_roc_pr_curves(y_true, y_proba_pred):
        """Tracer ROC et PR curves."""
        import matplotlib.pyplot as plt
        
        fpr, tpr, _ = roc_curve(y_true, y_proba_pred)
        roc_auc = auc(fpr, tpr)
        
        precision, recall, _ = precision_recall_curve(y_true, y_proba_pred)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # ROC curve
        axes[0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
        axes[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[0].set_xlabel('False Positive Rate')
        axes[0].set_ylabel('True Positive Rate')
        axes[0].set_title('ROC Curve')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # PR curve
        axes[1].plot(recall, precision, color='green', lw=2, label='PR')
        axes[1].set_xlabel('Recall')
        axes[1].set_ylabel('Precision')
        axes[1].set_title('Precision-Recall Curve')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        return fig
