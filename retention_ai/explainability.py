"""
Module pour explicabilité locale et globale avec SHAP.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class SHAPExplainer:
    """Wrapper pour SHAP avec support TreeExplainer et KernelExplainer."""
    
    def __init__(self, model, X_background=None, explainer_type: str = "auto"):
        """
        Args:
            model: Modèle entraîné ou pipeline
            X_background: Données de référence pour KernelExplainer
            explainer_type: 'tree', 'kernel', ou 'auto'
        """
        self.model = model
        self.X_background = X_background
        self.explainer_type = explainer_type
        self.explainer = None
        self.shap_values = None
        self.feature_names = None
        self._initialize_explainer()
    
    def _initialize_explainer(self):
        """Initialiser l'explainer SHAP approprié."""
        try:
            import shap
        except ImportError:
            raise ImportError("SHAP not installed. Install with: pip install shap")
        
        # Déterminer le type de modèle
        if self.explainer_type == "auto":
            try:
                # TreeExplainer pour modèles arbre
                self.explainer = shap.TreeExplainer(self.model)
                self.explainer_type = "tree"
            except Exception:
                # KernelExplainer par défaut
                if self.X_background is None:
                    raise ValueError("X_background required for KernelExplainer")
                self.explainer = shap.KernelExplainer(
                    self.model.predict_proba if hasattr(self.model, 'predict_proba') else self.model.predict,
                    self.X_background
                )
                self.explainer_type = "kernel"
        elif self.explainer_type == "tree":
            self.explainer = shap.TreeExplainer(self.model)
        elif self.explainer_type == "kernel":
            if self.X_background is None:
                raise ValueError("X_background required for KernelExplainer")
            self.explainer = shap.KernelExplainer(
                self.model.predict_proba if hasattr(self.model, 'predict_proba') else self.model.predict,
                self.X_background
            )
    
    def explain(self, X, max_samples=None) -> shap.Explanation:
        """
        Calculer les SHAP values pour X.
        
        Args:
            X: Données à expliquer
            max_samples: Limite pour KernelExplainer
        """
        if max_samples and self.explainer_type == "kernel":
            shap_values = self.explainer.shap_values(
                X.iloc[:max_samples] if isinstance(X, pd.DataFrame) else X[:max_samples]
            )
        else:
            shap_values = self.explainer.shap_values(X)
        
        self.shap_values = shap_values
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        
        return shap_values
    
    def feature_importance(self, class_index: int = 1) -> pd.DataFrame:
        """
        Importances globales basées sur SHAP.
        
        Args:
            class_index: Index de la classe (pour classification)
            
        Returns:
            DataFrame avec importances triées
        """
        if self.shap_values is None:
            raise ValueError("Call explain() first")
        
        # Gérer le cas binaire et multiclasse
        if isinstance(self.shap_values, list):
            shap_vals = self.shap_values[class_index]
        else:
            shap_vals = self.shap_values
        
        # Calculer les moyennes des valeurs absolues
        importances = np.abs(shap_vals).mean(axis=0)
        
        df = pd.DataFrame({
            "feature": self.feature_names or [f"Feature_{i}" for i in range(len(importances))],
            "importance": importances
        })
        
        return df.sort_values("importance", ascending=False)
    
    def summary_plot(self, plot_type: str = "bar", max_display: int = 20, **kwargs):
        """Créer un summary plot."""
        try:
            import shap
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("SHAP and matplotlib required")
        
        if self.shap_values is None:
            raise ValueError("Call explain() first")
        
        if plot_type == "bar":
            shap.summary_plot(
                self.shap_values,
                feature_names=self.feature_names,
                plot_type="bar",
                max_display=max_display,
                show=False,
                **kwargs
            )
        else:
            shap.summary_plot(
                self.shap_values,
                feature_names=self.feature_names,
                max_display=max_display,
                show=False,
                **kwargs
            )
        
        return plt.gcf()
    
    def force_plot(self, instance_index: int = 0):
        """Force plot pour une instance."""
        try:
            import shap
        except ImportError:
            raise ImportError("SHAP required")
        
        if self.shap_values is None:
            raise ValueError("Call explain() first")
        
        base_value = self.explainer.expected_value
        if isinstance(self.shap_values, list):
            shap_vals = self.shap_values[1][instance_index]  # Classe positive
        else:
            shap_vals = self.shap_values[instance_index]
        
        return shap.force_plot(
            base_value,
            shap_vals,
            feature_names=self.feature_names,
            matplotlib=True,
            show=False
        )
    
    def dependence_plot(self, feature_name: str, interaction_index: str = "auto", **kwargs):
        """Dependence plot pour une feature."""
        try:
            import shap
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("SHAP and matplotlib required")
        
        feature_index = self.feature_names.index(feature_name) if self.feature_names else 0
        
        shap.dependence_plot(
            feature_index,
            self.shap_values,
            feature_names=self.feature_names,
            interaction_index=interaction_index,
            show=False,
            **kwargs
        )
        
        return plt.gcf()


class FeatureContributionAnalyzer:
    """Analyse des contributions de features par groupe."""
    
    @staticmethod
    def group_feature_importance(
        shap_values,
        feature_names: list,
        feature_groups: dict[str, list]
    ) -> pd.DataFrame:
        """
        Agréger les importances par groupe de features.
        
        Args:
            shap_values: SHAP values
            feature_names: Noms des features
            feature_groups: Dict {group_name: [feature_names]}
            
        Returns:
            DataFrame avec importances par groupe
        """
        importances = np.abs(shap_values).mean(axis=0)
        
        group_importance = {}
        for group_name, features in feature_groups.items():
            indices = [i for i, fname in enumerate(feature_names) if fname in features]
            group_importance[group_name] = importances[indices].sum()
        
        df = pd.DataFrame(
            list(group_importance.items()),
            columns=["group", "importance"]
        )
        
        return df.sort_values("importance", ascending=False)
