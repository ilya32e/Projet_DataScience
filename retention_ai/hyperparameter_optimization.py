"""
Module pour optimisation d'hyperparamètres avec Optuna.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, make_scorer


class HyperparameterOptimizer:
    """Optimisation d'hyperparamètres avec Optuna."""
    
    def __init__(self, cv_folds: int = 5, random_state: int = 42):
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.study = None
        self.best_trial = None
    
    def optimize_logistic_regression(
        self,
        X_train,
        y_train,
        n_trials: int = 50,
        timeout: int = 3600
    ) -> dict:
        """Optimiser Logistic Regression."""
        try:
            import optuna
            from optuna.pruners import MedianPruner
            from sklearn.linear_model import LogisticRegression
        except ImportError:
            raise ImportError("Optuna required: pip install optuna")
        
        def objective(trial):
            C = trial.suggest_float("C", 1e-4, 1e2, log=True)
            solver = trial.suggest_categorical("solver", ["liblinear", "lbfgs"])
            max_iter = trial.suggest_int("max_iter", 100, 2000)
            
            model = LogisticRegression(
                C=C,
                solver=solver,
                max_iter=max_iter,
                class_weight="balanced",
                random_state=self.random_state,
                n_jobs=-1,
            )
            
            scores = cross_validate(
                model,
                X_train,
                y_train,
                cv=StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state),
                scoring={"roc_auc": "roc_auc", "f1": make_scorer(f1_score, zero_division=0)},
                n_jobs=-1,
            )
            
            return scores["test_roc_auc"].mean()
        
        self.study = optuna.create_study(
            direction="maximize",
            pruner=MedianPruner(),
        )
        self.study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)
        
        return self.study.best_params
    
    def optimize_random_forest(
        self,
        X_train,
        y_train,
        n_trials: int = 50,
        timeout: int = 3600
    ) -> dict:
        """Optimiser Random Forest."""
        try:
            import optuna
            from optuna.pruners import MedianPruner
            from sklearn.ensemble import RandomForestClassifier
        except ImportError:
            raise ImportError("Optuna required: pip install optuna")
        
        def objective(trial):
            n_estimators = trial.suggest_int("n_estimators", 50, 500)
            max_depth = trial.suggest_int("max_depth", 5, 50)
            min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
            max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", None])
            
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                class_weight="balanced_subsample",
                random_state=self.random_state,
                n_jobs=-1,
            )
            
            scores = cross_validate(
                model,
                X_train,
                y_train,
                cv=StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state),
                scoring={"roc_auc": "roc_auc", "f1": make_scorer(f1_score, zero_division=0)},
                n_jobs=-1,
            )
            
            return scores["test_roc_auc"].mean()
        
        self.study = optuna.create_study(
            direction="maximize",
            pruner=MedianPruner(),
        )
        self.study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)
        
        return self.study.best_params
    
    def optimize_gradient_boosting(
        self,
        X_train,
        y_train,
        n_trials: int = 50,
        timeout: int = 3600
    ) -> dict:
        """Optimiser Gradient Boosting."""
        try:
            import optuna
            from optuna.pruners import MedianPruner
            from sklearn.ensemble import GradientBoostingClassifier
        except ImportError:
            raise ImportError("Optuna required: pip install optuna")
        
        def objective(trial):
            n_estimators = trial.suggest_int("n_estimators", 50, 500)
            learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
            max_depth = trial.suggest_int("max_depth", 3, 15)
            min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
            subsample = trial.suggest_float("subsample", 0.5, 1.0)
            
            model = GradientBoostingClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                subsample=subsample,
                random_state=self.random_state,
            )
            
            scores = cross_validate(
                model,
                X_train,
                y_train,
                cv=StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state),
                scoring={"roc_auc": "roc_auc", "f1": make_scorer(f1_score, zero_division=0)},
                n_jobs=-1,
            )
            
            return scores["test_roc_auc"].mean()
        
        self.study = optuna.create_study(
            direction="maximize",
            pruner=MedianPruner(),
        )
        self.study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)
        
        return self.study.best_params
    
    def optimize_mlp_classifier(
        self,
        X_train,
        y_train,
        n_trials: int = 50,
        timeout: int = 3600
    ) -> dict:
        """Optimiser MLP Classifier."""
        try:
            import optuna
            from optuna.pruners import MedianPruner
            from sklearn.neural_network import MLPClassifier
        except ImportError:
            raise ImportError("Optuna required: pip install optuna")
        
        def objective(trial):
            hidden_layer_sizes = trial.suggest_categorical(
                "hidden_layer_sizes",
                [
                    (64,),
                    (128,),
                    (256,),
                    (64, 32),
                    (128, 64),
                    (256, 128),
                    (128, 64, 32),
                    (256, 128, 64),
                ]
            )
            learning_rate_init = trial.suggest_float("learning_rate_init", 1e-5, 1e-2, log=True)
            alpha = trial.suggest_float("alpha", 1e-6, 1e-2, log=True)
            
            model = MLPClassifier(
                hidden_layer_sizes=hidden_layer_sizes,
                learning_rate_init=learning_rate_init,
                alpha=alpha,
                activation="relu",
                max_iter=500,
                early_stopping=True,
                random_state=self.random_state,
                n_jobs=-1,
            )
            
            scores = cross_validate(
                model,
                X_train,
                y_train,
                cv=StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state),
                scoring={"roc_auc": "roc_auc", "f1": make_scorer(f1_score, zero_division=0)},
                n_jobs=-1,
            )
            
            return scores["test_roc_auc"].mean()
        
        self.study = optuna.create_study(
            direction="maximize",
            pruner=MedianPruner(),
        )
        self.study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)
        
        return self.study.best_params
    
    def get_best_params(self) -> dict:
        """Retourner les meilleurs paramètres."""
        if self.study is None:
            raise ValueError("Run optimize() first")
        return self.study.best_params
    
    def get_best_score(self) -> float:
        """Retourner le meilleur score."""
        if self.study is None:
            raise ValueError("Run optimize() first")
        return self.study.best_value
    
    def get_trials_df(self):
        """Retourner les essais sous forme DataFrame."""
        if self.study is None:
            raise ValueError("Run optimize() first")
        return self.study.trials_dataframe()
