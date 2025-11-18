"""
Optuna study manager for hyperparameter optimization.
"""

from typing import Dict, Callable, Optional, Any
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import joblib
from pathlib import Path
import config


class OptunaManager:
    """
    Manages Optuna hyperparameter optimization studies.

    Features:
    - Parallel trial execution
    - Study persistence to SQLite
    - Automatic overfitting detection
    - Progress tracking
    """

    def __init__(self, study_name: str, storage: Optional[str] = None,
                 direction: str = 'maximize'):
        """
        Initialize Optuna manager.

        Args:
            study_name: Name of the study
            storage: Database URL (default: uses config.OPTUNA_STORAGE)
            direction: 'maximize' or 'minimize'
        """
        self.study_name = study_name
        self.storage = storage or config.OPTUNA_STORAGE
        self.direction = direction

        # Create sampler and pruner from config
        sampler_name = config.OPTUNA_PARAMS.get('sampler', 'TPE')
        if sampler_name == 'TPE':
            self.sampler = TPESampler(
                n_startup_trials=config.OPTUNA_PARAMS.get('n_startup_trials', 20),
                seed=42
            )
        else:
            self.sampler = None  # Use Optuna default

        pruner_name = config.OPTUNA_PARAMS.get('pruner', 'Median')
        if pruner_name == 'Median':
            self.pruner = MedianPruner(
                n_startup_trials=10,
                n_warmup_steps=5
            )
        else:
            self.pruner = None

    def create_study(self, load_if_exists: bool = True) -> optuna.Study:
        """
        Create or load an Optuna study.

        Args:
            load_if_exists: Whether to load existing study with same name

        Returns:
            Optuna study object
        """
        study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            direction=self.direction,
            sampler=self.sampler,
            pruner=self.pruner,
            load_if_exists=load_if_exists
        )

        # Store metadata
        study.set_user_attr('created_at', str(pd.Timestamp.now()))
        study.set_user_attr('config_snapshot', {
            'n_trials': config.OPTUNA_PARAMS['n_trials'],
            'n_jobs': config.OPTUNA_PARAMS['n_jobs'],
            'sampler': config.OPTUNA_PARAMS['sampler'],
        })

        return study

    def optimize(self, objective: Callable, n_trials: Optional[int] = None,
                 n_jobs: Optional[int] = None, timeout: Optional[float] = None,
                 callbacks: Optional[list] = None, show_progress: bool = True) -> optuna.Study:
        """
        Run hyperparameter optimization.

        Args:
            objective: Objective function to optimize
            n_trials: Number of trials (default: from config)
            n_jobs: Number of parallel jobs (default: from config)
            timeout: Timeout in seconds (default: None)
            callbacks: List of callback functions
            show_progress: Whether to show progress bar

        Returns:
            Completed study
        """
        # Use config defaults if not specified
        if n_trials is None:
            n_trials = config.OPTUNA_PARAMS['n_trials']

        if n_jobs is None:
            n_jobs = config.OPTUNA_PARAMS['n_jobs']

        # Create study
        study = self.create_study(load_if_exists=True)

        print(f"Starting optimization: {self.study_name}")
        print(f"  Trials: {n_trials}")
        print(f"  Parallel jobs: {n_jobs}")
        print(f"  Current trials completed: {len(study.trials)}")

        # Run optimization
        study.optimize(
            objective,
            n_trials=n_trials,
            n_jobs=n_jobs,
            timeout=timeout,
            callbacks=callbacks,
            show_progress_bar=show_progress
        )

        print(f"\nOptimization complete!")
        print(f"  Best value: {study.best_value:.4f}")
        print(f"  Best params: {study.best_params}")

        return study

    def get_study(self) -> optuna.Study:
        """Load existing study."""
        return optuna.load_study(
            study_name=self.study_name,
            storage=self.storage
        )

    def get_best_params(self) -> Dict[str, Any]:
        """Get best parameters from study."""
        study = self.get_study()
        return study.best_params

    def check_overfitting(self, train_scores: list, test_scores: list,
                          threshold: float = 0.3) -> Dict:
        """
        Check if optimization has overfit.

        Args:
            train_scores: Training scores for each trial
            test_scores: Test scores for each trial
            threshold: Maximum acceptable train-test gap

        Returns:
            Dict with overfitting analysis
        """
        import numpy as np

        train_mean = np.mean(train_scores)
        test_mean = np.mean(test_scores)

        gap = train_mean - test_mean
        gap_pct = gap / abs(train_mean) if train_mean != 0 else 0

        is_overfit = gap_pct > threshold

        return {
            'train_mean': train_mean,
            'test_mean': test_mean,
            'gap': gap,
            'gap_pct': gap_pct,
            'threshold': threshold,
            'is_overfit': is_overfit,
            'warning': 'OVERFITTING DETECTED' if is_overfit else 'No overfitting detected'
        }

    def save_study(self, filepath: Path) -> None:
        """Save study to file for backup."""
        study = self.get_study()
        joblib.dump(study, filepath)
        print(f"Study saved to {filepath}")

    def visualize(self, study: Optional[optuna.Study] = None) -> None:
        """
        Generate optimization visualizations.

        Args:
            study: Optuna study (if None, loads from storage)
        """
        from optuna.visualization import (
            plot_optimization_history,
            plot_param_importances,
            plot_parallel_coordinate
        )

        if study is None:
            study = self.get_study()

        print("Generating visualizations...")

        # Optimization history
        fig = plot_optimization_history(study)
        fig.show()

        # Parameter importances
        if len(study.trials) > 10:
            fig = plot_param_importances(study)
            fig.show()

        # Parallel coordinate plot
        if len(study.trials) > 5:
            fig = plot_parallel_coordinate(study)
            fig.show()

    def __repr__(self) -> str:
        return f"OptunaManager(study_name='{self.study_name}', direction='{self.direction}')"


# Import pandas for timestamp
import pandas as pd
