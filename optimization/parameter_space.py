"""
Parameter space definitions for Optuna optimization.

Parses HYPERPARAMETERS.md and creates Optuna search space.
"""

from typing import Dict, Any, Tuple
import optuna


class ParameterSpace:
    """
    Defines search space for hyperparameter optimization.

    Converts parameter specifications from HYPERPARAMETERS.md format
    to Optuna trial suggestions.
    """

    @staticmethod
    def suggest_from_spec(trial: optuna.Trial, param_name: str,
                          param_spec: Tuple) -> Any:
        """
        Suggest parameter value from specification.

        Args:
            trial: Optuna trial object
            param_name: Name of parameter
            param_spec: Tuple of (type, min, max) or (type, choices)

        Returns:
            Suggested parameter value

        Example:
            ```python
            # Integer parameter
            spec = ('int', 5, 200)
            value = ParameterSpace.suggest_from_spec(trial, 'lookback', spec)

            # Float parameter
            spec = ('float', 0.0, 1.0)
            value = ParameterSpace.suggest_from_spec(trial, 'threshold', spec)

            # Categorical parameter
            spec = ('categorical', ['A', 'B', 'C'])
            value = ParameterSpace.suggest_from_spec(trial, 'method', spec)
            ```
        """
        param_type = param_spec[0]

        if param_type == 'int':
            min_val, max_val = param_spec[1], param_spec[2]
            return trial.suggest_int(param_name, min_val, max_val)

        elif param_type == 'float':
            min_val, max_val = param_spec[1], param_spec[2]
            # Check if log scale
            log_scale = len(param_spec) > 3 and param_spec[3] == 'log'
            return trial.suggest_float(param_name, min_val, max_val, log=log_scale)

        elif param_type == 'categorical':
            choices = param_spec[1]
            return trial.suggest_categorical(param_name, choices)

        else:
            raise ValueError(f"Unknown parameter type: {param_type}")

    @staticmethod
    def suggest_all(trial: optuna.Trial, param_space: Dict[str, Tuple]) -> Dict[str, Any]:
        """
        Suggest all parameters from parameter space dict.

        Args:
            trial: Optuna trial
            param_space: Dict of {param_name: param_spec}

        Returns:
            Dict of {param_name: suggested_value}

        Example:
            ```python
            param_space = {
                'lookback': ('int', 5, 200),
                'threshold': ('float', 0.0, 0.5),
                'method': ('categorical', ['A', 'B', 'C'])
            }

            params = ParameterSpace.suggest_all(trial, param_space)
            # params = {'lookback': 42, 'threshold': 0.3, 'method': 'B'}
            ```
        """
        params = {}
        for param_name, param_spec in param_space.items():
            params[param_name] = ParameterSpace.suggest_from_spec(
                trial, param_name, param_spec
            )
        return params

    @staticmethod
    def get_default_params(param_space: Dict[str, Tuple]) -> Dict[str, Any]:
        """
        Get default parameter values (midpoint for numeric, first for categorical).

        Args:
            param_space: Parameter space specification

        Returns:
            Dict of default parameters
        """
        defaults = {}

        for param_name, param_spec in param_space.items():
            param_type = param_spec[0]

            if param_type == 'int':
                # Midpoint
                min_val, max_val = param_spec[1], param_spec[2]
                defaults[param_name] = (min_val + max_val) // 2

            elif param_type == 'float':
                # Midpoint
                min_val, max_val = param_spec[1], param_spec[2]
                defaults[param_name] = (min_val + max_val) / 2

            elif param_type == 'categorical':
                # First choice
                defaults[param_name] = param_spec[1][0]

        return defaults


# Example parameter spaces for common signal types
EXAMPLE_MOMENTUM_SPACE = {
    'lookback': ('int', 5, 200),
    'threshold': ('float', 0.0, 0.5),
    'use_log_returns': ('categorical', [True, False]),
}

EXAMPLE_MEAN_REVERSION_SPACE = {
    'lookback': ('int', 5, 200),
    'entry_zscore': ('float', 0.5, 5.0),
    'exit_zscore': ('float', 0.0, 2.0),
    'use_bollinger': ('categorical', [True, False]),
}

EXAMPLE_ML_SPACE = {
    'feature_lookback': ('int', 10, 200),
    'n_estimators': ('int', 10, 500),
    'max_depth': ('int', 2, 20),
    'learning_rate': ('float', 0.001, 0.3, 'log'),
}
