# Hyperparameters - SignalTide v3

**Last Updated:** 2025-11-18

This document defines ALL tunable parameters in the SignalTide v3 system. These parameters are controlled by Optuna during optimization.

**CRITICAL PHILOSOPHY:**
- We do NOT filter parameter ranges prematurely
- Ranges are intentionally wide to let Optuna explore the space
- "Crazy" parameters are acceptable - let validation catch bad configurations
- Trust the optimization process and validation framework

---

## Global System Parameters

### Portfolio Configuration

```python
PORTFOLIO_PARAMS = {
    'max_positions': {
        'type': 'int',
        'range': [1, 20],
        'default': 5,
        'description': 'Maximum number of concurrent positions'
    },
    'position_sizing_method': {
        'type': 'categorical',
        'choices': ['equal_weight', 'volatility_scaled', 'kelly', 'risk_parity'],
        'default': 'volatility_scaled',
        'description': 'Method for determining position sizes'
    },
    'rebalance_frequency': {
        'type': 'categorical',
        'choices': ['1H', '4H', '1D', '1W'],
        'default': '1D',
        'description': 'How often to rebalance the portfolio'
    },
}
```

### Risk Management

```python
RISK_PARAMS = {
    'max_position_size': {
        'type': 'float',
        'range': [0.01, 0.50],  # 1% to 50% of portfolio
        'default': 0.20,
        'description': 'Maximum size for a single position as fraction of portfolio'
    },
    'stop_loss_pct': {
        'type': 'float',
        'range': [0.01, 0.30],  # 1% to 30% stop loss
        'default': 0.05,
        'description': 'Stop loss percentage from entry price'
    },
    'take_profit_pct': {
        'type': 'float',
        'range': [0.02, 1.00],  # 2% to 100% take profit
        'default': 0.15,
        'description': 'Take profit percentage from entry price'
    },
    'max_portfolio_drawdown': {
        'type': 'float',
        'range': [0.10, 0.50],  # 10% to 50% max drawdown (suggestion, not hard limit)
        'default': 0.25,
        'description': 'Suggested maximum portfolio drawdown before reducing exposure'
    },
    'drawdown_scale_factor': {
        'type': 'float',
        'range': [0.0, 1.0],
        'default': 0.5,
        'description': 'How aggressively to scale down positions during drawdown (0=none, 1=full)'
    },
}
```

### Transaction Costs

```python
TRANSACTION_PARAMS = {
    'commission_pct': {
        'type': 'float',
        'range': [0.0, 0.01],  # 0% to 1% commission
        'default': 0.001,  # 0.1% (10 bps)
        'description': 'Commission per trade as percentage'
    },
    'slippage_pct': {
        'type': 'float',
        'range': [0.0, 0.01],  # 0% to 1% slippage
        'default': 0.001,  # 0.1%
        'description': 'Expected slippage per trade as percentage'
    },
}
```

---

## Regime Detection Parameters

```python
REGIME_PARAMS = {
    'use_regime_detection': {
        'type': 'categorical',
        'choices': [True, False],
        'default': True,
        'description': 'Whether to use regime detection for signal weighting'
    },
    'n_regimes': {
        'type': 'int',
        'range': [2, 5],
        'default': 3,
        'description': 'Number of market regimes to detect'
    },
    'regime_lookback': {
        'type': 'int',
        'range': [20, 500],
        'default': 100,
        'description': 'Lookback period for regime detection (in bars)'
    },
    'regime_method': {
        'type': 'categorical',
        'choices': ['hmm', 'kmeans', 'volatility_threshold'],
        'default': 'hmm',
        'description': 'Method for regime detection'
    },
}
```

---

## Signal-Specific Parameters

### Base Signal Parameters (applies to all signals)

```python
BASE_SIGNAL_PARAMS = {
    'signal_weight': {
        'type': 'float',
        'range': [0.0, 1.0],
        'default': 1.0,
        'description': 'Weight of this signal in portfolio (0=disabled, 1=full weight)'
    },
    'min_confidence': {
        'type': 'float',
        'range': [0.0, 1.0],
        'default': 0.5,
        'description': 'Minimum confidence threshold for signal to trigger'
    },
}
```

### Momentum Signals

```python
MOMENTUM_PARAMS = {
    'lookback_period': {
        'type': 'int',
        'range': [5, 200],
        'default': 20,
        'description': 'Lookback period for momentum calculation'
    },
    'threshold': {
        'type': 'float',
        'range': [-1.0, 1.0],
        'default': 0.0,
        'description': 'Threshold for momentum signal trigger'
    },
    'use_log_returns': {
        'type': 'categorical',
        'choices': [True, False],
        'default': True,
        'description': 'Whether to use log returns vs simple returns'
    },
}
```

### Mean Reversion Signals

```python
MEAN_REVERSION_PARAMS = {
    'lookback_period': {
        'type': 'int',
        'range': [5, 200],
        'default': 20,
        'description': 'Lookback period for mean calculation'
    },
    'entry_zscore': {
        'type': 'float',
        'range': [0.5, 5.0],
        'default': 2.0,
        'description': 'Z-score threshold for entry'
    },
    'exit_zscore': {
        'type': 'float',
        'range': [0.0, 2.0],
        'default': 0.5,
        'description': 'Z-score threshold for exit'
    },
    'use_bollinger_bands': {
        'type': 'categorical',
        'choices': [True, False],
        'default': False,
        'description': 'Use Bollinger Bands instead of z-score'
    },
    'bollinger_std': {
        'type': 'float',
        'range': [1.0, 4.0],
        'default': 2.0,
        'description': 'Standard deviations for Bollinger Bands'
    },
}
```

### Volatility Signals

```python
VOLATILITY_PARAMS = {
    'vol_lookback': {
        'type': 'int',
        'range': [5, 100],
        'default': 20,
        'description': 'Lookback period for volatility calculation'
    },
    'vol_method': {
        'type': 'categorical',
        'choices': ['std', 'parkinson', 'garman_klass', 'rogers_satchell'],
        'default': 'std',
        'description': 'Method for volatility estimation'
    },
    'high_vol_threshold': {
        'type': 'float',
        'range': [1.0, 5.0],
        'default': 1.5,
        'description': 'Threshold for high volatility regime (in std devs)'
    },
    'low_vol_threshold': {
        'type': 'float',
        'range': [0.1, 1.0],
        'default': 0.5,
        'description': 'Threshold for low volatility regime (in std devs)'
    },
}
```

### Volume Signals

```python
VOLUME_PARAMS = {
    'volume_lookback': {
        'type': 'int',
        'range': [5, 100],
        'default': 20,
        'description': 'Lookback period for volume analysis'
    },
    'volume_spike_threshold': {
        'type': 'float',
        'range': [1.5, 10.0],
        'default': 2.0,
        'description': 'Multiple of average volume to consider a spike'
    },
    'use_relative_volume': {
        'type': 'categorical',
        'choices': [True, False],
        'default': True,
        'description': 'Use relative volume vs absolute volume'
    },
}
```

### Technical Indicator Signals (RSI, MACD, etc.)

```python
RSI_PARAMS = {
    'rsi_period': {
        'type': 'int',
        'range': [5, 50],
        'default': 14,
        'description': 'Period for RSI calculation'
    },
    'rsi_overbought': {
        'type': 'float',
        'range': [60, 90],
        'default': 70,
        'description': 'RSI overbought threshold'
    },
    'rsi_oversold': {
        'type': 'float',
        'range': [10, 40],
        'default': 30,
        'description': 'RSI oversold threshold'
    },
}

MACD_PARAMS = {
    'macd_fast': {
        'type': 'int',
        'range': [5, 20],
        'default': 12,
        'description': 'Fast EMA period for MACD'
    },
    'macd_slow': {
        'type': 'int',
        'range': [20, 50],
        'default': 26,
        'description': 'Slow EMA period for MACD'
    },
    'macd_signal': {
        'type': 'int',
        'range': [5, 15],
        'default': 9,
        'description': 'Signal line period for MACD'
    },
}
```

### Machine Learning Signal Parameters

```python
ML_PARAMS = {
    'feature_lookback': {
        'type': 'int',
        'range': [10, 200],
        'default': 50,
        'description': 'Lookback period for feature engineering'
    },
    'n_estimators': {
        'type': 'int',
        'range': [10, 500],
        'default': 100,
        'description': 'Number of trees (for tree-based models)'
    },
    'max_depth': {
        'type': 'int',
        'range': [2, 20],
        'default': 5,
        'description': 'Maximum tree depth (for tree-based models)'
    },
    'learning_rate': {
        'type': 'float',
        'range': [0.001, 0.3],
        'log_scale': True,
        'default': 0.01,
        'description': 'Learning rate for gradient boosting'
    },
    'regularization': {
        'type': 'float',
        'range': [0.0, 1.0],
        'default': 0.1,
        'description': 'L2 regularization parameter'
    },
}
```

---

## Validation Parameters

These parameters control the validation framework itself:

```python
VALIDATION_PARAMS = {
    'n_splits': {
        'type': 'int',
        'range': [3, 10],
        'default': 5,
        'description': 'Number of cross-validation splits'
    },
    'purge_pct': {
        'type': 'float',
        'range': [0.0, 0.2],
        'default': 0.05,
        'description': 'Percentage of data to purge between train/test splits'
    },
    'embargo_pct': {
        'type': 'float',
        'range': [0.0, 0.2],
        'default': 0.01,
        'description': 'Percentage of data to embargo after training period'
    },
    'min_sample_size': {
        'type': 'int',
        'range': [100, 1000],
        'default': 252,  # ~1 year of daily data
        'description': 'Minimum sample size for valid backtest'
    },
    'monte_carlo_n_trials': {
        'type': 'int',
        'range': [100, 10000],
        'default': 1000,
        'description': 'Number of Monte Carlo permutation trials'
    },
    'significance_level': {
        'type': 'float',
        'range': [0.01, 0.10],
        'default': 0.05,
        'description': 'Statistical significance level (p-value threshold)'
    },
}
```

---

## Optuna Optimization Parameters

```python
OPTUNA_PARAMS = {
    'n_trials': {
        'default': 100,
        'description': 'Number of optimization trials to run'
    },
    'n_jobs': {
        'default': -1,  # Use all available cores
        'description': 'Number of parallel jobs (-1 for all cores)'
    },
    'timeout': {
        'default': None,  # No timeout
        'description': 'Maximum time for optimization (seconds, None for no limit)'
    },
    'sampler': {
        'default': 'TPE',
        'choices': ['TPE', 'Random', 'CmaEs', 'Grid'],
        'description': 'Sampling algorithm for hyperparameter search'
    },
    'pruner': {
        'default': 'Median',
        'choices': ['Median', 'Hyperband', 'None'],
        'description': 'Pruning algorithm for early stopping of bad trials'
    },
}
```

---

## Notes on Parameter Ranges

1. **Wide Ranges Are Intentional**: Ranges are deliberately wide to allow exploration
2. **No Manual Filtering**: Do not narrow ranges based on intuition - let Optuna explore
3. **Validation Will Catch Bad Configs**: Trust the validation framework to reject overfitted parameters
4. **Log Scale When Appropriate**: Use log scale for parameters that span orders of magnitude
5. **Categorical vs Continuous**: Choose type based on parameter nature, not convenience
6. **Dependencies**: Some parameters only apply when others are set (e.g., bollinger_std only matters if use_bollinger_bands=True)

---

## Adding New Parameters

When adding a new parameter:

1. Add it to the appropriate section above
2. Specify type, range, default, and description
3. Update ParameterSpace class in optimization/parameter_space.py
4. Document any dependencies on other parameters
5. Update this file's last updated date
6. Do NOT prematurely narrow the range

---

## Parameter Versioning

Track which parameter set was used for each optimization study. Store in Optuna study metadata.

Example:
```python
study.set_user_attr('hyperparameters_version', '2025-11-18')
study.set_user_attr('hyperparameters_hash', 'sha256_hash_of_this_file')
```
