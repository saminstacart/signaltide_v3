# Optuna Optimization Guide - SignalTide v3

**Last Updated:** 2025-11-18

This document explains our hyperparameter optimization strategy using Optuna.

---

## Table of Contents

1. [Why Optuna?](#why-optuna)
2. [Core Concepts](#core-concepts)
3. [Implementation Guide](#implementation-guide)
4. [Best Practices](#best-practices)
5. [Common Pitfalls](#common-pitfalls)
6. [Advanced Features](#advanced-features)
7. [Troubleshooting](#troubleshooting)

---

## Why Optuna?

### Advantages Over Alternatives

**vs Grid Search**:
- Explores space more efficiently (doesn't try every combination)
- Handles high-dimensional spaces better
- Can be stopped early with good results

**vs Random Search**:
- Smarter sampling using past trials (TPE algorithm)
- Converges faster to optimal regions
- Built-in pruning for early stopping

**vs Bayesian Optimization (GPy, Hyperopt)**:
- Better parallelization support
- More modern API
- Better visualization tools
- Active development and community

**vs Manual Tuning**:
- Systematic and reproducible
- Explores parameter interactions
- Documents all attempts
- No human bias

### Key Features We Use

1. **Tree-structured Parzen Estimator (TPE)**: Smart Bayesian sampling
2. **Parallel Execution**: Utilize all CPU cores
3. **Study Persistence**: Resume interrupted optimizations
4. **Pruning**: Early stopping of unpromising trials
5. **Visualization**: Built-in plotting of optimization progress

---

## Core Concepts

### 1. Study

A study is an optimization session.

```python
import optuna

study = optuna.create_study(
    study_name='btc_momentum_v1',
    storage='sqlite:///optuna_studies.db',
    direction='maximize',  # or 'minimize'
    load_if_exists=True    # Resume if interrupted
)
```

**Key Points**:
- One study per optimization goal
- Persisted to SQLite for resumability
- Can be shared across processes

### 2. Trial

A single evaluation with a specific parameter set.

```python
def objective(trial):
    # Sample parameters
    lookback = trial.suggest_int('lookback', 5, 200)
    threshold = trial.suggest_float('threshold', 0.0, 1.0)

    # Evaluate
    score = evaluate_strategy(lookback, threshold)

    return score
```

**Key Points**:
- Trial suggests parameters from search space
- You evaluate and return objective value
- Optuna learns from all trials

### 3. Sampler

Algorithm for suggesting parameters.

```python
from optuna.samplers import TPESampler

sampler = TPESampler(
    n_startup_trials=10,    # Random trials before using TPE
    n_ei_candidates=24,     # Candidates for expected improvement
    seed=42                 # Reproducibility
)

study = optuna.create_study(sampler=sampler)
```

**Available Samplers**:
- **TPESampler** (default, recommended): Tree-structured Parzen Estimator
- **RandomSampler**: Pure random search
- **GridSampler**: Exhaustive grid search
- **CmaEsSampler**: Covariance Matrix Adaptation Evolution Strategy

### 4. Pruner

Stops unpromising trials early.

```python
from optuna.pruners import MedianPruner

pruner = MedianPruner(
    n_startup_trials=5,     # Don't prune first 5 trials
    n_warmup_steps=10,      # Wait 10 steps before pruning
    interval_steps=1        # Check every step
)

study = optuna.create_study(pruner=pruner)
```

**Available Pruners**:
- **MedianPruner**: Stop if worse than median of previous trials
- **PercentilePruner**: Stop if below certain percentile
- **HyperbandPruner**: Successive halving algorithm
- **NopPruner**: No pruning

### 5. Objective Function

Function to maximize or minimize.

```python
def objective(trial):
    # 1. Sample parameters
    params = {
        'lookback': trial.suggest_int('lookback', 5, 200),
        'threshold': trial.suggest_float('threshold', 0.0, 1.0),
        'method': trial.suggest_categorical('method', ['A', 'B', 'C'])
    }

    # 2. Evaluate strategy with cross-validation
    scores = []
    for train_idx, test_idx in kfold.split(data):
        score = backtest(data[test_idx], params)
        scores.append(score)

        # Optional: Report intermediate value for pruning
        trial.report(np.mean(scores), step=len(scores))

        # Check if trial should be pruned
        if trial.should_prune():
            raise optuna.TrialPruned()

    # 3. Return final objective value
    return np.mean(scores)
```

---

## Implementation Guide

### Basic Optimization

```python
import optuna
from signaltide.optimization import OptunaManager
from signaltide.validation import PurgedKFold

# 1. Define objective
def objective(trial):
    # Sample parameters
    lookback = trial.suggest_int('lookback', 5, 200)
    threshold = trial.suggest_float('threshold', 0.0, 1.0)

    # Create signal
    signal = MomentumSignal({'lookback': lookback, 'threshold': threshold})

    # Cross-validate
    kfold = PurgedKFold(n_splits=5)
    scores = []

    for train_idx, test_idx in kfold.split(data):
        test_data = data.iloc[test_idx]
        result = backtest(signal, test_data)
        scores.append(result['sharpe'])

    return np.mean(scores)

# 2. Create study
study = optuna.create_study(
    study_name='momentum_optimization',
    storage='sqlite:///optuna_studies.db',
    direction='maximize',
    load_if_exists=True
)

# 3. Optimize
study.optimize(objective, n_trials=100)

# 4. Get best parameters
print(f"Best Sharpe: {study.best_value}")
print(f"Best params: {study.best_params}")
```

### Parallel Optimization

```python
# Run with all CPU cores
study.optimize(objective, n_trials=100, n_jobs=-1)

# Or specify number of cores
study.optimize(objective, n_trials=100, n_jobs=4)
```

**Important**: Each parallel job needs its own database connection, so we use SQLite storage rather than in-memory.

### Multi-Objective Optimization

Optimize multiple metrics simultaneously (e.g., Sharpe AND max drawdown).

```python
def multi_objective(trial):
    params = {...}  # Sample parameters

    result = backtest(params)

    # Return tuple of objectives
    return result['sharpe'], -result['max_drawdown']  # Minimize drawdown

# Create multi-objective study
study = optuna.create_study(
    directions=['maximize', 'maximize']  # Both objectives
)

study.optimize(multi_objective, n_trials=100)

# Get Pareto front
pareto_trials = study.best_trials
```

### Conditional Parameters

Some parameters only apply when others are set.

```python
def objective(trial):
    use_bollinger = trial.suggest_categorical('use_bollinger', [True, False])

    if use_bollinger:
        # These parameters only used if use_bollinger is True
        bb_period = trial.suggest_int('bb_period', 10, 50)
        bb_std = trial.suggest_float('bb_std', 1.0, 4.0)
        params = {'use_bollinger': True, 'bb_period': bb_period, 'bb_std': bb_std}
    else:
        # Alternative parameters
        threshold = trial.suggest_float('threshold', 0.0, 1.0)
        params = {'use_bollinger': False, 'threshold': threshold}

    return evaluate(params)
```

### Log Scale Parameters

For parameters spanning orders of magnitude.

```python
def objective(trial):
    # Learning rate from 0.001 to 0.1 on log scale
    lr = trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True)

    # More evenly samples across 0.001, 0.01, 0.1
    # Rather than clustering near 0.1
```

---

## Best Practices

### 1. Start with Random Trials

```python
sampler = TPESampler(
    n_startup_trials=20,  # First 20 trials are random
    seed=42
)
```

**Why**: TPE needs some random exploration before it can learn patterns.

### 2. Use Appropriate Number of Trials

**Guidelines**:
- Simple strategies (< 5 parameters): 50-100 trials
- Medium complexity (5-10 parameters): 100-200 trials
- Complex strategies (> 10 parameters): 200-500 trials

**Diminishing Returns**: Improvement slows after initial trials.

### 3. Set Realistic Parameter Ranges

```python
# Too narrow: Might miss optimal value
lookback = trial.suggest_int('lookback', 18, 22)

# Too wide: Wastes trials on unrealistic values
lookback = trial.suggest_int('lookback', 1, 10000)

# Just right: Wide but reasonable
lookback = trial.suggest_int('lookback', 5, 200)
```

### 4. Use Cross-Validation

**Always** use cross-validation in objective function.

```python
def objective(trial):
    params = sample_params(trial)

    # BAD: Single train-test split
    # score = backtest(test_data, params)

    # GOOD: Cross-validation
    scores = []
    for train_idx, test_idx in kfold.split(data):
        score = backtest(data[test_idx], params)
        scores.append(score)

    return np.mean(scores)
```

### 5. Monitor Train-Test Gap

```python
def objective(trial):
    params = sample_params(trial)

    train_scores = []
    test_scores = []

    for train_idx, test_idx in kfold.split(data):
        train_score = backtest(data[train_idx], params)
        test_score = backtest(data[test_idx], params)

        train_scores.append(train_score)
        test_scores.append(test_score)

    # Check for overfitting
    train_mean = np.mean(train_scores)
    test_mean = np.mean(test_scores)

    if train_mean - test_mean > 0.5:  # Large gap
        trial.set_user_attr('overfitting_warning', True)

    return test_mean  # Optimize test performance
```

### 6. Store Metadata

```python
def objective(trial):
    params = sample_params(trial)

    result = backtest(params)

    # Store additional metrics as user attributes
    trial.set_user_attr('max_drawdown', result['max_drawdown'])
    trial.set_user_attr('win_rate', result['win_rate'])
    trial.set_user_attr('n_trades', result['n_trades'])

    return result['sharpe']

# Later: Access metadata
best_trial = study.best_trial
print(f"Max DD: {best_trial.user_attrs['max_drawdown']}")
```

### 7. Use Pruning for Long Evaluations

If each trial takes minutes:

```python
def objective(trial):
    params = sample_params(trial)

    scores = []
    for i, (train_idx, test_idx) in enumerate(kfold.split(data)):
        score = backtest(data[test_idx], params)
        scores.append(score)

        # Report intermediate result
        trial.report(np.mean(scores), step=i)

        # Check if should stop early
        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(scores)
```

**Effect**: Trials that perform poorly on first folds get stopped early, saving computation.

### 8. Reproducibility

```python
# Set seeds for reproducibility
sampler = TPESampler(seed=42)
study = optuna.create_study(sampler=sampler)

# Also set numpy/pandas seeds in objective
def objective(trial):
    np.random.seed(42)
    # ... rest of objective
```

### 9. Save Studies to Database

```python
# In-memory (not persistent, cannot parallelize)
study = optuna.create_study()

# SQLite (persistent, can parallelize)
study = optuna.create_study(
    storage='sqlite:///optuna_studies.db',
    study_name='my_optimization'
)

# PostgreSQL (for production, high concurrency)
study = optuna.create_study(
    storage='postgresql://user:pass@localhost/db',
    study_name='my_optimization'
)
```

---

## Common Pitfalls

### ❌ Pitfall 1: Optimizing on Same Data as Final Test

```python
# WRONG
study.optimize(objective, n_trials=100)
best_params = study.best_params
final_test_score = backtest(test_data, best_params)  # Same data used in optimization!
```

**Solution**: Reserve separate test set never seen during optimization.

### ❌ Pitfall 2: Not Using Cross-Validation

```python
# WRONG
def objective(trial):
    params = sample_params(trial)
    return backtest(data, params)  # No CV, will overfit
```

**Solution**: Always use cross-validation.

### ❌ Pitfall 3: Too Many Trials → Overfitting

More trials = more exploration = higher risk of finding spurious patterns.

**Solution**:
- Limit trials to reasonable number (100-500)
- Use deflated Sharpe to account for multiple testing
- Validate best params on held-out test set

### ❌ Pitfall 4: Ignoring Failed Trials

```python
def objective(trial):
    try:
        params = sample_params(trial)
        return backtest(params)
    except Exception as e:
        return 0  # WRONG: Misleading Optuna
```

**Solution**: Let exceptions propagate or return NaN.

```python
def objective(trial):
    try:
        params = sample_params(trial)
        return backtest(params)
    except Exception as e:
        # Log error
        trial.set_user_attr('error', str(e))
        # Return NaN to mark as failed
        return float('nan')
```

### ❌ Pitfall 5: Forgetting to Specify Ranges

```python
# WRONG: Using default range [0, 1]
lookback = trial.suggest_int('lookback')  # Error!

# RIGHT: Always specify range
lookback = trial.suggest_int('lookback', 5, 200)
```

### ❌ Pitfall 6: Not Monitoring Optimization Progress

**Solution**: Use callbacks and visualization.

```python
from optuna.visualization import plot_optimization_history, plot_param_importances

# During optimization
study.optimize(objective, n_trials=100, show_progress_bar=True)

# After optimization
plot_optimization_history(study).show()
plot_param_importances(study).show()
```

---

## Advanced Features

### Feature 1: Custom Sampler

Create your own sampling logic:

```python
class CustomSampler(optuna.samplers.BaseSampler):
    def sample_independent(self, study, trial, param_name, param_distribution):
        # Your custom sampling logic
        pass

    def sample_relative(self, study, trial, search_space):
        # Your custom logic for dependent parameters
        pass
```

### Feature 2: Callbacks

Execute code during optimization:

```python
def callback(study, trial):
    # Save checkpoint every 10 trials
    if trial.number % 10 == 0:
        joblib.dump(study, f'study_checkpoint_{trial.number}.pkl')

    # Print progress
    print(f"Trial {trial.number}: {trial.value}")

study.optimize(objective, n_trials=100, callbacks=[callback])
```

### Feature 3: Study Resume After Failure

```python
# Start optimization
study = optuna.create_study(
    storage='sqlite:///studies.db',
    study_name='my_study',
    load_if_exists=True  # Key: resume if exists
)

study.optimize(objective, n_trials=100)

# If interrupted, just run again - it will resume
study.optimize(objective, n_trials=100)  # Continues from where it left off
```

### Feature 4: Parameter Importance

Which parameters matter most?

```python
from optuna.importance import get_param_importances

importances = get_param_importances(study)

for param, importance in importances.items():
    print(f"{param}: {importance:.3f}")
```

**Use**: Focus on important parameters, fix unimportant ones.

### Feature 5: Visualization

```python
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
    plot_contour,
    plot_slice
)

# Optimization progress over time
plot_optimization_history(study).show()

# Which parameters are most important
plot_param_importances(study).show()

# Relationship between parameters and objective
plot_parallel_coordinate(study).show()

# 2D contour plot for parameter pairs
plot_contour(study, params=['lookback', 'threshold']).show()

# 1D slice plots
plot_slice(study).show()
```

---

## Troubleshooting

### Issue: Optimization Not Improving

**Symptoms**: Objective value plateaus early.

**Solutions**:
1. Widen parameter ranges
2. Increase n_trials
3. Check if objective has bugs
4. Try different sampler (CmaEs for continuous parameters)
5. Check if data has signal (maybe no edge exists!)

### Issue: High Variance in Trials

**Symptoms**: Similar parameters give wildly different results.

**Solutions**:
1. Use more CV folds (reduce variance)
2. Increase minimum sample size
3. Check for data quality issues
4. May indicate overfitting - use simpler model

### Issue: Best Trial is First Trial

**Symptoms**: Random trial beats all TPE trials.

**Solutions**:
1. Increase n_startup_trials (more random exploration)
2. May indicate flat objective landscape
3. Check parameter ranges (maybe too narrow)

### Issue: Trials Failing

**Symptoms**: Many trials return NaN or raise exceptions.

**Solutions**:
1. Check parameter ranges (may be invalid values)
2. Add validation in objective function
3. Use try-except to handle edge cases gracefully
4. Examine failed trials: `study.trials[i]` where `trial.state == TrialState.FAIL`

### Issue: Slow Optimization

**Symptoms**: Taking too long to complete trials.

**Solutions**:
1. Use pruning to stop bad trials early
2. Reduce CV folds (trade-off: less robust)
3. Parallelize with n_jobs=-1
4. Reduce data size (use sample of data)
5. Optimize code (vectorization, caching)

### Issue: Memory Issues

**Symptoms**: OOM errors during parallel optimization.

**Solutions**:
1. Reduce n_jobs
2. Use data caching (load once, reuse)
3. Free memory between trials (del variables)
4. Use database storage instead of in-memory

---

## Example: Complete Optimization Workflow

```python
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
import numpy as np
from signaltide.validation import PurgedKFold
from signaltide.backtest import BacktestEngine

# 1. Define objective function
def objective(trial):
    # Sample parameters
    params = {
        'lookback': trial.suggest_int('lookback', 5, 200),
        'threshold': trial.suggest_float('threshold', 0.0, 0.5),
        'stop_loss': trial.suggest_float('stop_loss', 0.01, 0.10),
        'take_profit': trial.suggest_float('take_profit', 0.02, 0.30),
    }

    # Cross-validation
    kfold = PurgedKFold(n_splits=5, purge_pct=0.05, embargo_pct=0.01)
    scores = []

    for i, (train_idx, test_idx) in enumerate(kfold.split(data)):
        test_data = data.iloc[test_idx]

        # Backtest
        engine = BacktestEngine()
        result = engine.run(signal, test_data, params)

        sharpe = result['metrics']['sharpe']
        scores.append(sharpe)

        # Store fold result
        trial.set_user_attr(f'fold_{i}_sharpe', sharpe)

        # Intermediate reporting for pruning
        trial.report(np.mean(scores), step=i)
        if trial.should_prune():
            raise optuna.TrialPruned()

    # Calculate mean and std
    mean_sharpe = np.mean(scores)
    std_sharpe = np.std(scores)

    # Store additional metrics
    trial.set_user_attr('std_sharpe', std_sharpe)
    trial.set_user_attr('max_drawdown', result['metrics']['max_drawdown'])
    trial.set_user_attr('win_rate', result['metrics']['win_rate'])

    return mean_sharpe

# 2. Create study
sampler = optuna.samplers.TPESampler(n_startup_trials=20, seed=42)
pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=2)

study = optuna.create_study(
    study_name='momentum_strategy_v1',
    storage='sqlite:///optuna_studies.db',
    direction='maximize',
    sampler=sampler,
    pruner=pruner,
    load_if_exists=True
)

# Store metadata
study.set_user_attr('data_start', str(data.index[0]))
study.set_user_attr('data_end', str(data.index[-1]))
study.set_user_attr('n_samples', len(data))

# 3. Optimize
print("Starting optimization...")
study.optimize(
    objective,
    n_trials=200,
    n_jobs=-1,  # Use all cores
    show_progress_bar=True
)

# 4. Analyze results
print(f"\nBest trial:")
print(f"  Value (Sharpe): {study.best_value:.3f}")
print(f"  Params: {study.best_params}")
print(f"  Max Drawdown: {study.best_trial.user_attrs['max_drawdown']:.2%}")
print(f"  Win Rate: {study.best_trial.user_attrs['win_rate']:.2%}")

# 5. Check for overfitting
best_trial = study.best_trial
fold_sharpes = [best_trial.user_attrs[f'fold_{i}_sharpe'] for i in range(5)]
print(f"\nFold Sharpes: {fold_sharpes}")
print(f"Std across folds: {np.std(fold_sharpes):.3f}")

if np.std(fold_sharpes) > 0.5:
    print("⚠️  Warning: High variance across folds - may be overfit")

# 6. Visualize
plot_optimization_history(study).show()
plot_param_importances(study).show()

# 7. Final validation on held-out test set
print("\n Running final validation on held-out test set...")
final_result = backtest(held_out_test_data, study.best_params)
print(f"Final test Sharpe: {final_result['sharpe']:.3f}")

if abs(final_result['sharpe'] - study.best_value) > 0.3:
    print("⚠️  Warning: Large gap between CV and test performance")
```

---

## Summary

**Key Takeaways**:

1. Always use cross-validation in objective function
2. Reserve separate test set for final validation
3. Use parallel execution (n_jobs=-1)
4. Monitor train-test gap
5. Use pruning for long evaluations
6. Store studies to database for persistence
7. Visualize results to understand optimization
8. Limit trials to avoid overfitting
9. Validate best parameters on held-out data
10. Trust the process - Optuna is smarter than manual tuning

**Remember**: Optimization is a tool, not magic. Garbage in = garbage out. Make sure your signals have edge before optimizing.
