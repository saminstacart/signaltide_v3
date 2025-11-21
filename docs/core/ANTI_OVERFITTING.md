# Anti-Overfitting Guide - SignalTide v3

**Last Updated:** 2025-11-18

This document explains our comprehensive approach to preventing overfitting, the #1 cause of quantitative strategy failure.

---

## Table of Contents

1. [What is Overfitting?](#what-is-overfitting)
2. [Why Overfitting Happens](#why-overfitting-happens)
3. [Our Multi-Layer Defense](#our-multi-layer-defense)
4. [Specific Techniques](#specific-techniques)
5. [Red Flags](#red-flags)
6. [Validation Checklist](#validation-checklist)

---

## What is Overfitting?

**Definition**: Overfitting occurs when a strategy performs well on historical data but fails on new data because it learned noise rather than signal.

### The Illusion

```
Backtest Results:
- Sharpe Ratio: 3.5
- Win Rate: 75%
- Max Drawdown: 5%

Live Trading Results:
- Sharpe Ratio: -0.5
- Win Rate: 45%
- Max Drawdown: 35%
```

**What happened?** The strategy fit to random patterns in historical data that don't generalize.

### Analogy

Imagine studying for an exam by memorizing the specific problems in a practice test rather than understanding the underlying concepts. You'll ace the practice test but fail the real exam.

Overfitting is memorizing the practice test (historical data) instead of learning the concepts (real market dynamics).

---

## Why Overfitting Happens

### 1. Optimization on the Same Data

```
Bad: Optimize → Test on same data → Deploy
Good: Train → Validate → Test on held-out data → Deploy
```

### 2. Too Many Parameters

More parameters = more ways to fit noise.

**Example**: A strategy with 20 parameters has astronomical combinations. Some WILL look great on historical data purely by chance.

### 3. Data Snooping

```
Attempt 1: MACD crossover → Bad results → Discard
Attempt 2: RSI mean reversion → Bad results → Discard
Attempt 3: Volume breakout → Bad results → Discard
...
Attempt 47: Combination of signals 3, 7, 12 → Great results! → DEPLOY

Problem: You implicitly tested 47 strategies. One was bound to look good by chance.
```

### 4. Lookahead Bias

Using information that wouldn't have been available at the time.

**Examples**:
- Using end-of-day closing price to generate entry signal for that same day
- Training on data that includes the test period
- Using future volatility to size past positions

### 5. Selection Bias

Only reporting the strategies that worked.

**The Graveyard of Failed Backtests**: For every published strategy, there are dozens of failed attempts that get quietly discarded.

### 6. Regime Change

Historical patterns may not persist due to:
- Market structure changes (algorithmic trading, new participants)
- Regulatory changes
- Technology shifts
- Economic regime changes

---

## Our Multi-Layer Defense

We employ a defense-in-depth strategy with multiple independent validation layers.

### Layer 1: Temporal Discipline (No Lookahead)

**Enforcement**:
- All data access is point-in-time
- DataManager enforces temporal ordering
- Automated tests verify no future data usage
- Code review for temporal violations

**Test**:
```python
def test_no_lookahead(signal, data):
    """Verify signal at time t only uses data up to time t."""
    for t in data.index:
        past_data = data[:t]
        signal_value = signal.generate(past_data).iloc[-1]
        # Verify signal_value doesn't change when we add future data
        future_data = data[:t+1]
        signal_value_future = signal.generate(future_data).iloc[-1]
        assert signal_value == signal_value_future
```

### Layer 2: Purged K-Fold Cross-Validation

**Standard K-Fold Problem**:
```
Train: [===A===][===B===][===C===][===D===]
Test:                                        [===E===]
```

Information from E may have leaked into A-D due to autocorrelation.

**Purged K-Fold Solution**:
```
Train: [===A===][===B===][===C===]
Purge:                             [--P--]
Embargo:                                   [E]
Test:                                         [===D===]
```

**Implementation**:
```python
class PurgedKFold:
    def __init__(self, n_splits=5, purge_pct=0.05, embargo_pct=0.01):
        # Remove purge_pct of samples that overlap with test
        # Add embargo_pct gap after training period
```

### Layer 3: Walk-Forward Analysis

**Procedure**:
```
Period 1:
  Train: Year 1-2
  Validate: Year 3 (optimize hyperparameters)
  Test: Year 4 (evaluate)

Period 2:
  Train: Year 2-3
  Validate: Year 4 (re-optimize if needed)
  Test: Year 5 (evaluate)

Period 3:
  Train: Year 3-4
  Validate: Year 5
  Test: Year 6
```

**Why it works**: Simulates real deployment where parameters are periodically re-optimized on rolling window.

### Layer 4: Monte Carlo Permutation Testing

**Question**: Is performance due to skill or luck?

**Procedure**:
```python
def monte_carlo_test(signal, data, n_trials=1000):
    # 1. Calculate actual performance
    actual_sharpe = backtest(signal, data).sharpe

    # 2. Permute signals randomly and calculate performance
    permuted_sharpes = []
    for _ in range(n_trials):
        permuted_signal = shuffle(signal)  # Break timing
        permuted_sharpe = backtest(permuted_signal, data).sharpe
        permuted_sharpes.append(permuted_sharpe)

    # 3. Calculate p-value
    p_value = sum(permuted_sharpes >= actual_sharpe) / n_trials

    return p_value
```

**Interpretation**:
- p < 0.05: Significant skill (reject null of pure luck)
- p ≥ 0.05: Cannot reject luck as explanation

### Layer 5: Deflated Sharpe Ratio

**Problem**: If you test 100 strategies, one will likely have Sharpe > 2 by pure luck.

**Solution**: Haircut Sharpe based on number of trials.

```python
def deflated_sharpe(observed_sharpe, n_trials, n_observations):
    """
    Calculate Deflated Sharpe Ratio accounting for multiple testing.

    Based on Bailey & López de Prado (2014).
    """
    # Expected maximum Sharpe from n_trials random strategies
    expected_max_sharpe = expected_max_sharpe_under_null(n_trials)

    # Standard error of maximum Sharpe
    se_max_sharpe = se_of_max_sharpe(n_trials, n_observations)

    # Deflated Sharpe
    dsr = (observed_sharpe - expected_max_sharpe) / se_max_sharpe

    # Convert to probability
    p_value = 1 - norm.cdf(dsr)

    return dsr, p_value
```

### Layer 6: Out-of-Sample Testing

**Critical**: Never optimize on your final test set.

```
Dataset Split:
- Training: 50% (for initial development)
- Validation: 25% (for hyperparameter optimization)
- Test: 25% (NEVER touched until final validation)
```

**Rules**:
1. Test set is locked away until final validation
2. Only look at test performance ONCE
3. If test performance fails, DO NOT re-optimize and re-test
4. If test performance is good, still verify with Monte Carlo

### Layer 7: Realistic Transaction Costs

**Include ALL costs**:
- Commission: Explicit fees per trade
- Slippage: Price movement during execution
- Spread: Bid-ask spread
- Market Impact: Your order moving the price

**Conservative Assumptions**:
```python
TRANSACTION_COSTS = {
    'commission': 0.001,  # 10 bps
    'slippage': 0.001,    # 10 bps
    'spread': 0.0005,     # 5 bps
    'total': 0.0025       # 25 bps round-trip
}
```

**Reality Check**: If your strategy makes 0.1% per trade but has 0.25% round-trip costs, it's not profitable.

### Layer 8: Parameter Sensitivity Analysis

**Test**: How sensitive is performance to parameter changes?

```python
def sensitivity_analysis(strategy, params, data):
    """Test strategy performance across parameter ranges."""
    base_sharpe = backtest(strategy, params, data).sharpe

    results = {}
    for param_name, param_value in params.items():
        # Vary parameter by ±20%
        for multiplier in [0.8, 0.9, 1.0, 1.1, 1.2]:
            modified_params = params.copy()
            modified_params[param_name] = param_value * multiplier
            sharpe = backtest(strategy, modified_params, data).sharpe
            results[f'{param_name}_{multiplier}'] = sharpe

    return results
```

**Red Flag**: If performance drops 50% with a 10% parameter change, the strategy is overfit.

**Good Sign**: If performance degrades gracefully with parameter changes, the strategy is robust.

### Layer 9: Statistical Significance Testing

**All metrics should include confidence intervals**:

```python
def sharpe_with_confidence(returns, confidence=0.95):
    """Calculate Sharpe ratio with confidence interval."""
    sharpe = returns.mean() / returns.std() * np.sqrt(252)

    # Standard error of Sharpe
    n = len(returns)
    se = np.sqrt((1 + sharpe**2 / 2) / n)

    # Confidence interval
    z = norm.ppf((1 + confidence) / 2)
    ci_lower = sharpe - z * se
    ci_upper = sharpe + z * se

    return {
        'sharpe': sharpe,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'significant': ci_lower > 0  # Positive even at lower bound
    }
```

**Requirement**: Lower bound of 95% CI must be positive for strategy to be considered significant.

### Layer 10: Minimum Sample Size

**Rule of Thumb**: Need at least 252 observations (1 year of daily data) for reliable statistics.

**Better**: 500+ observations (2 years).

**Why**: Standard error decreases with √n. Small samples have huge uncertainty.

**Test**:
```python
def validate_sample_size(data, min_samples=252):
    """Ensure sufficient data for reliable statistics."""
    if len(data) < min_samples:
        raise ValueError(f"Insufficient data: {len(data)} < {min_samples}")
```

---

## Specific Techniques

### Technique 1: Combinatorially Purged Cross-Validation

Ensure all test sets are purged from all training sets, not just their own:

```
Standard Purged K-Fold:
  Fold 1: Train[A,B,C], Purge, Test[D]
  Fold 2: Train[A,B,D], Purge, Test[C]
  Fold 3: Train[A,C,D], Purge, Test[B]
  Fold 4: Train[B,C,D], Purge, Test[A]

Problem: Training sets overlap with other folds' test sets.

Combinatorial Purged CV:
  Purge ALL test periods from ALL training sets.
```

### Technique 2: Backtest Overfitting Probability

Calculate probability that backtest performance is due to overfitting:

```python
def probability_of_backtest_overfitting(trials_data):
    """
    Calculate PBO from multiple strategy configurations.

    Args:
        trials_data: List of (train_sharpe, test_sharpe) tuples

    Returns:
        Probability that best in-sample strategy underperforms OOS
    """
    # Rank strategies by training performance
    ranked = sorted(trials_data, key=lambda x: x[0], reverse=True)

    # Count how many top in-sample have negative out-of-sample
    n_overfit = sum(1 for train_sr, test_sr in ranked[:len(ranked)//2]
                    if test_sr < 0)

    # PBO = proportion that are overfit
    pbo = n_overfit / (len(ranked) // 2)

    return pbo
```

**Interpretation**:
- PBO < 0.3: Likely robust
- PBO 0.3-0.5: Concerning
- PBO > 0.5: Likely overfit

### Technique 3: Regime-Based Validation

Test performance across different market regimes:

```
Regimes:
- Bull markets (trending up)
- Bear markets (trending down)
- High volatility
- Low volatility
- High volume
- Low volume
```

**Requirement**: Strategy should work across multiple regimes, not just one.

**Red Flag**: Only works in low-volatility uptrends (2017 crypto).

### Technique 4: Asset Cross-Validation

If strategy is designed for BTC, test on ETH, LTC, etc.

**Why**: True edge should generalize across similar assets.

**Red Flag**: Only works on BTC, fails on all other cryptocurrencies.

---

## Red Flags

### 🚩 Too Good to Be True

- Sharpe > 3 (extremely rare in real strategies)
- Win rate > 80%
- Max drawdown < 5%
- Every month is positive

**Action**: Increase skepticism, check for bugs, re-validate rigorously.

### 🚩 Cliff Edge Performance

Performance collapses with small parameter changes.

**Example**:
- Threshold = 0.49: Sharpe = -0.3
- Threshold = 0.50: Sharpe = 2.5
- Threshold = 0.51: Sharpe = -0.2

**Diagnosis**: Overfit to specific parameter value.

### 🚩 Train-Test Gap

Large performance difference between training and testing.

**Example**:
- Training Sharpe: 2.5
- Test Sharpe: 0.3

**Diagnosis**: Overfit to training data.

### 🚩 Decreasing Performance Over Time

Performance degrades in more recent data.

**Diagnosis**: Strategy exploited patterns that have disappeared (regime change, competition, market evolution).

### 🚩 Too Many Parameters

More than 10-15 optimized parameters is a red flag.

**Why**: Curse of dimensionality. With enough parameters, you can fit anything.

### 🚩 Perfect Fit

Equity curve with no drawdowns, smooth upward slope.

**Reality**: All strategies have drawdown periods. Perfection suggests overfitting or data errors.

### 🚩 Non-Significant Monte Carlo

p-value > 0.05 in permutation test.

**Diagnosis**: Performance is not statistically distinguishable from random chance.

### 🚩 Parameter Hunting

Tried 100 different parameter combinations, reporting the best.

**Problem**: Data snooping. That "best" combo is likely overfit.

### 🚩 Only Works in One Period

Great in 2017, terrible in 2018-2024.

**Diagnosis**: Exploited specific market regime that doesn't repeat.

---

## Validation Checklist

Before deploying ANY strategy, verify:

### Data Integrity
- [ ] No lookahead bias (verified by automated tests)
- [ ] Point-in-time data (no future revisions)
- [ ] Survivorship bias checked (if applicable)
- [ ] Data quality verified (no errors, gaps handled properly)

### Statistical Rigor
- [ ] Sample size ≥ 252 observations
- [ ] Purged K-Fold CV performed (5+ folds)
- [ ] Walk-forward analysis completed (3+ periods)
- [ ] Monte Carlo p-value < 0.05
- [ ] Deflated Sharpe ratio calculated and positive
- [ ] 95% confidence interval for Sharpe is positive

### Robustness
- [ ] Parameter sensitivity tested (±20% variation)
- [ ] Works across multiple regimes
- [ ] Works across multiple assets (if applicable)
- [ ] Train-test gap < 30%
- [ ] Probability of Backtest Overfitting < 0.3

### Realism
- [ ] Transaction costs included (commission + slippage + spread)
- [ ] Realistic execution assumptions (no perfect fills)
- [ ] Position size limits respected
- [ ] Liquidity constraints considered
- [ ] Slippage increases with position size

### Documentation
- [ ] Economic rationale documented
- [ ] All parameters documented in HYPERPARAMETERS.md
- [ ] Validation results recorded
- [ ] Known limitations documented
- [ ] Failure modes identified

### Final Test
- [ ] Out-of-sample test on held-out data
- [ ] Test performed only ONCE
- [ ] Test results match validation expectations
- [ ] No re-optimization after test

---

## When to Reject a Strategy

Reject if ANY of the following:

1. ❌ Monte Carlo p-value ≥ 0.05
2. ❌ Confidence interval includes zero or negative
3. ❌ Train-test Sharpe gap > 50%
4. ❌ Probability of Backtest Overfitting > 0.5
5. ❌ Lookahead bias detected
6. ❌ Performance not robust to parameter changes
7. ❌ Only works in one specific regime
8. ❌ Sharpe > 3 without extraordinary evidence
9. ❌ Cannot explain economic rationale

**Better to Reject**: Type I error (rejecting a good strategy) is better than Type II error (deploying a bad strategy).

---

## Living with Uncertainty

**Accept**: We can never be 100% certain a strategy will work.

**Goal**: Maximize probability of success while minimizing probability of catastrophic failure.

**Philosophy**:
- Prefer robust strategies with moderate returns
- Over fragile strategies with exceptional backtests
- "Slow and steady wins the race"

**Reality Check**: If it were easy to find 3+ Sharpe strategies, everyone would be rich. Be skeptical of exceptional results.

---

## References

1. Bailey, D. H., & López de Prado, M. (2014). "The Deflated Sharpe Ratio"
2. López de Prado, M. (2018). "Advances in Financial Machine Learning", Chapter 7-12
3. Harvey, C. R., & Liu, Y. (2015). "Backtesting"
4. Sullivan, R., Timmermann, A., & White, H. (1999). "Data-Snooping, Technical Trading Rule Performance"

---

**Remember**: The goal is not to create the best-looking backtest. The goal is to create a strategy that will actually make money in the future.

Overfitting gives you the former. Rigorous validation gives you the latter.
