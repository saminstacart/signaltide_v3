# Methodology - SignalTide v3

**Last Updated:** 2025-11-18

This document provides an academic-grade explanation of the quantitative methods used in SignalTide v3.

---

## Table of Contents

1. [Overview](#overview)
2. [Signal Generation](#signal-generation)
3. [Portfolio Construction](#portfolio-construction)
4. [Risk Management](#risk-management)
5. [Validation Framework](#validation-framework)
6. [Hyperparameter Optimization](#hyperparameter-optimization)
7. [Performance Measurement](#performance-measurement)
8. [References](#references)

---

## Overview

SignalTide v3 implements a multi-signal quantitative trading system based on modern portfolio theory, statistical machine learning, and rigorous empirical validation.

### Theoretical Foundation

Our approach draws from several academic traditions:

1. **Modern Portfolio Theory** (Markowitz, 1952): Optimal portfolio construction under risk-return tradeoffs
2. **Statistical Arbitrage** (Pole, 2007): Exploiting statistical mispricings in liquid markets
3. **Cross-Validation for Time Series** (Bergmeir & Benítez, 2012): Proper validation without lookahead bias
4. **Multiple Testing Correction** (Bailey et al., 2014): Accounting for multiple strategy testing
5. **Meta-Learning** (Vanschoren, 2018): Hyperparameter optimization for strategy configuration

### System Flow

```
Market Data → Signal Generation → Signal Aggregation → Portfolio Construction → Execution
                    ↓                     ↓                      ↓
              Validation ←──────── Optimization ←──────── Risk Management
```

---

## Signal Generation

### Philosophy

Each signal represents a testable hypothesis about market behavior. We do NOT:
- Cherry-pick signals based on backtested performance
- Overfit signals to historical data
- Ignore transaction costs
- Use future information

We DO:
- Test each signal through rigorous validation
- Document the economic rationale
- Account for all costs
- Maintain strict temporal discipline

### Signal Types

#### 1. Momentum Signals

**Economic Rationale**: Markets exhibit serial correlation due to:
- Behavioral biases (anchoring, herding)
- Gradual information diffusion
- Risk premium for trend exposure

**Mathematical Formulation**:

```
momentum_t = (P_t - P_{t-n}) / P_{t-n}

signal_t = normalize(momentum_t) ∈ [-1, 1]
```

Where:
- P_t is price at time t
- n is lookback period
- normalize() maps to [-1, 1] range

**Implementation Considerations**:
- Use log returns for better statistical properties
- Adjust for volatility to get risk-adjusted momentum
- Consider multiple timeframes (short, medium, long)

#### 2. Mean Reversion Signals

**Economic Rationale**: Prices deviate from fundamental value but revert due to:
- Arbitrage forces
- Overreaction to news
- Liquidity provision rewards

**Mathematical Formulation**:

```
z_t = (P_t - μ_t) / σ_t

signal_t = -tanh(z_t)  # Negative because we bet on reversion
```

Where:
- μ_t is rolling mean
- σ_t is rolling standard deviation
- tanh() provides smooth [-1, 1] mapping

**Implementation Considerations**:
- Ensure stationarity before applying mean reversion
- Use robust estimators (median, MAD) for outlier resistance
- Different lookback periods for μ and σ estimation

#### 3. Volatility Signals

**Economic Rationale**: Volatility clustering and volatility risk premium:
- High volatility predicts high future volatility
- Investors demand premium for volatility exposure
- Regime changes detectable through volatility

**Mathematical Formulation**:

```
σ_t = √(Σ(r_i - μ)² / n)  # Simple volatility

# Or more sophisticated:
σ_t = √(h_t)  where h_t follows GARCH(1,1):
h_t = ω + α·ε²_{t-1} + β·h_{t-1}
```

**Volatility Estimators**:

1. **Standard Deviation**: Simple but inefficient
2. **Parkinson**: Uses high-low range, more efficient
3. **Garman-Klass**: Uses OHLC, even more efficient
4. **Rogers-Satchell**: Accounts for drift, most efficient

#### 4. Volume Signals

**Economic Rationale**: Volume provides information about:
- Conviction behind price moves
- Liquidity availability
- Institutional participation

**Mathematical Formulation**:

```
relative_volume_t = V_t / MA(V, n)

signal_t = f(relative_volume_t, price_change_t)
```

Where signal strength increases when:
- High volume confirms price direction
- Low volume suggests reversal potential

---

## Portfolio Construction

### Signal Aggregation

Given K signals {S₁, S₂, ..., Sₖ}, we aggregate into a combined signal:

```
S_combined = Σ(w_i · S_i) / Σ(w_i)
```

Where:
- w_i is the weight for signal i
- Weights can be:
  - Equal (baseline)
  - Performance-based (Sharpe-weighted)
  - Regime-dependent (different weights per regime)

### Position Sizing

We support multiple position sizing methods:

#### 1. Equal Weight

```
position_i = equity / n_positions
```

Simple but ignores risk differences across assets.

#### 2. Volatility-Scaled

```
position_i = (equity / n_positions) · (σ_target / σ_i)
```

Scales positions inversely to volatility, targeting constant risk per position.

#### 3. Kelly Criterion

```
f* = (p·b - q) / b
```

Where:
- p = probability of win
- q = probability of loss
- b = win/loss ratio

**Note**: Full Kelly is too aggressive; we use fractional Kelly (f*/2 or f*/4).

#### 4. Risk Parity

```
position_i ∝ 1 / σ_i
```

Each position contributes equally to portfolio risk.

### Rebalancing

Rebalancing frequency is a hyperparameter (1H, 4H, 1D, 1W) because:
- More frequent = more responsive but higher costs
- Less frequent = lower costs but slower adaptation

---

## Risk Management

### Position-Level Risk

#### Stop Losses

```
stop_price = entry_price · (1 - stop_loss_pct)
```

Executed when price drops below stop_price.

**Considerations**:
- Too tight = stopped out by noise
- Too loose = large losses
- Optuna optimizes stop_loss_pct

#### Take Profits

```
take_profit_price = entry_price · (1 + take_profit_pct)
```

Locks in gains at predefined level.

### Portfolio-Level Risk

#### Maximum Drawdown Management

```
if current_drawdown > max_drawdown:
    scale_factor = (max_drawdown / current_drawdown) ^ scale_exponent
    reduce_positions_by(scale_factor)
```

Reduces exposure when drawdown exceeds threshold.

#### Position Limits

```
position_size = min(
    calculated_size,
    max_position_size · equity,
    available_capital
)
```

Prevents over-concentration.

---

## Validation Framework

**Critical Principle**: Validation prevents overfitting, the #1 cause of strategy failure.

### 1. Purged K-Fold Cross-Validation

Standard K-Fold CV fails for time series because:
- Training and test sets are not independent
- Information leaks across folds
- Overlapping samples create lookahead bias

**Solution**: Purged K-Fold (de Prado, 2018)

```
For each fold:
1. Split data into train (t₁ to t₂) and test (t₃ to t₄)
2. Purge training samples that overlap with test period
3. Add embargo period after t₂ to prevent leakage
4. Train on purged training set
5. Test on test set
```

**Mathematics**:

```
purge_period = int(n_samples · purge_pct)
embargo_period = int(n_samples · embargo_pct)

train_end = test_start - purge_period - embargo_period
```

### 2. Walk-Forward Analysis

```
For each period:
1. Train on expanding or rolling window
2. Optimize hyperparameters on validation set
3. Test on out-of-sample period
4. Move window forward
5. Repeat
```

This simulates realistic deployment where:
- We only have past data when making decisions
- Parameters may need re-optimization over time

### 3. Monte Carlo Permutation Testing

**Question**: Is the signal's performance due to skill or luck?

**Procedure**:
```
1. Calculate actual performance (e.g., Sharpe ratio)
2. For i in 1..N:
   a. Randomly permute signal values (breaking timing)
   b. Calculate performance of permuted signal
3. Compare actual to distribution of permuted performances
4. p-value = fraction of permuted ≥ actual
```

**Interpretation**:
- p < 0.05: Signal is statistically significant
- p ≥ 0.05: Cannot reject null (performance due to luck)

### 4. Deflated Sharpe Ratio

**Problem**: Multiple testing inflates significance.

**Solution**: Haircut-Sharpe Ratio (Bailey & de Prado, 2014)

```
DSR = Φ((SR - E[SR_max]) / σ[SR_max])
```

Where:
- SR is observed Sharpe ratio
- E[SR_max] is expected maximum Sharpe from N trials
- σ[SR_max] is standard deviation of maximum Sharpe
- Φ is cumulative normal distribution

**Calculation**:

```
E[SR_max] ≈ (1 - γ)·Φ⁻¹(1 - 1/N) + γ·Φ⁻¹(1 - 1/(N·e))

where γ ≈ 0.5772 (Euler-Mascheroni constant)
```

---

## Hyperparameter Optimization

### Why Optuna?

1. **Tree-structured Parzen Estimator (TPE)**: Smart sampling based on past trials
2. **Parallel Execution**: Utilize all CPU cores
3. **Pruning**: Early stopping of poor trials
4. **Study Persistence**: Resume interrupted optimizations
5. **Visualization**: Built-in plotting of optimization progress

### Objective Function

We maximize validated Sharpe ratio:

```
def objective(trial):
    # 1. Sample parameters
    params = sample_parameters(trial)

    # 2. Run purged K-fold CV
    fold_scores = []
    for train_idx, test_idx in purged_kfold.split(data):
        train_data = data.iloc[train_idx]
        test_data = data.iloc[test_idx]

        # 3. Backtest on test fold
        results = backtest(test_data, params)
        fold_scores.append(results['sharpe'])

    # 4. Return mean performance across folds
    return np.mean(fold_scores)
```

### Avoiding Overfitting in Optimization

**Risks**:
- Optimizing on same data used for validation
- Too many trials → overfitting to noise
- Cherry-picking best trial

**Mitigations**:
1. Use separate validation set from final test set
2. Apply deflated Sharpe to optimization results
3. Limit number of trials (100-500 typically sufficient)
4. Check train vs test performance gap
5. Run Monte Carlo on best parameters

### Search Space Design

**Principles**:
- Start wide, narrow based on evidence (not intuition)
- Use log scale for parameters spanning orders of magnitude
- Allow "crazy" values - let validation reject them
- Don't prematurely filter based on domain knowledge

**Example**:

```python
def sample_parameters(trial):
    return {
        'lookback': trial.suggest_int('lookback', 5, 200),
        'threshold': trial.suggest_float('threshold', 0.0, 1.0),
        'learning_rate': trial.suggest_float('lr', 1e-4, 1e-1, log=True),
        'method': trial.suggest_categorical('method', ['A', 'B', 'C']),
    }
```

---

## Performance Measurement

### Return Metrics

#### 1. Cumulative Return

```
R_cum = (V_final / V_initial) - 1
```

Simple but ignores path and risk.

#### 2. Annualized Return

```
R_annual = (1 + R_cum)^(252/n_days) - 1
```

Standardized for comparison across timeframes.

#### 3. Log Returns

```
r_log = log(P_t / P_{t-1})
```

Better statistical properties (additive, normality).

### Risk Metrics

#### 1. Volatility

```
σ_annual = σ_daily · √252
```

Standard deviation of returns, annualized.

#### 2. Maximum Drawdown

```
DD_t = (Peak_t - Valley_t) / Peak_t

MDD = max_t(DD_t)
```

Largest peak-to-trough decline.

#### 3. Value at Risk (VaR)

```
VaR_α = -Quantile(returns, α)
```

Worst loss at confidence level α (e.g., 95%).

#### 4. Conditional VaR (CVaR / Expected Shortfall)

```
CVaR_α = -E[returns | returns < -VaR_α]
```

Average loss beyond VaR threshold.

### Risk-Adjusted Returns

#### 1. Sharpe Ratio

```
SR = (R̄ - R_f) / σ
```

Return per unit of volatility.

**Interpretation**:
- SR < 1: Poor
- SR 1-2: Good
- SR > 2: Excellent (or suspicious - check for overfitting!)

#### 2. Sortino Ratio

```
Sortino = (R̄ - R_f) / σ_downside
```

Only penalizes downside volatility.

#### 3. Calmar Ratio

```
Calmar = R_annual / MDD
```

Return per unit of maximum drawdown.

#### 4. Information Ratio

```
IR = (R_p - R_b) / σ_tracking_error
```

Active return per unit of tracking error vs benchmark.

### Trade-Level Metrics

#### 1. Win Rate

```
win_rate = n_winning_trades / n_total_trades
```

Percentage of profitable trades.

#### 2. Profit Factor

```
PF = gross_profit / gross_loss
```

Ratio of winning to losing trades.

#### 3. Average Win/Loss Ratio

```
avg_win_loss = mean(winning_trades) / |mean(losing_trades)|
```

Size of average win vs average loss.

#### 4. Expectancy

```
expectancy = (win_rate · avg_win) - (loss_rate · avg_loss)
```

Expected value per trade.

---

## Statistical Inference

### Hypothesis Testing

**Null Hypothesis**: Strategy has no edge (return = 0 or = benchmark)

**Test Statistic**:

```
t = (R̄ - μ₀) / (σ / √n)
```

**p-value**: Probability of observing result if null is true

**Decision**:
- p < α (e.g., 0.05): Reject null, strategy has edge
- p ≥ α: Cannot reject null

### Multiple Testing Correction

When testing multiple strategies, use Bonferroni correction:

```
α_corrected = α / n_tests
```

Or false discovery rate (FDR) control for less conservative approach.

### Confidence Intervals

For Sharpe ratio:

```
CI = SR ± z_α/2 · SE(SR)

where SE(SR) ≈ √((1 + SR²/2) / n)
```

Provides range of plausible values.

---

## References

### Academic Papers

1. Markowitz, H. (1952). "Portfolio Selection." Journal of Finance.
2. Bergmeir, C., & Benítez, J. M. (2012). "On the use of cross-validation for time series predictor evaluation."
3. Bailey, D. H., & López de Prado, M. (2014). "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality."
4. López de Prado, M. (2018). "Advances in Financial Machine Learning." Wiley.
5. Pole, A. (2007). "Statistical Arbitrage: Algorithmic Trading Insights and Techniques." Wiley.

### Books

1. López de Prado, M. (2018). "Advances in Financial Machine Learning"
2. Chan, E. (2009). "Quantitative Trading: How to Build Your Own Algorithmic Trading Business"
3. Narang, R. K. (2013). "Inside the Black Box: A Simple Guide to Quantitative and High Frequency Trading"

### Online Resources

1. Quantopian Lectures: https://www.quantopian.com/lectures
2. QuantConnect Documentation: https://www.quantconnect.com/docs
3. Optuna Documentation: https://optuna.readthedocs.io

---

## Notation Guide

- R: Return
- σ: Volatility (standard deviation)
- μ: Mean
- P_t: Price at time t
- V_t: Volume at time t
- w_i: Weight of component i
- n: Sample size or lookback period
- α: Significance level
- Φ: Cumulative normal distribution
- E[X]: Expected value of X
- ∈: Element of (set membership)
- ∝: Proportional to
- ≈: Approximately equal
