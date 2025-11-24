"""
Simple validation metrics for Phase 2 Momentum optimization.

Provides basic performance metrics and overfitting controls without
the full complexity of López de Prado methods (those come in Phase 2.1+).

Functions:
- compute_basic_metrics: Sharpe, returns, drawdown from equity curve
- compute_regime_metrics: Performance by market regime
- simple_deflated_sharpe: Basic overfitting penalty
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from datetime import datetime


def compute_basic_metrics(equity_curve: pd.Series) -> Dict:
    """
    Compute basic performance metrics from equity curve.

    Args:
        equity_curve: Daily equity values (DatetimeIndex, values in $)

    Returns:
        Dict with keys:
            - total_return: Cumulative return (decimal)
            - annual_return: Annualized return (decimal)
            - volatility: Annualized volatility (decimal)
            - sharpe: Annualized Sharpe ratio
            - max_drawdown: Maximum drawdown (decimal, negative)
            - num_days: Number of trading days
            - num_months: Approximate number of months
    """
    if len(equity_curve) < 2:
        return {
            'total_return': 0.0,
            'annual_return': 0.0,
            'volatility': 0.0,
            'sharpe': 0.0,
            'max_drawdown': 0.0,
            'num_days': 0,
            'num_months': 0
        }

    # Daily returns
    daily_returns = equity_curve.pct_change().dropna()

    # Total return
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1

    # Annualized return (assuming 252 trading days)
    num_days = len(equity_curve)
    years = num_days / 252
    annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0

    # Annualized volatility
    volatility = daily_returns.std() * np.sqrt(252)

    # Sharpe ratio (annualized, assume 0% risk-free rate)
    sharpe = annual_return / volatility if volatility > 0 else 0.0

    # Maximum drawdown
    cumulative = (1 + daily_returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()

    # Approximate number of months
    num_months = int(num_days / 21)  # ~21 trading days per month

    return {
        'total_return': float(total_return),
        'annual_return': float(annual_return),
        'volatility': float(volatility),
        'sharpe': float(sharpe),
        'max_drawdown': float(max_drawdown),
        'num_days': int(num_days),
        'num_months': int(num_months)
    }


def compute_monthly_returns(equity_curve: pd.Series) -> pd.Series:
    """
    Convert daily equity curve to monthly returns.

    Args:
        equity_curve: Daily equity values (DatetimeIndex)

    Returns:
        Series of monthly returns (decimal), indexed by month-end dates
    """
    if len(equity_curve) < 2:
        return pd.Series(dtype=float)

    # Resample to month-end
    monthly_equity = equity_curve.resample('M').last()

    # Compute returns
    monthly_returns = monthly_equity.pct_change().dropna()

    return monthly_returns


def compute_regime_metrics(
    equity_curve: pd.Series,
    regime_definitions: Optional[Dict[str, tuple]] = None
) -> Dict:
    """
    Compute performance metrics by market regime.

    Args:
        equity_curve: Daily equity values (DatetimeIndex)
        regime_definitions: Dict mapping regime names to (start_date, end_date) tuples
                           If None, uses default regimes:
                           - covid: 2020-01-01 to 2020-12-31
                           - bear_2022: 2021-01-01 to 2022-12-31
                           - recent: 2023-01-01 to 2024-12-31

    Returns:
        Dict mapping regime names to metric dicts (return, sharpe, etc.)
    """
    if regime_definitions is None:
        regime_definitions = {
            'covid': ('2020-01-01', '2020-12-31'),
            'bear_2022': ('2021-01-01', '2022-12-31'),
            'recent': ('2023-01-01', '2024-12-31')
        }

    # Convert to monthly returns
    monthly_returns = compute_monthly_returns(equity_curve)

    regime_metrics = {}

    for regime_name, (start, end) in regime_definitions.items():
        # Filter monthly returns to regime
        regime_returns = monthly_returns[
            (monthly_returns.index >= start) &
            (monthly_returns.index <= end)
        ]

        if len(regime_returns) == 0:
            # No data in this regime
            regime_metrics[regime_name] = {
                'mean_return': 0.0,
                'volatility': 0.0,
                'sharpe': 0.0,
                'num_months': 0
            }
            continue

        # Monthly metrics
        mean_return = regime_returns.mean()
        volatility = regime_returns.std()
        sharpe = mean_return / volatility if volatility > 0 else 0.0

        regime_metrics[regime_name] = {
            'mean_return': float(mean_return),
            'volatility': float(volatility),
            'sharpe': float(sharpe),
            'num_months': int(len(regime_returns))
        }

    return regime_metrics


def simple_deflated_sharpe(
    sharpe: float,
    n_months: int,
    n_trials: int
) -> float:
    """
    Compute a simple Deflated Sharpe Ratio (DSR) proxy.

    This is a simplified version of the Bailey-López de Prado DSR.
    The full DSR requires estimating the distribution of returns and
    accounting for non-normality. This version uses a basic penalty
    based on the number of trials tested.

    Formula:
        DSR = Sharpe - sqrt(2 * ln(n_trials) / (n_months - 1))

    The penalty term grows with:
    - More trials tested (sqrt(ln(n_trials)))
    - Fewer months of data (1 / sqrt(n_months - 1))

    Args:
        sharpe: Observed Sharpe ratio (annualized)
        n_months: Number of months in sample
        n_trials: Total number of configurations tested

    Returns:
        Deflated Sharpe Ratio (can be negative)

    Example:
        >>> simple_deflated_sharpe(sharpe=0.5, n_months=100, n_trials=27)
        0.36  # Sharpe 0.5 deflated by ~0.14 due to 27 trials over 100 months

    Notes:
        - DSR > 1.0: Strong evidence of skill (unlikely to be data-mined)
        - DSR > 0.5: Moderate evidence
        - DSR > 0: Weak evidence
        - DSR < 0: Likely overfitting
    """
    if n_months <= 1 or n_trials <= 0:
        return 0.0

    # Penalty term: sqrt(2 * ln(n_trials) / (n_months - 1))
    penalty = np.sqrt(2 * np.log(n_trials) / (n_months - 1))

    dsr = sharpe - penalty

    return float(dsr)


def check_acceptance_gates(
    full_sharpe: float,
    oos_sharpe: float,
    regime_metrics: Dict,
    dsr: float
) -> Dict[str, bool]:
    """
    Check if metrics pass predeclared acceptance gates.

    Gates (from MOMENTUM_PHASE2_SPEC.md):
    1. Full-sample Sharpe > 0.15
    2. OOS Sharpe ≥ 0.2
    3. All regimes Sharpe > -0.3 (no catastrophic failures)
    4. DSR > 0.5

    Args:
        full_sharpe: Full-sample Sharpe ratio
        oos_sharpe: Out-of-sample Sharpe ratio
        regime_metrics: Dict from compute_regime_metrics
        dsr: Deflated Sharpe Ratio

    Returns:
        Dict with boolean flags:
            - passes_full_gate: Full Sharpe > 0.15
            - passes_oos_gate: OOS Sharpe ≥ 0.2
            - passes_regime_gate: All regimes > -0.3
            - passes_dsr_gate: DSR > 0.5
            - passes_all_gates: All 4 gates passed
    """
    # Gate 1: Full-sample Sharpe
    passes_full_gate = full_sharpe > 0.15

    # Gate 2: OOS Sharpe
    passes_oos_gate = oos_sharpe >= 0.2

    # Gate 3: Regime stability (no regime < -0.3 Sharpe)
    passes_regime_gate = True
    for regime_name, metrics in regime_metrics.items():
        if metrics['sharpe'] < -0.3:
            passes_regime_gate = False
            break

    # Gate 4: DSR
    passes_dsr_gate = dsr > 0.5

    # Overall
    passes_all_gates = (
        passes_full_gate and
        passes_oos_gate and
        passes_regime_gate and
        passes_dsr_gate
    )

    return {
        'passes_full_gate': passes_full_gate,
        'passes_oos_gate': passes_oos_gate,
        'passes_regime_gate': passes_regime_gate,
        'passes_dsr_gate': passes_dsr_gate,
        'passes_all_gates': passes_all_gates
    }
