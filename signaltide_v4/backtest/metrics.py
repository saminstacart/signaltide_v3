"""
Comprehensive performance metrics for backtest evaluation.

References:
    - Bailey & López de Prado (2012): "The Sharpe Ratio Efficient Frontier"
    - Bacon (2008): "Practical Portfolio Performance"
    - Sortino & van der Meer (1991): "Downside Risk"

Metrics include:
- Return metrics: Total return, CAGR, rolling returns
- Risk metrics: Volatility, downside deviation, VaR, CVaR
- Risk-adjusted: Sharpe, Sortino, Calmar, Information Ratio
- Factor attribution: Alpha, Beta, R-squared
"""

import logging
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

import pandas as pd
import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for all performance metrics."""

    # Return metrics
    total_return: float
    cagr: float
    annual_return: float

    # Risk metrics
    volatility: float
    downside_deviation: float
    max_drawdown: float
    avg_drawdown: float
    max_drawdown_duration: int  # Days

    # Risk-adjusted metrics
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float

    # VaR metrics
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float

    # Distribution metrics
    skewness: float
    kurtosis: float

    # Trading metrics
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float

    # Time metrics
    trading_days: int
    years: float

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'total_return': self.total_return,
            'cagr': self.cagr,
            'annual_return': self.annual_return,
            'volatility': self.volatility,
            'downside_deviation': self.downside_deviation,
            'max_drawdown': self.max_drawdown,
            'avg_drawdown': self.avg_drawdown,
            'max_drawdown_duration': self.max_drawdown_duration,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'calmar_ratio': self.calmar_ratio,
            'var_95': self.var_95,
            'var_99': self.var_99,
            'cvar_95': self.cvar_95,
            'cvar_99': self.cvar_99,
            'skewness': self.skewness,
            'kurtosis': self.kurtosis,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'trading_days': self.trading_days,
            'years': self.years,
        }


@dataclass
class BenchmarkComparison:
    """Comparison metrics vs benchmark."""

    alpha: float              # Annualized alpha
    beta: float               # Market beta
    correlation: float        # Correlation with benchmark
    tracking_error: float     # Annualized tracking error
    information_ratio: float  # Alpha / Tracking Error
    up_capture: float         # Upside capture ratio
    down_capture: float       # Downside capture ratio
    capture_ratio: float      # Up / Down capture
    r_squared: float          # R-squared of regression

    # Statistical significance
    alpha_t_stat: float
    alpha_p_value: float
    alpha_significant: bool  # p < 0.05

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'alpha': self.alpha,
            'beta': self.beta,
            'correlation': self.correlation,
            'tracking_error': self.tracking_error,
            'information_ratio': self.information_ratio,
            'up_capture': self.up_capture,
            'down_capture': self.down_capture,
            'capture_ratio': self.capture_ratio,
            'r_squared': self.r_squared,
            'alpha_t_stat': self.alpha_t_stat,
            'alpha_p_value': self.alpha_p_value,
            'alpha_significant': self.alpha_significant,
        }


class MetricsCalculator:
    """
    Calculate comprehensive performance metrics.

    Handles:
    - Daily/weekly/monthly return series
    - Benchmark comparisons
    - Rolling metrics
    - Distribution analysis
    """

    # Risk-free rate assumption (adjust as needed)
    RISK_FREE_RATE = 0.04  # 4% annual
    TRADING_DAYS_PER_YEAR = 252

    def __init__(
        self,
        risk_free_rate: Optional[float] = None,
        annualization_factor: int = 252,
    ):
        """
        Initialize calculator.

        Args:
            risk_free_rate: Annual risk-free rate (decimal)
            annualization_factor: Days per year for annualization
        """
        self.risk_free_rate = risk_free_rate or self.RISK_FREE_RATE
        self.annualization_factor = annualization_factor

        # Daily risk-free rate
        self.daily_rf = self.risk_free_rate / self.annualization_factor

    def calculate_metrics(
        self,
        returns: pd.Series,
        portfolio_values: Optional[pd.Series] = None,
    ) -> PerformanceMetrics:
        """
        Calculate all performance metrics.

        Args:
            returns: Daily return series
            portfolio_values: Optional portfolio value series

        Returns:
            PerformanceMetrics dataclass
        """
        if len(returns) == 0:
            return self._empty_metrics()

        returns = returns.dropna()

        if len(returns) < 2:
            return self._empty_metrics()

        # Basic time info
        trading_days = len(returns)
        years = trading_days / self.annualization_factor

        # Return metrics
        total_return = (1 + returns).prod() - 1
        cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        annual_return = returns.mean() * self.annualization_factor

        # Risk metrics
        volatility = returns.std() * np.sqrt(self.annualization_factor)

        negative_returns = returns[returns < 0]
        downside_deviation = (
            negative_returns.std() * np.sqrt(self.annualization_factor)
            if len(negative_returns) > 0 else volatility
        )

        # Drawdown analysis
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        avg_drawdown = drawdown[drawdown < 0].mean() if len(drawdown[drawdown < 0]) > 0 else 0

        # Max drawdown duration
        dd_duration = self._calculate_max_dd_duration(drawdown)

        # Risk-adjusted metrics
        excess_return = returns.mean() - self.daily_rf
        sharpe = (
            (excess_return * self.annualization_factor) / volatility
            if volatility > 0 else 0
        )

        sortino = (
            (excess_return * self.annualization_factor) / downside_deviation
            if downside_deviation > 0 else 0
        )

        calmar = cagr / abs(max_drawdown) if max_drawdown != 0 else 0

        # VaR and CVaR
        var_95 = returns.quantile(0.05)
        var_99 = returns.quantile(0.01)
        cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else var_95
        cvar_99 = returns[returns <= var_99].mean() if len(returns[returns <= var_99]) > 0 else var_99

        # Distribution metrics
        skewness = returns.skew()
        kurtosis = returns.kurtosis()

        # Trading metrics
        winning = returns[returns > 0]
        losing = returns[returns < 0]

        win_rate = len(winning) / len(returns) if len(returns) > 0 else 0
        avg_win = winning.mean() if len(winning) > 0 else 0
        avg_loss = losing.mean() if len(losing) > 0 else 0

        total_wins = winning.sum() if len(winning) > 0 else 0
        total_losses = abs(losing.sum()) if len(losing) > 0 else 1
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')

        return PerformanceMetrics(
            total_return=float(total_return),
            cagr=float(cagr),
            annual_return=float(annual_return),
            volatility=float(volatility),
            downside_deviation=float(downside_deviation),
            max_drawdown=float(max_drawdown),
            avg_drawdown=float(avg_drawdown),
            max_drawdown_duration=dd_duration,
            sharpe_ratio=float(sharpe),
            sortino_ratio=float(sortino),
            calmar_ratio=float(calmar),
            var_95=float(var_95),
            var_99=float(var_99),
            cvar_95=float(cvar_95),
            cvar_99=float(cvar_99),
            skewness=float(skewness),
            kurtosis=float(kurtosis),
            win_rate=float(win_rate),
            profit_factor=float(profit_factor),
            avg_win=float(avg_win),
            avg_loss=float(avg_loss),
            trading_days=trading_days,
            years=float(years),
        )

    def compare_to_benchmark(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series,
    ) -> BenchmarkComparison:
        """
        Compare strategy returns to benchmark.

        Args:
            returns: Strategy daily returns
            benchmark_returns: Benchmark daily returns

        Returns:
            BenchmarkComparison dataclass
        """
        # Align series
        combined = pd.DataFrame({
            'strategy': returns,
            'benchmark': benchmark_returns,
        }).dropna()

        if len(combined) < 30:
            return self._empty_comparison()

        strat = combined['strategy']
        bench = combined['benchmark']

        # Correlation
        correlation = strat.corr(bench)

        # OLS regression: strategy = alpha + beta * benchmark + epsilon
        X = bench.values.reshape(-1, 1)
        X_with_const = np.column_stack([np.ones(len(X)), X])
        y = strat.values

        try:
            beta_coeffs = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
            alpha_daily = beta_coeffs[0]
            beta = beta_coeffs[1]
        except np.linalg.LinAlgError:
            alpha_daily = 0.0
            beta = 1.0

        # Annualized alpha
        alpha = alpha_daily * self.annualization_factor

        # R-squared
        y_pred = X_with_const @ beta_coeffs
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Tracking error
        excess = strat - bench
        tracking_error = excess.std() * np.sqrt(self.annualization_factor)

        # Information ratio
        information_ratio = alpha / tracking_error if tracking_error > 0 else 0

        # Capture ratios
        up_days = bench > 0
        down_days = bench < 0

        up_capture = (
            strat[up_days].mean() / bench[up_days].mean() * 100
            if bench[up_days].mean() != 0 else 100
        )
        down_capture = (
            strat[down_days].mean() / bench[down_days].mean() * 100
            if bench[down_days].mean() != 0 else 100
        )
        capture_ratio = up_capture / down_capture if down_capture != 0 else 1

        # Alpha significance test
        residuals = y - y_pred
        n = len(y)
        k = 2  # Number of parameters (alpha + beta)

        se_residuals = np.sqrt(np.sum(residuals ** 2) / (n - k))
        se_alpha = se_residuals / np.sqrt(n)

        t_stat = alpha_daily / se_alpha if se_alpha > 0 else 0
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - k))

        return BenchmarkComparison(
            alpha=float(alpha),
            beta=float(beta),
            correlation=float(correlation),
            tracking_error=float(tracking_error),
            information_ratio=float(information_ratio),
            up_capture=float(up_capture),
            down_capture=float(down_capture),
            capture_ratio=float(capture_ratio),
            r_squared=float(r_squared),
            alpha_t_stat=float(t_stat),
            alpha_p_value=float(p_value),
            alpha_significant=p_value < 0.05,
        )

    def calculate_rolling_metrics(
        self,
        returns: pd.Series,
        window: int = 252,
    ) -> Dict[str, pd.Series]:
        """
        Calculate rolling performance metrics.

        Args:
            returns: Daily return series
            window: Rolling window in days

        Returns:
            Dict of rolling metric series
        """
        rolling = {}

        # Rolling return (annualized)
        rolling_return = returns.rolling(window).apply(
            lambda x: (1 + x).prod() ** (252 / len(x)) - 1 if len(x) > 0 else 0
        )
        rolling['rolling_return'] = rolling_return

        # Rolling volatility
        rolling_vol = returns.rolling(window).std() * np.sqrt(252)
        rolling['rolling_volatility'] = rolling_vol

        # Rolling Sharpe
        rolling_sharpe = rolling_return / rolling_vol
        rolling['rolling_sharpe'] = rolling_sharpe

        # Rolling max drawdown
        def max_dd(x):
            cumret = (1 + x).cumprod()
            running_max = cumret.cummax()
            dd = (cumret - running_max) / running_max
            return dd.min()

        rolling_dd = returns.rolling(window).apply(max_dd)
        rolling['rolling_max_drawdown'] = rolling_dd

        return rolling

    def _calculate_max_dd_duration(self, drawdown: pd.Series) -> int:
        """Calculate maximum drawdown duration in days."""
        in_drawdown = drawdown < 0

        if not in_drawdown.any():
            return 0

        # Find consecutive drawdown periods
        groups = (in_drawdown != in_drawdown.shift()).cumsum()
        dd_groups = drawdown.groupby(groups)

        max_duration = 0
        for _, group in dd_groups:
            if (group < 0).any():
                max_duration = max(max_duration, len(group))

        return max_duration

    def _empty_metrics(self) -> PerformanceMetrics:
        """Return empty metrics."""
        return PerformanceMetrics(
            total_return=0.0,
            cagr=0.0,
            annual_return=0.0,
            volatility=0.0,
            downside_deviation=0.0,
            max_drawdown=0.0,
            avg_drawdown=0.0,
            max_drawdown_duration=0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            var_95=0.0,
            var_99=0.0,
            cvar_95=0.0,
            cvar_99=0.0,
            skewness=0.0,
            kurtosis=0.0,
            win_rate=0.0,
            profit_factor=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            trading_days=0,
            years=0.0,
        )

    def _empty_comparison(self) -> BenchmarkComparison:
        """Return empty comparison."""
        return BenchmarkComparison(
            alpha=0.0,
            beta=1.0,
            correlation=0.0,
            tracking_error=0.0,
            information_ratio=0.0,
            up_capture=100.0,
            down_capture=100.0,
            capture_ratio=1.0,
            r_squared=0.0,
            alpha_t_stat=0.0,
            alpha_p_value=1.0,
            alpha_significant=False,
        )


def calculate_probabilistic_sharpe(
    observed_sharpe: float,
    benchmark_sharpe: float,
    n_observations: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """
    Calculate Probabilistic Sharpe Ratio.

    From Bailey & López de Prado (2012).

    Args:
        observed_sharpe: Observed Sharpe ratio
        benchmark_sharpe: Benchmark Sharpe to beat
        n_observations: Number of return observations
        skewness: Return skewness
        kurtosis: Return kurtosis

    Returns:
        Probability that true Sharpe exceeds benchmark
    """
    # Standard error of Sharpe
    se_sharpe = np.sqrt(
        (1 + 0.5 * observed_sharpe ** 2 - skewness * observed_sharpe +
         (kurtosis - 3) / 4 * observed_sharpe ** 2) / (n_observations - 1)
    )

    # Probabilistic Sharpe
    z = (observed_sharpe - benchmark_sharpe) / se_sharpe
    prob_sharpe = stats.norm.cdf(z)

    return prob_sharpe
