"""
Deflated Sharpe Ratio calculation per Bailey & López de Prado (2014).

Reference:
    Bailey, D. H., & López de Prado, M. (2014).
    "The Deflated Sharpe Ratio: Correcting for Selection Bias,
    Backtest Overfitting, and Non-Normality".
    Journal of Portfolio Management, 40(5), 94-107.

Key insight: Raw Sharpe ratios are inflated by:
1. Multiple testing (trying many strategies)
2. Non-normal returns (skewness, kurtosis)
3. Short sample sizes

DSR corrects for these to give the TRUE probability the strategy has skill.
"""

import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

import pandas as pd
import numpy as np
from scipy import stats

from signaltide_v4.config.settings import get_settings

logger = logging.getLogger(__name__)


@dataclass
class DSRResult:
    """Container for Deflated Sharpe Ratio results."""

    observed_sharpe: float  # Raw Sharpe ratio
    deflated_sharpe: float  # DSR-adjusted Sharpe
    p_value: float  # Probability this Sharpe is due to luck
    confidence_level: float  # 1 - p_value
    is_significant: bool  # Above threshold?
    expected_max_sharpe: float  # E[max(SR)] under null
    n_trials: int  # Number of strategy trials assumed
    sample_size: int  # Number of return observations
    skewness: float
    kurtosis: float
    diagnostics: Dict[str, Any] = None


class DeflatedSharpeCalculator:
    """
    Calculate Deflated Sharpe Ratio to correct for multiple testing.

    The DSR answers: "What is the probability that an observed Sharpe
    ratio is due to luck rather than skill?"
    """

    def __init__(
        self,
        n_trials: Optional[int] = None,
        min_confidence: Optional[float] = None,
    ):
        """
        Initialize DSR calculator.

        Args:
            n_trials: Number of strategy trials to correct for
            min_confidence: Minimum confidence level required
        """
        settings = get_settings()
        self.n_trials = n_trials or settings.dsr_trials_adjustment
        self.min_confidence = min_confidence or settings.dsr_min_confidence

        logger.info(
            f"DSRCalculator: {self.n_trials} trials, "
            f"{self.min_confidence:.0%} confidence required"
        )

    def calculate(
        self,
        returns: pd.Series,
        benchmark_sharpe: float = 0.0,
    ) -> DSRResult:
        """
        Calculate Deflated Sharpe Ratio.

        Args:
            returns: Series of strategy returns
            benchmark_sharpe: Sharpe ratio of null hypothesis (default 0)

        Returns:
            DSRResult with deflated Sharpe and significance
        """
        returns = returns.dropna()
        n = len(returns)

        if n < 30:
            logger.warning(f"Insufficient data for DSR: {n} observations")
            return self._empty_result(n)

        # Calculate observed Sharpe
        mean_ret = returns.mean()
        std_ret = returns.std()

        if std_ret == 0:
            return self._empty_result(n)

        # Annualize (assuming daily returns)
        sr_observed = (mean_ret / std_ret) * np.sqrt(252)

        # Calculate moments for non-normality adjustment
        skew = float(stats.skew(returns))
        kurt = float(stats.kurtosis(returns))  # Excess kurtosis

        # Expected maximum Sharpe under null (multiple testing adjustment)
        e_max_sr = self._expected_max_sharpe(n)

        # Standard error of Sharpe ratio (accounting for non-normality)
        sr_std = self._sharpe_std_error(n, sr_observed, skew, kurt)

        # Deflated Sharpe
        if sr_std > 0:
            dsr = (sr_observed - e_max_sr) / sr_std
            p_value = 1 - stats.norm.cdf(dsr)
        else:
            dsr = 0.0
            p_value = 1.0

        confidence = 1 - p_value
        is_significant = confidence >= self.min_confidence

        result = DSRResult(
            observed_sharpe=float(sr_observed),
            deflated_sharpe=float(dsr),
            p_value=float(p_value),
            confidence_level=float(confidence),
            is_significant=is_significant,
            expected_max_sharpe=float(e_max_sr),
            n_trials=self.n_trials,
            sample_size=n,
            skewness=skew,
            kurtosis=kurt,
            diagnostics={
                'sharpe_std_error': float(sr_std),
                'mean_return_annual': float(mean_ret * 252),
                'volatility_annual': float(std_ret * np.sqrt(252)),
                'min_confidence_required': self.min_confidence,
            },
        )

        logger.info(
            f"DSR: Observed SR={sr_observed:.3f}, DSR={dsr:.3f}, "
            f"p={p_value:.3f}, significant={is_significant}"
        )

        return result

    def _expected_max_sharpe(self, n: int) -> float:
        """
        Calculate expected maximum Sharpe ratio under null.

        This is E[max(SR_1, ..., SR_T)] when all strategies have true SR=0.

        Formula from Bailey & López de Prado (2014):
        E[max] ≈ sqrt(V[SR]) * ((1-γ)*Φ^(-1)(1-1/T) + γ*Φ^(-1)(1-1/T*e))

        Simplified approximation used here.
        """
        if self.n_trials <= 1:
            return 0.0

        # Variance of Sharpe ratio estimate
        sr_var = (1 + 0.5 * 0) / n  # Simplified (skew=0 approx)

        # Expected maximum of T standard normals
        # Approximation: E[max] ≈ sqrt(2 * log(T))
        e_max_z = np.sqrt(2 * np.log(self.n_trials))

        return e_max_z * np.sqrt(sr_var) * np.sqrt(252)  # Annualized

    def _sharpe_std_error(
        self,
        n: int,
        sharpe: float,
        skew: float,
        kurt: float,
    ) -> float:
        """
        Calculate standard error of Sharpe ratio.

        Accounts for non-normality via Lo (2002) correction:
        Var(SR) ≈ (1 + 0.5*SR^2 - skew*SR + (kurt-3)/4*SR^2) / n
        """
        # Lo (2002) formula for Sharpe variance
        sr_annual = sharpe / np.sqrt(252)  # De-annualize for formula

        var_sr = (
            1 +
            0.5 * sr_annual**2 -
            skew * sr_annual +
            (kurt / 4) * sr_annual**2
        ) / n

        # Annualize the standard error
        std_sr = np.sqrt(max(var_sr, 1e-10)) * np.sqrt(252)

        return std_sr

    def _empty_result(self, n: int) -> DSRResult:
        """Return empty result for insufficient data."""
        return DSRResult(
            observed_sharpe=0.0,
            deflated_sharpe=0.0,
            p_value=1.0,
            confidence_level=0.0,
            is_significant=False,
            expected_max_sharpe=0.0,
            n_trials=self.n_trials,
            sample_size=n,
            skewness=0.0,
            kurtosis=0.0,
            diagnostics={'error': 'insufficient_data'},
        )

    def minimum_track_record(
        self,
        target_sharpe: float,
        confidence: float = 0.95,
    ) -> int:
        """
        Calculate minimum track record length for given Sharpe to be significant.

        From Bailey & López de Prado:
        MinTRL = (1 + (1-skew*SR + (kurt-3)/4*SR^2)) / ((SR - SR_0) / Z_α)^2

        Simplified version assuming normal returns.
        """
        z_alpha = stats.norm.ppf(confidence)
        sr_0 = self._expected_max_sharpe(252)  # Rough annual sample

        if target_sharpe <= sr_0:
            return 99999  # Never significant

        min_n = ((z_alpha / (target_sharpe - sr_0)) ** 2) * 252

        return int(np.ceil(min_n))
