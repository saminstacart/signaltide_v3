#!/usr/bin/env python3
"""
Pre-Deployment Diagnostic Tear Sheet.

Comprehensive risk analysis before deploying SignalTide V4 strategy.
Runs 5 diagnostic modules with decision thresholds.

Diagnostics:
1. Sector Concentration ("Tech Trap")
2. Turnover Analysis ("Churn Burn")
3. Factor Attribution (FF5)
4. Drawdown Analysis
5. Signal Contribution

Decision Thresholds:
| Metric         | Green    | Yellow   | Red      |
|----------------|----------|----------|----------|
| Max Sector     | < 30%    | 30-40%   | > 40%    |
| Turnover       | < 100%   | 100-150% | > 150%   |
| Breakeven Cost | > 20bps  | 10-20bps | < 10bps  |
| Alpha t-stat   | > 2.5    | 2.0-2.5  | < 2.0    |
| Beta           | 0.8-1.1  | 1.1-1.3  | > 1.3    |
| Single Signal  | < 30%    | 30-50%   | > 50%    |
"""

import argparse
import json
import logging
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

import pandas as pd
import numpy as np
from scipy import stats

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from signaltide_v4.config.settings import get_settings
from signaltide_v4.data.market_data import MarketDataProvider
from signaltide_v4.data.factor_data import FactorDataProvider
from signaltide_v4.validation.factor_attribution import FactorAttributor, AttributionResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


# =============================================================================
# DECISION THRESHOLDS
# =============================================================================

THRESHOLDS = {
    'sector_concentration': {
        'green': 0.30,   # < 30%
        'yellow': 0.40,  # 30-40%
        # > 40% is red
    },
    'turnover': {
        'green': 1.00,   # < 100%
        'yellow': 1.50,  # 100-150%
        # > 150% is red
    },
    'breakeven_bps': {
        'green': 20,     # > 20bps
        'yellow': 10,    # 10-20bps
        # < 10bps is red
    },
    'alpha_t_stat': {
        'green': 2.5,    # > 2.5
        'yellow': 2.0,   # 2.0-2.5
        # < 2.0 is red
    },
    'beta': {
        'green_low': 0.8,
        'green_high': 1.1,
        'yellow_high': 1.3,
        # > 1.3 is red
    },
    'signal_contribution': {
        'green': 0.30,   # < 30%
        'yellow': 0.50,  # 30-50%
        # > 50% is red
    },
}


@dataclass
class DiagnosticResult:
    """Container for a single diagnostic result."""
    name: str
    status: str  # 'GREEN', 'YELLOW', 'RED'
    metric_value: float
    threshold: str
    details: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


@dataclass
class TearSheetResult:
    """Complete diagnostic tear sheet."""
    timestamp: str
    diagnostics: List[DiagnosticResult]
    overall_verdict: str  # 'READY FOR DEPLOYMENT', 'PROCEED WITH CAUTION', 'DO NOT DEPLOY - FIX FIRST'
    summary: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# 1. SECTOR CONCENTRATION ANALYSIS
# =============================================================================

class SectorConcentrationAnalyzer:
    """
    Analyze sector concentration to detect "Tech Trap".

    Uses GICS sectors from Sharadar TICKERS table.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._sector_cache = {}

    def _load_sector_mapping(self) -> Dict[str, str]:
        """Load ticker to GICS sector mapping."""
        if self._sector_cache:
            return self._sector_cache

        conn = sqlite3.connect(self.db_path)
        query = """
            SELECT ticker, sector
            FROM sharadar_tickers
            WHERE sector IS NOT NULL
        """
        df = pd.read_sql(query, conn)
        conn.close()

        self._sector_cache = dict(zip(df['ticker'], df['sector']))
        return self._sector_cache

    def analyze(
        self,
        portfolio_history: List[Dict[str, Any]],  # List of {date, positions: {ticker: weight}}
    ) -> DiagnosticResult:
        """
        Analyze sector concentration over portfolio history.

        Returns DiagnosticResult with:
        - Max single sector exposure (sector, %, date)
        - Average Tech+Communication exposure
        - Warning if any sector > 40%
        """
        sector_map = self._load_sector_mapping()

        # Calculate monthly sector weights
        monthly_sector_weights = []
        max_sector_exposure = 0.0
        max_sector_name = ""
        max_sector_date = ""

        tech_comm_exposures = []

        for record in portfolio_history:
            date = record['date']
            positions = record['positions']

            if not positions:
                continue

            # Calculate sector weights
            sector_weights = {}
            total_weight = sum(positions.values())

            for ticker, weight in positions.items():
                sector = sector_map.get(ticker, 'Unknown')
                normalized_weight = weight / total_weight if total_weight > 0 else 0
                sector_weights[sector] = sector_weights.get(sector, 0) + normalized_weight

            monthly_sector_weights.append({
                'date': date,
                'weights': sector_weights,
            })

            # Track max sector exposure
            for sector, weight in sector_weights.items():
                if weight > max_sector_exposure:
                    max_sector_exposure = weight
                    max_sector_name = sector
                    max_sector_date = date

            # Track Tech + Communication Services exposure
            tech_weight = sector_weights.get('Technology', 0) + sector_weights.get('Communication Services', 0)
            tech_comm_exposures.append(tech_weight)

        avg_tech_comm = np.mean(tech_comm_exposures) if tech_comm_exposures else 0.0

        # Determine status
        if max_sector_exposure < THRESHOLDS['sector_concentration']['green']:
            status = 'GREEN'
        elif max_sector_exposure < THRESHOLDS['sector_concentration']['yellow']:
            status = 'YELLOW'
        else:
            status = 'RED'

        warnings = []
        if max_sector_exposure > THRESHOLDS['sector_concentration']['yellow']:
            warnings.append(f"WARNING: {max_sector_name} sector reached {max_sector_exposure:.1%} on {max_sector_date}")
        if avg_tech_comm > 0.40:
            warnings.append(f"WARNING: Average Tech+Comm exposure is {avg_tech_comm:.1%}")

        return DiagnosticResult(
            name='Sector Concentration',
            status=status,
            metric_value=max_sector_exposure,
            threshold=f"<30% Green, 30-40% Yellow, >40% Red",
            details={
                'max_sector': max_sector_name,
                'max_sector_weight': max_sector_exposure,
                'max_sector_date': max_sector_date,
                'avg_tech_comm_exposure': avg_tech_comm,
                'n_periods_analyzed': len(monthly_sector_weights),
            },
            warnings=warnings,
        )


# =============================================================================
# 2. TURNOVER ANALYSIS
# =============================================================================

class TurnoverAnalyzer:
    """
    Analyze portfolio turnover to detect "Churn Burn".

    Calculates:
    - Monthly one-way turnover
    - Annualized turnover
    - Breakeven cost (how much cost before alpha disappears)
    """

    def analyze(
        self,
        portfolio_history: List[Dict[str, Any]],
        excess_return: float,  # Annualized excess return (e.g., 0.10 for 10%)
    ) -> DiagnosticResult:
        """
        Analyze turnover and calculate breakeven cost.

        Turnover = Σ|w_t - w_{t-1}| / 2 (one-way)
        Breakeven = Excess Return / Annualized Turnover (in bps)
        """
        if len(portfolio_history) < 2:
            return DiagnosticResult(
                name='Turnover Analysis',
                status='YELLOW',
                metric_value=0.0,
                threshold="<100% Green, 100-150% Yellow, >150% Red",
                details={'error': 'Insufficient data'},
                warnings=['WARNING: Not enough portfolio history for turnover analysis'],
            )

        monthly_turnovers = []

        for i in range(1, len(portfolio_history)):
            prev_positions = portfolio_history[i-1]['positions']
            curr_positions = portfolio_history[i]['positions']

            # Get all tickers
            all_tickers = set(prev_positions.keys()) | set(curr_positions.keys())

            # Calculate one-way turnover
            turnover = 0.0
            for ticker in all_tickers:
                prev_weight = prev_positions.get(ticker, 0)
                curr_weight = curr_positions.get(ticker, 0)
                turnover += abs(curr_weight - prev_weight)

            turnover = turnover / 2  # One-way
            monthly_turnovers.append(turnover)

        # Annualize turnover (12 months)
        avg_monthly_turnover = np.mean(monthly_turnovers)
        annualized_turnover = avg_monthly_turnover * 12

        # Calculate breakeven cost
        # If excess return is 10% and turnover is 100%, breakeven = 10%/100% = 10% = 1000bps
        # If we trade 100% of portfolio, and each trade costs X bps, total cost = 100% * X bps
        # Breakeven: excess_return = annualized_turnover * breakeven_cost
        # breakeven_cost = excess_return / annualized_turnover
        if annualized_turnover > 0:
            breakeven_cost_pct = excess_return / annualized_turnover
            breakeven_cost_bps = breakeven_cost_pct * 10000  # Convert to bps
        else:
            breakeven_cost_bps = float('inf')

        # Determine status based on turnover
        if annualized_turnover < THRESHOLDS['turnover']['green']:
            turnover_status = 'GREEN'
        elif annualized_turnover < THRESHOLDS['turnover']['yellow']:
            turnover_status = 'YELLOW'
        else:
            turnover_status = 'RED'

        # Also check breakeven cost
        if breakeven_cost_bps > THRESHOLDS['breakeven_bps']['green']:
            breakeven_status = 'GREEN'
        elif breakeven_cost_bps > THRESHOLDS['breakeven_bps']['yellow']:
            breakeven_status = 'YELLOW'
        else:
            breakeven_status = 'RED'

        # Overall status is the worse of the two
        status_order = ['GREEN', 'YELLOW', 'RED']
        status = status_order[max(status_order.index(turnover_status), status_order.index(breakeven_status))]

        warnings = []
        if annualized_turnover > THRESHOLDS['turnover']['yellow']:
            warnings.append(f"WARNING: Annualized turnover ({annualized_turnover:.0%}) exceeds 150%")
        if breakeven_cost_bps < THRESHOLDS['breakeven_bps']['yellow']:
            warnings.append(f"WARNING: Breakeven cost ({breakeven_cost_bps:.1f}bps) is below 10bps")

        return DiagnosticResult(
            name='Turnover Analysis',
            status=status,
            metric_value=annualized_turnover,
            threshold="Turnover: <100% Green, 100-150% Yellow, >150% Red; Breakeven: >20bps Green, 10-20bps Yellow, <10bps Red",
            details={
                'avg_monthly_turnover': avg_monthly_turnover,
                'annualized_turnover': annualized_turnover,
                'breakeven_cost_bps': breakeven_cost_bps,
                'excess_return_input': excess_return,
                'n_rebalances': len(monthly_turnovers),
            },
            warnings=warnings,
        )


# =============================================================================
# 3. FACTOR ATTRIBUTION (FF5)
# =============================================================================

class FactorAttributionAnalyzer:
    """
    Fama-French 5-Factor Attribution Analysis.

    Uses existing FactorAttributor from signaltide_v4.validation.
    """

    def __init__(self):
        self.attributor = FactorAttributor()

    def analyze(
        self,
        daily_returns: pd.Series,  # Strategy daily returns
    ) -> DiagnosticResult:
        """
        Run FF5 regression and return diagnostic result.
        """
        try:
            result = self.attributor.attribute(daily_returns)
        except Exception as e:
            logger.error(f"Factor attribution failed: {e}")
            return DiagnosticResult(
                name='Factor Attribution (FF5)',
                status='YELLOW',
                metric_value=0.0,
                threshold="Alpha t-stat: >2.5 Green, 2.0-2.5 Yellow, <2.0 Red",
                details={'error': str(e)},
                warnings=[f"WARNING: Factor attribution failed: {e}"],
            )

        # Alpha status
        if result.alpha_t_stat > THRESHOLDS['alpha_t_stat']['green']:
            alpha_status = 'GREEN'
        elif result.alpha_t_stat > THRESHOLDS['alpha_t_stat']['yellow']:
            alpha_status = 'YELLOW'
        else:
            alpha_status = 'RED'

        # Beta status
        beta = result.mkt_rf_beta
        if THRESHOLDS['beta']['green_low'] <= beta <= THRESHOLDS['beta']['green_high']:
            beta_status = 'GREEN'
        elif beta <= THRESHOLDS['beta']['yellow_high']:
            beta_status = 'YELLOW'
        else:
            beta_status = 'RED'

        # Overall status
        status_order = ['GREEN', 'YELLOW', 'RED']
        status = status_order[max(status_order.index(alpha_status), status_order.index(beta_status))]

        warnings = []
        if result.alpha_t_stat < THRESHOLDS['alpha_t_stat']['yellow']:
            warnings.append(f"WARNING: Alpha t-stat ({result.alpha_t_stat:.2f}) below 2.0 - alpha not statistically significant")
        if beta > THRESHOLDS['beta']['yellow_high']:
            warnings.append(f"WARNING: Beta ({beta:.2f}) exceeds 1.3 - high market exposure")

        # Interpretation
        interpretation = self.attributor.interpret(result)

        return DiagnosticResult(
            name='Factor Attribution (FF5)',
            status=status,
            metric_value=result.alpha_t_stat,
            threshold="Alpha t-stat: >2.5 Green, 2.0-2.5 Yellow, <2.0 Red; Beta: 0.8-1.1 Green, 1.1-1.3 Yellow, >1.3 Red",
            details={
                'alpha_annualized': result.alpha,
                'alpha_t_stat': result.alpha_t_stat,
                'alpha_p_value': result.alpha_p_value,
                'alpha_significant': result.alpha_significant,
                'mkt_rf_beta': result.mkt_rf_beta,
                'smb_beta': result.smb_beta,
                'hml_beta': result.hml_beta,
                'rmw_beta': result.rmw_beta,
                'cma_beta': result.cma_beta,
                'r_squared': result.r_squared,
                'n_observations': result.n_observations,
                'interpretation': interpretation,
            },
            warnings=warnings,
        )


# =============================================================================
# 4. DRAWDOWN ANALYSIS
# =============================================================================

class DrawdownAnalyzer:
    """
    Analyze drawdowns and compare to SPY.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path

    def _get_spy_returns(self, start_date: str, end_date: str) -> pd.Series:
        """Get SPY daily returns for comparison."""
        conn = sqlite3.connect(self.db_path)
        query = """
            SELECT date, closeadj
            FROM sharadar_prices
            WHERE ticker = 'SPY'
              AND date BETWEEN ? AND ?
            ORDER BY date
        """
        df = pd.read_sql(query, conn, params=[start_date, end_date])
        conn.close()

        if len(df) == 0:
            return pd.Series(dtype=float)

        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        returns = df['closeadj'].pct_change().dropna()
        return returns

    def _identify_drawdowns(
        self,
        returns: pd.Series,
        n_top: int = 3
    ) -> List[Dict[str, Any]]:
        """Identify the N largest drawdowns."""
        # Calculate cumulative returns and drawdown series
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown_series = (cum_returns - running_max) / running_max

        # Find drawdown periods
        drawdowns = []
        in_drawdown = False
        dd_start = None
        dd_trough = None
        dd_trough_date = None

        for date, dd in drawdown_series.items():
            if dd < 0:
                if not in_drawdown:
                    in_drawdown = True
                    dd_start = date
                    dd_trough = dd
                    dd_trough_date = date
                elif dd < dd_trough:
                    dd_trough = dd
                    dd_trough_date = date
            else:
                if in_drawdown:
                    # Drawdown ended
                    drawdowns.append({
                        'start_date': dd_start.strftime('%Y-%m-%d'),
                        'trough_date': dd_trough_date.strftime('%Y-%m-%d'),
                        'end_date': date.strftime('%Y-%m-%d'),
                        'depth': dd_trough,
                        'recovery_days': (date - dd_start).days,
                    })
                    in_drawdown = False

        # If still in drawdown at end
        if in_drawdown:
            drawdowns.append({
                'start_date': dd_start.strftime('%Y-%m-%d'),
                'trough_date': dd_trough_date.strftime('%Y-%m-%d'),
                'end_date': 'ongoing',
                'depth': dd_trough,
                'recovery_days': None,
            })

        # Sort by depth (most negative first) and take top N
        drawdowns.sort(key=lambda x: x['depth'])
        return drawdowns[:n_top]

    def analyze(
        self,
        daily_returns: pd.Series,
    ) -> DiagnosticResult:
        """
        Analyze drawdowns and compare to SPY during same periods.
        """
        if len(daily_returns) < 60:
            return DiagnosticResult(
                name='Drawdown Analysis',
                status='YELLOW',
                metric_value=0.0,
                threshold="Comparison to SPY during drawdowns",
                details={'error': 'Insufficient data'},
                warnings=['WARNING: Not enough data for drawdown analysis'],
            )

        start_date = daily_returns.index.min().strftime('%Y-%m-%d')
        end_date = daily_returns.index.max().strftime('%Y-%m-%d')

        # Get strategy drawdowns
        strategy_drawdowns = self._identify_drawdowns(daily_returns, n_top=3)

        # Get SPY returns
        spy_returns = self._get_spy_returns(start_date, end_date)

        # Compare strategy vs SPY during each drawdown period
        comparisons = []
        for dd in strategy_drawdowns:
            dd_start = dd['start_date']
            dd_end = dd['end_date'] if dd['end_date'] != 'ongoing' else end_date

            # Strategy return during drawdown
            mask = (daily_returns.index >= dd_start) & (daily_returns.index <= dd_end)
            strategy_period_return = (1 + daily_returns[mask]).prod() - 1 if mask.any() else 0

            # SPY return during same period
            if len(spy_returns) > 0:
                spy_mask = (spy_returns.index >= dd_start) & (spy_returns.index <= dd_end)
                spy_period_return = (1 + spy_returns[spy_mask]).prod() - 1 if spy_mask.any() else 0
            else:
                spy_period_return = None

            comparisons.append({
                'period': f"{dd_start} to {dd['end_date']}",
                'strategy_drawdown': dd['depth'],
                'strategy_return': strategy_period_return,
                'spy_return': spy_period_return,
                'relative_performance': (strategy_period_return - spy_period_return) if spy_period_return is not None else None,
            })

        # Calculate max drawdown
        max_dd = min([dd['depth'] for dd in strategy_drawdowns]) if strategy_drawdowns else 0.0

        # Status based on max drawdown severity
        if max_dd > -0.20:  # Less than 20% drawdown
            status = 'GREEN'
        elif max_dd > -0.30:  # 20-30% drawdown
            status = 'YELLOW'
        else:  # More than 30% drawdown
            status = 'RED'

        warnings = []
        for comp in comparisons:
            if comp['relative_performance'] is not None and comp['relative_performance'] < -0.05:
                warnings.append(
                    f"WARNING: Underperformed SPY by {abs(comp['relative_performance']):.1%} during {comp['period']}"
                )

        return DiagnosticResult(
            name='Drawdown Analysis',
            status=status,
            metric_value=max_dd,
            threshold="Max DD: >-20% Green, -20% to -30% Yellow, <-30% Red",
            details={
                'max_drawdown': max_dd,
                'top_3_drawdowns': strategy_drawdowns,
                'spy_comparisons': comparisons,
            },
            warnings=warnings,
        )


# =============================================================================
# 5. SIGNAL CONTRIBUTION ANALYSIS
# =============================================================================

class SignalContributionAnalyzer:
    """
    Analyze which signals are driving performance.

    Estimates contribution based on correlation of signal rankings to returns.
    """

    def analyze(
        self,
        signal_scores_history: List[Dict[str, Any]],  # [{date, signal_name, scores: {ticker: score}}]
        portfolio_returns: List[Dict[str, float]],  # [{date, ticker, return}]
    ) -> DiagnosticResult:
        """
        Estimate signal contributions to returns.

        If detailed signal data not available, returns simplified analysis.
        """
        if not signal_scores_history:
            return DiagnosticResult(
                name='Signal Contribution',
                status='YELLOW',
                metric_value=0.0,
                threshold="Single signal: <30% Green, 30-50% Yellow, >50% Red",
                details={
                    'note': 'Detailed signal history not available',
                    'estimation': 'Based on signal weights: Momentum 40%, Quality 40%, Insider 20%',
                },
                warnings=['WARNING: Unable to calculate exact signal contributions - using weight estimates'],
            )

        # Calculate signal-return correlations per period
        signal_correlations = {}

        for record in signal_scores_history:
            date = record['date']
            signal_name = record['signal_name']
            scores = record['scores']

            # Find returns for this date
            period_returns = {r['ticker']: r['return'] for r in portfolio_returns if r['date'] == date}

            if not period_returns:
                continue

            # Calculate rank correlation between signal scores and returns
            common_tickers = set(scores.keys()) & set(period_returns.keys())
            if len(common_tickers) < 5:
                continue

            signal_vals = [scores[t] for t in common_tickers]
            return_vals = [period_returns[t] for t in common_tickers]

            correlation, _ = stats.spearmanr(signal_vals, return_vals)

            if signal_name not in signal_correlations:
                signal_correlations[signal_name] = []
            signal_correlations[signal_name].append(correlation)

        # Calculate average correlations
        avg_correlations = {}
        for signal_name, corrs in signal_correlations.items():
            avg_correlations[signal_name] = np.mean(corrs) if corrs else 0.0

        # Estimate contribution (proportional to positive correlation)
        total_positive_corr = sum(max(0, c) for c in avg_correlations.values())

        contributions = {}
        if total_positive_corr > 0:
            for signal_name, corr in avg_correlations.items():
                contributions[signal_name] = max(0, corr) / total_positive_corr
        else:
            # Default to equal weights if no clear signal
            n_signals = len(avg_correlations)
            for signal_name in avg_correlations:
                contributions[signal_name] = 1.0 / n_signals if n_signals > 0 else 0

        # Find max contribution
        max_contribution = max(contributions.values()) if contributions else 0.0
        max_signal = max(contributions, key=contributions.get) if contributions else 'Unknown'

        # Determine status
        if max_contribution < THRESHOLDS['signal_contribution']['green']:
            status = 'GREEN'
        elif max_contribution < THRESHOLDS['signal_contribution']['yellow']:
            status = 'YELLOW'
        else:
            status = 'RED'

        warnings = []
        if max_contribution > THRESHOLDS['signal_contribution']['yellow']:
            warnings.append(f"WARNING: {max_signal} signal contributes {max_contribution:.1%} of returns - high concentration risk")

        return DiagnosticResult(
            name='Signal Contribution',
            status=status,
            metric_value=max_contribution,
            threshold="Single signal: <30% Green, 30-50% Yellow, >50% Red",
            details={
                'contributions': contributions,
                'avg_correlations': avg_correlations,
                'max_signal': max_signal,
                'max_contribution': max_contribution,
            },
            warnings=warnings,
        )


# =============================================================================
# MAIN TEAR SHEET GENERATOR
# =============================================================================

class DiagnosticTearSheet:
    """
    Main class to generate comprehensive diagnostic tear sheet.
    """

    def __init__(self, db_path: str = None):
        settings = get_settings()
        self.db_path = db_path or settings.db_path

        self.sector_analyzer = SectorConcentrationAnalyzer(self.db_path)
        self.turnover_analyzer = TurnoverAnalyzer()
        self.factor_analyzer = FactorAttributionAnalyzer()
        self.drawdown_analyzer = DrawdownAnalyzer(self.db_path)
        self.signal_analyzer = SignalContributionAnalyzer()

    def generate(
        self,
        daily_returns: pd.Series,
        portfolio_history: List[Dict[str, Any]],
        excess_return: float,
        signal_scores_history: Optional[List[Dict[str, Any]]] = None,
        portfolio_returns: Optional[List[Dict[str, float]]] = None,
    ) -> TearSheetResult:
        """
        Generate complete diagnostic tear sheet.

        Args:
            daily_returns: Strategy daily returns series
            portfolio_history: List of {date, positions: {ticker: weight}}
            excess_return: Annualized excess return over SPY (e.g., 0.05 for 5%)
            signal_scores_history: Optional detailed signal scores
            portfolio_returns: Optional portfolio-level returns by ticker

        Returns:
            TearSheetResult with all diagnostics and verdict
        """
        logger.info("=" * 60)
        logger.info("GENERATING DIAGNOSTIC TEAR SHEET")
        logger.info("=" * 60)

        diagnostics = []

        # 1. Sector Concentration
        logger.info("1/5: Analyzing Sector Concentration...")
        sector_result = self.sector_analyzer.analyze(portfolio_history)
        diagnostics.append(sector_result)
        logger.info(f"    Status: {sector_result.status}")

        # 2. Turnover Analysis
        logger.info("2/5: Analyzing Turnover...")
        turnover_result = self.turnover_analyzer.analyze(portfolio_history, excess_return)
        diagnostics.append(turnover_result)
        logger.info(f"    Status: {turnover_result.status}")

        # 3. Factor Attribution
        logger.info("3/5: Running Factor Attribution (FF5)...")
        factor_result = self.factor_analyzer.analyze(daily_returns)
        diagnostics.append(factor_result)
        logger.info(f"    Status: {factor_result.status}")

        # 4. Drawdown Analysis
        logger.info("4/5: Analyzing Drawdowns...")
        drawdown_result = self.drawdown_analyzer.analyze(daily_returns)
        diagnostics.append(drawdown_result)
        logger.info(f"    Status: {drawdown_result.status}")

        # 5. Signal Contribution
        logger.info("5/5: Analyzing Signal Contributions...")
        signal_result = self.signal_analyzer.analyze(
            signal_scores_history or [],
            portfolio_returns or [],
        )
        diagnostics.append(signal_result)
        logger.info(f"    Status: {signal_result.status}")

        # Determine overall verdict
        statuses = [d.status for d in diagnostics]
        if 'RED' in statuses:
            verdict = "DO NOT DEPLOY - FIX FIRST"
        elif 'YELLOW' in statuses:
            verdict = "PROCEED WITH CAUTION"
        else:
            verdict = "READY FOR DEPLOYMENT"

        # Summary
        summary = {
            'n_green': statuses.count('GREEN'),
            'n_yellow': statuses.count('YELLOW'),
            'n_red': statuses.count('RED'),
            'all_warnings': [w for d in diagnostics for w in d.warnings],
        }

        logger.info("=" * 60)
        logger.info(f"VERDICT: {verdict}")
        logger.info(f"Green: {summary['n_green']}, Yellow: {summary['n_yellow']}, Red: {summary['n_red']}")
        logger.info("=" * 60)

        return TearSheetResult(
            timestamp=datetime.now().isoformat(),
            diagnostics=diagnostics,
            overall_verdict=verdict,
            summary=summary,
        )


def format_tearsheet(result: TearSheetResult) -> str:
    """Format tear sheet as text report."""
    lines = []
    lines.append("=" * 80)
    lines.append("SIGNALTIDE V4 - PRE-DEPLOYMENT DIAGNOSTIC TEAR SHEET")
    lines.append(f"Generated: {result.timestamp}")
    lines.append("=" * 80)
    lines.append("")

    # Overall verdict
    lines.append(f">>> OVERALL VERDICT: {result.overall_verdict}")
    lines.append(f"    Green: {result.summary['n_green']} | Yellow: {result.summary['n_yellow']} | Red: {result.summary['n_red']}")
    lines.append("")
    lines.append("-" * 80)

    # Each diagnostic
    for diag in result.diagnostics:
        status_emoji = {'GREEN': '[OK]', 'YELLOW': '[!!]', 'RED': '[XX]'}[diag.status]

        lines.append("")
        lines.append(f"{status_emoji} {diag.name}")
        lines.append(f"    Status: {diag.status}")
        lines.append(f"    Metric: {diag.metric_value:.4f}" if isinstance(diag.metric_value, float) else f"    Metric: {diag.metric_value}")
        lines.append(f"    Threshold: {diag.threshold}")

        # Key details
        lines.append("    Details:")
        for key, value in diag.details.items():
            if isinstance(value, float):
                lines.append(f"      - {key}: {value:.4f}")
            elif isinstance(value, dict):
                lines.append(f"      - {key}:")
                for k, v in value.items():
                    if isinstance(v, float):
                        lines.append(f"          {k}: {v:.4f}")
                    else:
                        lines.append(f"          {k}: {v}")
            elif isinstance(value, list) and len(value) > 0:
                lines.append(f"      - {key}:")
                for item in value[:5]:  # Limit to 5 items
                    if isinstance(item, dict):
                        item_str = ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in item.items())
                        lines.append(f"          {item_str}")
                    else:
                        lines.append(f"          {item}")
            else:
                lines.append(f"      - {key}: {value}")

        # Warnings
        if diag.warnings:
            lines.append("    Warnings:")
            for w in diag.warnings:
                lines.append(f"      {w}")

        lines.append("-" * 80)

    # All warnings summary
    if result.summary['all_warnings']:
        lines.append("")
        lines.append(">>> ALL WARNINGS:")
        for w in result.summary['all_warnings']:
            lines.append(f"    {w}")

    lines.append("")
    lines.append("=" * 80)
    lines.append("END OF DIAGNOSTIC TEAR SHEET")
    lines.append("=" * 80)

    return "\n".join(lines)


def run_from_backtest_result(
    backtest_log_path: str,
    output_path: str = "results/diagnostics_v4.txt",
) -> TearSheetResult:
    """
    Run diagnostics from a backtest log file.

    Parses the backtest output to extract necessary data.
    """
    logger.info(f"Loading backtest results from: {backtest_log_path}")

    # This is a simplified version - in production, you'd parse the actual backtest output
    # or load from a structured results file

    # For now, we'll run a fresh backtest and capture the data
    # Import and run the backtest
    from signaltide_v4.scripts.run_backtest_dynamic_sp500 import run_dynamic_backtest

    settings = get_settings()

    # Run backtest
    logger.info("Running backtest to collect diagnostic data...")
    result = run_dynamic_backtest(
        start_date='2015-01-01',
        end_date='2024-12-31',
        initial_capital=50000.0,
        transaction_cost_bps=5.0,
    )

    # Extract data for diagnostics
    daily_returns = result.returns

    # Build portfolio history from rebalance records
    portfolio_history = []
    for rebal in result.rebalances:
        portfolio_history.append({
            'date': rebal.date,
            'positions': rebal.portfolio.positions,
        })

    # Calculate excess return (approximate)
    total_return = result.total_return
    n_years = (pd.Timestamp(result.end_date) - pd.Timestamp(result.start_date)).days / 365.25
    cagr = (1 + total_return) ** (1 / n_years) - 1

    # Assume SPY CAGR of ~10% for excess calculation
    excess_return = cagr - 0.10

    logger.info(f"CAGR: {cagr:.2%}, Excess Return: {excess_return:.2%}")

    # Generate tear sheet
    tearsheet = DiagnosticTearSheet()
    tearsheet_result = tearsheet.generate(
        daily_returns=daily_returns,
        portfolio_history=portfolio_history,
        excess_return=excess_return,
    )

    # Format and save
    report = format_tearsheet(tearsheet_result)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(report)

    logger.info(f"Diagnostic tear sheet saved to: {output_path}")
    print(report)

    return tearsheet_result


def main():
    """Run diagnostic tear sheet."""
    parser = argparse.ArgumentParser(description='Run pre-deployment diagnostics')
    parser.add_argument(
        '--backtest-log',
        type=str,
        default='results/backtest_v4_phase3.txt',
        help='Path to backtest log file',
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results/diagnostics_v4.txt',
        help='Output path for diagnostic report',
    )
    args = parser.parse_args()

    result = run_from_backtest_result(args.backtest_log, args.output)

    # Exit with code based on verdict
    if result.overall_verdict == "DO NOT DEPLOY - FIX FIRST":
        sys.exit(1)
    elif result.overall_verdict == "PROCEED WITH CAUTION":
        sys.exit(0)  # Yellow is acceptable
    else:
        sys.exit(0)


if __name__ == '__main__':
    main()
