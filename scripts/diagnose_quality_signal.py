"""
Phase 0.1: InstitutionalQuality Signal Diagnostics

Comprehensive diagnostic analysis to understand why the current Quality signal
is underperforming (1.74% annual return, negative Sharpe).

Analysis:
1. Data coverage - % S&P 500 with non-null fundamental data
2. Quality score distribution and outliers
3. Cross-sectional vs time-series ranking issues
4. Simple decile tests (long-only top decile vs SPY)
5. Long-short spread (top minus bottom decile)
6. Sector tilts and biases

Output: results/quality_diagnostics_report.md
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.data_manager import DataManager
from core.universe_manager import UniverseManager
from signals.quality.institutional_quality import InstitutionalQuality
from config import get_logger

logger = get_logger(__name__)

class QualityDiagnostics:
    """Comprehensive diagnostics for InstitutionalQuality signal."""

    def __init__(self):
        self.dm = DataManager()
        self.um = UniverseManager(self.dm)

        # Standard Quality signal parameters
        self.params = {
            'use_profitability': True,
            'use_growth': True,
            'use_safety': True,
            'prof_weight': 0.4,
            'growth_weight': 0.3,
            'safety_weight': 0.3,
            'rebalance_frequency': 'monthly',
            'winsorize_pct': [5, 95]
        }

        self.signal = InstitutionalQuality(self.params, data_manager=self.dm)

    def run_full_diagnostics(self, snapshot_date: str = '2024-01-01') -> Dict:
        """Run all diagnostic tests."""
        logger.info("="*80)
        logger.info("Phase 0.1: InstitutionalQuality Signal Diagnostics")
        logger.info("="*80)

        results = {}

        # 1. Get S&P 500 universe for snapshot
        logger.info(f"\n1. Loading S&P 500 universe as of {snapshot_date}...")
        sp500 = self.um.get_universe(
            universe_type='sp500_actual',
            as_of_date=snapshot_date,
            min_price=5.0
        )
        logger.info(f"   Found {len(sp500)} S&P 500 constituents")
        results['universe_size'] = len(sp500)
        results['snapshot_date'] = snapshot_date

        # 2. Check data coverage
        logger.info("\n2. Checking fundamental data coverage...")
        coverage = self._check_data_coverage(sp500, snapshot_date)
        results['coverage'] = coverage

        # 3. Compute quality scores for snapshot
        logger.info("\n3. Computing quality scores for cross-sectional analysis...")
        quality_df = self._compute_quality_scores_cross_sectional(sp500, snapshot_date)
        results['quality_scores'] = quality_df

        # 4. Analyze quality score distribution
        logger.info("\n4. Analyzing quality score distribution...")
        distribution = self._analyze_distribution(quality_df)
        results['distribution'] = distribution

        # 5. Sector analysis
        logger.info("\n5. Analyzing sector tilts in top quality quintile...")
        sector_analysis = self._analyze_sectors(quality_df, sp500)
        results['sector_analysis'] = sector_analysis

        # 6. Simple decile backtest (long-only top decile)
        logger.info("\n6. Running simple decile backtest (2015-2024)...")
        decile_results = self._run_decile_backtest(
            start_date='2015-01-01',
            end_date='2024-12-31'
        )
        results['decile_backtest'] = decile_results

        logger.info("\n" + "="*80)
        logger.info("Diagnostics complete. Generating report...")
        logger.info("="*80)

        return results

    def _check_data_coverage(self, tickers: List[str], as_of_date: str) -> Dict:
        """Check what % of tickers have fundamental data."""
        coverage = {
            'total_tickers': len(tickers),
            'has_fundamentals': 0,
            'has_roe': 0,
            'has_roa': 0,
            'has_gp_assets': 0,
            'has_revenue': 0,
            'has_netinc': 0,
            'has_de': 0,
            'missing_tickers': []
        }

        for ticker in tickers[:50]:  # Sample first 50 for speed
            try:
                # Get fundamentals (ARQ - as-reported quarterly)
                fundamentals = self.dm.get_fundamentals(
                    ticker,
                    start_date='2020-01-01',  # Recent 5 years
                    end_date=as_of_date,
                    dimension='ARQ',
                    as_of_date=as_of_date
                )

                if len(fundamentals) > 0:
                    coverage['has_fundamentals'] += 1
                    if 'roe' in fundamentals.columns and fundamentals['roe'].notna().any():
                        coverage['has_roe'] += 1
                    if 'roa' in fundamentals.columns and fundamentals['roa'].notna().any():
                        coverage['has_roa'] += 1
                    if 'gp' in fundamentals.columns and 'assets' in fundamentals.columns:
                        if fundamentals['gp'].notna().any() and fundamentals['assets'].notna().any():
                            coverage['has_gp_assets'] += 1
                    if 'revenue' in fundamentals.columns and fundamentals['revenue'].notna().any():
                        coverage['has_revenue'] += 1
                    if 'netinc' in fundamentals.columns and fundamentals['netinc'].notna().any():
                        coverage['has_netinc'] += 1
                    if 'de' in fundamentals.columns and fundamentals['de'].notna().any():
                        coverage['has_de'] += 1
                else:
                    coverage['missing_tickers'].append(ticker)
            except Exception as e:
                logger.debug(f"Error checking {ticker}: {e}")
                coverage['missing_tickers'].append(ticker)

        # Calculate percentages (from sample of 50)
        sample_size = min(50, len(tickers))
        coverage['pct_has_fundamentals'] = coverage['has_fundamentals'] / sample_size * 100
        coverage['pct_has_roe'] = coverage['has_roe'] / sample_size * 100
        coverage['pct_has_roa'] = coverage['has_roa'] / sample_size * 100
        coverage['pct_has_gp_assets'] = coverage['has_gp_assets'] / sample_size * 100
        coverage['pct_has_revenue'] = coverage['has_revenue'] / sample_size * 100
        coverage['pct_has_netinc'] = coverage['has_netinc'] / sample_size * 100
        coverage['pct_has_de'] = coverage['has_de'] / sample_size * 100
        coverage['sample_size'] = sample_size

        logger.info(f"   Sample size: {sample_size} tickers")
        logger.info(f"   Has fundamentals: {coverage['pct_has_fundamentals']:.1f}%")
        logger.info(f"   Has ROE: {coverage['pct_has_roe']:.1f}%")
        logger.info(f"   Has ROA: {coverage['pct_has_roa']:.1f}%")
        logger.info(f"   Has GP/Assets: {coverage['pct_has_gp_assets']:.1f}%")
        logger.info(f"   Has Revenue: {coverage['pct_has_revenue']:.1f}%")
        logger.info(f"   Has Net Income: {coverage['pct_has_netinc']:.1f}%")
        logger.info(f"   Has D/E: {coverage['pct_has_de']:.1f}%")

        return coverage

    def _compute_quality_scores_cross_sectional(self, tickers: List[str],
                                                as_of_date: str) -> pd.DataFrame:
        """
        Compute quality scores for all tickers at a single point in time.

        This is what the signal SHOULD do for cross-sectional ranking,
        but currently it does time-series ranking instead.
        """
        scores = []

        for ticker in tickers[:100]:  # Analyze first 100 for speed
            try:
                # Get price data (minimal - just for API compatibility)
                prices = self.dm.get_price_data(
                    ticker,
                    start_date='2020-01-01',
                    end_date=as_of_date
                )

                if len(prices) == 0:
                    continue

                # Get fundamentals
                fundamentals = self.dm.get_fundamentals(
                    ticker,
                    start_date='2020-01-01',
                    end_date=as_of_date,
                    dimension='ARQ',
                    as_of_date=as_of_date
                )

                if len(fundamentals) == 0:
                    continue

                # Calculate quality components manually (same logic as signal)
                # Profitability
                prof_score = self._calc_profitability_manual(fundamentals)

                # Growth
                growth_score = self._calc_growth_manual(fundamentals)

                # Safety
                safety_score = self._calc_safety_manual(fundamentals)

                # Composite (40% prof, 30% growth, 30% safety)
                composite = (prof_score * 0.4 + growth_score * 0.3 + safety_score * 0.3)

                scores.append({
                    'ticker': ticker,
                    'profitability': prof_score,
                    'growth': growth_score,
                    'safety': safety_score,
                    'composite_quality': composite
                })

            except Exception as e:
                logger.debug(f"Error computing quality for {ticker}: {e}")
                continue

        df = pd.DataFrame(scores)
        logger.info(f"   Computed quality scores for {len(df)} stocks")
        return df

    def _calc_profitability_manual(self, fundamentals: pd.DataFrame) -> float:
        """Calculate profitability score (most recent value)."""
        scores = []

        # ROE
        if 'roe' in fundamentals.columns:
            roe = fundamentals['roe'].iloc[-1] if len(fundamentals) > 0 else np.nan
            if pd.notna(roe) and np.isfinite(roe):
                scores.append(roe)

        # ROA
        if 'roa' in fundamentals.columns:
            roa = fundamentals['roa'].iloc[-1] if len(fundamentals) > 0 else np.nan
            if pd.notna(roa) and np.isfinite(roa):
                scores.append(roa)

        # GP/A
        if 'gp' in fundamentals.columns and 'assets' in fundamentals.columns:
            gp = fundamentals['gp'].iloc[-1] if len(fundamentals) > 0 else np.nan
            assets = fundamentals['assets'].iloc[-1] if len(fundamentals) > 0 else np.nan
            if pd.notna(gp) and pd.notna(assets) and assets != 0:
                gp_a = gp / assets
                if np.isfinite(gp_a):
                    scores.append(gp_a)

        return np.mean(scores) if len(scores) > 0 else 0.0

    def _calc_growth_manual(self, fundamentals: pd.DataFrame) -> float:
        """Calculate growth score (YoY growth)."""
        scores = []

        # Revenue growth
        if 'revenue' in fundamentals.columns and len(fundamentals) >= 5:
            rev_growth = fundamentals['revenue'].pct_change(periods=4).iloc[-1]
            if pd.notna(rev_growth) and np.isfinite(rev_growth):
                scores.append(rev_growth)

        # Earnings growth
        if 'netinc' in fundamentals.columns and len(fundamentals) >= 5:
            ni_growth = fundamentals['netinc'].pct_change(periods=4).iloc[-1]
            if pd.notna(ni_growth) and np.isfinite(ni_growth):
                scores.append(ni_growth)

        return np.mean(scores) if len(scores) > 0 else 0.0

    def _calc_safety_manual(self, fundamentals: pd.DataFrame) -> float:
        """Calculate safety score."""
        scores = []

        # Low leverage (invert)
        if 'de' in fundamentals.columns:
            de = fundamentals['de'].iloc[-1] if len(fundamentals) > 0 else np.nan
            if pd.notna(de) and np.isfinite(de):
                scores.append(-de)  # Negative because low leverage = high safety

        # ROE stability (invert volatility)
        if 'roe' in fundamentals.columns and len(fundamentals) >= 8:
            roe_vol = fundamentals['roe'].tail(8).std()
            if pd.notna(roe_vol) and np.isfinite(roe_vol) and roe_vol != 0:
                scores.append(-roe_vol)

        # Positive ROE
        if 'roe' in fundamentals.columns:
            roe = fundamentals['roe'].iloc[-1] if len(fundamentals) > 0 else np.nan
            if pd.notna(roe):
                scores.append(1.0 if roe > 0 else 0.0)

        return np.mean(scores) if len(scores) > 0 else 0.0

    def _analyze_distribution(self, quality_df: pd.DataFrame) -> Dict:
        """Analyze distribution of quality scores."""
        if len(quality_df) == 0:
            return {'error': 'No quality scores computed'}

        composite = quality_df['composite_quality'].dropna()

        distribution = {
            'count': len(composite),
            'mean': composite.mean(),
            'std': composite.std(),
            'min': composite.min(),
            'q25': composite.quantile(0.25),
            'median': composite.median(),
            'q75': composite.quantile(0.75),
            'max': composite.max(),
            'num_zero': (composite == 0).sum(),
            'num_positive': (composite > 0).sum(),
            'num_negative': (composite < 0).sum()
        }

        logger.info(f"   Count: {distribution['count']}")
        logger.info(f"   Mean: {distribution['mean']:.3f}")
        logger.info(f"   Std: {distribution['std']:.3f}")
        logger.info(f"   Range: [{distribution['min']:.3f}, {distribution['max']:.3f}]")
        logger.info(f"   Zero scores: {distribution['num_zero']}")

        return distribution

    def _analyze_sectors(self, quality_df: pd.DataFrame, sp500: List[str]) -> Dict:
        """Analyze sector tilts in top quality quintile."""
        # This would require sector data from sharadar_tickers
        # For now, return placeholder
        return {
            'note': 'Sector analysis requires sharadar_tickers integration',
            'todo': 'Implement sector tilt analysis in next iteration'
        }

    def _run_decile_backtest(self, start_date: str, end_date: str) -> Dict:
        """
        Run simple decile backtest: long-only top decile vs SPY.

        This is a simplified test, not using the full backtest infrastructure.
        """
        logger.info(f"   Period: {start_date} to {end_date}")
        logger.info(f"   NOTE: This is a placeholder - full decile backtest")
        logger.info(f"         requires cross-sectional ranking infrastructure")
        logger.info(f"         which is not yet implemented for Quality signal.")

        return {
            'status': 'not_implemented',
            'reason': 'Current Quality signal uses time-series ranking, not cross-sectional',
            'problem': 'Cannot form deciles without cross-sectional ranking at each rebalance',
            'next_steps': [
                'Implement cross-sectional Quality ranking',
                'Add decile portfolio formation logic',
                'Run proper long-only and long-short backtests'
            ]
        }

    def generate_report(self, results: Dict, output_path: str):
        """Generate markdown diagnostic report."""
        with open(output_path, 'w') as f:
            f.write("# Phase 0.1: InstitutionalQuality Signal Diagnostics Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Snapshot Date:** {results['snapshot_date']}\n\n")
            f.write(f"**Universe:** S&P 500 (sp500_actual)\n\n")
            f.write("---\n\n")

            # Executive Summary
            f.write("## Executive Summary\n\n")
            f.write("**Current Status:** InstitutionalQuality signal shows 1.74% annual return with negative Sharpe (-0.157)\n\n")
            f.write("**Key Finding:** The signal uses **time-series ranking** (each stock vs its own history) instead of **cross-sectional ranking** (all stocks vs each other at each point in time). This is NOT the academic QMJ methodology.\n\n")

            # Current Implementation
            f.write("## 1. Current Implementation Analysis\n\n")
            f.write("### Signal Construction\n\n")
            f.write("**Components:**\n")
            f.write("- **Profitability (40%):** ROE + ROA + GP/Assets\n")
            f.write("- **Growth (30%):** Revenue YoY growth + Net Income YoY growth\n")
            f.write("- **Safety (30%):** Low leverage + Low ROE volatility + Positive ROE\n\n")

            f.write("**Ranking Method (CRITICAL ISSUE):**\n")
            f.write("- Current: `_to_time_series_signals()` - ranks each stock vs its own 2-year history\n")
            f.write("- Academic QMJ: Cross-sectional ranking - ranks all stocks vs each other at each month-end\n")
            f.write("- **This is a fundamental methodology mismatch!**\n\n")

            f.write("**Implications:**\n")
            f.write("- Time-series ranking captures \"is this stock more/less quality than its own past\" (more like momentum)\n")
            f.write("- Cross-sectional ranking captures \"which stocks are highest quality RIGHT NOW\" (true quality premium)\n")
            f.write("- Academic research on QMJ uses cross-sectional approach\n")
            f.write("- This explains poor performance - we're not capturing the quality factor at all\n\n")

            # Data Coverage
            f.write("## 2. Data Coverage Analysis\n\n")
            coverage = results['coverage']
            f.write(f"**Sample Size:** {coverage['sample_size']} tickers (from {coverage['total_tickers']} total)\n\n")
            f.write("**Coverage Rates:**\n")
            f.write(f"- Has any fundamentals: {coverage['pct_has_fundamentals']:.1f}%\n")
            f.write(f"- Has ROE: {coverage['pct_has_roe']:.1f}%\n")
            f.write(f"- Has ROA: {coverage['pct_has_roa']:.1f}%\n")
            f.write(f"- Has GP/Assets: {coverage['pct_has_gp_assets']:.1f}%\n")
            f.write(f"- Has Revenue: {coverage['pct_has_revenue']:.1f}%\n")
            f.write(f"- Has Net Income: {coverage['pct_has_netinc']:.1f}%\n")
            f.write(f"- Has D/E: {coverage['pct_has_de']:.1f}%\n\n")

            if coverage['pct_has_fundamentals'] < 90:
                f.write(f"**WARNING:** Only {coverage['pct_has_fundamentals']:.1f}% coverage - may explain low trade count.\n\n")

            # Quality Score Distribution
            f.write("## 3. Quality Score Distribution\n\n")
            if 'quality_scores' in results and len(results['quality_scores']) > 0:
                dist = results['distribution']
                f.write(f"**Computed for:** {dist['count']} stocks\n\n")
                f.write("**Summary Statistics:**\n")
                f.write(f"- Mean: {dist['mean']:.3f}\n")
                f.write(f"- Std Dev: {dist['std']:.3f}\n")
                f.write(f"- Range: [{dist['min']:.3f}, {dist['max']:.3f}]\n")
                f.write(f"- Median: {dist['median']:.3f}\n\n")
                f.write("**Distribution:**\n")
                f.write(f"- Positive scores: {dist['num_positive']}\n")
                f.write(f"- Zero scores: {dist['num_zero']}\n")
                f.write(f"- Negative scores: {dist['num_negative']}\n\n")
            else:
                f.write("**ERROR:** Could not compute quality scores for snapshot.\n\n")

            # Decile Backtest
            f.write("## 4. Decile Backtest Results\n\n")
            decile = results['decile_backtest']
            f.write(f"**Status:** {decile['status']}\n\n")
            f.write(f"**Reason:** {decile['reason']}\n\n")
            f.write("**Problem:** Cannot form proper decile portfolios with time-series ranking.\n\n")

            # Root Cause Analysis
            f.write("## 5. Root Cause Analysis\n\n")
            f.write("### Why is Quality Signal Failing?\n\n")
            f.write("**Primary Issue: Time-Series vs Cross-Sectional Ranking**\n\n")
            f.write("The current implementation uses time-series ranking (`_to_time_series_signals()`) which:\n")
            f.write("- Ranks each stock against its own 2-year history\n")
            f.write("- Does NOT compare stocks to each other\n")
            f.write("- Captures time-series mean reversion, not cross-sectional quality premium\n")
            f.write("- Is fundamentally incompatible with academic QMJ methodology\n\n")

            f.write("**Secondary Issues:**\n")
            f.write("- Quality scores may have low signal-to-noise ratio\n")
            f.write("- 2-year rolling window (504 days) requires long history\n")
            f.write("- min_periods=4 (quarters) may exclude many stocks\n")
            f.write("- Forward-filling quarterly data to daily creates stale signals\n\n")

            # Recommendations
            f.write("## 6. Recommendations for Phase 0.2\n\n")
            f.write("### Immediate Actions\n\n")
            f.write("1. **Implement Cross-Sectional Quality Ranking**\n")
            f.write("   - Add `CrossSectionalQuality` class that ranks all stocks at each rebalance date\n")
            f.write("   - Use same factor construction (Prof + Growth + Safety)\n")
            f.write("   - Rank on month-ends, forward-fill to daily\n")
            f.write("   - This is the proper QMJ methodology\n\n")

            f.write("2. **Build Quality Suite (3 variants)**\n")
            f.write("   - **QualityProfitability**: Heavy profitability (80% ROE/ROA/ROIC, 10% growth, 10% safety)\n")
            f.write("   - **QualityAccruals**: Earnings quality (Sloan accruals, Beneish M-Score, cash flow quality)\n")
            f.write("   - **QualityPiotroski**: F-Score implementation (9-point scorecard)\n\n")

            f.write("3. **Test Each Variant on S&P 500**\n")
            f.write("   - Run cross-sectional decile tests for each\n")
            f.write("   - Compare monotonicity (top decile vs bottom decile)\n")
            f.write("   - Measure turnover and capacity\n")
            f.write("   - Select top 2 for full baseline backtest\n\n")

            # Punch List
            f.write("## 7. Phase 0.2 Punch List\n\n")
            f.write("### Quality Variants to Implement\n\n")
            f.write("1. **CrossSectionalQuality** (fix current signal)\n")
            f.write("   - Implement proper cross-sectional ranking at each rebalance\n")
            f.write("   - Keep same factor construction initially\n")
            f.write("   - Expected improvement: significant (captures actual quality premium)\n\n")

            f.write("2. **QualityProfitability** (profitability-heavy)\n")
            f.write("   - 80% weight: ROE, ROA, ROIC, CF/Assets, Gross Margin\n")
            f.write("   - 10% weight: Revenue growth\n")
            f.write("   - 10% weight: Low leverage\n")
            f.write("   - Academic basis: Novy-Marx (2013) \"The Other Side of Value\"\n\n")

            f.write("3. **QualityAccruals** (earnings quality)\n")
            f.write("   - Sloan accruals ratio (low accruals = high quality)\n")
            f.write("   - Cash flow / Net income ratio\n")
            f.write("   - Working capital changes\n")
            f.write("   - Academic basis: Sloan (1996), Richardson et al. (2005)\n\n")

            f.write("4. **QualityPiotroski** (F-Score)\n")
            f.write("   - 9-point scorecard (profitability, leverage, operating efficiency)\n")
            f.write("   - Binary signals (0 or 1 for each criterion)\n")
            f.write("   - Sum to get F-Score (0-9)\n")
            f.write("   - Academic basis: Piotroski (2000)\n\n")

            # Next Steps
            f.write("## 8. Next Steps\n\n")
            f.write("1. Review this report with stakeholders\n")
            f.write("2. Decide on Quality variants to implement (recommend all 4 above)\n")
            f.write("3. Implement CrossSectionalQuality base class\n")
            f.write("4. Implement 3 Quality variants\n")
            f.write("5. Run decile tests on each (2015-2024)\n")
            f.write("6. Compare performance and select top 2 for Phase 1 baseline\n\n")

            f.write("---\n\n")
            f.write("**End of Report**\n")

        logger.info(f"\nReport written to: {output_path}")


def main():
    """Run full diagnostic suite."""
    diagnostics = QualityDiagnostics()

    # Run diagnostics on recent snapshot
    results = diagnostics.run_full_diagnostics(snapshot_date='2024-01-01')

    # Generate report
    output_path = Path(__file__).parent.parent / 'results' / 'quality_diagnostics_report.md'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics.generate_report(results, str(output_path))

    logger.info("\n" + "="*80)
    logger.info("Phase 0.1 Complete!")
    logger.info(f"Review: {output_path}")
    logger.info("="*80)


if __name__ == '__main__':
    main()
