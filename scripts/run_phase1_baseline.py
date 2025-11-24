"""
Phase 1 Baseline Runner

Runs standardized S&P 500 PIT baselines for institutional signals.
This is DIAGNOSTIC ONLY - no optimization, no ensemble use.

Usage:
    # Momentum baseline
    python3 scripts/run_phase1_baseline.py --signal momentum

    # Insider baseline
    python3 scripts/run_phase1_baseline.py --signal insider

    # Quality v1 baseline
    python3 scripts/run_phase1_baseline.py --signal quality_v1

    # All baselines
    python3 scripts/run_phase1_baseline.py --signal all

Outputs:
    results/baselines/{signal}_sp500.json - Summary metrics
    results/baselines/{signal}_sp500_equity.csv - Daily equity curve
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime
import argparse
import json
from typing import Dict, List, Optional

from data.data_manager import DataManager
from signals.momentum.institutional_momentum import InstitutionalMomentum
from signals.insider.institutional_insider import InstitutionalInsider
from signals.quality.cross_sectional_quality import CrossSectionalQuality
from core.portfolio import Portfolio
from core.universe_manager import UniverseManager
from core.schedules import get_rebalance_dates
from config import get_logger

logger = get_logger(__name__)


class Phase1BaselineRunner:
    """
    Phase 1 baseline runner for institutional signals.

    Standardized settings:
    - Universe: S&P 500 PIT (sp500_actual)
    - Period: 2015-01-01 to latest available
    - Rebalance: Monthly (month-end)
    - Capital: $50,000
    - Transaction costs: 5 bps (default Schwab model)
    """

    def __init__(self,
                 start_date: str = '2015-01-01',
                 end_date: Optional[str] = None,
                 initial_capital: float = 50000):
        """
        Initialize baseline runner.

        Args:
            start_date: Backtest start (default 2015-01-01)
            end_date: Backtest end (default: latest available)
            initial_capital: Starting capital (default $50k)
        """
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.initial_capital = initial_capital

        self.dm = DataManager()
        self.um = UniverseManager(self.dm)

        # Standard transaction costs (5 bps total)
        self.transaction_costs = {
            'commission_pct': 0.0,      # $0 commission
            'slippage_pct': 0.0002,     # 2 bps
            'spread_pct': 0.0003,       # 3 bps
        }

        logger.info("=" * 80)
        logger.info("Phase 1 Baseline Runner - DIAGNOSTIC ONLY")
        logger.info("=" * 80)
        logger.info(f"Period: {self.start_date} to {self.end_date}")
        logger.info(f"Capital: ${self.initial_capital:,.0f}")
        logger.info(f"Transaction costs: 5.0 bps")
        logger.info(f"Universe: S&P 500 PIT (sp500_actual)")
        logger.info("=" * 80)

    def run_signal_baseline(self, signal_name: str) -> Dict:
        """
        Run baseline for a single signal.

        Args:
            signal_name: One of {'momentum', 'insider', 'quality_v1'}

        Returns:
            Dict with baseline results
        """
        logger.info(f"\n{'=' * 80}")
        logger.info(f"BASELINE: {signal_name.upper()}")
        logger.info(f"{'=' * 80}")

        # Initialize signal
        if signal_name == 'momentum':
            signal_class = InstitutionalMomentum
            signal_params = {
                'formation_period': 252,  # 12 months
                'skip_period': 21,        # 1 month
                'quintiles': True
            }
            needs_dm = False
        elif signal_name == 'insider':
            signal_class = InstitutionalInsider
            signal_params = {
                'lookback_days': 90,
                'min_transaction_value': 10000,
                'cluster_window': 7,
                'cluster_min_insiders': 3
            }
            needs_dm = True
        elif signal_name == 'quality_v1':
            signal_class = CrossSectionalQuality
            signal_params = {
                'w_profitability': 0.4,
                'w_growth': 0.3,
                'w_safety': 0.3,
                'winsorize_pct': [5, 95],
                'quintiles': True,
                'min_coverage': 0.5
            }
            needs_dm = True
        else:
            raise ValueError(f"Unknown signal: {signal_name}")

        # Initialize signal
        if needs_dm:
            signal = signal_class(signal_params, data_manager=self.dm)
        else:
            signal = signal_class(signal_params)

        # Get S&P 500 PIT universe
        logger.info(f"Building S&P 500 PIT universe as of {self.start_date}...")
        universe = self.um.get_universe(
            universe_type='sp500_actual',
            as_of_date=self.start_date,
            min_price=5.0
        )
        logger.info(f"Universe: {len(universe)} stocks")

        # Get rebalance dates (monthly, month-end)
        rebalance_dates = get_rebalance_dates(
            schedule='M',  # Monthly (month-end)
            dm=self.dm,
            start_date=self.start_date,
            end_date=self.end_date
        )
        logger.info(f"Rebalance dates: {len(rebalance_dates)} month-ends")

        # Initialize portfolio
        portfolio_params = {
            'max_position_size': 1.0,
            'max_positions': len(universe),
            'stop_loss_pct': None,
            'take_profit_pct': None,
            'max_portfolio_drawdown': 0.25,
            'drawdown_scale_factor': 0.5,
        }
        portfolio_params.update(self.transaction_costs)  # Merge transaction costs

        portfolio = Portfolio(
            initial_capital=self.initial_capital,
            params=portfolio_params
        )

        # Get price data
        logger.info("Fetching price data...")
        price_data = self.dm.get_multi_prices(
            tickers=universe,
            start_date=self.start_date,
            end_date=self.end_date
        )

        # Generate signals
        logger.info("Generating signals...")
        if signal_name == 'quality_v1':
            # Quality v1 uses cross-sectional method
            signals_df = signal.generate_signals_cross_sectional(
                universe_tickers=universe,
                rebalance_dates=rebalance_dates,
                start_date=self.start_date,
                end_date=self.end_date
            )
        else:
            # Momentum and Insider use standard method
            signals_df = signal.generate_signals(price_data)

        logger.info(f"Signals shape: {signals_df.shape}")
        logger.info(f"Signal range: [{signals_df.min().min():.2f}, {signals_df.max().max():.2f}]")

        # Run backtest
        logger.info("Running backtest...")
        equity_curve = portfolio.run_backtest(
            signals_df=signals_df,
            price_data=price_data,
            rebalance_dates=rebalance_dates
        )

        # Compute metrics
        returns = equity_curve.pct_change().dropna()

        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
        cagr = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe = (cagr - 0.02) / volatility if volatility > 0 else 0

        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdowns.min()

        win_rate = (returns > 0).mean()

        results = {
            'signal': signal_name,
            'universe': 'sp500_actual',
            'start_date': self.start_date,
            'end_date': self.end_date,
            'num_stocks': len(universe),
            'num_rebalances': len(rebalance_dates),
            'initial_capital': self.initial_capital,
            'final_equity': float(equity_curve.iloc[-1]),
            'total_return': float(total_return),
            'cagr': float(cagr),
            'volatility': float(volatility),
            'sharpe': float(sharpe),
            'max_drawdown': float(max_drawdown),
            'win_rate': float(win_rate),
            'num_trades': int(portfolio.trade_log.shape[0]) if hasattr(portfolio, 'trade_log') else 0
        }

        logger.info("\nBaseline Results:")
        logger.info(f"  Final Equity: ${results['final_equity']:,.0f}")
        logger.info(f"  Total Return: {results['total_return']:.2%}")
        logger.info(f"  CAGR: {results['cagr']:.2%}")
        logger.info(f"  Volatility: {results['volatility']:.2%}")
        logger.info(f"  Sharpe: {results['sharpe']:.2f}")
        logger.info(f"  Max Drawdown: {results['max_drawdown']:.2%}")
        logger.info(f"  Win Rate: {results['win_rate']:.2%}")

        # Save results
        output_dir = Path(__file__).parent.parent / 'results' / 'baselines'
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save JSON summary
        json_path = output_dir / f"{signal_name}_sp500.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"\nSummary saved: {json_path}")

        # Save equity curve
        equity_df = pd.DataFrame({
            'date': equity_curve.index,
            'equity': equity_curve.values
        })
        csv_path = output_dir / f"{signal_name}_sp500_equity.csv"
        equity_df.to_csv(csv_path, index=False)
        logger.info(f"Equity curve saved: {csv_path}")

        return results


def main():
    parser = argparse.ArgumentParser(
        description='Phase 1 Baseline Runner - DIAGNOSTIC ONLY',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Momentum baseline
  %(prog)s --signal momentum

  # Insider baseline
  %(prog)s --signal insider

  # Quality v1 baseline
  %(prog)s --signal quality_v1

  # All baselines
  %(prog)s --signal all

Outputs:
  results/baselines/{signal}_sp500.json - Summary metrics
  results/baselines/{signal}_sp500_equity.csv - Daily equity curve
        """
    )

    parser.add_argument('--signal', required=True,
                       choices=['momentum', 'insider', 'quality_v1', 'all'],
                       help='Signal to run baseline for')
    parser.add_argument('--start', default='2015-01-01',
                       help='Start date (default: 2015-01-01)')
    parser.add_argument('--end', default=None,
                       help='End date (default: latest available)')
    parser.add_argument('--capital', type=float, default=50000,
                       help='Initial capital (default: $50,000)')

    args = parser.parse_args()

    # Initialize runner
    runner = Phase1BaselineRunner(
        start_date=args.start,
        end_date=args.end,
        initial_capital=args.capital
    )

    # Run baseline(s)
    if args.signal == 'all':
        signals = ['momentum', 'insider', 'quality_v1']
    else:
        signals = [args.signal]

    for signal in signals:
        try:
            runner.run_signal_baseline(signal)
        except Exception as e:
            logger.error(f"Failed to run {signal} baseline: {e}")
            import traceback
            traceback.print_exc()

    logger.info("\n" + "=" * 80)
    logger.info("✅ PHASE 1 BASELINES COMPLETE")
    logger.info("=" * 80)
    logger.info("Results in: results/baselines/")


if __name__ == '__main__':
    main()
