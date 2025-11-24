"""
Optuna Optimization for Simple Signals

Uses PurgedKFold cross-validation to find optimal parameters for each signal.
Runs in parallel using all available cores.

Usage:
    python scripts/optimize_signals.py momentum 200
    python scripts/optimize_signals.py quality 200
    python scripts/optimize_signals.py insider 200
    python scripts/optimize_signals.py all 200
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # scripts/ -> repo root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import optuna
from optuna.trial import Trial
import warnings
warnings.filterwarnings('ignore')

from data.data_manager import DataManager
from signals.momentum.simple_momentum import SimpleMomentum
from signals.quality.simple_quality import SimpleQuality
from signals.insider.simple_insider import SimpleInsider
from validation.purged_kfold import PurgedKFold


class SignalOptimizer:
    """Optimize signal parameters using Optuna with PurgedKFold CV."""

    def __init__(self,
                 tickers: list = None,
                 start_date: str = '2020-01-01',
                 end_date: str = '2023-12-31',
                 db_path: str = 'results/optimization/optuna_studies.db'):
        """
        Initialize optimizer.

        Args:
            tickers: List of tickers to optimize on
            start_date: Start date for optimization
            end_date: End date for optimization
            db_path: Path to Optuna database
        """
        self.dm = DataManager()

        # Use a diverse universe for optimization
        if tickers is None:
            self.tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',
                           'NVDA', 'TSLA', 'JPM', 'JNJ', 'XOM']
        else:
            self.tickers = tickers

        self.start_date = start_date
        self.end_date = end_date
        self.db_path = db_path

        # Preload all price data
        print(f"Loading price data for {len(self.tickers)} tickers...")
        self.price_data = {}
        for ticker in self.tickers:
            prices = self.dm.get_prices(ticker, start_date, end_date)
            if 'ticker' in prices.columns:
                prices = prices.drop(columns=['ticker'])
            prices['ticker'] = ticker
            self.price_data[ticker] = prices
        print(f"✓ Loaded {sum(len(p) for p in self.price_data.values())} price rows")

    def backtest_with_params(self,
                            signal,
                            prices: pd.DataFrame,
                            long_threshold: float = 0.5,
                            short_threshold: float = -0.5) -> dict:
        """
        Backtest signal on given prices.

        Args:
            signal: Signal instance with parameters set
            prices: Price DataFrame
            long_threshold: Go long when signal > this
            short_threshold: Go short when signal < this

        Returns:
            Dict with performance metrics
        """
        try:
            # Generate signals
            signals = signal.generate_signals(prices)

            # Calculate returns
            price_returns = prices['close'].pct_change()

            # Generate positions
            positions = pd.Series(0.0, index=signals.index)
            positions[signals > long_threshold] = 1.0
            positions[signals < short_threshold] = -1.0

            # Calculate strategy returns (shift positions to avoid lookahead)
            strategy_returns = positions.shift(1) * price_returns
            strategy_returns = strategy_returns.dropna()

            if len(strategy_returns) == 0 or strategy_returns.std() == 0:
                return {'sharpe': 0.0, 'total_return': 0.0, 'max_dd': 0.0, 'n_trades': 0}

            # Calculate metrics
            total_return = (1 + strategy_returns).prod() - 1
            annual_return = (1 + total_return) ** (252 / len(strategy_returns)) - 1
            annual_vol = strategy_returns.std() * np.sqrt(252)
            sharpe = annual_return / annual_vol if annual_vol > 0 else 0.0

            # Max drawdown
            cum_returns = (1 + strategy_returns).cumprod()
            cum_max = cum_returns.cummax()
            drawdown = (cum_returns - cum_max) / cum_max
            max_dd = drawdown.min()

            # Number of trades
            n_trades = (positions.diff() != 0).sum()

            return {
                'sharpe': sharpe,
                'total_return': total_return,
                'max_dd': max_dd,
                'n_trades': n_trades,
                'n_days': len(strategy_returns)
            }

        except Exception as e:
            print(f"  Backtest error: {str(e)}")
            return {'sharpe': 0.0, 'total_return': 0.0, 'max_dd': 0.0, 'n_trades': 0}

    def objective_momentum(self, trial: Trial) -> float:
        """Objective function for SimpleMomentum."""
        # Sample parameters with constraint: rank_window >= lookback
        lookback = trial.suggest_int('lookback', 5, 252)
        # Ensure rank_window >= lookback
        rank_window = trial.suggest_int('rank_window', max(lookback, 60), 252)
        long_threshold = trial.suggest_float('long_threshold', 0.1, 0.9)
        short_threshold = trial.suggest_float('short_threshold', -0.9, -0.1)

        params = {
            'lookback': lookback,
            'rank_window': rank_window
        }

        # Use PurgedKFold CV
        kfold = PurgedKFold(n_splits=5, embargo_pct=0.01)
        fold_sharpes = []

        try:
            signal = SimpleMomentum(params)
        except ValueError as e:
            # Parameter validation failed
            return -999.0  # Return very bad Sharpe

        for ticker in self.tickers[:5]:  # Use subset for speed
            prices = self.price_data[ticker]

            # Split data
            for train_idx, test_idx in kfold.split(prices):
                # Only backtest on test fold
                test_prices = prices.iloc[test_idx]

                if len(test_prices) < lookback + 50:
                    continue

                # Backtest
                metrics = self.backtest_with_params(
                    signal, test_prices, long_threshold, short_threshold
                )

                fold_sharpes.append(metrics['sharpe'])

                # Store additional metrics (convert numpy types to Python types)
                trial.set_user_attr(f'max_dd_{ticker}', float(metrics['max_dd']))
                trial.set_user_attr(f'n_trades_{ticker}', int(metrics['n_trades']))

        # Return mean Sharpe across folds
        mean_sharpe = np.mean(fold_sharpes) if fold_sharpes else 0.0

        # Penalize if too few trades
        avg_trades = np.mean([trial.user_attrs.get(f'n_trades_{t}', 0)
                             for t in self.tickers[:5]])
        if avg_trades < 10:
            mean_sharpe *= 0.5  # Penalize low trading frequency

        return mean_sharpe

    def objective_quality(self, trial: Trial) -> float:
        """Objective function for SimpleQuality."""
        # Sample parameters
        rank_window = trial.suggest_int('rank_window', 252, 252 * 3)
        long_threshold = trial.suggest_float('long_threshold', 0.0, 0.3)
        short_threshold = trial.suggest_float('short_threshold', -0.3, 0.0)

        params = {
            'rank_window': rank_window
        }

        # Use PurgedKFold CV
        kfold = PurgedKFold(n_splits=5, embargo_pct=0.01)
        fold_sharpes = []

        for ticker in self.tickers[:5]:
            prices = self.price_data[ticker]

            for train_idx, test_idx in kfold.split(prices):
                test_prices = prices.iloc[test_idx]

                if len(test_prices) < 100:
                    continue

                # Create signal
                signal = SimpleQuality(params, data_manager=self.dm)

                # Backtest
                metrics = self.backtest_with_params(
                    signal, test_prices, long_threshold, short_threshold
                )

                fold_sharpes.append(metrics['sharpe'])

        return np.mean(fold_sharpes) if fold_sharpes else 0.0

    def objective_insider(self, trial: Trial) -> float:
        """Objective function for SimpleInsider."""
        # Sample parameters
        lookback_days = trial.suggest_int('lookback_days', 10, 180)
        rank_window = trial.suggest_int('rank_window', 60, 252)
        long_threshold = trial.suggest_float('long_threshold', 0.1, 0.9)
        short_threshold = trial.suggest_float('short_threshold', -0.9, -0.1)

        params = {
            'lookback_days': lookback_days,
            'rank_window': rank_window
        }

        # Use PurgedKFold CV
        kfold = PurgedKFold(n_splits=5, embargo_pct=0.01)
        fold_sharpes = []

        for ticker in self.tickers[:5]:
            prices = self.price_data[ticker]

            for train_idx, test_idx in kfold.split(prices):
                test_prices = prices.iloc[test_idx]

                if len(test_prices) < lookback_days + 50:
                    continue

                # Create signal
                signal = SimpleInsider(params, data_manager=self.dm)

                # Backtest
                metrics = self.backtest_with_params(
                    signal, test_prices, long_threshold, short_threshold
                )

                fold_sharpes.append(metrics['sharpe'])

        return np.mean(fold_sharpes) if fold_sharpes else 0.0

    def optimize(self, signal_type: str, n_trials: int = 200, n_jobs: int = 12):
        """
        Run optimization for a signal type.

        Args:
            signal_type: 'momentum', 'quality', or 'insider'
            n_trials: Number of trials to run
            n_jobs: Number of parallel jobs
        """
        print("=" * 80)
        print(f"Optimizing {signal_type.upper()} Signal")
        print("=" * 80)
        print(f"Trials: {n_trials}")
        print(f"Parallel jobs: {n_jobs}")
        print(f"Tickers: {', '.join(self.tickers[:5])}")
        print(f"Period: {self.start_date} to {self.end_date}")
        print(f"Database: {self.db_path}")
        print("=" * 80)

        # Create or load study
        storage = f'sqlite:///{self.db_path}'
        study_name = f'{signal_type}_optimization_{datetime.now().strftime("%Y%m%d")}'

        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction='maximize',
            load_if_exists=True
        )

        # Select objective function
        if signal_type == 'momentum':
            objective = self.objective_momentum
        elif signal_type == 'quality':
            objective = self.objective_quality
        elif signal_type == 'insider':
            objective = self.objective_insider
        else:
            raise ValueError(f"Unknown signal type: {signal_type}")

        # Run optimization
        print(f"\nStarting optimization at {datetime.now().strftime('%H:%M:%S')}...")
        print(f"Using {n_jobs} parallel jobs\n")

        study.optimize(
            objective,
            n_trials=n_trials,
            n_jobs=n_jobs,
            show_progress_bar=True
        )

        # Report results
        print("\n" + "=" * 80)
        print("OPTIMIZATION COMPLETE")
        print("=" * 80)

        best_trial = study.best_trial
        print(f"\nBest Sharpe: {best_trial.value:.4f}")
        print(f"\nBest Parameters:")
        for key, value in best_trial.params.items():
            print(f"  {key}: {value}")

        # Save summary
        self.save_optimization_summary(signal_type, study)

        return study

    def save_optimization_summary(self, signal_type: str, study):
        """Save optimization summary to file."""
        output_dir = Path('results/optimization')
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / f'{signal_type}_summary.txt'

        with open(output_file, 'w') as f:
            f.write(f"Optimization Summary - {signal_type.upper()}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Number of trials: {len(study.trials)}\n")
            f.write(f"Best Sharpe: {study.best_value:.4f}\n\n")

            f.write("Best Parameters:\n")
            for key, value in study.best_params.items():
                f.write(f"  {key}: {value}\n")

            f.write("\n\nTop 10 Trials:\n")
            f.write("-" * 80 + "\n")

            # Get top trials sorted by value
            sorted_trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else -999, reverse=True)[:10]

            for trial in sorted_trials:
                f.write(f"\nTrial {trial.number}:\n")
                sharpe_str = f"{trial.value:.4f}" if trial.value is not None else "0.0000"
                f.write(f"  Sharpe: {sharpe_str}\n")
                f.write(f"  Params:\n")
                for key, value in trial.params.items():
                    f.write(f"    {key}: {value}\n")

        print(f"\n✓ Summary saved to {output_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Optimize signal parameters with Optuna')
    parser.add_argument('signal', choices=['momentum', 'quality', 'insider', 'all'],
                       help='Signal type to optimize')
    parser.add_argument('trials', type=int, default=200,
                       help='Number of trials (default: 200)')
    parser.add_argument('--jobs', type=int, default=12,
                       help='Number of parallel jobs (default: 12)')

    args = parser.parse_args()

    optimizer = SignalOptimizer()

    if args.signal == 'all':
        # Run all optimizations
        for signal_type in ['momentum', 'quality', 'insider']:
            print(f"\n\n{'='*80}")
            print(f"OPTIMIZING {signal_type.upper()}")
            print(f"{'='*80}\n")
            optimizer.optimize(signal_type, args.trials, args.jobs)
    else:
        optimizer.optimize(args.signal, args.trials, args.jobs)

    print("\n✓ Optimization complete!")
    print(f"\nResults saved to: results/optimization/")
    print(f"Optuna database: results/optimization/optuna_studies.db")


if __name__ == '__main__':
    main()
