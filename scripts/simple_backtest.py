"""
Simple Backtest for Signals on Real Data

Tests each signal individually with basic long/short strategy:
- Long when signal > 0.5
- Short when signal < -0.5
- Flat otherwise

Compares to buy-and-hold benchmark.

Writes results to: results/simple_backtest_results.txt
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # scripts/ -> repo root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

from data.data_manager import DataManager
from signals.momentum.simple_momentum import SimpleMomentum
from signals.quality.simple_quality import SimpleQuality
from signals.insider.simple_insider import SimpleInsider


class SimpleBacktest:
    """Simple backtester for individual signals."""

    def __init__(self, output_file: str = 'results/simple_backtest_results.txt'):
        self.output_file = output_file
        self.dm = DataManager()
        self.results = []

        # Test universe
        self.tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',
                       'NVDA', 'TSLA', 'JPM', 'JNJ', 'XOM']
        self.start_date = '2020-01-01'
        self.end_date = '2023-12-31'

    def log(self, message: str):
        """Log message to both console and results."""
        print(message)
        self.results.append(message)

    def write_results(self):
        """Write all results to file."""
        output_path = Path(self.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            f.write('\n'.join(self.results))

        self.log(f"\n✓ Results written to {self.output_file}")

    def calculate_metrics(self, returns: pd.Series, label: str) -> dict:
        """Calculate performance metrics."""
        # Cumulative return
        cum_returns = (1 + returns).cumprod()
        total_return = cum_returns.iloc[-1] - 1

        # Annualized return (assuming ~252 trading days)
        n_years = len(returns) / 252
        annual_return = (1 + total_return) ** (1 / n_years) - 1

        # Volatility
        annual_vol = returns.std() * np.sqrt(252)

        # Sharpe ratio (assuming 0% risk-free rate)
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0

        # Max drawdown
        cum_max = cum_returns.cummax()
        drawdown = (cum_returns - cum_max) / cum_max
        max_drawdown = drawdown.min()

        # Win rate
        win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0

        # Number of trades (position changes)
        positions = pd.Series(0, index=returns.index)
        positions[returns != 0] = np.sign(returns[returns != 0])
        n_trades = (positions.diff() != 0).sum()

        return {
            'label': label,
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_vol': annual_vol,
            'sharpe': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'n_trades': n_trades,
            'n_days': len(returns)
        }

    def backtest_signal(self, signal, signal_name: str, ticker: str,
                       long_threshold: float = 0.5,
                       short_threshold: float = -0.5) -> dict:
        """
        Backtest a signal on a single ticker.

        Args:
            signal: Signal instance
            signal_name: Name for reporting
            ticker: Ticker to test
            long_threshold: Go long when signal > this
            short_threshold: Go short when signal < this

        Returns:
            Dict with performance metrics
        """
        try:
            # Get price data
            prices = self.dm.get_prices(ticker, self.start_date, self.end_date)

            if len(prices) == 0:
                return None

            # Prepare data for signal
            if 'ticker' in prices.columns:
                prices = prices.drop(columns=['ticker'])
            prices['ticker'] = ticker

            # Generate signals
            signals = signal.generate_signals(prices)

            # Calculate returns (using close prices)
            price_returns = prices['close'].pct_change()

            # Generate positions based on signal
            positions = pd.Series(0.0, index=signals.index)
            positions[signals > long_threshold] = 1.0   # Long
            positions[signals < short_threshold] = -1.0  # Short
            # Else: 0 (flat)

            # Calculate strategy returns
            # Shift positions by 1 to avoid lookahead (trade on tomorrow's open)
            strategy_returns = positions.shift(1) * price_returns

            # Drop NaN from shift
            strategy_returns = strategy_returns.dropna()

            # Calculate metrics
            metrics = self.calculate_metrics(strategy_returns, f"{signal_name} on {ticker}")

            # Also calculate buy-and-hold for comparison
            bh_metrics = self.calculate_metrics(price_returns.dropna(), f"Buy-Hold {ticker}")

            return {
                'ticker': ticker,
                'signal': signal_name,
                'strategy': metrics,
                'buyhold': bh_metrics
            }

        except Exception as e:
            self.log(f"  ✗ Error backtesting {signal_name} on {ticker}: {str(e)}")
            return None

    def run_backtest(self, signal, signal_name: str, params: dict):
        """Run backtest for a signal across all tickers."""
        self.log("=" * 80)
        self.log(f"Backtesting {signal_name}")
        self.log(f"Parameters: {params}")
        self.log("=" * 80)

        all_results = []

        for ticker in self.tickers:
            self.log(f"\n{ticker}:")

            result = self.backtest_signal(signal, signal_name, ticker)

            if result:
                all_results.append(result)

                strat = result['strategy']
                bh = result['buyhold']

                self.log(f"  Strategy:")
                self.log(f"    Total Return: {strat['total_return']*100:7.2f}%")
                self.log(f"    Annual Return: {strat['annual_return']*100:6.2f}%")
                self.log(f"    Sharpe Ratio:  {strat['sharpe']:6.3f}")
                self.log(f"    Max Drawdown: {strat['max_drawdown']*100:7.2f}%")
                self.log(f"    Win Rate:     {strat['win_rate']*100:6.2f}%")
                self.log(f"    Trades:        {strat['n_trades']}")

                self.log(f"  Buy-Hold:")
                self.log(f"    Total Return: {bh['total_return']*100:7.2f}%")
                self.log(f"    Sharpe Ratio:  {bh['sharpe']:6.3f}")

                # Compare
                outperformance = strat['total_return'] - bh['total_return']
                self.log(f"  Outperformance: {outperformance*100:+7.2f}%")

        # Aggregate statistics
        if all_results:
            self.log("\n" + "=" * 80)
            self.log(f"AGGREGATE STATISTICS - {signal_name}")
            self.log("=" * 80)

            strategy_returns = [r['strategy']['total_return'] for r in all_results]
            bh_returns = [r['buyhold']['total_return'] for r in all_results]
            sharpes = [r['strategy']['sharpe'] for r in all_results]
            outperfs = [r['strategy']['total_return'] - r['buyhold']['total_return']
                       for r in all_results]

            self.log(f"\nAverage Total Return:    {np.mean(strategy_returns)*100:7.2f}%")
            self.log(f"Average Sharpe Ratio:    {np.mean(sharpes):6.3f}")
            self.log(f"Average Outperformance:  {np.mean(outperfs)*100:+7.2f}%")
            self.log(f"Win Rate (vs Buy-Hold):  {(np.array(outperfs) > 0).sum()}/{len(outperfs)}")

            self.log(f"\nBest Performer:  {all_results[np.argmax(strategy_returns)]['ticker']}")
            self.log(f"  Return: {max(strategy_returns)*100:.2f}%")

            self.log(f"\nWorst Performer: {all_results[np.argmin(strategy_returns)]['ticker']}")
            self.log(f"  Return: {min(strategy_returns)*100:.2f}%")

        return all_results

    def run_all_backtests(self):
        """Run backtests for all signals."""
        self.log("SignalTide v3 - Simple Backtest")
        self.log(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log(f"Period: {self.start_date} to {self.end_date}")
        self.log(f"Universe: {', '.join(self.tickers)}")
        self.log("=" * 80)

        # SimpleMomentum
        self.log("\n\n")
        momentum_params = {'lookback': 20, 'rank_window': 252}
        momentum = SimpleMomentum(momentum_params)
        momentum_results = self.run_backtest(momentum, 'SimpleMomentum', momentum_params)

        # SimpleQuality
        self.log("\n\n")
        quality_params = {'rank_window': 252 * 2}
        quality = SimpleQuality(quality_params, data_manager=self.dm)
        quality_results = self.run_backtest(quality, 'SimpleQuality', quality_params)

        # SimpleInsider
        self.log("\n\n")
        insider_params = {'lookback_days': 30, 'rank_window': 252}
        insider = SimpleInsider(insider_params, data_manager=self.dm)
        insider_results = self.run_backtest(insider, 'SimpleInsider', insider_params)

        # Summary comparison
        self.log("\n\n")
        self.log("=" * 80)
        self.log("SUMMARY COMPARISON")
        self.log("=" * 80)

        if momentum_results:
            mom_sharpe = np.mean([r['strategy']['sharpe'] for r in momentum_results])
            self.log(f"SimpleMomentum:  Avg Sharpe = {mom_sharpe:.3f}")

        if quality_results:
            qual_sharpe = np.mean([r['strategy']['sharpe'] for r in quality_results])
            self.log(f"SimpleQuality:   Avg Sharpe = {qual_sharpe:.3f}")

        if insider_results:
            ins_sharpe = np.mean([r['strategy']['sharpe'] for r in insider_results])
            self.log(f"SimpleInsider:   Avg Sharpe = {ins_sharpe:.3f}")

        self.log("\n✓ Backtest complete!")
        self.log("\nNote: These are simple backtests with basic long/short rules.")
        self.log("Real portfolio optimization would improve performance significantly.")

        self.write_results()


if __name__ == '__main__':
    print("Starting simple backtest...")
    print("This will test signals with basic long/short strategy.\n")

    backtester = SimpleBacktest()
    backtester.run_all_backtests()

    print("\n✓ Backtest complete. Check results/simple_backtest_results.txt")
