#!/usr/bin/env python3
"""
Portfolio generator for SignalTide v4.

Generates current portfolio holdings for deployment.

Usage:
    python -m signaltide_v4.scripts.generate_portfolio --help
    python -m signaltide_v4.scripts.generate_portfolio
    python -m signaltide_v4.scripts.generate_portfolio --capital 100000 --date 2024-01-15
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from signaltide_v4.config.settings import get_settings
from signaltide_v4.data.integration import DataIntegration
from signaltide_v4.signals.residual_momentum import ResidualMomentumSignal
from signaltide_v4.signals.quality import QualitySignal
from signaltide_v4.signals.insider import OpportunisticInsiderSignal
from signaltide_v4.portfolio.scoring import SignalAggregator
from signaltide_v4.portfolio.construction import PortfolioConstructor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='SignalTide v4 Portfolio Generator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate portfolio for today
    python -m signaltide_v4.scripts.generate_portfolio

    # Generate with custom capital
    python -m signaltide_v4.scripts.generate_portfolio --capital 100000

    # Generate for specific date
    python -m signaltide_v4.scripts.generate_portfolio --date 2024-01-15

    # Generate with custom universe
    python -m signaltide_v4.scripts.generate_portfolio --universe sp100
        """
    )

    parser.add_argument(
        '--capital',
        type=float,
        help='Portfolio capital (default from settings)'
    )
    parser.add_argument(
        '--date',
        type=str,
        help='As-of date for portfolio (YYYY-MM-DD, default: today)'
    )
    parser.add_argument(
        '--universe',
        type=str,
        default='sp500',
        choices=['sp500', 'sp100', 'default'],
        help='Universe to use'
    )
    parser.add_argument(
        '--top-n',
        type=int,
        help='Number of positions (default from settings)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results/current_portfolio_v4.json',
        help='Output file for portfolio'
    )
    parser.add_argument(
        '--csv',
        type=str,
        help='Also output as CSV for easy import'
    )
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    return parser.parse_args()


def generate_signals(
    tickers: List[str],
    as_of_date: str,
    data_integration: DataIntegration,
) -> Dict[str, pd.Series]:
    """Generate all signals for universe."""
    logger.info(f"Generating signals for {len(tickers)} tickers as of {as_of_date}")

    signals = {}

    # Residual Momentum
    try:
        momentum_signal = ResidualMomentumSignal()
        momentum_result = momentum_signal.generate(tickers, as_of_date)
        signals['momentum'] = momentum_result.scores
        logger.info(f"Momentum signal: {len(momentum_result.scores)} scores")
    except Exception as e:
        logger.warning(f"Momentum signal failed: {e}")
        signals['momentum'] = pd.Series(dtype=float)

    # Quality
    try:
        quality_signal = QualitySignal()
        quality_result = quality_signal.generate(tickers, as_of_date)
        signals['quality'] = quality_result.scores
        logger.info(f"Quality signal: {len(quality_result.scores)} scores")
    except Exception as e:
        logger.warning(f"Quality signal failed: {e}")
        signals['quality'] = pd.Series(dtype=float)

    # Insider
    try:
        insider_signal = OpportunisticInsiderSignal()
        insider_result = insider_signal.generate(tickers, as_of_date)
        signals['insider'] = insider_result.scores
        logger.info(f"Insider signal: {len(insider_result.scores)} scores")
    except Exception as e:
        logger.warning(f"Insider signal failed: {e}")
        signals['insider'] = pd.Series(dtype=float)

    return signals


def aggregate_signals(signals: Dict[str, pd.Series]) -> pd.Series:
    """Aggregate multiple signals into combined score."""
    # Default weights (can be configurable)
    weights = {
        'momentum': 0.40,
        'quality': 0.35,
        'insider': 0.25,
    }

    combined = pd.Series(dtype=float)

    for signal_name, signal_scores in signals.items():
        if len(signal_scores) > 0:
            weight = weights.get(signal_name, 0.25)
            if len(combined) == 0:
                combined = signal_scores * weight
            else:
                # Align and combine
                aligned = combined.align(signal_scores, fill_value=0)
                combined = aligned[0] + aligned[1] * weight

    # Normalize to [-1, 1]
    if len(combined) > 0 and combined.std() > 0:
        combined = (combined - combined.mean()) / combined.std()
        combined = combined.clip(-3, 3) / 3  # Clip outliers

    return combined


def construct_portfolio(
    scores: pd.Series,
    top_n: int,
    universe_data: 'UniverseSnapshot',
) -> Dict[str, float]:
    """Construct portfolio from scores."""
    # Select top N stocks
    top_stocks = scores.nlargest(top_n)

    if len(top_stocks) == 0:
        return {}

    # Get volatility for inverse-vol weighting
    volatilities = universe_data.volatility.reindex(top_stocks.index)

    # Inverse volatility weighting
    valid_vol = volatilities.dropna()
    valid_vol = valid_vol[valid_vol > 0]

    if len(valid_vol) == 0:
        # Equal weight fallback
        weights = pd.Series(
            index=top_stocks.index,
            data=1.0 / len(top_stocks)
        )
    else:
        inv_vol = 1.0 / valid_vol
        weights = inv_vol / inv_vol.sum()

        # Ensure all selected stocks have weights
        for ticker in top_stocks.index:
            if ticker not in weights.index:
                weights[ticker] = 1.0 / len(top_stocks)

        weights = weights / weights.sum()

    return weights.to_dict()


def calculate_shares(
    weights: Dict[str, float],
    capital: float,
    prices: pd.Series,
) -> Dict[str, int]:
    """Calculate number of shares to buy."""
    shares = {}

    for ticker, weight in weights.items():
        allocation = capital * weight
        price = prices.get(ticker, 0)

        if price > 0:
            n_shares = int(allocation / price)
            if n_shares > 0:
                shares[ticker] = n_shares

    return shares


def print_portfolio(portfolio: Dict, capital: float):
    """Print formatted portfolio."""
    print("\n" + "=" * 80)
    print("GENERATED PORTFOLIO")
    print("=" * 80)
    print(f"\nAs of Date: {portfolio['metadata']['as_of_date']}")
    print(f"Capital: ${capital:,.2f}")
    print(f"Number of Positions: {portfolio['n_positions']}")

    print(f"\n{'Ticker':<10} {'Weight':>10} {'Shares':>10} {'Price':>12} {'Value':>12}")
    print("-" * 80)

    total_value = 0
    for holding in portfolio['holdings']:
        ticker = holding['ticker']
        weight = holding['weight']
        shares = holding['shares']
        price = holding['price']
        value = holding['value']
        total_value += value

        print(f"{ticker:<10} {weight:>10.2%} {shares:>10} ${price:>11,.2f} ${value:>11,.2f}")

    print("-" * 80)
    print(f"{'TOTAL':<10} {portfolio['total_weight']:>10.2%} {'':<10} {'':<12} ${total_value:>11,.2f}")
    print(f"{'Cash':<10} {'':<10} {'':<10} {'':<12} ${portfolio['cash']:>11,.2f}")
    print("=" * 80)


def main():
    """Main entry point."""
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create output directory
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    settings = get_settings()
    capital = args.capital or settings.initial_capital
    as_of_date = args.date or datetime.now().strftime('%Y-%m-%d')
    top_n = args.top_n or settings.top_n_positions

    logger.info("=" * 60)
    logger.info("SignalTide v4 Portfolio Generator")
    logger.info("=" * 60)
    logger.info(f"Date: {as_of_date}")
    logger.info(f"Capital: ${capital:,.2f}")
    logger.info(f"Universe: {args.universe}")
    logger.info(f"Target positions: {top_n}")

    # Initialize data integration
    data_integration = DataIntegration()

    try:
        # Get universe
        tickers = data_integration.get_universe(args.universe, as_of_date)
        logger.info(f"Universe: {len(tickers)} tickers")

        if len(tickers) == 0:
            logger.error("No tickers in universe")
            return 1

        # Get universe snapshot
        universe_data = data_integration.get_universe_snapshot(tickers, as_of_date)

        # Generate signals
        signals = generate_signals(tickers, as_of_date, data_integration)

        # Check if we have any signals
        total_signals = sum(len(s) for s in signals.values())
        if total_signals == 0:
            logger.error("No signals generated")
            return 1

        # Aggregate signals
        combined_scores = aggregate_signals(signals)
        logger.info(f"Combined scores for {len(combined_scores)} tickers")

        # Construct portfolio
        weights = construct_portfolio(combined_scores, top_n, universe_data)
        logger.info(f"Portfolio: {len(weights)} positions")

        if len(weights) == 0:
            logger.error("Empty portfolio")
            return 1

        # Calculate shares
        shares = calculate_shares(weights, capital, universe_data.prices)

        # Build output
        holdings = []
        total_value = 0

        for ticker in sorted(weights.keys()):
            weight = weights[ticker]
            n_shares = shares.get(ticker, 0)
            price = universe_data.prices.get(ticker, 0)
            value = n_shares * price

            holdings.append({
                'ticker': ticker,
                'weight': weight,
                'shares': n_shares,
                'price': price,
                'value': value,
                'sector': universe_data.sector.get(ticker, 'Unknown'),
            })

            total_value += value

        # Sort by weight descending
        holdings.sort(key=lambda x: x['weight'], reverse=True)

        portfolio = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'as_of_date': as_of_date,
                'capital': capital,
                'universe': args.universe,
                'target_positions': top_n,
            },
            'holdings': holdings,
            'n_positions': len(holdings),
            'total_weight': sum(h['weight'] for h in holdings),
            'invested_value': total_value,
            'cash': capital - total_value,
            'signal_coverage': {
                name: len(scores) for name, scores in signals.items()
            },
        }

        # Print portfolio
        print_portfolio(portfolio, capital)

        # Save JSON
        with open(args.output, 'w') as f:
            json.dump(portfolio, f, indent=2, default=str)

        logger.info(f"\nPortfolio saved to: {args.output}")

        # Save CSV if requested
        if args.csv:
            df = pd.DataFrame(holdings)
            df.to_csv(args.csv, index=False)
            logger.info(f"CSV saved to: {args.csv}")

        return 0

    except Exception as e:
        logger.error(f"Portfolio generation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        data_integration.close()


if __name__ == '__main__':
    sys.exit(main())
