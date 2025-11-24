"""
Institutional-Grade Signal Base Classes and Utilities

Provides standard utilities used by all institutional signals:
- Cross-sectional ranking and z-scoring
- Winsorization
- Quintile construction
- Sector neutralization
- Monthly rebalancing alignment

References:
- Asness, Frazzini, Pedersen (2013) - "Quality Minus Junk"
- Jegadeesh, Titman (1993) - "Returns to Buying Winners"
- Fama, French (1992, 1993, 2015) - Factor methodology
"""

from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from datetime import datetime
from core.base_signal import BaseSignal


class InstitutionalSignal(BaseSignal):
    """
    Base class for institutional-grade signals.

    Provides standard methods used across all factor strategies:
    - Cross-sectional processing
    - Robust statistical transforms
    - Professional signal construction
    """

    def __init__(self, params: Dict[str, Any], name: str):
        # Set defaults for institutional parameters BEFORE validation
        params.setdefault('winsorize_pct', [5, 95])
        params.setdefault('sector_neutral', False)
        params.setdefault('rebalance_frequency', 'monthly')
        params.setdefault('quintiles', True)
        params.setdefault('quintile_mode', 'adaptive')  # Default to current behavior

        super().__init__(params, name)

        # Standard institutional parameters (now guaranteed to exist)
        self.winsorize_pct = params['winsorize_pct']
        self.sector_neutral = params['sector_neutral']
        self.rebalance_frequency = params['rebalance_frequency']
        self.quintiles = params['quintiles']
        self.quintile_mode = params['quintile_mode']

    @staticmethod
    def winsorize(values: pd.Series,
                  lower_pct: float = 5,
                  upper_pct: float = 95) -> pd.Series:
        """
        Winsorize values at specified percentiles.

        Standard practice: 5th/95th percentile to handle outliers
        while preserving cross-sectional ordering.

        Args:
            values: Series to winsorize
            lower_pct: Lower percentile (default 5)
            upper_pct: Upper percentile (default 95)

        Returns:
            Winsorized series
        """
        if len(values) == 0:
            return values

        lower = np.percentile(values.dropna(), lower_pct)
        upper = np.percentile(values.dropna(), upper_pct)

        return values.clip(lower, upper)

    @staticmethod
    def cross_sectional_zscore(df: pd.DataFrame,
                               value_column: str,
                               date_column: str = 'date',
                               group_column: Optional[str] = None) -> pd.Series:
        """
        Calculate cross-sectional z-scores at each point in time.

        Professional standard: Rank across stocks at each date,
        then convert to z-scores for standard interpretation.

        Args:
            df: DataFrame with dates and values
            value_column: Column to z-score
            date_column: Date column for grouping
            group_column: Optional sector/industry for within-group z-scoring

        Returns:
            Series of z-scores
        """
        def zscore_group(group):
            mean = group.mean()
            std = group.std()
            if std == 0 or pd.isna(std):
                return pd.Series(0, index=group.index)
            return (group - mean) / std

        if group_column:
            # Sector-neutral: z-score within sector at each date
            return df.groupby([date_column, group_column])[value_column].transform(zscore_group)
        else:
            # Market-wide: z-score across all stocks at each date
            return df.groupby(date_column)[value_column].transform(zscore_group)

    @staticmethod
    def cross_sectional_rank(df: pd.DataFrame,
                            value_column: str,
                            date_column: str = 'date',
                            ascending: bool = True,
                            pct: bool = True) -> pd.Series:
        """
        Rank values cross-sectionally at each point in time.

        Args:
            df: DataFrame with dates and values
            value_column: Column to rank
            date_column: Date column for grouping
            ascending: True if higher values get higher ranks
            pct: True to return percentile ranks [0, 1]

        Returns:
            Series of ranks (percentile if pct=True)
        """
        def rank_group(group):
            return group.rank(ascending=ascending, pct=pct)

        return df.groupby(date_column)[value_column].transform(rank_group)

    def to_quintiles(self,
                     values: pd.Series,
                     labels: Optional[List] = None,
                     mode: Optional[str] = None) -> pd.Series:
        """
        Convert continuous values to quintile signals.

        Standard quintile mapping:
        - Q1 (bottom 20%): -1.0
        - Q2: -0.5
        - Q3 (middle): 0.0
        - Q4: 0.5
        - Q5 (top 20%): 1.0

        Quintile Modes:
        - 'adaptive' (default): Uses pd.qcut with duplicates='drop'.
          When values cluster, bins merge, potentially selecting >20% in top/bottom bins.
          This is the current production behavior.

        - 'hard_20pct': Rank-based assignment ensuring exactly 20% per quintile.
          Uses rank(method='first') to break ties consistently.
          This matches Trial 11 manual logic.

        Args:
            values: Continuous values to discretize
            labels: Custom labels (default: [-1, -0.5, 0, 0.5, 1])
            mode: 'adaptive' or 'hard_20pct' (overrides self.quintile_mode if provided)

        Returns:
            Quintile-assigned values
        """
        if labels is None:
            labels = [-1.0, -0.5, 0.0, 0.5, 1.0]

        # Use provided mode or fall back to instance default
        quintile_mode = mode if mode is not None else self.quintile_mode

        if quintile_mode == 'hard_20pct':
            # Hard threshold: Exactly 20% per quintile using rank
            try:
                # Rank with method='first' for consistent tie-breaking
                ranks = values.rank(method='first')
                # Use qcut on ranks (which are unique) to get exact 20% bins
                quintiles = pd.qcut(ranks, q=5, labels=labels, duplicates='raise')
                return quintiles
            except ValueError:
                # Handle case where not enough values (< 5)
                return pd.Series(0, index=values.index)

        elif quintile_mode == 'adaptive':
            # Adaptive: Current behavior with qcut + duplicates='drop'
            try:
                return pd.qcut(values, q=5, labels=labels, duplicates='drop')
            except ValueError:
                # Handle case where not enough unique values
                return pd.Series(0, index=values.index)

        else:
            raise ValueError(f"Unknown quintile_mode: {quintile_mode}. "
                           f"Must be 'adaptive' or 'hard_20pct'.")

    @staticmethod
    def sector_neutralize(df: pd.DataFrame,
                         value_column: str,
                         sector_column: str,
                         date_column: str = 'date') -> pd.Series:
        """
        Create sector-neutral signals by ranking within sectors.

        Common in equity long-short: Removes sector beta,
        focuses on stock selection within industries.

        Args:
            df: DataFrame with dates, sectors, and values
            value_column: Column to neutralize
            sector_column: Sector/industry column
            date_column: Date column

        Returns:
            Sector-neutral signals
        """
        def rank_within_sector(group):
            return group.rank(pct=True) - 0.5  # Center at zero

        return df.groupby([date_column, sector_column])[value_column].transform(rank_within_sector)

    @staticmethod
    def align_to_month_end(dates: pd.DatetimeIndex) -> pd.Series:
        """
        Map dates to month-end for monthly rebalancing.

        Professional practice: Signals calculated daily but
        positions only updated at month-end.

        Args:
            dates: DatetimeIndex to align

        Returns:
            Series mapping each date to its month-end
        """
        return pd.Series(
            dates.to_period('M').to_timestamp('M'),
            index=dates
        )

    @staticmethod
    def demean_cross_sectional(df: pd.DataFrame,
                               value_column: str,
                               date_column: str = 'date') -> pd.Series:
        """
        Demean values cross-sectionally (market-neutral).

        Args:
            df: DataFrame with dates and values
            value_column: Column to demean
            date_column: Date column

        Returns:
            Demeaned values
        """
        return df.groupby(date_column)[value_column].transform(lambda x: x - x.mean())

    @staticmethod
    def standardize_cross_sectional(df: pd.DataFrame,
                                    value_column: str,
                                    date_column: str = 'date',
                                    target_vol: float = 0.15) -> pd.Series:
        """
        Standardize cross-sectional values to target volatility.

        Professional practice: Scale signals to consistent
        volatility for portfolio construction.

        Args:
            df: DataFrame with dates and values
            value_column: Column to standardize
            date_column: Date column
            target_vol: Target annualized volatility (default 15%)

        Returns:
            Standardized values
        """
        def scale_group(group):
            std = group.std()
            if std == 0 or pd.isna(std):
                return group
            return group / std * target_vol

        return df.groupby(date_column)[value_column].transform(scale_group)

    def validate_no_lookahead(self,
                             signals: pd.Series,
                             data: pd.DataFrame,
                             lookback: int = None) -> bool:
        """
        Verify no lookahead bias in signal construction.

        Professional requirement: All signals must be
        constructible with data available at signal date.

        Args:
            signals: Generated signals
            data: Input data
            lookback: Required lookback period

        Returns:
            True if no lookahead detected
        """
        if lookback is None:
            return True

        # Check that first signal occurs after sufficient data
        first_signal_date = signals.first_valid_index()
        first_data_date = data.index.min()

        required_date = first_data_date + pd.Timedelta(days=lookback)

        if first_signal_date < required_date:
            print(f"⚠️ Potential lookahead: First signal at {first_signal_date}, "
                  f"required data through {required_date}")
            return False

        return True

    @staticmethod
    def calculate_ic(predictions: pd.Series,
                    returns: pd.Series,
                    forward_periods: int = 21) -> float:
        """
        Calculate Information Coefficient (IC).

        Professional metric: Correlation between signal
        and forward returns.

        Args:
            predictions: Signal values
            returns: Realized returns
            forward_periods: Horizon in days (default 21 = 1 month)

        Returns:
            Spearman IC
        """
        # Align predictions with forward returns
        forward_returns = returns.shift(-forward_periods)

        # Drop NaN values
        aligned = pd.DataFrame({
            'pred': predictions,
            'ret': forward_returns
        }).dropna()

        if len(aligned) < 10:
            return 0.0

        # Spearman correlation (rank-based, more robust)
        ic = aligned['pred'].corr(aligned['ret'], method='spearman')

        return ic if not pd.isna(ic) else 0.0


class FactorAnalyzer:
    """
    Standard factor analytics used in institutional research.

    Produces metrics matching academic factor papers:
    - Quintile spreads
    - Information ratios
    - Turnover statistics
    - Factor correlation matrices
    """

    @staticmethod
    def quintile_analysis(signals: pd.DataFrame,
                         returns: pd.DataFrame,
                         signal_column: str = 'signal',
                         return_column: str = 'return') -> pd.DataFrame:
        """
        Analyze returns by signal quintile.

        Standard factor test: Sort stocks into quintiles,
        calculate average return for each bucket.

        Args:
            signals: DataFrame with dates, tickers, signals
            returns: DataFrame with returns
            signal_column: Signal column name
            return_column: Return column name

        Returns:
            DataFrame with quintile statistics
        """
        # Merge signals and returns
        df = signals.join(returns[return_column], how='inner')

        # Assign quintiles
        df['quintile'] = pd.qcut(df[signal_column], q=5, labels=[1, 2, 3, 4, 5])

        # Calculate quintile returns
        quintile_returns = df.groupby('quintile')[return_column].agg([
            ('mean', 'mean'),
            ('std', 'std'),
            ('sharpe', lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0),
            ('count', 'count')
        ])

        # Calculate spread (Q5 - Q1)
        if len(quintile_returns) >= 5:
            quintile_returns.loc['Spread', 'mean'] = (
                quintile_returns.loc[5, 'mean'] - quintile_returns.loc[1, 'mean']
            )

        return quintile_returns

    @staticmethod
    def calculate_turnover(signals: pd.DataFrame,
                          signal_column: str = 'signal',
                          date_column: str = 'date') -> float:
        """
        Calculate monthly signal turnover.

        Turnover = Average fraction of portfolio changed each month

        Args:
            signals: DataFrame with dates and signals
            signal_column: Signal column
            date_column: Date column

        Returns:
            Monthly turnover as decimal
        """
        # Get month-end signals
        monthly_signals = signals.groupby(
            signals[date_column].dt.to_period('M')
        )[signal_column].last()

        # Calculate changes
        changes = monthly_signals.diff().abs()

        # Average turnover
        return changes.mean()

    def generate_cross_sectional_scores(
        self,
        rebal_date: pd.Timestamp,
        universe: List[str],
        data_manager: "DataManager",
    ) -> pd.Series:
        """
        Generate cross-sectional signal scores for a universe of tickers.

        This is a contract-only method - subclasses MUST override.

        Args:
            rebal_date: Rebalance date (all data must be point-in-time as of this date)
            universe: List of ticker symbols to score
            data_manager: DataManager instance for fetching price/fundamental data

        Returns:
            pd.Series indexed by ticker with signal scores. NaNs allowed.
            Score scale is signal-specific (quintiles, z-scores, raw returns, etc.)

        Raises:
            NotImplementedError: This is a contract-only method

        Notes:
            - Must maintain point-in-time correctness (no lookahead bias)
            - Returned Series may be subset of universe (tickers with insufficient data excluded)
            - Score interpretation depends on signal implementation
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement generate_cross_sectional_scores(). "
            "This method is required for backtest integration."
        )


def institutional_signal_template() -> str:
    """
    Template for implementing institutional signals.

    Returns string showing the standard structure.
    """
    return """
class MyInstitutionalSignal(InstitutionalSignal):
    '''Docstring with academic reference.'''

    def __init__(self, params, data_manager=None):
        super().__init__(params, name='MySignal')
        self.data_manager = data_manager

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        # 1. Calculate raw factor values
        # 2. Winsorize outliers
        # 3. Cross-sectional rank or z-score
        # 4. Convert to quintiles (optional)
        # 5. Align to month-end (if monthly rebalancing)
        # 6. Return signals in [-1, 1] range
        pass
"""
