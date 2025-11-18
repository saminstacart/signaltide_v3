"""
Purged K-Fold Cross-Validation for time series.

Based on "Advances in Financial Machine Learning" by Marcos López de Prado.
Prevents information leakage in time series cross-validation.
"""

from typing import Generator, Tuple
import numpy as np
import pandas as pd


class PurgedKFold:
    """
    Purged K-Fold cross-validation for time series.

    Standard K-Fold CV fails for time series because:
    1. Training and test sets overlap in time
    2. Information leaks due to autocorrelation
    3. Overlapping samples create lookahead bias

    Purged K-Fold fixes this by:
    1. Removing (purging) training samples that overlap with test period
    2. Adding embargo period after training to prevent information leakage
    3. Respecting temporal ordering
    """

    def __init__(self, n_splits: int = 5, purge_pct: float = 0.05,
                 embargo_pct: float = 0.01):
        """
        Initialize Purged K-Fold.

        Args:
            n_splits: Number of folds
            purge_pct: Percentage of samples to purge before test set
            embargo_pct: Percentage of samples to embargo after training set
        """
        if n_splits < 2:
            raise ValueError(f"n_splits must be >= 2, got {n_splits}")

        if not 0 <= purge_pct < 0.5:
            raise ValueError(f"purge_pct must be in [0, 0.5), got {purge_pct}")

        if not 0 <= embargo_pct < 0.5:
            raise ValueError(f"embargo_pct must be in [0, 0.5), got {embargo_pct}")

        self.n_splits = n_splits
        self.purge_pct = purge_pct
        self.embargo_pct = embargo_pct

    def split(self, data: pd.DataFrame) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate purged train/test splits.

        Args:
            data: DataFrame with DatetimeIndex

        Yields:
            (train_indices, test_indices) tuples

        Example:
            ```python
            kfold = PurgedKFold(n_splits=5)
            for train_idx, test_idx in kfold.split(data):
                train_data = data.iloc[train_idx]
                test_data = data.iloc[test_idx]
                # Train and evaluate...
            ```
        """
        n_samples = len(data)
        indices = np.arange(n_samples)

        # Calculate test set size
        test_size = n_samples // self.n_splits

        # Calculate purge and embargo sizes
        purge_size = int(test_size * self.purge_pct)
        embargo_size = int(test_size * self.embargo_pct)

        for i in range(self.n_splits):
            # Test set for this fold
            test_start = i * test_size
            test_end = test_start + test_size if i < self.n_splits - 1 else n_samples
            test_indices = indices[test_start:test_end]

            # Training set: All data except test period
            train_indices = np.concatenate([
                indices[:test_start],
                indices[test_end:]
            ])

            # Purge: Remove training samples near test set (before test)
            if purge_size > 0 and test_start > 0:
                purge_start = max(0, test_start - purge_size)
                purge_mask = (train_indices < purge_start) | (train_indices >= test_start)
                train_indices = train_indices[purge_mask]

            # Embargo: Remove training samples after test set
            if embargo_size > 0 and test_end < n_samples:
                embargo_end = min(n_samples, test_end + embargo_size)
                embargo_mask = (train_indices < test_end) | (train_indices >= embargo_end)
                train_indices = train_indices[embargo_mask]

            yield train_indices, test_indices

    def get_n_splits(self) -> int:
        """Get number of splits."""
        return self.n_splits

    def __repr__(self) -> str:
        return (f"PurgedKFold(n_splits={self.n_splits}, "
                f"purge_pct={self.purge_pct}, embargo_pct={self.embargo_pct})")


class CombinatorialPurgedKFold(PurgedKFold):
    """
    More aggressive purging that removes ALL test periods from ALL training sets.

    Standard Purged K-Fold only purges each test period from its own training set.
    Combinatorial Purged K-Fold purges all test periods from all training sets,
    ensuring maximum independence between folds.

    This is more conservative and reduces training data, but provides stronger
    guarantees against information leakage.
    """

    def split(self, data: pd.DataFrame) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate combinatorially purged train/test splits.

        Args:
            data: DataFrame with DatetimeIndex

        Yields:
            (train_indices, test_indices) tuples
        """
        n_samples = len(data)
        indices = np.arange(n_samples)
        test_size = n_samples // self.n_splits

        # Pre-compute all test sets
        all_test_indices = []
        for i in range(self.n_splits):
            test_start = i * test_size
            test_end = test_start + test_size if i < self.n_splits - 1 else n_samples
            all_test_indices.append((test_start, test_end))

        # For each fold
        for i in range(self.n_splits):
            test_start, test_end = all_test_indices[i]
            test_indices = indices[test_start:test_end]

            # Start with all non-test indices
            train_mask = np.ones(n_samples, dtype=bool)
            train_mask[test_start:test_end] = False

            # Purge ALL test periods (not just current one)
            purge_size = int(test_size * self.purge_pct)
            embargo_size = int(test_size * self.embargo_pct)

            for other_test_start, other_test_end in all_test_indices:
                # Purge before test period
                if purge_size > 0:
                    purge_start = max(0, other_test_start - purge_size)
                    train_mask[purge_start:other_test_start] = False

                # Embargo after test period
                if embargo_size > 0:
                    embargo_end = min(n_samples, other_test_end + embargo_size)
                    train_mask[other_test_end:embargo_end] = False

            train_indices = indices[train_mask]

            yield train_indices, test_indices


def calculate_purge_embargo_sizes(n_samples: int, n_splits: int,
                                   purge_pct: float, embargo_pct: float) -> dict:
    """
    Calculate actual purge and embargo sizes for given configuration.

    Useful for understanding how much data will be removed.

    Args:
        n_samples: Total number of samples
        n_splits: Number of folds
        purge_pct: Purge percentage
        embargo_pct: Embargo percentage

    Returns:
        Dict with size information
    """
    test_size = n_samples // n_splits
    purge_size = int(test_size * purge_pct)
    embargo_size = int(test_size * embargo_pct)

    total_removed_per_fold = purge_size + embargo_size
    avg_train_size = n_samples - test_size - total_removed_per_fold

    return {
        'n_samples': n_samples,
        'n_splits': n_splits,
        'test_size': test_size,
        'purge_size': purge_size,
        'embargo_size': embargo_size,
        'total_removed_per_fold': total_removed_per_fold,
        'avg_train_size': avg_train_size,
        'train_pct': avg_train_size / n_samples,
        'test_pct': test_size / n_samples,
        'removed_pct': total_removed_per_fold / n_samples
    }


if __name__ == "__main__":
    # Example usage
    import pandas as pd

    # Create sample data
    dates = pd.date_range('2020-01-01', periods=1000, freq='D')
    data = pd.DataFrame({'value': np.random.randn(1000)}, index=dates)

    # Create Purged K-Fold
    pkf = PurgedKFold(n_splits=5, purge_pct=0.05, embargo_pct=0.01)

    # Show split sizes
    info = calculate_purge_embargo_sizes(
        n_samples=len(data),
        n_splits=5,
        purge_pct=0.05,
        embargo_pct=0.01
    )

    print("Purged K-Fold Configuration:")
    print(f"  Total samples: {info['n_samples']}")
    print(f"  Splits: {info['n_splits']}")
    print(f"  Test size: {info['test_size']} ({info['test_pct']:.1%})")
    print(f"  Purge size: {info['purge_size']}")
    print(f"  Embargo size: {info['embargo_size']}")
    print(f"  Avg train size: {info['avg_train_size']} ({info['train_pct']:.1%})")
    print(f"  Removed per fold: {info['total_removed_per_fold']} ({info['removed_pct']:.1%})")

    print("\nSplits:")
    for i, (train_idx, test_idx) in enumerate(pkf.split(data)):
        print(f"  Fold {i+1}: Train={len(train_idx)}, Test={len(test_idx)}")
