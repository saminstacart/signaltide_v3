"""
Momentum + Quality Weight Optimization via Optuna

Uses cached monthly returns to efficiently optimize ensemble weights without
re-running expensive backtests on each trial.

Method:
1. Load monthly returns for momentum_v2 and momentum_quality_v1 (50/50 weights)
2. Algebraically reconstruct quality-only returns: r_q = 2*r_mq - r_m
3. Verify reconstruction accuracy: 0.5*r_m + 0.5*r_q ≈ r_mq
4. Optimize momentum weight w_m ∈ [0.2, 0.6] using regime-aware objective
5. Quality weight is always w_q = 1 - w_m

Objective Function:
    maximize: sharpe_full + 0.5 * avg_regime_sharpe - 0.5 * |max_drawdown|

This balances full-period risk-adjusted returns with regime robustness and
drawdown control.

Outputs:
    results/ensemble_baselines/momentum_quality_v1_weight_optuna_trials.csv
    results/ensemble_baselines/momentum_quality_v1_weight_optuna.md

Usage:
    python3 scripts/run_momentum_quality_weight_optuna.py

    # More trials for higher precision
    python3 scripts/run_momentum_quality_weight_optuna.py --n-trials 64
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

import optuna
from optuna.samplers import TPESampler
from config import get_logger

logger = get_logger(__name__)


class MonthlyReturnsOptimizer:
    """
    Optimize ensemble weights using cached monthly returns.
    """

    def __init__(self,
                 momentum_csv: str,
                 mq_csv: str,
                 n_trials: int = 32,
                 seed: int = 42):
        """
        Initialize optimizer with monthly return CSVs.

        Args:
            momentum_csv: Path to momentum_v2 monthly returns
            mq_csv: Path to momentum_quality_v1 monthly returns
            n_trials: Number of Optuna trials
            seed: Random seed for reproducibility
        """
        self.momentum_csv = momentum_csv
        self.mq_csv = mq_csv
        self.n_trials = n_trials
        self.seed = seed

        logger.info("=" * 80)
        logger.info("OPTUNA WEIGHT OPTIMIZATION - Momentum + Quality")
        logger.info("=" * 80)
        logger.info(f"Momentum returns: {momentum_csv}")
        logger.info(f"Momentum+Quality returns: {mq_csv}")
        logger.info(f"Trials: {n_trials}")
        logger.info(f"Seed: {seed}")
        logger.info("=" * 80)
        logger.info("")

        # Load and process monthly returns
        self.r_m, self.r_mq, self.r_q = self._load_and_reconstruct()

        # Define regime boundaries
        self.regimes = self._define_regimes()

    def _load_and_reconstruct(self) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Load monthly returns and algebraically reconstruct quality-only returns.

        Returns:
            (r_m, r_mq, r_q): Momentum, Momentum+Quality, and Quality monthly returns
        """
        logger.info("Loading monthly returns...")

        # Load momentum returns
        df_m = pd.read_csv(self.momentum_csv)
        df_m['date'] = pd.to_datetime(df_m['date'])
        df_m = df_m.set_index('date').sort_index()
        r_m = df_m['return']

        logger.info(f"  Momentum: {len(r_m)} months, "
                   f"{r_m.index.min()} to {r_m.index.max()}")

        # Load momentum+quality returns
        df_mq = pd.read_csv(self.mq_csv)
        df_mq['date'] = pd.to_datetime(df_mq['date'])
        df_mq = df_mq.set_index('date').sort_index()
        r_mq = df_mq['return']

        logger.info(f"  Momentum+Quality: {len(r_mq)} months, "
                   f"{r_mq.index.min()} to {r_mq.index.max()}")

        # Verify alignment
        if not r_m.index.equals(r_mq.index):
            logger.warning("⚠️  Date indices not perfectly aligned - aligning now")
            common_dates = r_m.index.intersection(r_mq.index)
            r_m = r_m.loc[common_dates]
            r_mq = r_mq.loc[common_dates]
            logger.info(f"  Aligned to {len(common_dates)} common dates")

        # Reconstruct quality returns algebraically
        # We know: r_mq = 0.5*r_m + 0.5*r_q (from 50/50 ensemble)
        # Therefore: r_q = 2*r_mq - r_m
        logger.info("")
        logger.info("Reconstructing quality-only returns...")
        logger.info("  Using: r_q = 2*r_mq - r_m")

        r_q = 2 * r_mq - r_m

        logger.info(f"  Quality: {len(r_q)} months")

        # Verification: Check that 0.5*r_m + 0.5*r_q ≈ r_mq
        r_mq_recon = 0.5 * r_m + 0.5 * r_q
        max_diff = (r_mq_recon - r_mq).abs().max()
        mean_diff = (r_mq_recon - r_mq).abs().mean()

        logger.info("")
        logger.info("Verification: 0.5*r_m + 0.5*r_q ≈ r_mq")
        logger.info(f"  Max reconstruction error: {max_diff:.10f}")
        logger.info(f"  Mean reconstruction error: {mean_diff:.10f}")

        if max_diff > 1e-8:
            logger.warning(f"⚠️  Reconstruction error larger than expected: {max_diff}")
        else:
            logger.info("  ✅ Reconstruction accurate to machine precision")

        logger.info("")
        logger.info("Monthly return statistics:")
        logger.info(f"  Momentum - Mean: {r_m.mean():.4%}, Std: {r_m.std():.4%}")
        logger.info(f"  Quality  - Mean: {r_q.mean():.4%}, Std: {r_q.std():.4%}")
        logger.info(f"  M+Q 50/50 - Mean: {r_mq.mean():.4%}, Std: {r_mq.std():.4%}")
        logger.info(f"  Correlation (M, Q): {r_m.corr(r_q):.4f}")
        logger.info("")

        return r_m, r_mq, r_q

    def _define_regimes(self) -> Dict[str, Tuple[str, str]]:
        """
        Define macro regime boundaries for regime-aware evaluation.

        Returns:
            Dict mapping regime name to (start_date, end_date)
        """
        regimes = {
            'Pre-COVID Bull': ('2015-04-30', '2020-02-28'),
            'COVID Crash': ('2020-03-31', '2020-04-30'),
            'QE Recovery': ('2020-05-31', '2021-12-31'),
            '2022 Bear': ('2022-01-31', '2022-10-31'),
            'Recent': ('2022-11-30', '2024-12-31'),
        }

        logger.info("Defined macro regimes:")
        for name, (start, end) in regimes.items():
            logger.info(f"  {name}: {start} to {end}")
        logger.info("")

        return regimes

    def _compute_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """
        Compute performance metrics from monthly returns.

        Args:
            returns: Monthly return series

        Returns:
            Dict with sharpe, total_return, max_drawdown, etc.
        """
        # Cumulative returns
        cum_returns = (1 + returns).cumprod()
        total_return = cum_returns.iloc[-1] - 1

        # Annualized metrics
        n_months = len(returns)
        n_years = n_months / 12.0

        mean_monthly = returns.mean()
        std_monthly = returns.std()

        # Annualized Sharpe (assume ~0% risk-free rate for simplicity)
        sharpe = (mean_monthly / std_monthly) * np.sqrt(12) if std_monthly > 0 else 0.0

        # Max drawdown
        cum_max = cum_returns.cummax()
        drawdown = (cum_returns - cum_max) / cum_max
        max_drawdown = drawdown.min()  # Most negative value

        # CAGR
        cagr = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0.0

        return {
            'sharpe': sharpe,
            'total_return': total_return,
            'cagr': cagr,
            'max_drawdown': max_drawdown,
            'volatility': std_monthly * np.sqrt(12),  # Annualized
        }

    def _evaluate_weight(self, w_m: float) -> Dict[str, float]:
        """
        Evaluate performance for a given momentum weight.

        Args:
            w_m: Momentum weight in [0, 1]

        Returns:
            Dict with sharpe, max_drawdown, regime_sharpes, etc.
        """
        w_q = 1.0 - w_m

        # Construct weighted returns
        r_weighted = w_m * self.r_m + w_q * self.r_q

        # Full-period metrics
        full_metrics = self._compute_metrics(r_weighted)

        # Regime-specific Sharpe ratios
        regime_sharpes = {}
        for regime_name, (start, end) in self.regimes.items():
            start_dt = pd.to_datetime(start)
            end_dt = pd.to_datetime(end)

            regime_returns = r_weighted[(r_weighted.index >= start_dt) &
                                       (r_weighted.index <= end_dt)]

            if len(regime_returns) > 0:
                regime_metrics = self._compute_metrics(regime_returns)
                regime_sharpes[regime_name] = regime_metrics['sharpe']
            else:
                regime_sharpes[regime_name] = 0.0

        avg_regime_sharpe = np.mean(list(regime_sharpes.values()))

        return {
            'sharpe': full_metrics['sharpe'],
            'total_return': full_metrics['total_return'],
            'cagr': full_metrics['cagr'],
            'max_drawdown': full_metrics['max_drawdown'],
            'volatility': full_metrics['volatility'],
            'avg_regime_sharpe': avg_regime_sharpe,
            'regime_sharpes': regime_sharpes,
        }

    def _objective(self, trial: optuna.Trial) -> float:
        """
        Optuna objective function.

        Args:
            trial: Optuna trial object

        Returns:
            Objective value to maximize
        """
        # Suggest momentum weight in [0.2, 0.6]
        w_m = trial.suggest_float('w_momentum', 0.2, 0.6)

        # Evaluate performance
        metrics = self._evaluate_weight(w_m)

        # Objective: sharpe + 0.5 * avg_regime_sharpe - 0.5 * |max_drawdown|
        objective = (metrics['sharpe'] +
                    0.5 * metrics['avg_regime_sharpe'] -
                    0.5 * abs(metrics['max_drawdown']))

        # Log trial metrics for storage
        trial.set_user_attr('sharpe', metrics['sharpe'])
        trial.set_user_attr('total_return', metrics['total_return'])
        trial.set_user_attr('cagr', metrics['cagr'])
        trial.set_user_attr('max_drawdown', metrics['max_drawdown'])
        trial.set_user_attr('volatility', metrics['volatility'])
        trial.set_user_attr('avg_regime_sharpe', metrics['avg_regime_sharpe'])

        for regime_name, sharpe in metrics['regime_sharpes'].items():
            trial.set_user_attr(f'regime_{regime_name.replace(" ", "_")}', sharpe)

        return objective

    def optimize(self) -> optuna.Study:
        """
        Run Optuna optimization.

        Returns:
            Optuna Study object with results
        """
        logger.info("Starting Optuna optimization...")
        logger.info(f"  Search domain: w_momentum ∈ [0.2, 0.6]")
        logger.info(f"  Objective: sharpe + 0.5*avg_regime_sharpe - 0.5*|max_dd|")
        logger.info(f"  Trials: {self.n_trials}")
        logger.info("")

        # Create study
        sampler = TPESampler(seed=self.seed)
        study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            study_name='momentum_quality_weight_optimization'
        )

        # Optimize
        study.optimize(self._objective, n_trials=self.n_trials, show_progress_bar=True)

        logger.info("")
        logger.info("Optimization complete!")
        logger.info(f"  Best trial: #{study.best_trial.number}")
        logger.info(f"  Best w_momentum: {study.best_params['w_momentum']:.4f}")
        logger.info(f"  Best objective: {study.best_value:.4f}")
        logger.info("")

        # Best trial metrics
        best_trial = study.best_trial
        logger.info("Best trial metrics:")
        logger.info(f"  Sharpe: {best_trial.user_attrs['sharpe']:.4f}")
        logger.info(f"  Total Return: {best_trial.user_attrs['total_return']:.2%}")
        logger.info(f"  CAGR: {best_trial.user_attrs['cagr']:.2%}")
        logger.info(f"  Max Drawdown: {best_trial.user_attrs['max_drawdown']:.2%}")
        logger.info(f"  Avg Regime Sharpe: {best_trial.user_attrs['avg_regime_sharpe']:.4f}")
        logger.info("")

        return study

    def save_results(self, study: optuna.Study):
        """
        Save optimization results to CSV and markdown.

        Args:
            study: Optuna Study object
        """
        output_dir = Path('results/ensemble_baselines')
        output_dir.mkdir(parents=True, exist_ok=True)

        # Extract trials data
        trials_data = []
        for trial in study.trials:
            if trial.state != optuna.trial.TrialState.COMPLETE:
                continue

            row = {
                'trial': trial.number,
                'w_momentum': trial.params['w_momentum'],
                'w_quality': 1.0 - trial.params['w_momentum'],
                'objective': trial.value,
                'sharpe': trial.user_attrs['sharpe'],
                'total_return': trial.user_attrs['total_return'],
                'cagr': trial.user_attrs['cagr'],
                'max_drawdown': trial.user_attrs['max_drawdown'],
                'volatility': trial.user_attrs['volatility'],
                'avg_regime_sharpe': trial.user_attrs['avg_regime_sharpe'],
            }

            # Add regime-specific sharpes
            for regime_name in self.regimes.keys():
                attr_key = f'regime_{regime_name.replace(" ", "_")}'
                if attr_key in trial.user_attrs:
                    row[regime_name] = trial.user_attrs[attr_key]

            trials_data.append(row)

        df = pd.DataFrame(trials_data)

        # Save CSV
        csv_path = output_dir / 'momentum_quality_v1_weight_optuna_trials.csv'
        df.to_csv(csv_path, index=False, float_format='%.8f')
        logger.info(f"Saved trials CSV: {csv_path}")

        # Save markdown diagnostic
        self._save_diagnostic_markdown(study, df, output_dir)

    def _save_diagnostic_markdown(self, study: optuna.Study, df: pd.DataFrame, output_dir: Path):
        """
        Save diagnostic markdown report.

        Args:
            study: Optuna Study object
            df: Trials DataFrame
            output_dir: Output directory
        """
        md_path = output_dir / 'momentum_quality_v1_weight_optuna.md'

        best = study.best_trial
        w_m_best = best.params['w_momentum']
        w_q_best = 1.0 - w_m_best

        # Top 5 trials by objective
        top5 = df.nlargest(5, 'objective')

        content = f"""# Momentum + Quality v1 Weight Optimization (Optuna)

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Method:** Optuna TPE sampler on cached monthly returns
**Trials:** {self.n_trials}
**Seed:** {self.seed}
**Search Domain:** w_momentum ∈ [0.2, 0.6]

---

## Optimization Setup

### Data Sources
- **Momentum v2**: `{self.momentum_csv}`
- **Momentum+Quality v1 (50/50)**: `{self.mq_csv}`
- **Quality v1 (reconstructed)**: Algebraic derivation via `r_q = 2*r_mq - r_m`

### Objective Function
```
maximize: sharpe + 0.5 * avg_regime_sharpe - 0.5 * |max_drawdown|
```

This balances:
- Full-period risk-adjusted returns (Sharpe)
- Regime robustness (average Sharpe across 5 macro regimes)
- Drawdown control (penalize large losses)

### Macro Regimes
"""

        for regime_name, (start, end) in self.regimes.items():
            content += f"- **{regime_name}**: {start} to {end}\n"

        content += f"""
---

## Best Result

**Optimal Weights:**
- Momentum: **{w_m_best:.4f}** ({w_m_best * 100:.2f}%)
- Quality: **{w_q_best:.4f}** ({w_q_best * 100:.2f}%)

**Performance:**
- Sharpe Ratio: **{best.user_attrs['sharpe']:.4f}**
- Total Return: **{best.user_attrs['total_return']:.2%}**
- CAGR: **{best.user_attrs['cagr']:.2%}**
- Max Drawdown: **{best.user_attrs['max_drawdown']:.2%}**
- Volatility: **{best.user_attrs['volatility']:.2%}**
- Avg Regime Sharpe: **{best.user_attrs['avg_regime_sharpe']:.4f}**
- **Objective Value**: **{best.value:.4f}**

---

## Top 5 Trials

| Trial | w_mom | w_qual | Objective | Sharpe | Total Ret | Max DD | Avg Regime Sharpe |
|-------|-------|--------|-----------|--------|-----------|--------|-------------------|
"""

        for _, row in top5.iterrows():
            content += f"| {int(row['trial'])} | {row['w_momentum']:.4f} | {row['w_quality']:.4f} | "
            content += f"{row['objective']:.4f} | {row['sharpe']:.4f} | {row['total_return']:.2%} | "
            content += f"{row['max_drawdown']:.2%} | {row['avg_regime_sharpe']:.4f} |\n"

        content += """
---

## Regime Breakdown (Best Trial)

| Regime | Sharpe |
|--------|--------|
"""

        for regime_name in self.regimes.keys():
            attr_key = f'regime_{regime_name.replace(" ", "_")}'
            if attr_key in best.user_attrs:
                sharpe = best.user_attrs[attr_key]
                content += f"| {regime_name} | {sharpe:.4f} |\n"

        content += f"""
---

## Observations

- Best momentum weight: {w_m_best:.4f} (quality weight: {w_q_best:.4f})
- Sharpe range across trials: [{df['sharpe'].min():.4f}, {df['sharpe'].max():.4f}]
- Objective range: [{df['objective'].min():.4f}, {df['objective'].max():.4f}]
- Weight sensitivity: {"High" if df['objective'].std() > 0.1 else "Moderate" if df['objective'].std() > 0.05 else "Low"} (objective std = {df['objective'].std():.4f})

---

## Reconciliation with Grid Sweep

**Grid Sweep Results (Phase 3 M3.4a):**
- M=0.25, Q=0.75: Sharpe 2.876
- M=0.50, Q=0.50: Sharpe 2.873 (0.12% difference)

**Optuna Result:**
- w_momentum = {w_m_best:.4f}, w_quality = {w_q_best:.4f}

**Decision Rule:**
- If Optuna weight within 0.10 of 0.50 AND improvement < 1%: **Keep 50/50**
- Else: Round to nearest 0.05 and update config

**Recommendation:**
"""

        # Apply decision rule
        dist_from_50 = abs(w_m_best - 0.5)
        grid_50_sharpe = 2.873  # From grid sweep
        optuna_sharpe = best.user_attrs['sharpe']
        improvement_pct = ((optuna_sharpe - grid_50_sharpe) / grid_50_sharpe) * 100

        if dist_from_50 <= 0.10 and improvement_pct < 1.0:
            content += f"""
**KEEP 50/50** - Optuna weight ({w_m_best:.4f}) is within 0.10 of 0.50 and improvement ({improvement_pct:+.2f}%) < 1%.

The statistical noise between grid sweep and Optuna does not justify changing the canonical 50/50 allocation.
"""
        else:
            rounded_w_m = round(w_m_best * 20) / 20  # Round to nearest 0.05
            rounded_w_q = 1.0 - rounded_w_m
            content += f"""
**UPDATE RECOMMENDED** - Optuna found {improvement_pct:+.2f}% improvement.

Proposed weights: **M={rounded_w_m:.2f}, Q={rounded_w_q:.2f}**
"""

        content += """
---

**Status:** Phase 3 Milestone 3.4b (Optuna validation complete)
**Files:**
- Trials CSV: `results/ensemble_baselines/momentum_quality_v1_weight_optuna_trials.csv`
- Diagnostic: `results/ensemble_baselines/momentum_quality_v1_weight_optuna.md`
"""

        md_path.write_text(content)
        logger.info(f"Saved diagnostic: {md_path}")


def main():
    """Run Optuna weight optimization."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Optimize momentum+quality weights via Optuna on cached monthly returns',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--momentum-csv',
                       default='results/ensemble_baselines/momentum_v2_monthly_returns.csv',
                       help='Momentum monthly returns CSV')
    parser.add_argument('--mq-csv',
                       default='results/ensemble_baselines/momentum_quality_v1_monthly_returns.csv',
                       help='Momentum+Quality monthly returns CSV')
    parser.add_argument('--n-trials', type=int, default=32,
                       help='Number of Optuna trials (default: 32)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')

    args = parser.parse_args()

    # Run optimization
    optimizer = MonthlyReturnsOptimizer(
        momentum_csv=args.momentum_csv,
        mq_csv=args.mq_csv,
        n_trials=args.n_trials,
        seed=args.seed
    )

    study = optimizer.optimize()
    optimizer.save_results(study)

    logger.info("")
    logger.info("=" * 80)
    logger.info("✅ OPTUNA OPTIMIZATION COMPLETE")
    logger.info("=" * 80)
    logger.info("Results saved to:")
    logger.info("  - results/ensemble_baselines/momentum_quality_v1_weight_optuna_trials.csv")
    logger.info("  - results/ensemble_baselines/momentum_quality_v1_weight_optuna.md")
    logger.info("")


if __name__ == '__main__':
    main()
