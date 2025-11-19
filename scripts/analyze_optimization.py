"""
Analyze Optuna Optimization Results

Loads completed optimization studies and generates analysis:
- Optimization history plots
- Parameter importance
- Top trial analysis
- Comprehensive markdown report

Usage:
    python scripts/analyze_optimization.py
"""

import sys
sys.path.insert(0, '/Users/samuelksherman/signaltide_v3')

import optuna
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Try to import matplotlib, but make it optional
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style('whitegrid')
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("⚠️  matplotlib not available - skipping plots")


class OptimizationAnalyzer:
    """Analyze Optuna optimization results."""

    def __init__(self, db_path: str = 'results/optimization/optuna_studies.db'):
        """
        Initialize analyzer.

        Args:
            db_path: Path to Optuna database
        """
        self.db_path = db_path
        self.storage = f'sqlite:///{db_path}'
        self.output_dir = Path('results/optimization')
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def list_studies(self) -> list:
        """List all available studies in database."""
        summaries = optuna.study.get_all_study_summaries(storage=self.storage)
        return [s.study_name for s in summaries]

    def load_study(self, study_name: str):
        """Load a study from database."""
        try:
            study = optuna.load_study(
                study_name=study_name,
                storage=self.storage
            )
            return study
        except KeyError:
            print(f"❌ Study '{study_name}' not found")
            return None

    def analyze_study(self, study_name: str) -> dict:
        """
        Analyze a single study.

        Returns:
            Dict with analysis results
        """
        study = self.load_study(study_name)
        if study is None:
            return None

        print(f"\n{'='*80}")
        print(f"Analyzing: {study_name}")
        print(f"{'='*80}")

        # Basic stats
        n_trials = len(study.trials)
        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        failed = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]

        print(f"Total trials: {n_trials}")
        print(f"Completed: {len(completed)}")
        print(f"Failed: {len(failed)}")

        if len(completed) == 0:
            print("❌ No completed trials found")
            return None

        # Best trial
        best_trial = study.best_trial
        print(f"\nBest Trial: #{best_trial.number}")
        print(f"Best Sharpe: {best_trial.value:.4f}")
        print(f"\nBest Parameters:")
        for key, value in best_trial.params.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

        # Collect data for analysis
        trial_numbers = [t.number for t in completed]
        values = [t.value for t in completed]

        # Get parameter values for each trial
        param_names = list(best_trial.params.keys())
        param_data = {name: [] for name in param_names}
        for trial in completed:
            for name in param_names:
                param_data[name].append(trial.params.get(name, np.nan))

        # Create DataFrame
        df = pd.DataFrame({
            'trial': trial_numbers,
            'sharpe': values,
            **param_data
        })

        # Statistics
        print(f"\nSharpe Statistics:")
        print(f"  Mean: {df['sharpe'].mean():.4f}")
        print(f"  Std: {df['sharpe'].std():.4f}")
        print(f"  Min: {df['sharpe'].min():.4f}")
        print(f"  Max: {df['sharpe'].max():.4f}")
        print(f"  Median: {df['sharpe'].median():.4f}")

        # Top 10 trials
        top_10 = df.nlargest(10, 'sharpe')
        print(f"\nTop 10 Trials:")
        print(top_10[['trial', 'sharpe']].to_string(index=False))

        # Parameter ranges in top trials
        print(f"\nParameter Ranges in Top 10 Trials:")
        for param in param_names:
            top_vals = top_10[param]
            print(f"  {param}: [{top_vals.min():.2f}, {top_vals.max():.2f}] (mean: {top_vals.mean():.2f})")

        return {
            'study': study,
            'study_name': study_name,
            'df': df,
            'best_trial': best_trial,
            'n_trials': n_trials,
            'n_completed': len(completed),
            'n_failed': len(failed),
            'top_10': top_10,
            'param_names': param_names
        }

    def plot_optimization_history(self, analysis: dict, output_file: str = None):
        """Plot optimization history over trials."""
        if not HAS_MATPLOTLIB:
            return

        df = analysis['df']
        study_name = analysis['study_name']

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Plot 1: All trials
        ax1.scatter(df['trial'], df['sharpe'], alpha=0.6, s=30)
        ax1.axhline(y=analysis['best_trial'].value, color='r', linestyle='--',
                   label=f'Best: {analysis["best_trial"].value:.4f}')
        ax1.set_xlabel('Trial Number')
        ax1.set_ylabel('Sharpe Ratio')
        ax1.set_title(f'{study_name} - Optimization History')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Best-so-far
        best_so_far = df['sharpe'].cummax()
        ax2.plot(df['trial'], best_so_far, linewidth=2, color='green')
        ax2.set_xlabel('Trial Number')
        ax2.set_ylabel('Best Sharpe So Far')
        ax2.set_title('Best Performance Over Time')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if output_file is None:
            signal_type = study_name.split('_')[0]
            output_file = self.output_dir / f'{signal_type}_optimization_history.png'

        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\n✓ Plot saved: {output_file}")
        plt.close()

    def plot_parameter_distributions(self, analysis: dict, output_file: str = None):
        """Plot parameter distributions for all trials vs top 10."""
        if not HAS_MATPLOTLIB:
            return

        df = analysis['df']
        top_10 = analysis['top_10']
        param_names = analysis['param_names']
        study_name = analysis['study_name']

        n_params = len(param_names)
        fig, axes = plt.subplots(n_params, 1, figsize=(12, 4 * n_params))
        if n_params == 1:
            axes = [axes]

        for i, param in enumerate(param_names):
            # All trials
            axes[i].hist(df[param], bins=30, alpha=0.5, label='All Trials', color='blue')
            # Top 10 trials
            axes[i].hist(top_10[param], bins=15, alpha=0.7, label='Top 10', color='green')
            # Best trial
            axes[i].axvline(x=analysis['best_trial'].params[param], color='red',
                           linestyle='--', linewidth=2, label='Best')
            axes[i].set_xlabel(param)
            axes[i].set_ylabel('Count')
            axes[i].set_title(f'{param} Distribution')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)

        plt.suptitle(f'{study_name} - Parameter Distributions', fontsize=14, y=1.001)
        plt.tight_layout()

        if output_file is None:
            signal_type = study_name.split('_')[0]
            output_file = self.output_dir / f'{signal_type}_parameter_distributions.png'

        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✓ Plot saved: {output_file}")
        plt.close()

    def plot_parameter_importance(self, analysis: dict, output_file: str = None):
        """Plot parameter correlations with Sharpe ratio."""
        if not HAS_MATPLOTLIB:
            return

        df = analysis['df']
        param_names = analysis['param_names']
        study_name = analysis['study_name']

        # Calculate correlations
        correlations = {}
        for param in param_names:
            corr = df[[param, 'sharpe']].corr().iloc[0, 1]
            correlations[param] = corr

        # Sort by absolute correlation
        sorted_params = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

        fig, ax = plt.subplots(figsize=(10, 6))

        params = [p[0] for p in sorted_params]
        corrs = [p[1] for p in sorted_params]
        colors = ['green' if c > 0 else 'red' for c in corrs]

        ax.barh(params, corrs, color=colors, alpha=0.7)
        ax.set_xlabel('Correlation with Sharpe Ratio')
        ax.set_title(f'{study_name} - Parameter Importance')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()

        if output_file is None:
            signal_type = study_name.split('_')[0]
            output_file = self.output_dir / f'{signal_type}_parameter_importance.png'

        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✓ Plot saved: {output_file}")
        plt.close()

        return correlations

    def generate_report(self, analyses: dict, output_file: str = None):
        """
        Generate comprehensive markdown report.

        Args:
            analyses: Dict mapping signal_type -> analysis dict
            output_file: Output markdown file path
        """
        if output_file is None:
            output_file = self.output_dir / 'optimization_report.md'

        with open(output_file, 'w') as f:
            f.write("# SignalTide v3 - Optimization Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("---\n\n")

            f.write("## Executive Summary\n\n")

            # Summary table
            f.write("| Signal | Trials | Best Sharpe | Improvement | Status |\n")
            f.write("|--------|--------|-------------|-------------|--------|\n")

            baseline_sharpes = {
                'momentum': -0.084,
                'quality': 0.000,
                'insider': 0.049
            }

            for signal_type, analysis in analyses.items():
                if analysis is None:
                    continue

                signal_name = signal_type.split('_')[0]
                baseline = baseline_sharpes.get(signal_name, 0.0)
                best_sharpe = analysis['best_trial'].value
                improvement = best_sharpe - baseline

                status = "✅ Complete" if analysis['n_completed'] > 100 else "⚠️ Incomplete"

                f.write(f"| {signal_name.capitalize()} | {analysis['n_completed']} | "
                       f"{best_sharpe:.4f} | +{improvement:.4f} | {status} |\n")

            f.write("\n---\n\n")

            # Detailed results for each signal
            for signal_type, analysis in analyses.items():
                if analysis is None:
                    continue

                signal_name = signal_type.split('_')[0]
                f.write(f"## {signal_name.capitalize()} Signal\n\n")

                f.write(f"**Study:** {analysis['study_name']}\n\n")
                f.write(f"**Trials:**\n")
                f.write(f"- Total: {analysis['n_trials']}\n")
                f.write(f"- Completed: {analysis['n_completed']}\n")
                f.write(f"- Failed: {analysis['n_failed']}\n\n")

                f.write(f"**Best Result:**\n")
                f.write(f"- Trial #{analysis['best_trial'].number}\n")
                f.write(f"- Sharpe Ratio: **{analysis['best_trial'].value:.4f}**\n\n")

                f.write(f"**Best Parameters:**\n")
                for key, value in analysis['best_trial'].params.items():
                    if isinstance(value, float):
                        f.write(f"- `{key}`: {value:.4f}\n")
                    else:
                        f.write(f"- `{key}`: {value}\n")

                f.write("\n**Top 10 Trials:**\n\n")
                f.write("| Trial | Sharpe |\n")
                f.write("|-------|--------|\n")
                for _, row in analysis['top_10'].iterrows():
                    f.write(f"| {int(row['trial'])} | {row['sharpe']:.4f} |\n")

                f.write(f"\n**Parameter Ranges (Top 10):**\n\n")
                for param in analysis['param_names']:
                    top_vals = analysis['top_10'][param]
                    f.write(f"- `{param}`: [{top_vals.min():.2f}, {top_vals.max():.2f}] "
                           f"(mean: {top_vals.mean():.2f})\n")

                # Add plots if available
                if HAS_MATPLOTLIB:
                    f.write(f"\n**Visualizations:**\n\n")
                    f.write(f"- ![Optimization History]({signal_name}_optimization_history.png)\n")
                    f.write(f"- ![Parameter Distributions]({signal_name}_parameter_distributions.png)\n")
                    f.write(f"- ![Parameter Importance]({signal_name}_parameter_importance.png)\n")

                f.write("\n---\n\n")

            # Next steps
            f.write("## Next Steps\n\n")
            f.write("1. **Apply Best Parameters:** Update signal configs with optimized values\n")
            f.write("2. **Backtest with Best Params:** Run full backtest on 10-ticker universe\n")
            f.write("3. **Monte Carlo Validation:** Test statistical significance\n")
            f.write("4. **Ensemble Optimization:** Optimize signal combination weights\n")
            f.write("5. **Out-of-Sample Testing:** Validate on 2024 data\n\n")

            f.write("---\n\n")
            f.write(f"**Database:** `{self.db_path}`\n")
            f.write(f"**Analysis Script:** `scripts/analyze_optimization.py`\n")

        print(f"\n✓ Report saved: {output_file}")
        return output_file


def main():
    """Main entry point."""
    analyzer = OptimizationAnalyzer()

    print("="*80)
    print("SignalTide v3 - Optimization Analysis")
    print("="*80)

    # List available studies
    study_names = analyzer.list_studies()
    print(f"\nFound {len(study_names)} studies:")
    for name in study_names:
        print(f"  - {name}")

    # Analyze each study
    analyses = {}
    for study_name in study_names:
        analysis = analyzer.analyze_study(study_name)
        if analysis is not None:
            analyses[study_name] = analysis

            # Generate plots
            if HAS_MATPLOTLIB:
                analyzer.plot_optimization_history(analysis)
                analyzer.plot_parameter_distributions(analysis)
                analyzer.plot_parameter_importance(analysis)

    # Generate comprehensive report
    if analyses:
        print(f"\n{'='*80}")
        print("Generating Comprehensive Report")
        print("="*80)
        report_file = analyzer.generate_report(analyses)
        print(f"\n✓ Analysis complete!")
        print(f"\nResults:")
        print(f"  - Report: {report_file}")
        print(f"  - Plots: results/optimization/*.png")
        print(f"  - Database: {analyzer.db_path}")
    else:
        print("\n❌ No completed studies found to analyze")


if __name__ == '__main__':
    main()
