"""
Academic parameters with citations for SignalTide v4.

Each parameter is linked to its source paper for reproducibility.
"""

from typing import Dict, Any

ACADEMIC_PARAMS: Dict[str, Dict[str, Any]] = {
    # Residual Momentum
    'residual_momentum': {
        'citation': 'Blitz, D., Huij, J., & Martens, M. (2011). "Residual Momentum". Journal of Empirical Finance, 18(3), 506-521.',
        'lookback_days': 252,
        'skip_days': 21,
        'ff3_regression_window': 252,
        'description': 'Momentum computed on FF3-adjusted returns, crash-resistant',
    },

    # Cash-Based Operating Profitability
    'cash_based_quality': {
        'citation': 'Ball, R., Gerakos, J., Linnainmaa, J. T., & Nikolaev, V. (2016). "Accruals, Cash Flows, and Operating Profitability in the Cross Section of Stock Returns". Journal of Financial Economics, 121(1), 28-45.',
        'cbop_formula': '(OCF - Delta_AR - Delta_Inv + Delta_AP) / Total_Assets',
        'lookback_quarters': 4,
        'description': 'Cash-based operating profitability (CbOP) superior to accrual-based',
    },

    # Betting Against Beta
    'bab': {
        'citation': 'Frazzini, A., & Pedersen, L. H. (2014). "Betting Against Beta". Journal of Financial Economics, 111(1), 1-25.',
        'low_beta_threshold': 1.0,
        'description': 'Low-beta stocks outperform high-beta on risk-adjusted basis',
    },

    # Buyback Yield
    'buyback_yield': {
        'citation': 'Grullon, G., & Michaely, R. (2004). "The Information Content of Share Repurchase Programs". Journal of Finance, 59(2), 651-680.',
        'lookback_quarters': 4,
        'description': 'Companies buying back shares signal undervaluation',
    },

    # Asset Growth Screen
    'asset_growth': {
        'citation': 'Cooper, M. J., Gulen, H., & Schill, M. J. (2008). "Asset Growth and the Cross-Section of Stock Returns". Journal of Finance, 63(4), 1609-1651.',
        'max_growth_rate': 0.30,
        'description': 'Negative relationship between asset growth and returns',
    },

    # Opportunistic Insider Trading
    'opportunistic_insider': {
        'citation': 'Cohen, L., Malloy, C., & Pomorski, L. (2012). "Decoding Inside Information". Journal of Finance, 67(3), 1009-1043.',
        'routine_trade_threshold': 3,  # Trades in 3+ consecutive years = routine
        'lookback_years': 3,
        'description': 'Opportunistic trades (non-routine) have 8.6% annual alpha',
    },

    # Semantic Tone Analysis
    'semantic_tone': {
        'citation': 'Tetlock, P. C., Saar-Tsechansky, M., & Macskassy, S. (2008). "More Than Words: Quantifying Language to Measure Firms\' Fundamentals". Journal of Finance, 63(3), 1437-1467.',
        'tone_window_days': 30,
        'description': 'Negative tone in SEC filings predicts lower earnings and returns',
    },

    # Deflated Sharpe Ratio
    'deflated_sharpe': {
        'citation': 'Bailey, D. H., & López de Prado, M. (2014). "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality". Journal of Portfolio Management, 40(5), 94-107.',
        'min_confidence': 0.95,
        'description': 'Corrects for multiple testing and strategy selection bias',
    },

    # Fama-French 5 Factor Model
    'ff5': {
        'citation': 'Fama, E. F., & French, K. R. (2015). "A Five-Factor Asset Pricing Model". Journal of Financial Economics, 116(1), 1-22.',
        'factors': ['MKT-RF', 'SMB', 'HML', 'RMW', 'CMA'],
        'description': 'Market, Size, Value, Profitability, Investment factors',
    },

    # Walk-Forward Validation
    'walk_forward': {
        'citation': 'Pardo, R. (2008). "The Evaluation and Optimization of Trading Strategies". Wiley Trading.',
        'train_months': 60,
        'test_months': 12,
        'min_folds': 5,
        'description': 'Rolling OOS validation for time series, not k-fold CV',
    },
}
