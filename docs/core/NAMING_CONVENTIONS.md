# SignalTide v3 - Master Naming Conventions

**Version:** 1.0.0
**Last Updated:** 2025-11-20
**Status:** SOURCE OF TRUTH - All code must follow these conventions

---

## Philosophy

**Consistency > Personal Preference**

This document defines the ONE correct way to name everything in SignalTide v3. When in doubt, refer here. If something isn't covered, add it here FIRST, then use it everywhere.

---

## General Principles

1. **Use institutional quant terminology** - Match academic papers and professional practice
2. **snake_case for Python** - Functions, variables, files (PEP 8)
3. **PascalCase for classes** - Class names only
4. **SCREAMING_SNAKE_CASE for constants** - Module-level constants
5. **Explicit over implicit** - `as_of_date` not `as_of`, `filing_date` not `date`
6. **American spelling** - "optimization" not "optimisation"
7. **Full words over abbreviations** - Except standard quant terms (see below)

---

## Temporal Variables (CRITICAL FOR DATA INTEGRITY)

**The Rule:** Always use the MOST SPECIFIC name possible.

### Date Variables

| Concept | Variable Name | Type | Example |
|---------|---------------|------|---------|
| Signal generation date | `as_of_date` | str/datetime | `'2023-03-15'` |
| Quarter end date | `calendar_date` | datetime | `2023-03-31` |
| SEC filing date | `filing_date` | datetime | `2023-05-05` |
| Transaction date (insider) | `transaction_date` | datetime | `2023-03-13` |
| Report period end | `report_period` | datetime | `2023-03-31` |
| Start of backtest | `start_date` | str | `'2020-01-01'` |
| End of backtest | `end_date` | str | `'2024-12-31'` |
| Date data was updated | `last_updated` | datetime | `2025-08-01` |

**Database Columns (Sharadar convention):**
- `calendardate` - Quarter/annual period end
- `datekey` - Filing/availability date (**USE THIS for as_of filtering**)
- `filingdate` - SEC filing date (insider data)
- `reportperiod` - Period covered by report
- `lastupdated` - Database refresh date

**Database Column Naming (SignalTide Tables):**

For **new** SignalTide dimension and fact tables, use explicit column names:
- `calendar_date` - For rows in trading calendar or any calendar dimension
- `trading_date` - For fact tables representing actual trading days
- `filing_date` - SEC filing dates
- `report_period` - Financial reporting period end
- `as_of_date` - Point-in-time reference date
- `membership_start_date` / `membership_end_date` - Universe membership boundaries
- **NEVER** use bare `date` as a column name in new tables

**Context:**
- Python variables follow the guidance above (always use `_date` suffix)
- Sharadar vendor tables use their own convention (`calendardate`, `datekey`) - respect as-is
- SignalTide dimension/fact tables must use explicit, underscored names

### NEVER Use These Ambiguous Names:
- ❌ `date` - Too generic (both code and new database columns)
- ❌ `as_of` - Missing "date"
- ❌ `filing` - Missing "date"
- ❌ `t` or `T` - Not explicit enough

### Time Period Variables

| Concept | Variable Name | Unit | Example |
|---------|---------------|------|---------|
| Formation period | `formation_period_days` | int | `252` |
| Lookback window | `lookback_days` | int | `90` |
| Skip period (momentum) | `skip_period_days` | int | `21` |
| Rolling window | `window_days` | int | `504` |
| Rebalancing frequency | `rebalance_frequency` | str | `'monthly'` |

**Standard Periods:**
- 1 month = `21` trading days
- 1 quarter = `63` trading days
- 1 year = `252` trading days
- 2 years = `504` trading days

---

## Signal Variables

### Signal Values

| Concept | Variable Name | Range | Example |
|---------|---------------|-------|---------|
| Raw signal | `signal_raw` | float | `0.73` |
| Normalized signal | `signal` | [-1, 1] | `0.5` |
| Quintile signal | `signal_quintile` | {-1, -0.5, 0, 0.5, 1} | `1.0` |
| Z-score | `z_score` | float | `1.96` |
| Percentile rank | `percentile_rank` | [0, 1] | `0.85` |

### Signal Components

| Concept | Variable Name | Example |
|---------|---------------|---------|
| Momentum return | `momentum_return` | `0.25` |
| Profitability score | `profitability_score` | `0.82` |
| Growth score | `growth_score` | `0.15` |
| Safety score | `safety_score` | `0.91` |
| Insider activity | `insider_activity_score` | `2.3` |
| Weighted score | `weighted_score` | `1.47` |

---

## Financial Metrics

### Returns

| Concept | Variable Name | Format | Example |
|---------|---------------|--------|---------|
| Simple return | `return_simple` | decimal | `0.10` (10%) |
| Log return | `return_log` | decimal | `0.0953` |
| Total return | `total_return` | decimal | `2.5398` (253.98%) |
| Annual return | `annual_return` | decimal | `0.2882` (28.82%) |
| Excess return | `excess_return` | decimal | `0.05` |
| Risk-free rate | `risk_free_rate` | decimal | `0.02` (2%) |

### Risk Metrics

| Concept | Variable Name | Format | Example |
|---------|---------------|--------|---------|
| Volatility (annualized) | `volatility_annual` | decimal | `0.3543` (35.43%) |
| Downside deviation | `downside_deviation` | decimal | `0.25` |
| Maximum drawdown | `max_drawdown` | decimal (negative) | `-0.2872` (-28.72%) |
| Sharpe ratio | `sharpe_ratio` | float | `0.757` |
| Sortino ratio | `sortino_ratio` | float | `0.975` |
| Information ratio | `information_ratio` | float | `0.426` |
| Calmar ratio | `calmar_ratio` | float | `1.20` |

**CRITICAL:** Drawdowns are ALWAYS negative. Less negative = better.
```python
# CORRECT
max_drawdown = -0.2872  # -28.72%
assert max_drawdown < 0, "Drawdowns must be negative"

# WRONG
max_drawdown = 0.2872  # Confusing!
```

### Performance Metrics

| Concept | Variable Name | Format |
|---------|---------------|--------|
| Alpha (annualized) | `alpha_annual` | decimal |
| Beta | `beta` | float |
| R-squared | `r_squared` | [0, 1] |
| Tracking error | `tracking_error` | decimal |
| Win rate | `win_rate` | [0, 1] |
| Profit factor | `profit_factor` | float |

---

## Transaction Costs

| Concept | Variable Name | Unit | Example |
|---------|---------------|------|---------|
| Commission | `commission_bps` | basis points | `0` (Schwab) |
| Slippage | `slippage_bps` | basis points | `2` |
| Spread | `spread_bps` | basis points | `3` |
| Total cost | `total_cost_bps` | basis points | `5` |
| Commission % | `commission_pct` | decimal | `0.0000` |
| Slippage % | `slippage_pct` | decimal | `0.0002` |
| Total cost % | `total_cost_pct` | decimal | `0.0005` (5 bps) |

**Standard:** Use `_bps` suffix for basis points, `_pct` for decimal percentages.

**Conversions:**
```python
# 5 basis points
cost_bps = 5
cost_pct = cost_bps / 10000  # 0.0005
cost_percent = cost_pct * 100  # 0.05%
```

---

## Database & Data

### Table Names (Sharadar Convention)

| Table | Purpose |
|-------|---------|
| `sharadar_prices` | Daily OHLCV data |
| `sharadar_sf1` | Fundamentals (quarterly/annual) |
| `sharadar_insiders` | Insider transactions |
| `sharadar_tickers` | Ticker metadata |
| `sharadar_events` | Corporate events |

### Column Naming

**Price Data:**
- `open`, `high`, `low`, `close` - OHLC prices
- `volume` - Trading volume
- `adj_close` - Split-adjusted close (if separate)

**Fundamental Data:**
- `roe` - Return on equity
- `roa` - Return on assets
- `revenue` - Total revenue
- `netinc` - Net income
- `assets` - Total assets
- `debt` - Total debt
- `de` - Debt-to-equity ratio
- `gp` - Gross profit

**Insider Data:**
- `transactioncode` - P (purchase) or S (sale)
- `transactionshares` - Number of shares
- `transactionpricepershare` - Price per share
- `transactionvalue` - Dollar value
- `officertitle` - Insider's title

### Database Object Naming

**Table Prefixes:**

SignalTide uses a three-tier data architecture with clear naming conventions:

| Prefix | Tier | Purpose | Examples |
|--------|------|---------|----------|
| `sharadar_*` | Tier 1: Vendor | Raw Sharadar data (immutable) | `sharadar_prices`, `sharadar_sf1`, `sharadar_sp500` |
| `dim_*` | Tier 2: Warehouse | Dimension tables | `dim_trading_calendar`, `dim_universe_membership` |
| `fact_*` | Tier 2: Warehouse | Fact/panel tables | `fact_signals_panel`, `fact_positions` |
| `backtest_*` or `run_*` | Tier 3: Research | Backtest runs and experiments | `backtest_results`, `run_metadata` |

**Column Naming Rules:**

For **dimension and fact tables**:
- Date columns **must** end in `_date` (e.g., `calendar_date`, `trading_date`, `membership_start_date`)
- Universe identifiers use `universe_name` (not `universe_id` or `universe`)
- Use `membership_start_date` and `membership_end_date` for slowly changing dimensions
- Boolean flags use `is_*` prefix (e.g., `is_trading_day`, `is_current_member`)
- Timestamp columns use `*_at` suffix (e.g., `created_at`, `updated_at`, `added_at`)

**Foreign Keys and Constraints:**

- Always enable `PRAGMA foreign_keys = ON;` in SQLite connections
- Use `WITHOUT ROWID` for tables with composite primary keys
- Add indexes on frequently queried date ranges and lookup columns

**Example Dimension Table:**

```sql
CREATE TABLE dim_universe_membership (
    universe_name TEXT NOT NULL,           -- Not 'universe_id'
    ticker TEXT NOT NULL,
    membership_start_date DATE NOT NULL,   -- Not 'start_date'
    membership_end_date DATE,              -- NULL = still active
    source TEXT NOT NULL,
    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (universe_name, ticker, membership_start_date)
) WITHOUT ROWID;
```

---

## Function & Method Names

### Signal Methods

| Method | Purpose | Returns |
|--------|---------|---------|
| `generate_signals()` | Generate signal values | `pd.Series` [-1, 1] |
| `calculate_momentum()` | Calculate momentum metric | `pd.Series` (raw) |
| `calculate_profitability()` | Calculate profitability score | `pd.Series` |
| `winsorize()` | Winsorize outliers | `pd.Series` |
| `to_quintiles()` | Convert to quintile signals | `pd.Series` |
| `apply_monthly_rebalancing()` | Monthly rebalance logic | `pd.Series` |

**NEVER:**
- ❌ `calculate()` - Too generic
- ❌ `generate()` - Missing "signals"
- ❌ `get_signal()` - Use `generate_signals()`
- ❌ `compute()` - Use `calculate_`

### Data Access Methods

| Method | Purpose | Required Params |
|--------|---------|-----------------|
| `get_fundamentals()` | Fetch fundamental data | `ticker, start_date, end_date, dimension, as_of_date` |
| `get_insider_trades()` | Fetch insider data | `ticker, start_date, end_date, as_of_date` |
| `get_prices()` | Fetch price data | `ticker, start_date, end_date` |
| `get_tickers()` | Get ticker list | `category, is_delisted` |

**Point-in-Time Rule:** ALL data methods MUST have `as_of_date` parameter.

### Validation Methods

| Method | Purpose |
|--------|---------|
| `validate_no_lookahead()` | Check for lookahead bias |
| `validate_universe_timeline()` | Verify point-in-time universe |
| `validate_filing_lag()` | Check fundamental filing lag |
| `check_survivorship_bias()` | Detect survivorship bias |

---

## Class Names

### Signal Classes

```python
# CORRECT (PascalCase, descriptive)
class InstitutionalMomentum(InstitutionalSignal):
    pass

class InstitutionalQuality(InstitutionalSignal):
    pass

class InstitutionalInsider(InstitutionalSignal):
    pass

# WRONG
class Momentum:  # Not specific enough
class momentum_signal:  # snake_case for classes
class MomSignal:  # Abbreviation
```

### Base Classes

| Class | Purpose |
|-------|---------|
| `InstitutionalSignal` | Base class for all signals |
| `TransactionCostModel` | Transaction cost modeling |
| `Portfolio` | Portfolio state management |
| `DataManager` | Database interface |
| `BacktestEngine` | Backtest execution |

---

## File & Module Names

### File Naming (snake_case)

**Signals:**
```
signals/
├── momentum/
│   └── institutional_momentum.py
├── quality/
│   └── institutional_quality.py
└── insider/
    └── institutional_insider.py
```

**Core Modules:**
```
core/
├── institutional_base.py  # Base classes
├── portfolio.py           # Portfolio management
├── execution.py           # Transaction costs
└── types.py              # Type definitions
```

**Scripts:**
```
scripts/
├── run_institutional_backtest.py
├── analyze_spy_benchmark.py
├── validate_data_integrity.py
└── optimize_signals.py
```

### Documentation Files (SCREAMING_SNAKE_CASE or Title Case)

```
docs/
├── ARCHITECTURE.md
├── NAMING_CONVENTIONS.md
├── ERROR_PREVENTION_ARCHITECTURE.md
└── TRANSACTION_COST_ANALYSIS.md
```

---

## Parameter Names (Dict Keys)

### Signal Parameters

```python
signal_params = {
    # Momentum
    'formation_period_days': 252,
    'skip_period_days': 21,
    'winsorize_lower_pct': 5,
    'winsorize_upper_pct': 95,
    'use_quintiles': True,

    # Quality
    'use_profitability': True,
    'use_growth': True,
    'use_safety': True,
    'profitability_weight': 0.4,
    'growth_weight': 0.3,
    'safety_weight': 0.3,

    # Insider
    'lookback_days': 90,
    'min_transaction_value_usd': 10000,
    'cluster_window_days': 7,
    'cluster_min_insiders': 3,
    'ceo_weight': 3.0,
    'cfo_weight': 2.5,
}
```

**Convention:**
- Include units in name: `_days`, `_usd`, `_pct`, `_bps`
- Boolean flags: `use_`, `is_`, `has_`, `enable_`
- Weights: `_weight` suffix
- Thresholds: `min_`, `max_` prefix

### Portfolio Parameters

```python
portfolio_params = {
    'initial_capital_usd': 50000,
    'max_positions': 5,
    'max_position_size_pct': 0.20,  # 20%
    'rebalance_frequency': 'monthly',
    'position_sizing_method': 'equal_weight',
}
```

### Risk Parameters

```python
risk_params = {
    'stop_loss_pct': 0.05,  # 5%
    'take_profit_pct': 0.15,  # 15%
    'max_portfolio_drawdown_pct': 0.25,  # 25%
    'drawdown_scale_factor': 0.5,  # Reduce exposure 50% in drawdown
}
```

---

## Standard Abbreviations (ALLOWED)

These are standard in quantitative finance and academia:

| Abbreviation | Full Term | Context |
|--------------|-----------|---------|
| ROE | Return on Equity | Fundamental metric |
| ROA | Return on Assets | Fundamental metric |
| QMJ | Quality Minus Junk | Asness et al. quality factor |
| ADV | Average Daily Volume | Liquidity measure |
| OHLCV | Open High Low Close Volume | Price data |
| bps | Basis points | Transaction costs |
| pct | Percent | Generic percentage |
| USD | US Dollars | Currency |
| YoY | Year-over-Year | Growth rates |
| TTM | Trailing Twelve Months | Fundamentals |
| ARQ | As-Reported Quarterly | Sharadar dimension |
| ARY | As-Reported Yearly | Sharadar dimension |
| MRQ | Most Recent Quarter | Sharadar dimension |
| MRY | Most Recent Year | Sharadar dimension |

**NOT Allowed:**
- ❌ `mom` - Use `momentum`
- ❌ `qual` - Use `quality`
- ❌ `ins` - Use `insider`
- ❌ `ret` - Use `return`
- ❌ `vol` - Ambiguous (volume or volatility)

---

## Comments & Docstrings

### Function Docstrings (Google Style)

```python
def get_fundamentals(self,
                     ticker: str,
                     start_date: str,
                     end_date: str,
                     dimension: str = 'ARQ',
                     as_of_date: Optional[str] = None) -> pd.DataFrame:
    """
    Get fundamental data with point-in-time filtering.

    CRITICAL: Always pass as_of_date to prevent lookahead bias.
    Filters by datekey (filing date), NOT calendardate (quarter end).

    Args:
        ticker: Stock symbol (e.g., 'AAPL')
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        dimension: Data dimension (ARQ, MRQ, ARY, MRY)
        as_of_date: Point-in-time date for filtering (YYYY-MM-DD)

    Returns:
        DataFrame with fundamental metrics, indexed by calendar_date

    Raises:
        ValueError: If as_of_date is None (prevents accidental lookahead)

    Example:
        >>> dm = DataManager()
        >>> fundamentals = dm.get_fundamentals(
        ...     ticker='AAPL',
        ...     start_date='2023-01-01',
        ...     end_date='2023-12-31',
        ...     dimension='ARQ',
        ...     as_of_date='2023-06-30'  # Only data filed by June 30
        ... )

    References:
        - Sharadar documentation on filing lag
        - See ERROR_PREVENTION_ARCHITECTURE.md for temporal discipline
    """
```

### Inline Comments

```python
# Calculate 12-month momentum, skip most recent month
# Jegadeesh-Titman (1993) standard: 12-1 formation
momentum_return = prices.pct_change(periods=252).shift(21)

# Filter by filing date (datekey), NOT quarter-end (calendardate)
# This respects 30-45 day filing lag for fundamentals
if as_of_date:
    query += " AND datekey <= ?"  # Filing date
```

**Guidelines:**
- Explain WHY, not WHAT
- Reference papers when using academic methods
- Flag temporal discipline concerns
- Note edge cases and assumptions

---

## Constants

### Module-Level Constants

```python
# Time constants
TRADING_DAYS_PER_YEAR = 252
TRADING_DAYS_PER_MONTH = 21
TRADING_DAYS_PER_QUARTER = 63

# Annualization factors
ANNUALIZATION_DAILY = 252
ANNUALIZATION_MONTHLY = 12
ANNUALIZATION_QUARTERLY = 4

# Risk-free rate (updated periodically)
RISK_FREE_RATE_ANNUAL = 0.02  # 2%

# Transaction costs (Schwab $50K model)
COMMISSION_BPS = 0  # $0 commissions
SLIPPAGE_BPS = 2    # 2 bps
SPREAD_BPS = 3      # 3 bps
TOTAL_COST_BPS = 5  # 5 bps total

# Signal ranges
SIGNAL_MIN = -1.0
SIGNAL_MAX = 1.0
SIGNAL_QUINTILES = [-1.0, -0.5, 0.0, 0.5, 1.0]

# Filing lags (days)
FUNDAMENTAL_FILING_LAG_MIN = 30
FUNDAMENTAL_FILING_LAG_MAX = 45
INSIDER_FILING_LAG_SEC_REQUIRED = 2
```

---

## Error Messages & Logging

### Error Messages

```python
# GOOD - Explicit, actionable
raise ValueError(
    f"as_of_date is required to prevent lookahead bias. "
    f"Got: as_of_date={as_of_date}. "
    f"Pass the signal generation date as as_of_date parameter."
)

# BAD - Generic, unhelpful
raise ValueError("Invalid date")
```

### Log Messages

```python
# GOOD - Structured, informative
logger.info(
    f"Fetching fundamentals: ticker={ticker}, "
    f"period={start_date} to {end_date}, "
    f"as_of={as_of_date}, "
    f"dimension={dimension}"
)

logger.warning(
    f"Potential lookahead bias: as_of_date not provided. "
    f"Using all data up to {end_date}. "
    f"File: {__file__}, Line: {lineno}"
)

# BAD - Vague, hard to debug
logger.info("Getting data")
```

---

## Validation Checklist

Before committing code, verify:

- [ ] All date variables end in `_date`
- [ ] Time periods end in `_days`, `_months`, or `_years`
- [ ] Percentages use `_pct` (decimal) or `_percent` (0-100)
- [ ] Basis points use `_bps`
- [ ] Signal methods named `generate_signals()` or `calculate_*`
- [ ] Data methods have `as_of_date` parameter
- [ ] Class names are PascalCase
- [ ] Function/variable names are snake_case
- [ ] Constants are SCREAMING_SNAKE_CASE
- [ ] No ambiguous abbreviations (except standard list)
- [ ] Docstrings follow Google style
- [ ] Comments explain WHY, not WHAT

---

## Migration from Old Names

If you find inconsistent names in existing code:

1. **Add to this document** if not already covered
2. **Global find/replace** across entire codebase
3. **Update tests** to use new names
4. **Update documentation** to match
5. **Add to ERROR_PREVENTION_ARCHITECTURE.md** if it caused bugs

### Common Migrations

| Old Name (WRONG) | New Name (CORRECT) | Reason |
|------------------|-------------------|--------|
| `as_of` | `as_of_date` | More explicit |
| `calendardate` filter | `datekey` filter | Respects filing lag |
| `calculate()` | `generate_signals()` | Standard API |
| `transaction_costs` | `total_cost_bps` | Units specified |
| `max_dd` | `max_drawdown` | No abbreviation |
| `vol` | `volatility_annual` or `volume` | Disambiguation |

---

## References

**Style Guides:**
- PEP 8 - Python style guide
- Google Python Style Guide - Docstrings
- NumPy documentation style

**Quant Terminology:**
- Grinold & Kahn (2000) "Active Portfolio Management"
- Jegadeesh & Titman (1993) - Momentum terminology
- Asness et al. (2018) - Quality factor terminology
- Cohen et al. (2012) - Insider trading terminology

**Database:**
- Sharadar documentation
- SEC EDGAR filing requirements

---

**Last Updated:** 2025-11-20
**Next Review:** 2026-02-20 (Quarterly)
**Maintainer:** See CLAUDE.md for governance process
