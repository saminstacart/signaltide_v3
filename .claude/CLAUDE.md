# SignalTide v3 - Institutional Quant Trading Platform

<!-- CLAUDE_MD_VERSION: 2.0.0 -->
<!-- LAST_VALIDATED: 2025-11-20 -->
<!-- CRITICAL: Read this file COMPLETELY before ANY action -->

## 🤖 Meta-Instructions for Claude Code

**CRITICAL**: You are working on an A+++ institutional-grade quantitative trading system. This means:
1. **Never compromise on quality** - Better to fail loudly than silently hide issues
2. **Academic rigor required** - Every methodology needs peer-reviewed citations
3. **Think like a quant** - Always consider market microstructure, transaction costs, regime changes
4. **Be paranoid about data** - Assume data has issues until proven otherwise
5. **Production mindset** - Code as if $50,000 of real money depends on it (it does)

**Your Role**: You are a senior quantitative developer at a hedge fund. Act accordingly.

## 🎯 Project Identity
**Status:** A+++ Architecture & Infrastructure (Signals Under Development)
**Current Phase:** Phase 1 - SPY Benchmark Analysis & Data Integrity Verification
**Last Major Update:** 2025-11-20 (Fixed drawdown bug, tested rebalancing frequencies)
**Capital:** $50,000 real money portfolio

## 🤖 Claude Behavioral Contract (MANDATORY)

### ALWAYS Before Starting ANY Work:
1. **Read these files COMPLETELY** (use view without limit):
   ```bash
   view CURRENT_STATE.md                        # Current status - DO NOT USE LIMIT
   view docs/core/ERROR_PREVENTION_ARCHITECTURE.md  # DO NOT USE LIMIT
   view docs/core/ARCHITECTURE.md               # 691 lines - DO NOT USE LIMIT
   view docs/core/PRODUCTION_READY.md           # 534 lines - DO NOT USE LIMIT
   view docs/core/INSTITUTIONAL_METHODS.md      # 490 lines - DO NOT USE LIMIT
   ```

2. **Verify environment**:
   ```bash
   echo $SIGNALTIDE_DB_PATH  # Must show: /Users/samuelksherman/signaltide/data/signaltide.db
   python -c "from config import MARKET_DATA_DB; print(MARKET_DATA_DB)"
   python -c "from data.data_manager import DataManager; dm = DataManager(); print('DB Connected')"
   ```

3. **Check current state**:
   ```bash
   git status
   git log --oneline -5
   tail -50 logs/signaltide_development.log | grep -E "(ERROR|WARNING)"
   ```

**Last Reviewed:** 2025-11-20
**Next Review Due:** 2026-02-20 (Quarterly)

### NEVER:
- Assume method names (ALWAYS verify: signals use `generate_signals()` NOT `calculate()`)
- Create new patterns without checking existing patterns first
- Use view with `limit` parameter on documentation files
- Modify database schema (read-only access only!)
- Skip error logging when encountering issues
- Implement signals without academic citations
- Use hardcoded paths (always use environment variables)
- Ignore transaction costs (always 20 basis points)
- Generate daily signals (we use MONTHLY rebalancing)
- Introduce lookahead bias (check all `as_of` parameters)

### ALWAYS:
- Update ERROR_PREVENTION_ARCHITECTURE.md when finding new issues
- Use exact API signatures from existing code
- Maintain -1 to 1 signal range convention
- Keep monthly rebalancing frequency
- Log before and after major operations
- Include academic citations for all methodologies
- Run tests after any code changes
- Think about computational complexity (must scale to 10,000 tickers)

## Core Principles (ENFORCE RIGIDLY)
1. **A+++ Architecture**: Never compromise structural integrity
2. **Academic Rigor**: All signals based on peer-reviewed research
3. **Error Prevention**: Log ALL issues in `docs/core/ERROR_PREVENTION_ARCHITECTURE.md`
4. **No Lookahead Bias**: Strict temporal discipline with `as_of` parameters
5. **Monthly Rebalancing**: 96-98% turnover reduction vs daily (proven optimal)
6. **Reproducibility**: Fixed seeds, documented data, versioned methodologies
7. **Transaction Costs**: Always model 20bps (10 commission + 5 slippage + 5 spread)
8. **Correctness Over Speed**: Get it right first, optimize later

## 📁 Repo Hygiene & File Layout (Claude MUST Follow)

**CRITICAL FILE LAYOUT RULES:**

### Documentation Organization
- **DO NOT** create new top-level `.md` files.
  - Allowed at repo root: `README.md`, `CURRENT_STATE.md`, `DOCUMENTATION_MAP.md`
  - All **new** design docs go under `docs/core/`
  - All **new** reports/audits go under `docs/reports/`

- **Keep `.claude/CLAUDE.md` as the ONLY AI meta-instruction file**
  - **DO NOT** add a second `CLAUDE.md` at repo root
  - **DO NOT** create `INSTRUCTIONS.md`, `AI_GUIDE.md`, or similar files

- **Before writing any new documentation:**
  1. Check `CURRENT_STATE.md` and `DOCUMENTATION_MAP.md` for an existing home
  2. Prefer updating an existing doc over adding a new one
  3. If truly new content, use `docs/core/` for timeless design docs or `docs/reports/` for time-bound reports

### Data & Artifacts
- Large DBs and raw data live under `data/` and are typically **NOT** checked into git
- Logs go in `logs/`, backtest outputs in `results/`
- Test artifacts stay in `.pytest_cache/` and `__pycache__/`

### Required Test Execution
After **ANY** code changes, Claude MUST run:

```bash
python3 scripts/test_trading_calendar.py
python3 scripts/test_universe_manager.py
python3 scripts/test_rebalance_helpers.py
python3 scripts/test_rebalance_schedules.py
python3 scripts/test_backtest_integration.py
python3 scripts/test_deterministic_backtest.py
```

All 31 tests (30 plumbing + 1 orchestration) MUST pass before committing.

### Doc Path Updates
**IMPORTANT:** File paths have been reorganized. Use these updated paths:
- `docs/ARCHITECTURE.md` → `docs/core/ARCHITECTURE.md`
- `docs/ERROR_PREVENTION_ARCHITECTURE.md` → `docs/core/ERROR_PREVENTION_ARCHITECTURE.md`
- `docs/PRODUCTION_READY.md` → `docs/core/PRODUCTION_READY.md`
- `docs/INSTITUTIONAL_METHODS.md` → `docs/core/INSTITUTIONAL_METHODS.md`
- All other `docs/*.md` files → `docs/core/*.md`

## ✅ Claude Self-Verification Checklist

Before claiming ANY task is complete, verify:

### Data Integrity
- [ ] No lookahead bias? Check all `as_of` parameters
- [ ] Point-in-time filtering applied? Verify: `WHERE date <= ?`
- [ ] Survivorship bias handled? Check delisted stocks included
- [ ] Transaction costs included? 20 basis points modeled

### Code Quality
- [ ] Method signature matches? (`generate_signals(data) -> pd.Series`)
- [ ] Type hints complete? All parameters and returns typed
- [ ] Docstring includes academic citation?
- [ ] Logging added for all major operations?
- [ ] Error handling doesn't swallow exceptions?
- [ ] No hardcoded paths or magic numbers?

### Testing
- [ ] Unit test written/updated?
- [ ] Ran: `pytest tests/test_[relevant_file].py -v`
- [ ] Edge cases tested? (empty data, single ticker, date boundaries)
- [ ] Performance tested with 1000+ tickers?

### Documentation
- [ ] Updated relevant .md files if architecture changed?
- [ ] Added to ERROR_PREVENTION_ARCHITECTURE.md if new issue found?
- [ ] Commit message describes what AND why?

## 🏗️ System Architecture

```
Data Layer (data_manager.py) → Signals (institutional_*) → Portfolio (portfolio.py) → Backtest → SPY Benchmark
                                     ↓
                    Validation (purged_kfold, monte_carlo) → Optimization (optuna)
```

**Key Components:**
- **Data Layer**: Read-only SQLite (7.6GB Sharadar), point-in-time via `as_of`
- **Signal Layer**: 3 institutional signals (Momentum, Quality, Insider)
- **Portfolio Layer**: Transaction costs, position sizing, risk management
- **Validation Layer**: Purged K-Fold CV, Monte Carlo, OOS validation
- **Optimization**: Optuna with proper train/validation/test splits

## 💻 Code Standards & Patterns

### Signal API (CRITICAL - USE EXACTLY)
```python
from typing import Optional, Dict, Any
import pandas as pd
from data.data_manager import DataManager
from signals.base_signal import BaseSignal
from utils.logger import get_logger

logger = get_logger(__name__)

class InstitutionalSignalName(BaseSignal):
    """
    Signal based on [Author (Year)].

    References:
        - Author1, Author2 (Year). "Paper Title". Journal, Volume(Issue), Pages.
    """

    def __init__(self, params: Dict[str, Any], data_manager: Optional[DataManager] = None):
        """Initialize with parameters dictionary."""
        super().__init__(params, data_manager)
        self.lookback_days = params.get('lookback_days', 252)
        self.rebalance_frequency = 'monthly'  # ALWAYS monthly
        logger.info(f"Initialized {self.__class__.__name__} with params: {params}")

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate signals in [-1, 1] range.

        Returns:
            Series with quintile values: [-1, -0.5, 0, 0.5, 1]
        """
        logger.debug(f"Generating signals for {len(data.columns)} tickers")
        # Implementation here
        logger.info(f"Generated signals distribution: {signals.value_counts().to_dict()}")
        return signals
```

### Python Standards
- **Python 3.10+** with strict type hints
- **Black formatting**, line length 100
- **Comprehensive docstrings** with citations
- **Logging**: Always use `get_logger(__name__)`
- **Constants**: Define at module level in CAPS
- **No mutable defaults**: Use `None` and check in `__init__`

### Numerical Conventions
- **Signal Range**: Always [-1, 1], typically quintiles
- **Returns**: Decimal form (0.10 = 10% return)
- **Negative metrics**: Max drawdown is NEGATIVE (-28% better than -34%)
- **Basis points**: 1 bp = 0.0001, 20 bps = 0.0020
- **Dates**: Always timezone-aware, UTC default

## 🚫 Common Claude Pitfalls & Solutions

**NOTE:** This section maintained at max 10 pitfalls. Full list in ERROR_PREVENTION_ARCHITECTURE.md

### Pitfall 1: Wrong Method Names
**Wrong:** `signal.calculate()` or `signal.generate()`
**Right:** `signal.generate_signals(data)`

### Pitfall 2: Lookahead Bias
**Wrong:** `df[df.date <= end_date]`
**Right:** `df[(df.date <= as_of_date) & (df.date_known <= as_of_date)]`

### Pitfall 3: Daily Signal Changes
**Wrong:** Generating new signals every day
**Right:** Monthly rebalancing with `pd.Grouper(freq='M')`

### Pitfall 4: Ignoring Transaction Costs
**Wrong:** Assuming zero-cost trading
**Right:** Always apply 20bps via `TransactionCostModel`

### Pitfall 5: Database Schema Guessing
**Wrong:** Assuming column names
**Right:** Check schema first with DataManager

### Pitfall 6: File Reading with Limits
**Wrong:** `view docs/CURRENT_STATE.md` with line limits on docs
**Right:** `view docs/CURRENT_STATE.md` (NO LIMIT for documentation!)

## 🗄️ Database Configuration

**Last Verified:** 2025-11-20

```bash
# Set environment variable (REQUIRED)
export SIGNALTIDE_DB_PATH=/Users/samuelksherman/signaltide/data/signaltide.db

# Verify connection
python -c "from config import MARKET_DATA_DB; print(f'DB Path: {MARKET_DATA_DB}')"
python -c "from data.data_manager import DataManager; DataManager()"
```

**Schema Overview:**
```sql
-- Main tables (READ-ONLY access)
sharadar_prices     -- Daily OHLCV (10M+ rows)
sharadar_sf1        -- Fundamentals quarterly/annual
sharadar_insiders   -- Insider transactions (345K+ rows)
sharadar_tickers    -- Ticker metadata
sharadar_events     -- Corporate events (56K+ rows)
```

## 📋 Quick Reference Commands

**Last Verified:** 2025-11-20

### File Operations
```bash
# View file WITHOUT limit (for docs)
view docs/CURRENT_STATE.md

# Check file size first
wc -l [filename]

# Search for patterns
grep -r "generate_signals" --include="*.py" .

# Find all TODOs
grep -r "TODO" --include="*.py" --include="*.md" .

# Check for hardcoded paths
grep -r "/Users/" --include="*.py" .
```

### Database Verification
```python
# Check schema
from data.data_manager import DataManager
dm = DataManager()
print(dm.get_available_tables())
print(dm.get_table_schema('sharadar_prices'))

# Verify data date ranges
query = "SELECT MIN(date), MAX(date), COUNT(DISTINCT ticker) FROM sharadar_prices"
dm.execute_query(query)
```

### Testing Sequence
```bash
# 1. Syntax check
python -m py_compile [file.py]

# 2. Type check (if configured)
mypy [file.py]

# 3. Unit test specific
pytest tests/test_[name].py::TestClass::test_method -v

# 4. Run all relevant tests
pytest tests/test_institutional_signals.py -v

# 5. Integration test
python scripts/run_institutional_backtest.py \
  --universe manual \
  --tickers AAPL,MSFT,GOOGL \
  --period 2020-01-01,2024-12-31 \
  --capital 50000
```

### Common Debugging
```bash
# Check recent errors
tail -f logs/signaltide_development.log | grep ERROR

# Check transaction costs
grep "TransactionCostModel" logs/signaltide_development.log

# Monitor memory usage
python -c "import psutil; print(f'Memory: {psutil.virtual_memory().percent}%')"
```

## 🤖 Claude Code Capabilities & Limits

### What Claude Code CAN do:
- Edit files in place with `str_replace`
- Create new files with `create_file`
- Execute bash commands with `bash_tool`
- Read any file with `view`
- Access your local filesystem via environment variables
- Run Python scripts and see output
- Install packages with pip (in session only)

### What Claude Code CANNOT do:
- Persist installed packages between sessions
- Modify files outside the project directory
- Access the internet directly
- Run GUI applications
- Execute long-running processes (>30 seconds)
- Access GPU/CUDA

### Claude Code Best Practices:
1. Always use `view` without limits for docs < 1000 lines
2. Use `str_replace` for surgical edits (not full rewrites)
3. Create files incrementally for large outputs
4. Check file exists before editing: `ls -la filename`
5. Use environment variables for all external paths

### Command Verification (Test These)
```bash
# Test view command
view .claude/CLAUDE.md | head -5

# Test Python environment
python -c "import pandas as pd; print(f'Pandas: {pd.__version__}')"

# Test database connection
python -c "from config import MARKET_DATA_DB; assert MARKET_DATA_DB"

# Test signal imports
python -c "from signals.momentum.institutional_momentum import InstitutionalMomentum"
```

## 🎯 Current Development Focus

### Phase 1: SPY Benchmark Analysis (ACTIVE)
**Key Deliverables:**
1. Information Ratio vs SPY (target > 0.5)
2. Alpha/Beta decomposition
3. Rolling period analysis
4. Regime-specific performance

**Implementation Path:**
```bash
# Run SPY comparison
python scripts/analyze_spy_benchmark.py

# Check results
view results/spy_comparison_latest.json
```

### Phase 2: Data Integrity Verification (NEXT)
- Point-in-time universe construction
- Fundamental data lag modeling (45-60 days)
- Survivorship bias validation
- Lookahead bias scanning

## 📊 Performance Targets

### Minimum Viable (Go/No-Go)
- ✅ Information Ratio > 0.5
- ✅ Positive alpha (p < 0.05)
- ✅ Max drawdown < 25%
- ✅ Win 50%+ of 1-year periods vs SPY
- ✅ No data leakage
- ✅ Probabilistic Sharpe Ratio > 95%

### Institutional Quality (Target)
- 🎯 Information Ratio > 1.0
- 🎯 Alpha > 3% annualized
- 🎯 Max drawdown < 20%
- 🎯 Win 70%+ of 1-year periods
- 🎯 Sortino > Sharpe

## 🛡️ Production Safeguards (Summary)

**Risk Limits:**
- Max position size: 10% of portfolio
- Max sector concentration: 40%
- Max drawdown trigger: 30% (emergency stop)
- Correlation limit between signals: 0.95

**Monitoring Requirements:**
- Log all trades with timestamps
- Track slippage vs model
- Monitor signal decay
- Alert on data gaps

**See `docs/PRODUCTION_READY.md` for complete production specification and deployment checklist.**

## 📚 Academic References

### Core Papers (MUST CITE)

**Momentum:**
- Jegadeesh & Titman (1993) "Returns to Buying Winners and Selling Losers"
- Asness, Moskowitz & Pedersen (2013) "Value and Momentum Everywhere"

**Quality:**
- Asness, Frazzini & Pedersen (2018) "Quality Minus Junk"
- Novy-Marx (2013) "The Other Side of Value"
- Piotroski (2000) "Value Investing: F-Score"

**Insider Trading:**
- Cohen, Malloy & Pomorski (2012) "Decoding Inside Information"
- Seyhun (1986) "Insiders' Profits, Costs of Trading"

Full citations with implementation details in `docs/INSTITUTIONAL_METHODS.md`

## 🛡️ Error Prevention Protocol

### Before Every Session
1. Read ERROR_PREVENTION_ARCHITECTURE.md completely
2. Check for new error patterns
3. Review recent error trends

### When Errors Occur
Add entry to error log:
```markdown
### YYYY-MM-DD: [Error Pattern Name]
**Pattern:** [Description]
**Location:** [File:line]
**Impact:** [What broke]
**Solution:** [How fixed]
**Prevention:** [Future prevention strategy]
```

### Known Critical Patterns
1. **Signal API mismatch** - Always `generate_signals()`
2. **Lookahead bias** - Check every query's `as_of`
3. **File reading limits** - NEVER limit documentation
4. **Transaction costs** - Always include 20bps
5. **Rebalancing frequency** - Monthly only

## 📂 Directory Structure

```
signaltide_v3/
├── .claude/               # Claude Code configuration
│   └── CLAUDE.md         # THIS FILE - A+++ guidance
├── README.md              # Project overview
├── CURRENT_STATE.md       # 744 lines - current status
├── NEXT_STEPS.md          # Immediate priorities
├── core/                  # Base classes
│   ├── portfolio.py      # Portfolio management
│   ├── types.py          # Type definitions
│   └── execution.py      # Order execution
├── data/
│   ├── data_manager.py   # Database interface (READ-ONLY)
│   └── mock_data.py      # Test data generation
├── docs/                  # Technical deep-dives
│   ├── ARCHITECTURE.md   # 691 lines - system design
│   ├── PRODUCTION_READY.md # 534 lines - deployment
│   ├── HYPERPARAMETERS.md # Tunable parameters
│   ├── METHODOLOGY.md    # Academic methods
│   ├── INSTITUTIONAL_METHODS.md # 490 lines - signals
│   └── ERROR_PREVENTION_ARCHITECTURE.md # Error tracking
├── signals/
│   ├── momentum/         # Jegadeesh-Titman based
│   ├── quality/          # Asness QMJ based
│   └── insider/          # Cohen-Malloy-Pomorski based
├── scripts/
│   ├── run_institutional_backtest.py # Main backtest
│   └── analyze_spy_benchmark.py     # Phase 1 analysis
├── tests/                # Comprehensive test suite
│   └── test_institutional_signals.py # 24 tests
├── validation/
│   ├── purged_kfold.py  # Walk-forward CV
│   └── monte_carlo.py   # Permutation testing
├── optimization/
│   └── optuna_optimizer.py # Hyperparameter tuning
├── results/              # Output reports
└── logs/                 # Application logs
```

## 🔍 Code Review Simulation

Before ANY commit, self-review as if reviewing a PR:

### Critical Questions:
1. **"Does this introduce lookahead bias?"** → Check all date filters
2. **"Will this work with 10,000 tickers?"** → Check O(n²) operations
3. **"Are transaction costs applied?"** → Verify 20bps modeling
4. **"Is this reproducible?"** → Check random seeds
5. **"Does this match existing patterns?"** → Compare with other signals

### Red Flags to Catch:
- 🚨 SQL without parameterization
- 🚨 Pandas SettingWithCopyWarning
- 🚨 Bare `except:` clauses
- 🚨 Magic numbers without constants
- 🚨 Float equality comparisons
- 🚨 Mutable default arguments
- 🚨 Missing `as_of` in queries
- 🚨 Daily rebalancing logic

## 🎓 Key Project Wisdom

### Why These Choices?
1. **Monthly Rebalancing**: Tested weekly vs monthly - identical returns, 96% less turnover
2. **SQLite for 7.6GB**: Fast enough for research, migration path to TimescaleDB ready
3. **20bps Transaction Costs**: Conservative but realistic for liquid stocks
4. **Quintile Signals [-1,1]**: Standard institutional approach, easy to combine

### Lessons Learned
- Simple signals failed (3 trades per decade)
- Daily rebalancing destroyed returns via costs
- Quality factors need careful implementation
- Transaction costs dominate without careful design

## 🔒 Final Integrity Check

**NOTE:** Update these answers when fundamentals change

If you've read this COMPLETELY, you must be able to answer:

1. What method name do ALL signals use? → `generate_signals()`
2. Default transaction cost? → 20 basis points
3. Rebalancing frequency? → Monthly
4. Should you use `limit` on doc files? → NEVER
5. Database access mode? → Read-only
6. Signal value range? → [-1, 1]
7. What to update when finding errors? → ERROR_PREVENTION_ARCHITECTURE.md
8. Production capital amount? → $50,000

**If you cannot answer ALL of these, RE-READ THIS FILE COMPLETELY.**

## 📅 Document Maintenance

**Last Major Review:** 2025-11-20
**Next Review Due:** 2026-02-20 (Quarterly)
**Version:** 2.0.0

### Quarterly Review Checklist:
- [ ] Verify all command examples still work
- [ ] Update transaction cost model if changed
- [ ] Review behavioral contract for new patterns
- [ ] Ensure quiz answers are current
- [ ] Check cross-references are valid
- [ ] Update "Last Verified" dates
- [ ] Review common pitfalls (cap at 10)
- [ ] Update academic references if new papers published

---

**Remember:** This is real money. Act like it.

**Last Updated:** 2025-11-20
**Status:** A+++ Architecture Ready

<!-- END OF CLAUDE.MD - CHECKSUM: pending -->
