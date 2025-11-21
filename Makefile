.PHONY: help install test lint format clean validate optimize backtest run

# Default target
help:
	@echo "SignalTide v3 - Available Commands"
	@echo "=================================="
	@echo ""
	@echo "Setup:"
	@echo "  make install      - Install dependencies"
	@echo "  make setup        - Initial setup (install + create dirs)"
	@echo ""
	@echo "Development:"
	@echo "  make test         - Run test suite"
	@echo "  make test-plumbing - Run market plumbing tests (31 tests)"
	@echo "  make test-fast    - Run tests in parallel"
	@echo "  make test-cov     - Run tests with coverage report"
	@echo "  make lint         - Run ruff linter"
	@echo "  make format       - Run ruff formatter (alias: fmt)"
	@echo "  make typecheck    - Run mypy type checker"
	@echo "  make clean        - Clean temporary files"
	@echo ""
	@echo "Validation:"
	@echo "  make validate     - Run full validation framework"
	@echo "  make validate-fast - Run quick validation checks"
	@echo ""
	@echo "Optimization:"
	@echo "  make optimize     - Run Optuna hyperparameter optimization"
	@echo "  make optimize-resume - Resume interrupted optimization"
	@echo ""
	@echo "Backtesting:"
	@echo "  make backtest     - Run backtest with current parameters"
	@echo "  make backtest-report - Generate backtest report with visualizations"
	@echo ""
	@echo "Data:"
	@echo "  make fetch-data   - Fetch latest market data"
	@echo "  make check-data   - Validate data integrity"
	@echo ""
	@echo "Utilities:"
	@echo "  make notebook     - Start Jupyter notebook"
	@echo "  make shell        - Start IPython shell with context loaded"
	@echo "  make config       - Show current configuration"
	@echo ""

# ============================================================================
# Setup
# ============================================================================

install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt
	@echo "✓ Dependencies installed"

setup: install
	@echo "Creating directories..."
	mkdir -p data/databases data/cache logs results
	@echo "Creating .env file from template..."
	[ -f .env ] || cp .env.template .env
	@echo "✓ Setup complete"
	@echo ""
	@echo "⚠️  Don't forget to edit .env with your API keys!"

# ============================================================================
# Testing
# ============================================================================

test:
	@echo "Running test suite..."
	pytest tests/ -v

test-plumbing:
	@echo "Running market plumbing tests (31 tests)..."
	python3 scripts/test_trading_calendar.py && \
	python3 scripts/test_universe_manager.py && \
	python3 scripts/test_rebalance_helpers.py && \
	python3 scripts/test_rebalance_schedules.py && \
	python3 scripts/test_backtest_integration.py && \
	python3 scripts/test_deterministic_backtest.py
	@echo ""
	@echo "✓ All 31 plumbing tests passed!"

test-fast:
	@echo "Running tests in parallel..."
	pytest tests/ -v -n auto

test-cov:
	@echo "Running tests with coverage..."
	pytest tests/ -v --cov=. --cov-report=html --cov-report=term
	@echo "✓ Coverage report generated in htmlcov/"

test-watch:
	@echo "Running tests in watch mode..."
	pytest-watch tests/ -v

# ============================================================================
# Code Quality
# ============================================================================

lint:
	@echo "Running ruff linter..."
	@command -v ruff >/dev/null 2>&1 || { echo "⚠️  ruff not installed. Run: pip install ruff"; exit 1; }
	@ruff check .
	@echo "✓ Linting complete"

format:
	@echo "Running ruff formatter..."
	@command -v ruff >/dev/null 2>&1 || { echo "⚠️  ruff not installed. Run: pip install ruff"; exit 1; }
	@ruff format .
	@echo "✓ Formatting complete"

fmt: format
	@# Alias for format

format-check:
	@echo "Checking format..."
	@command -v ruff >/dev/null 2>&1 || { echo "⚠️  ruff not installed. Run: pip install ruff"; exit 1; }
	@ruff format --check .
	@echo "Checking lint..."
	@ruff check .

typecheck:
	@echo "Running mypy type checker..."
	@command -v mypy >/dev/null 2>&1 || { echo "⚠️  mypy not installed. Run: pip install mypy"; exit 1; }
	@mypy .
	@echo "✓ Type checking complete"

# ============================================================================
# Validation
# ============================================================================

validate:
	@echo "Running full validation framework..."
	python -m scripts.run_validation --full
	@echo "✓ Validation complete"

validate-fast:
	@echo "Running quick validation..."
	python -m scripts.run_validation --quick
	@echo "✓ Quick validation complete"

validate-signal:
	@echo "Validating specific signal..."
	@read -p "Enter signal name: " signal; \
	python -m scripts.run_validation --signal $$signal

# ============================================================================
# Optimization
# ============================================================================

optimize:
	@echo "Starting hyperparameter optimization..."
	@echo "This may take a while. Progress will be shown below."
	python -m scripts.run_optimization

optimize-resume:
	@echo "Resuming optimization from last checkpoint..."
	python -m scripts.run_optimization --resume

optimize-trials:
	@echo "Running optimization with custom trial count..."
	@read -p "Enter number of trials: " trials; \
	python -m scripts.run_optimization --n-trials $$trials

# ============================================================================
# Backtesting
# ============================================================================

backtest:
	@echo "Running backtest..."
	python -m scripts.run_backtest

backtest-report:
	@echo "Generating backtest report..."
	python -m scripts.run_backtest --report
	@echo "✓ Report generated in results/"

backtest-compare:
	@echo "Comparing multiple strategies..."
	python -m scripts.run_backtest --compare

# ============================================================================
# Data Management
# ============================================================================

fetch-data:
	@echo "Fetching latest market data..."
	python -m scripts.fetch_data

fetch-data-full:
	@echo "Fetching full historical data..."
	@echo "⚠️  This may take a long time and use significant API quota"
	@read -p "Continue? [y/N]: " confirm; \
	if [ "$$confirm" = "y" ]; then \
		python -m scripts.fetch_data --full; \
	fi

check-data:
	@echo "Validating data integrity..."
	python -m scripts.check_data_integrity
	@echo "✓ Data validation complete"

# ============================================================================
# Utilities
# ============================================================================

notebook:
	@echo "Starting Jupyter notebook..."
	jupyter notebook

shell:
	@echo "Starting IPython shell..."
	ipython -i -c "from config import *; from core import *; print('SignalTide v3 shell loaded')"

config:
	@echo "Current configuration:"
	python config.py

clean:
	@echo "Cleaning temporary files..."
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.coverage" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ *.egg-info/
	@echo "✓ Cleanup complete"

clean-all: clean
	@echo "⚠️  This will delete all data, logs, and results!"
	@read -p "Continue? [y/N]: " confirm; \
	if [ "$$confirm" = "y" ]; then \
		rm -rf data/ logs/ results/ cache/; \
		mkdir -p data/databases data/cache logs results; \
		echo "✓ Full cleanup complete"; \
	fi

# ============================================================================
# Database Management
# ============================================================================

db-shell:
	@echo "Opening database shell..."
	sqlite3 data/databases/market_data.db

db-backup:
	@echo "Backing up databases..."
	mkdir -p backups
	cp -r data/databases backups/databases_$$(date +%Y%m%d_%H%M%S)
	@echo "✓ Backup created in backups/"

db-reset:
	@echo "⚠️  This will delete all database data!"
	@read -p "Continue? [y/N]: " confirm; \
	if [ "$$confirm" = "y" ]; then \
		rm -f data/databases/*.db; \
		echo "✓ Databases reset"; \
	fi

# ============================================================================
# Documentation
# ============================================================================

docs:
	@echo "Opening documentation..."
	@open README.md || xdg-open README.md || echo "Please open README.md manually"

docs-serve:
	@echo "Starting documentation server..."
	python -m http.server 8000 --directory .

# ============================================================================
# Git Helpers
# ============================================================================

git-status:
	@echo "Repository status:"
	@git status

commit:
	@echo "Creating commit..."
	@$(MAKE) format
	@$(MAKE) lint
	@$(MAKE) test-fast
	@git add -A
	@git status
	@echo ""
	@read -p "Commit message: " msg; \
	git commit -m "$$msg"

# ============================================================================
# Profiling & Performance
# ============================================================================

profile:
	@echo "Profiling backtest performance..."
	python -m cProfile -o profile.stats scripts/run_backtest.py
	@echo "✓ Profile saved to profile.stats"
	@echo "View with: python -m pstats profile.stats"

benchmark:
	@echo "Running performance benchmarks..."
	python -m scripts.benchmark

# ============================================================================
# CI/CD Helpers
# ============================================================================

ci: format-check lint test-cov
	@echo "✓ All CI checks passed"

pre-commit: format lint test-fast
	@echo "✓ Pre-commit checks passed"
