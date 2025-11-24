#!/usr/bin/env python3
"""
Insider data coverage pre-flight check for M3.6 3-signal ensemble.

Runs coverage queries from docs/ENSEMBLES_M3.6_THREE_SIGNAL_SPEC.md Section 6.1
to determine if insider data is sufficient for M3.6 baseline.

Decision criteria:
- GO: ≥75% of universe covered with reasonable temporal density
- NO-GO: <50% coverage or very sparse/patchy data
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.data_manager import DataManager
import pandas as pd


def main():
    print("=" * 80)
    print("INSIDER DATA COVERAGE PRE-FLIGHT CHECK (M3.6)")
    print("=" * 80)
    print()

    dm = DataManager()
    conn = dm._get_connection()

    # Query 1: Overall coverage (2015-2024)
    print("Query 1: Overall insider data coverage (2015-2024)")
    print("-" * 80)

    query1 = """
    SELECT
      MIN(filingdate) as earliest,
      MAX(filingdate) as latest,
      COUNT(DISTINCT ticker) as unique_tickers,
      COUNT(*) as total_trades
    FROM sharadar_insiders
    WHERE filingdate >= '2015-01-01'
      AND filingdate <= '2024-12-31'
    """

    result1 = pd.read_sql_query(query1, conn)
    print(result1)
    print()

    if len(result1) > 0:
        earliest = result1.iloc[0]['earliest']
        latest = result1.iloc[0]['latest']
        unique_tickers = result1.iloc[0]['unique_tickers']
        total_trades = result1.iloc[0]['total_trades']

        print(f"  Earliest filing: {earliest}")
        print(f"  Latest filing: {latest}")
        print(f"  Unique tickers: {unique_tickers:,}")
        print(f"  Total trades: {total_trades:,}")
        print()

    # Query 2: Trades per year (density check)
    print("Query 2: Trades per year (density check)")
    print("-" * 80)

    query2 = """
    SELECT
      SUBSTR(filingdate, 1, 4) as year,
      COUNT(*) as trades_count,
      COUNT(DISTINCT ticker) as unique_tickers
    FROM sharadar_insiders
    WHERE filingdate >= '2015-01-01'
      AND filingdate <= '2024-12-31'
    GROUP BY year
    ORDER BY year
    """

    result2 = pd.read_sql_query(query2, conn)
    print(result2.to_string(index=False))
    print()

    # Calculate coverage statistics
    print("Query 3: Transaction type distribution")
    print("-" * 80)

    query3 = """
    SELECT
      transactioncode,
      COUNT(*) as count,
      ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) as pct
    FROM sharadar_insiders
    WHERE filingdate >= '2015-01-01'
      AND filingdate <= '2024-12-31'
    GROUP BY transactioncode
    ORDER BY count DESC
    LIMIT 10
    """

    result3 = pd.read_sql_query(query3, conn)
    print(result3.to_string(index=False))
    print()

    # Query 4: Filing lag statistics
    print("Query 4: Filing lag statistics (filingdate - transactiondate)")
    print("-" * 80)

    query4 = """
    SELECT
      COUNT(*) as total_with_dates,
      AVG(JULIANDAY(filingdate) - JULIANDAY(transactiondate)) as avg_lag_days,
      MIN(JULIANDAY(filingdate) - JULIANDAY(transactiondate)) as min_lag_days,
      MAX(JULIANDAY(filingdate) - JULIANDAY(transactiondate)) as max_lag_days
    FROM sharadar_insiders
    WHERE filingdate >= '2015-01-01'
      AND filingdate <= '2024-12-31'
      AND transactiondate IS NOT NULL
      AND filingdate IS NOT NULL
    """

    result4 = pd.read_sql_query(query4, conn)
    print(result4.to_string(index=False))
    print()

    # Additional query: Check S&P 500 overlap (approximate using large-cap tickers)
    print("Query 5: Coverage for common S&P 500 tickers (sample)")
    print("-" * 80)

    # Sample of well-known S&P 500 tickers
    sp500_sample = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK.B',
        'JPM', 'JNJ', 'V', 'PG', 'MA', 'HD', 'CVX', 'MRK', 'ABBV', 'PEP',
        'KO', 'AVGO', 'COST', 'WMT', 'LLY', 'TMO', 'DIS', 'CSCO', 'ADBE',
        'ACN', 'NFLX', 'CRM', 'NKE', 'ABT', 'MCD', 'VZ', 'PM', 'INTC',
        'AMD', 'TXN', 'UNP', 'NEE', 'UNH', 'RTX', 'QCOM', 'DHR', 'HON',
        'AMGN', 'SBUX', 'LOW', 'BMY', 'BA', 'CAT', 'GE', 'IBM', 'GILD'
    ]

    # Check which tickers have data
    tickers_with_data = set(result2['year'].unique()) if 'year' in result2.columns else set()

    query5 = f"""
    SELECT
      COUNT(DISTINCT ticker) as tickers_with_data,
      COUNT(*) as total_trades
    FROM sharadar_insiders
    WHERE filingdate >= '2015-01-01'
      AND filingdate <= '2024-12-31'
      AND ticker IN ({','.join(f"'{t}'" for t in sp500_sample)})
    """

    result5 = pd.read_sql_query(query5, conn)
    print(f"Sample tickers checked: {len(sp500_sample)}")
    print(result5.to_string(index=False))

    covered_count = result5.iloc[0]['tickers_with_data'] if len(result5) > 0 else 0
    coverage_pct = 100.0 * covered_count / len(sp500_sample)
    print(f"  Coverage: {covered_count}/{len(sp500_sample)} = {coverage_pct:.1f}%")
    print()

    # Summary and decision
    print("=" * 80)
    print("SUMMARY & GO/NO-GO DECISION")
    print("=" * 80)
    print()

    print("Coverage Statistics:")
    print(f"  - Total unique tickers (2015-2024): {unique_tickers:,}")
    print(f"  - Total trades: {total_trades:,}")
    print(f"  - Sample S&P 500 coverage: {coverage_pct:.1f}% ({covered_count}/{len(sp500_sample)})")
    print()

    # Decision logic
    if coverage_pct >= 75:
        decision = "GO"
        reasoning = f"Coverage is {coverage_pct:.1f}%, which meets the ≥75% threshold."
    elif coverage_pct >= 50:
        decision = "MARGINAL"
        reasoning = f"Coverage is {coverage_pct:.1f}%, between 50-75%. Proceed with caution."
    else:
        decision = "NO-GO"
        reasoning = f"Coverage is {coverage_pct:.1f}%, below the 50% minimum threshold."

    print(f"DECISION: {decision}")
    print(f"REASONING: {reasoning}")
    print()

    if decision == "GO":
        print("✅ Insider data has sufficient coverage for M3.6 3-signal ensemble.")
        print("   Recommend proceeding with Priority 3.2 (ensemble wiring).")
    elif decision == "MARGINAL":
        print("⚠️  Insider data has marginal coverage.")
        print("   Consider proceeding but document limitations.")
    else:
        print("❌ Insider data coverage is insufficient for M3.6.")
        print("   Recommend marking M3.6 as 'DATA LIMITED - DEFERRED'.")

    print()
    print("=" * 80)

    return decision


if __name__ == "__main__":
    try:
        decision = main()
        sys.exit(0 if decision == "GO" else 1)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(2)
