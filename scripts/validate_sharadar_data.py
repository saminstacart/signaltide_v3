"""
Sharadar Database Validation Suite

Comprehensive validation of Sharadar data quality before migration.

Validates:
1. Schema integrity (tables, columns)
2. Temporal discipline (filing lags, datekey vs calendardate)
3. Data completeness (coverage, missing data)
4. Data quality (ranges, anomalies)
5. Survivorship bias (delisted stocks)
6. Point-in-time correctness

Grade: A++++ requires all checks to pass
"""

import sqlite3
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import get_logger
from core.db import get_read_only_connection

logger = get_logger(__name__)


class SharadarDataValidator:
    """Comprehensive Sharadar database validator."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.conn = None
        self.issues = []
        self.warnings = []

    def connect(self):
        """Connect to database using centralized core.db helper."""
        self.conn = get_read_only_connection(db_path=self.db_path)
        logger.info(f"Connected to database: {self.db_path}")
        logger.info(f"Database size: {self.db_path.stat().st_size / 1e9:.2f} GB")

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()

    def validate_schema(self):
        """Validate database schema."""
        logger.info("\n" + "="*60)
        logger.info("1. SCHEMA VALIDATION")
        logger.info("="*60)

        # Check required tables
        required_tables = {
            'sharadar_prices': ['ticker', 'date', 'open', 'high', 'low', 'close', 'volume', 'closeadj', 'lastupdated'],
            'sharadar_sf1': ['ticker', 'dimension', 'calendardate', 'datekey', 'reportperiod', 'lastupdated', 'revenue', 'netinc', 'equity', 'roe'],
            'sharadar_insiders': ['ticker', 'filingdate', 'transactiondate', 'ownername', 'transactioncode', 'transactionshares', 'transactionpricepershare'],
            'sharadar_tickers': ['ticker', 'name', 'category', 'isdelisted']
        }

        cursor = self.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}

        for table, columns in required_tables.items():
            if table not in tables:
                self.issues.append(f"Missing required table: {table}")
                continue

            # Check columns
            cursor.execute(f"PRAGMA table_info({table})")
            existing_cols = {row[1] for row in cursor.fetchall()}

            missing_cols = set(columns) - existing_cols
            if missing_cols:
                self.issues.append(f"Table {table} missing columns: {missing_cols}")
            else:
                logger.info(f"✅ {table}: All {len(columns)} required columns present")

        # Count rows
        for table in required_tables.keys():
            if table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                logger.info(f"   - {count:,} rows")

    def validate_temporal_discipline(self):
        """Validate filing lag and date consistency."""
        logger.info("\n" + "="*60)
        logger.info("2. TEMPORAL DISCIPLINE VALIDATION")
        logger.info("="*60)

        # Check fundamentals: datekey should be >= reportperiod (filing after period end)
        query = """
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN datekey < reportperiod THEN 1 ELSE 0 END) as violations,
                AVG(julianday(datekey) - julianday(reportperiod)) as avg_lag_days
            FROM sharadar_sf1
            WHERE dimension = 'ARQ'
        """

        df = pd.read_sql_query(query, self.conn)
        total = df['total'].iloc[0]
        violations = df['violations'].iloc[0]
        avg_lag = df['avg_lag_days'].iloc[0]

        logger.info(f"Fundamentals (ARQ):")
        logger.info(f"  Total records: {total:,}")
        logger.info(f"  Average filing lag: {avg_lag:.1f} days")

        if violations > 0:
            self.issues.append(f"CRITICAL: {violations} records where datekey < reportperiod")
        else:
            logger.info(f"  ✅ No temporal violations (datekey >= reportperiod)")

        # Check expected lag (should be ~33 days for fundamentals)
        if avg_lag < 20:
            self.warnings.append(f"Filing lag suspiciously low: {avg_lag:.1f} days (expected ~33)")
        elif avg_lag > 50:
            self.warnings.append(f"Filing lag suspiciously high: {avg_lag:.1f} days (expected ~33)")
        else:
            logger.info(f"  ✅ Filing lag within expected range (20-50 days)")

        # Check insider trades: filingdate should be >= transactiondate
        query = """
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN filingdate < transactiondate THEN 1 ELSE 0 END) as violations,
                AVG(julianday(filingdate) - julianday(transactiondate)) as avg_lag_days
            FROM sharadar_insiders
        """

        df = pd.read_sql_query(query, self.conn)
        total = df['total'].iloc[0]
        violations = df['violations'].iloc[0]
        avg_lag = df['avg_lag_days'].iloc[0]

        logger.info(f"\nInsider Trades:")
        logger.info(f"  Total records: {total:,}")
        logger.info(f"  Average filing lag: {avg_lag:.1f} days")

        if violations > 0:
            self.issues.append(f"CRITICAL: {violations} insider records where filingdate < transactiondate")
        else:
            logger.info(f"  ✅ No temporal violations (filingdate >= transactiondate)")

        # Check expected lag (should be 1-2 days for insider trades)
        if avg_lag < 0.5:
            self.warnings.append(f"Insider filing lag suspiciously low: {avg_lag:.1f} days")
        elif avg_lag > 5:
            self.warnings.append(f"Insider filing lag suspiciously high: {avg_lag:.1f} days")
        else:
            logger.info(f"  ✅ Insider filing lag within expected range (0.5-5 days)")

    def validate_data_completeness(self):
        """Validate data completeness and coverage."""
        logger.info("\n" + "="*60)
        logger.info("3. DATA COMPLETENESS VALIDATION")
        logger.info("="*60)

        # Check price data coverage
        query = """
            SELECT
                MIN(date) as min_date,
                MAX(date) as max_date,
                COUNT(DISTINCT ticker) as num_tickers,
                COUNT(*) as total_rows
            FROM sharadar_prices
        """
        df = pd.read_sql_query(query, self.conn)

        logger.info(f"Price Data:")
        logger.info(f"  Date range: {df['min_date'].iloc[0]} to {df['max_date'].iloc[0]}")
        logger.info(f"  Tickers: {df['num_tickers'].iloc[0]:,}")
        logger.info(f"  Total rows: {df['total_rows'].iloc[0]:,}")

        # Check for critical NULLs in prices
        query = """
            SELECT
                SUM(CASE WHEN close IS NULL THEN 1 ELSE 0 END) as null_close,
                SUM(CASE WHEN volume IS NULL THEN 1 ELSE 0 END) as null_volume
            FROM sharadar_prices
        """
        df = pd.read_sql_query(query, self.conn)

        if df['null_close'].iloc[0] > 0:
            self.issues.append(f"CRITICAL: {df['null_close'].iloc[0]} NULL close prices")
        else:
            logger.info(f"  ✅ No NULL close prices")

        # Check fundamentals coverage
        query = """
            SELECT
                MIN(calendardate) as min_date,
                MAX(calendardate) as max_date,
                COUNT(DISTINCT ticker) as num_tickers,
                COUNT(*) as total_rows
            FROM sharadar_sf1
            WHERE dimension = 'ARQ'
        """
        df = pd.read_sql_query(query, self.conn)

        logger.info(f"\nFundamentals (ARQ):")
        logger.info(f"  Date range: {df['min_date'].iloc[0]} to {df['max_date'].iloc[0]}")
        logger.info(f"  Tickers: {df['num_tickers'].iloc[0]:,}")
        logger.info(f"  Total rows: {df['total_rows'].iloc[0]:,}")

        # Check for critical NULLs in fundamentals
        query = """
            SELECT
                SUM(CASE WHEN revenue IS NULL THEN 1 ELSE 0 END) as null_revenue,
                SUM(CASE WHEN equity IS NULL THEN 1 ELSE 0 END) as null_equity,
                SUM(CASE WHEN datekey IS NULL THEN 1 ELSE 0 END) as null_datekey
            FROM sharadar_sf1
            WHERE dimension = 'ARQ'
        """
        df = pd.read_sql_query(query, self.conn)

        if df['null_datekey'].iloc[0] > 0:
            self.issues.append(f"CRITICAL: {df['null_datekey'].iloc[0]} NULL datekey values")
        else:
            logger.info(f"  ✅ No NULL datekey values (required for point-in-time)")

    def validate_survivorship_bias(self):
        """Validate delisted stocks are included."""
        logger.info("\n" + "="*60)
        logger.info("4. SURVIVORSHIP BIAS VALIDATION")
        logger.info("="*60)

        # Query for US domestic stocks (multiple category patterns)
        query = """
            SELECT
                COUNT(*) as total_tickers,
                SUM(CASE WHEN isdelisted = 'Y' THEN 1 ELSE 0 END) as delisted,
                SUM(CASE WHEN isdelisted = 'N' THEN 1 ELSE 0 END) as active
            FROM sharadar_tickers
            WHERE category LIKE 'Domestic%'
        """

        df = pd.read_sql_query(query, self.conn)
        total = df['total_tickers'].iloc[0] or 0
        delisted = df['delisted'].iloc[0] or 0
        active = df['active'].iloc[0] or 0

        logger.info(f"Domestic Tickers:")
        logger.info(f"  Total: {total:,}")

        if total > 0:
            logger.info(f"  Active: {active:,} ({active/total*100:.1f}%)")
            logger.info(f"  Delisted: {delisted:,} ({delisted/total*100:.1f}%)")

            if delisted == 0:
                self.issues.append("CRITICAL: No delisted stocks found (survivorship bias!)")
            elif delisted < total * 0.10:  # Less than 10% delisted
                self.warnings.append(f"Low delisted percentage: {delisted/total*100:.1f}% (expected 15-25%)")
            else:
                logger.info(f"  ✅ Delisted stocks included (prevents survivorship bias)")
        else:
            # No domestic category - check all tickers
            logger.info("  No 'Domestic' category found, checking all tickers...")
            query = """
                SELECT
                    COUNT(*) as total_tickers,
                    SUM(CASE WHEN isdelisted = 'Y' THEN 1 ELSE 0 END) as delisted,
                    SUM(CASE WHEN isdelisted = 'N' THEN 1 ELSE 0 END) as active
                FROM sharadar_tickers
            """
            df = pd.read_sql_query(query, self.conn)
            total = df['total_tickers'].iloc[0] or 0
            delisted = df['delisted'].iloc[0] or 0
            active = df['active'].iloc[0] or 0

            logger.info(f"  Total (all categories): {total:,}")
            if total > 0:
                logger.info(f"  Active: {active:,} ({active/total*100:.1f}%)")
                logger.info(f"  Delisted: {delisted:,} ({delisted/total*100:.1f}%)")

                if delisted == 0:
                    self.issues.append("CRITICAL: No delisted stocks found (survivorship bias!)")
                elif delisted < total * 0.10:
                    self.warnings.append(f"Low delisted percentage: {delisted/total*100:.1f}% (expected 15-25%)")
                else:
                    logger.info(f"  ✅ Delisted stocks included (prevents survivorship bias)")

        # Check if delisted stocks have price data
        query = """
            SELECT COUNT(DISTINCT p.ticker) as delisted_with_prices
            FROM sharadar_prices p
            INNER JOIN sharadar_tickers t ON p.ticker = t.ticker
            WHERE t.isdelisted = 'Y'
        """
        df = pd.read_sql_query(query, self.conn)
        delisted_with_prices = df['delisted_with_prices'].iloc[0] or 0

        if delisted > 0:
            logger.info(f"  Delisted stocks with price data: {delisted_with_prices:,}")

            if delisted_with_prices < delisted * 0.5:
                self.warnings.append(f"Many delisted stocks missing price data: {delisted_with_prices}/{delisted}")

    def validate_data_quality(self):
        """Validate data quality and ranges."""
        logger.info("\n" + "="*60)
        logger.info("5. DATA QUALITY VALIDATION")
        logger.info("="*60)

        # Check for price anomalies
        query = """
            SELECT
                SUM(CASE WHEN close <= 0 THEN 1 ELSE 0 END) as negative_prices,
                SUM(CASE WHEN high < low THEN 1 ELSE 0 END) as high_lt_low,
                SUM(CASE WHEN close > high OR close < low THEN 1 ELSE 0 END) as close_out_of_range,
                SUM(CASE WHEN volume < 0 THEN 1 ELSE 0 END) as negative_volume
            FROM sharadar_prices
        """

        df = pd.read_sql_query(query, self.conn)

        logger.info(f"Price Data Quality:")
        anomalies = 0
        for col in df.columns:
            count = df[col].iloc[0]
            if count > 0:
                self.issues.append(f"Price anomaly: {count} rows with {col}")
                anomalies += count
            else:
                logger.info(f"  ✅ No {col} anomalies")

        if anomalies == 0:
            logger.info(f"  ✅ All price data within valid ranges")

        # Check for fundamental anomalies
        query = """
            SELECT
                SUM(CASE WHEN equity <= 0 AND equity IS NOT NULL THEN 1 ELSE 0 END) as negative_equity,
                SUM(CASE WHEN roe < -5 OR roe > 5 THEN 1 ELSE 0 END) as extreme_roe
            FROM sharadar_sf1
            WHERE dimension = 'ARQ'
        """

        df = pd.read_sql_query(query, self.conn)

        logger.info(f"\nFundamentals Data Quality:")

        neg_equity = df['negative_equity'].iloc[0]
        if neg_equity > 0:
            self.warnings.append(f"{neg_equity} companies with negative equity (may be valid)")
            logger.info(f"  ⚠️  {neg_equity} rows with negative equity (may be bankruptcies)")

        extreme_roe = df['extreme_roe'].iloc[0]
        if extreme_roe > 0:
            logger.info(f"  ⚠️  {extreme_roe} rows with extreme ROE (|ROE| > 500%)")
        else:
            logger.info(f"  ✅ All ROE values in reasonable range")

    def generate_report(self):
        """Generate final validation report."""
        logger.info("\n" + "="*60)
        logger.info("VALIDATION SUMMARY")
        logger.info("="*60)

        if not self.issues and not self.warnings:
            logger.info("\n🎉 A++++ DATA QUALITY ACHIEVED!")
            logger.info("\n✅ All validation checks passed")
            logger.info("✅ No critical issues found")
            logger.info("✅ No warnings")
            logger.info("\nDatabase is ready for production use!")
            return "A++++"

        if not self.issues and self.warnings:
            logger.info("\n✅ A+++ DATA QUALITY")
            logger.info(f"\n✅ No critical issues")
            logger.info(f"⚠️  {len(self.warnings)} warnings:")
            for i, warning in enumerate(self.warnings, 1):
                logger.info(f"   {i}. {warning}")
            logger.info("\nDatabase is acceptable for production use.")
            return "A+++"

        if self.issues:
            logger.info("\n❌ DATA QUALITY ISSUES FOUND")
            logger.info(f"\n❌ {len(self.issues)} critical issues:")
            for i, issue in enumerate(self.issues, 1):
                logger.info(f"   {i}. {issue}")

            if self.warnings:
                logger.info(f"\n⚠️  {len(self.warnings)} warnings:")
                for i, warning in enumerate(self.warnings, 1):
                    logger.info(f"   {i}. {warning}")

            logger.info("\n⛔ Database NOT ready for production!")
            logger.info("Fix critical issues before using.")
            return "FAIL"

    def run_all_validations(self):
        """Run all validation checks."""
        try:
            self.connect()

            self.validate_schema()
            self.validate_temporal_discipline()
            self.validate_data_completeness()
            self.validate_survivorship_bias()
            self.validate_data_quality()

            grade = self.generate_report()

            return grade

        finally:
            self.close()


def main():
    """Run validation on Sharadar database."""
    print("\n" + "="*60)
    print("SHARADAR DATABASE VALIDATION SUITE")
    print("="*60)
    print("\nValidating: /Users/samuelksherman/signaltide/data/signaltide.db")
    print("Target Grade: A++++")
    print("="*60)

    db_path = Path("/Users/samuelksherman/signaltide/data/signaltide.db")

    validator = SharadarDataValidator(db_path)
    grade = validator.run_all_validations()

    print(f"\n{'='*60}")
    print(f"FINAL GRADE: {grade}")
    print(f"{'='*60}\n")

    return grade


if __name__ == '__main__':
    grade = main()
    sys.exit(0 if grade in ["A++++", "A+++"] else 1)
