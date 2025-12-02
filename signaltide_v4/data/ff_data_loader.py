"""
Fama-French Factor Data Loader.

Downloads real FF3/FF5 factors from Kenneth French's Data Library
and populates the ff_factors table in the database.

References:
    Fama, E.F. and French, K.R. (1993). "Common risk factors in the returns
    on stocks and bonds." Journal of Financial Economics, 33(1), 3-56.

    Fama, E.F. and French, K.R. (2015). "A five-factor asset pricing model."
    Journal of Financial Economics, 116(1), 1-22.
"""

import io
import logging
import sqlite3
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.request import urlopen

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Kenneth French Data Library URLs
FF3_DAILY_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"
FF5_DAILY_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"


class FFDataLoader:
    """
    Loader for Fama-French factor data from Kenneth French's Data Library.

    Downloads real academic factor data and populates database table.
    """

    def __init__(self, db_path: str):
        """
        Initialize the loader.

        Args:
            db_path: Path to the SQLite database
        """
        self.db_path = db_path

    def download_ff3_factors(self) -> pd.DataFrame:
        """
        Download FF3 daily factors from Kenneth French's website.

        Returns:
            DataFrame with columns: date, MKT-RF, SMB, HML, RF
        """
        logger.info("Downloading FF3 factors from Kenneth French's Data Library...")

        try:
            with urlopen(FF3_DAILY_URL, timeout=60) as response:
                zip_data = response.read()
        except Exception as e:
            logger.error(f"Failed to download FF3 data: {e}")
            raise

        # Extract CSV from zip (handle both .CSV and .csv extensions)
        with zipfile.ZipFile(io.BytesIO(zip_data)) as z:
            csv_names = [n for n in z.namelist() if n.lower().endswith('.csv')]
            if not csv_names:
                logger.error(f"No CSV found in zip. Contents: {z.namelist()}")
                raise ValueError(f"No CSV file in FF zip. Files: {z.namelist()}")
            csv_name = csv_names[0]
            with z.open(csv_name) as f:
                content = f.read().decode('utf-8')

        # Parse the CSV (skip header rows, find data start)
        lines = content.split('\n')
        data_start = None
        data_end = None

        for i, line in enumerate(lines):
            # Data starts after header with column names
            if line.strip().startswith('19') or line.strip().startswith('20'):
                if data_start is None:
                    data_start = i
                data_end = i
            elif data_start is not None and not line.strip():
                # Empty line after data
                data_end = i - 1
                break

        if data_start is None:
            raise ValueError("Could not find data in FF3 CSV")

        # Parse data rows
        data = []
        for line in lines[data_start:data_end + 1]:
            parts = line.strip().split(',')
            if len(parts) >= 5:
                try:
                    date_str = parts[0].strip()
                    if len(date_str) == 8:  # YYYYMMDD format
                        date = datetime.strptime(date_str, '%Y%m%d')
                        mkt_rf = float(parts[1]) / 100  # Convert from percentage
                        smb = float(parts[2]) / 100
                        hml = float(parts[3]) / 100
                        rf = float(parts[4]) / 100
                        data.append({
                            'date': date,
                            'MKT-RF': mkt_rf,
                            'SMB': smb,
                            'HML': hml,
                            'RF': rf
                        })
                except (ValueError, IndexError):
                    continue

        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()

        logger.info(f"Downloaded {len(df)} days of FF3 factors from {df.index.min()} to {df.index.max()}")

        return df

    def download_ff5_factors(self) -> pd.DataFrame:
        """
        Download FF5 daily factors from Kenneth French's website.

        Returns:
            DataFrame with columns: date, MKT-RF, SMB, HML, RMW, CMA, RF
        """
        logger.info("Downloading FF5 factors from Kenneth French's Data Library...")

        try:
            with urlopen(FF5_DAILY_URL, timeout=60) as response:
                zip_data = response.read()
        except Exception as e:
            logger.error(f"Failed to download FF5 data: {e}")
            raise

        # Extract CSV from zip (handle both .CSV and .csv extensions)
        with zipfile.ZipFile(io.BytesIO(zip_data)) as z:
            csv_names = [n for n in z.namelist() if n.lower().endswith('.csv')]
            if not csv_names:
                logger.error(f"No CSV found in zip. Contents: {z.namelist()}")
                raise ValueError(f"No CSV file in FF zip. Files: {z.namelist()}")
            csv_name = csv_names[0]
            with z.open(csv_name) as f:
                content = f.read().decode('utf-8')

        # Parse the CSV
        lines = content.split('\n')
        data_start = None
        data_end = None

        for i, line in enumerate(lines):
            if line.strip().startswith('19') or line.strip().startswith('20'):
                if data_start is None:
                    data_start = i
                data_end = i
            elif data_start is not None and not line.strip():
                data_end = i - 1
                break

        if data_start is None:
            raise ValueError("Could not find data in FF5 CSV")

        # Parse data rows
        data = []
        for line in lines[data_start:data_end + 1]:
            parts = line.strip().split(',')
            if len(parts) >= 7:
                try:
                    date_str = parts[0].strip()
                    if len(date_str) == 8:
                        date = datetime.strptime(date_str, '%Y%m%d')
                        mkt_rf = float(parts[1]) / 100
                        smb = float(parts[2]) / 100
                        hml = float(parts[3]) / 100
                        rmw = float(parts[4]) / 100
                        cma = float(parts[5]) / 100
                        rf = float(parts[6]) / 100
                        data.append({
                            'date': date,
                            'MKT-RF': mkt_rf,
                            'SMB': smb,
                            'HML': hml,
                            'RMW': rmw,
                            'CMA': cma,
                            'RF': rf
                        })
                except (ValueError, IndexError):
                    continue

        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()

        logger.info(f"Downloaded {len(df)} days of FF5 factors from {df.index.min()} to {df.index.max()}")

        return df

    def create_ff_table(self):
        """Create the ff_factors table if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ff_factors (
                date TEXT PRIMARY KEY,
                mkt_rf REAL NOT NULL,
                smb REAL NOT NULL,
                hml REAL NOT NULL,
                rmw REAL,
                cma REAL,
                rf REAL NOT NULL
            )
        """)

        # Create index for date range queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_ff_factors_date
            ON ff_factors(date)
        """)

        conn.commit()
        conn.close()
        logger.info("Created ff_factors table")

    def populate_ff_factors(self, use_ff5: bool = True) -> int:
        """
        Download and populate FF factors in database.

        Args:
            use_ff5: If True, download FF5 factors. Otherwise FF3.

        Returns:
            Number of rows inserted
        """
        # Create table first
        self.create_ff_table()

        # Download factors
        if use_ff5:
            factors = self.download_ff5_factors()
        else:
            factors = self.download_ff3_factors()
            factors['RMW'] = np.nan
            factors['CMA'] = np.nan

        # Insert into database
        conn = sqlite3.connect(self.db_path)

        # Clear existing data
        conn.execute("DELETE FROM ff_factors")

        # Prepare data for insert
        records = []
        for date, row in factors.iterrows():
            records.append((
                date.strftime('%Y-%m-%d'),
                row['MKT-RF'],
                row['SMB'],
                row['HML'],
                row.get('RMW', None),
                row.get('CMA', None),
                row['RF']
            ))

        # Batch insert
        conn.executemany(
            "INSERT INTO ff_factors (date, mkt_rf, smb, hml, rmw, cma, rf) VALUES (?, ?, ?, ?, ?, ?, ?)",
            records
        )

        conn.commit()
        conn.close()

        logger.info(f"Populated ff_factors table with {len(records)} rows")
        return len(records)

    def verify_data(self) -> dict:
        """
        Verify the FF factors data integrity.

        Returns:
            Dict with verification metrics
        """
        conn = sqlite3.connect(self.db_path)

        # Check row count
        row_count = conn.execute("SELECT COUNT(*) FROM ff_factors").fetchone()[0]

        # Check date range
        date_range = conn.execute(
            "SELECT MIN(date), MAX(date) FROM ff_factors"
        ).fetchone()

        # Check factor statistics (should have meaningful std devs)
        stats = conn.execute("""
            SELECT
                AVG(smb) as smb_mean,
                AVG(hml) as hml_mean,
                AVG(ABS(smb)) as smb_abs_mean,
                AVG(ABS(hml)) as hml_abs_mean
            FROM ff_factors
        """).fetchone()

        # Sample some data
        sample = conn.execute("""
            SELECT date, mkt_rf, smb, hml, rf
            FROM ff_factors
            WHERE date >= '2020-01-01'
            LIMIT 5
        """).fetchall()

        conn.close()

        result = {
            'row_count': row_count,
            'date_min': date_range[0],
            'date_max': date_range[1],
            'smb_mean': stats[0],
            'hml_mean': stats[1],
            'smb_abs_mean': stats[2],
            'hml_abs_mean': stats[3],
            'sample_data': sample,
            'is_valid': row_count > 5000 and stats[2] > 0.001  # SMB should have real variance
        }

        return result


def main():
    """Download and populate FF factors."""
    import os

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Get database path
    db_path = os.environ.get(
        'SIGNALTIDE_DB_PATH',
        '/Users/samuelksherman/signaltide/data/signaltide.db'
    )

    print(f"Using database: {db_path}")

    # Load data
    loader = FFDataLoader(db_path)

    print("\n=== DOWNLOADING FAMA-FRENCH FACTORS ===")
    n_rows = loader.populate_ff_factors(use_ff5=True)

    print(f"\n=== VERIFICATION ===")
    verification = loader.verify_data()

    print(f"Rows inserted: {verification['row_count']}")
    print(f"Date range: {verification['date_min']} to {verification['date_max']}")
    print(f"SMB mean: {verification['smb_mean']:.6f}")
    print(f"HML mean: {verification['hml_mean']:.6f}")
    print(f"SMB abs mean: {verification['smb_abs_mean']:.6f}")
    print(f"HML abs mean: {verification['hml_abs_mean']:.6f}")
    print(f"\nSample data (2020+):")
    for row in verification['sample_data']:
        print(f"  {row[0]}: MKT-RF={row[1]:.4f}, SMB={row[2]:.4f}, HML={row[3]:.4f}, RF={row[4]:.6f}")

    print(f"\nData valid: {verification['is_valid']}")

    return verification['is_valid']


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
