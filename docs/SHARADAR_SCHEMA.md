# Sharadar Database Schema Documentation

**Source:** SignalTide v2 Database (`/Users/samuelksherman/signaltide/data/signaltide.db`)
**Date Documented:** 2025-11-18
**Purpose:** Reference for building SignalTide v3 data layer

---

## Overview

This document describes the Sharadar data tables used in SignalTide. The schema is taken from the production SignalTide v2 database and serves as the authoritative reference for v3 implementation.

**DO NOT** copy any Python code or signal logic from v2. This document is ONLY for understanding the database structure.

---

## Core Tables

### 1. sharadar_prices (Daily OHLCV Data)

**Purpose:** Daily price and volume data for all securities.

```sql
CREATE TABLE sharadar_prices(
  ticker TEXT,
  date NUM,
  open REAL,
  high REAL,
  low REAL,
  close REAL,
  volume REAL,
  closeadj REAL,         -- Adjusted close (splits/dividends)
  closeunadj REAL,       -- Unadjusted close
  lastupdated NUM
);

CREATE INDEX idx_sharadar_prices_ticker_date ON sharadar_prices(ticker, date);
CREATE INDEX idx_sharadar_prices_date ON sharadar_prices(date);
```

**Sample Data (AAPL):**
```
ticker | date       | open   | high   | low    | close  | volume
AAPL   | 2025-10-17 | 248.02 | 253.38 | 247.27 | 252.29 | 48876000
AAPL   | 2025-10-16 | 248.25 | 249.04 | 245.13 | 247.45 | 39698000
```

**Key Fields:**
- `date`: Date of trading day (YYYY-MM-DD format stored as NUM)
- `closeadj`: Use this for backtesting (accounts for splits/dividends)
- `lastupdated`: When this data was added to database (for point-in-time filtering)

---

### 2. sharadar_sf1 (Fundamental Data)

**Purpose:** Comprehensive fundamental metrics (income statement, balance sheet, cash flow, ratios).

This is Sharadar's **SF1** dataset - the core fundamental data.

```sql
CREATE TABLE sharadar_sf1(
  ticker TEXT,
  dimension TEXT,         -- ARQ, ARY, MRQ, MRY (frequency)
  calendardate TEXT,      -- When data was available (filing date)
  datekey TEXT,
  reportperiod TEXT,      -- Actual period the data covers
  fiscalperiod TEXT,      -- Q1, Q2, Q3, Q4, FY
  lastupdated TEXT,

  -- Income Statement
  revenue REAL,
  revenueusd REAL,
  cor REAL,               -- Cost of revenue
  grossmargin REAL,
  gp REAL,                -- Gross profit
  opex REAL,              -- Operating expenses
  opinc REAL,             -- Operating income
  netinc REAL,            -- Net income
  netinccmn REAL,         -- Net income common
  netinccmnusd REAL,
  eps REAL,
  epsdil REAL,            -- Diluted EPS
  epsusd REAL,

  -- Balance Sheet
  assets REAL,            -- Total assets
  assetsavg REAL,
  assetsc REAL,           -- Current assets
  assetsnc REAL,          -- Non-current assets
  cashneq REAL,           -- Cash and equivalents
  cashnequsd REAL,
  receivables REAL,
  inventory REAL,
  liabilities REAL,       -- Total liabilities
  liabilitiesc REAL,      -- Current liabilities
  liabilitiesnc REAL,     -- Non-current liabilities
  debt REAL,              -- Total debt
  debtc REAL,             -- Current debt
  debtnc REAL,            -- Non-current debt
  debtusd REAL,
  equity REAL,            -- Shareholders equity
  equityavg REAL,
  equityusd REAL,
  payables REAL,

  -- Cash Flow
  ncf REAL,               -- Net cash flow
  ncfo REAL,              -- Operating cash flow
  ncfi REAL,              -- Investing cash flow
  ncff REAL,              -- Financing cash flow
  fcf REAL,               -- Free cash flow
  fcfps REAL,             -- FCF per share
  capex REAL,             -- Capital expenditures
  depamor REAL,           -- Depreciation & amortization

  -- Ratios
  roa REAL,               -- Return on assets
  roe REAL,               -- Return on equity
  roic REAL,              -- Return on invested capital
  ros REAL,               -- Return on sales
  currentratio REAL,
  de REAL,                -- Debt to equity
  pb REAL,                -- Price to book
  pe REAL,                -- Price to earnings
  pe1 REAL,
  ps REAL,                -- Price to sales
  ps1 REAL,
  ebit REAL,
  ebitda REAL,
  ebitdamargin REAL,
  ebitdausd REAL,
  ebitusd REAL,
  marketcap REAL,
  ev REAL,                -- Enterprise value
  evebit REAL,
  evebitda REAL,

  -- Other
  divyield REAL,
  dps REAL,               -- Dividends per share
  workingcapital REAL,
  tangibles REAL,
  intangibles REAL,
  taxexp REAL,
  taxassets REAL,
  taxliabilities REAL,
  shareswa REAL,          -- Weighted average shares
  shareswadil REAL,       -- Diluted shares
  sharesbas REAL,         -- Basic shares

  -- Many more fields (120+ columns total)...
);

CREATE INDEX idx_sharadar_sf1_ticker_date ON sharadar_sf1(ticker, calendardate);
CREATE INDEX idx_sharadar_sf1_dimension ON sharadar_sf1(dimension);
```

**Dimension Field Values:**
- `ARQ`: As-Reported Quarterly
- `ARY`: As-Reported Annual (Yearly)
- `MRQ`: Most Recent Quarterly (restated)
- `MRY`: Most Recent Annual (restated)

**CRITICAL - Point-in-Time Access:**
- Use `calendardate` for point-in-time filtering (when we knew about it)
- Use `reportperiod` to know what period the data covers
- `calendardate` >= `reportperiod` (data becomes available after the period ends)

**Sample Data (AAPL ARQ):**
```
ticker | dimension | calendardate | reportperiod | revenue      | netinc       | assets       | equity
AAPL   | ARQ       | 2025-06-30   | 2025-06-28   | 94036000000  | 23434000000  | 331495000000 | 65830000000
AAPL   | ARQ       | 2025-03-31   | 2025-03-29   | 95359000000  | 24780000000  | 331233000000 | 66796000000
```

---

### 3. sharadar_insiders (Insider Trading)

**Purpose:** Corporate insider trading transactions (Form 4 filings).

```sql
CREATE TABLE sharadar_insiders (
  ticker TEXT,
  filingdate TIMESTAMP,           -- When Form 4 was filed (point-in-time!)
  formtype TEXT,                  -- Usually "4" or "4/A"
  issuername TEXT,                -- Company name
  ownername TEXT,                 -- Insider name
  officertitle TEXT,              -- CEO, CFO, Director, etc.
  isdirector TEXT,                -- "true" or "false"
  isofficer TEXT,
  istenpercentowner TEXT,
  transactiondate TIMESTAMP,      -- Actual trade date
  securityadcode TEXT,
  transactioncode TEXT,           -- P=Purchase, S=Sale, M=Option Exercise, etc.
  sharesownedbeforetransaction REAL,
  transactionshares INTEGER,      -- Number of shares traded
  sharesownedfollowingtransaction REAL,
  transactionpricepershare REAL,
  transactionvalue REAL,
  securitytitle TEXT,
  directorindirect TEXT,
  natureofownership TEXT,
  dateexercisable TIMESTAMP,
  priceexercisable REAL,
  expirationdate TIMESTAMP,
  rownum INTEGER
);

CREATE INDEX idx_sharadar_insiders_filing ON sharadar_insiders(filingdate);
CREATE INDEX idx_sharadar_insiders_ticker ON sharadar_insiders(ticker);
```

**Transaction Codes:**
- `P`: Purchase (bullish)
- `S`: Sale (bearish)
- `M`: Option exercise
- `A`: Award/grant
- Many others (see SEC Form 4 documentation)

**CRITICAL - Point-in-Time Access:**
- Use `filingdate` for point-in-time filtering (not `transactiondate`)
- Insiders have 2 business days to file after transaction
- `filingdate` > `transactiondate` (filing comes after trade)

**Sample Data (AAPL):**
```
ticker | filingdate | transactiondate | ownername    | officertitle | transactioncode | transactionshares | transactionpricepershare
AAPL   | 2025-10-17 | 2025-10-15      | PAREKH KEVAN | SVP CFO      | M               | -5111            |
```

---

### 4. sharadar_daily (Daily Metrics)

**Purpose:** Daily calculated metrics (valuation ratios, enterprise value).

```sql
CREATE TABLE sharadar_daily(
  ticker TEXT,
  date TEXT,
  lastupdated TEXT,
  ev REAL,              -- Enterprise value
  evebit REAL,          -- EV / EBIT
  evebitda REAL,        -- EV / EBITDA
  marketcap REAL,       -- Market capitalization
  pb REAL,              -- Price to book
  pe REAL,              -- Price to earnings (trailing)
  ps REAL               -- Price to sales
);

CREATE INDEX idx_sharadar_daily_ticker_date ON sharadar_daily(ticker, date);
```

**Note:** These are calculated daily using latest fundamentals + current price.

---

### 5. sharadar_tickers (Ticker Metadata)

**Purpose:** Company information, sector classification, listing status.

```sql
CREATE TABLE sharadar_tickers (
  table TEXT,
  permaticker INTEGER,        -- Permanent ticker ID
  ticker TEXT,
  name TEXT,                  -- Company name
  exchange TEXT,              -- NYSE, NASDAQ, etc.
  isdelisted TEXT,            -- "Y" or "N"
  category TEXT,              -- Domestic, Canadian, ADR, etc.
  cusips TEXT,
  siccode REAL,               -- SIC industry code
  sicsector TEXT,
  sicindustry TEXT,
  famasector TEXT,            -- Fama-French sector
  famaindustry TEXT,
  sector TEXT,                -- GICS-like sector
  industry TEXT,
  scalemarketcap TEXT,        -- 1-Nano to 6-Mega
  scalerevenue TEXT,
  relatedtickers TEXT,
  currency TEXT,
  location TEXT,              -- Country/state
  lastupdated TIMESTAMP,
  firstadded TIMESTAMP,
  firstpricedate TIMESTAMP,
  lastpricedate TIMESTAMP,
  firstquarter DATE,
  lastquarter DATE,
  secfilings TEXT,
  companysite TEXT
);

CREATE INDEX idx_tickers_ticker ON sharadar_tickers(ticker);
CREATE INDEX idx_tickers_category ON sharadar_tickers(category);
```

**Key Fields:**
- `isdelisted`: Use to filter active companies
- `category`: "Domestic" for US companies (most common filter)
- `scalemarketcap`: Market cap bucket (useful for universe construction)
- `sector`/`industry`: Use for sector-neutral signals

---

## Point-in-Time Access Rules

**CRITICAL FOR PREVENTING LOOKAHEAD BIAS**

### Prices (sharadar_prices)
```python
# Get prices as of date X:
# Use: date <= X AND lastupdated <= X
```

### Fundamentals (sharadar_sf1)
```python
# Get fundamentals as of date X:
# Use: calendardate <= X
# This ensures we only see data that was publicly filed by date X
```

### Insiders (sharadar_insiders)
```python
# Get insider trades as of date X:
# Use: filingdate <= X
# NOT transactiondate! Filing date is when we learned about it.
```

---

## Dimension Field (SF1 Fundamentals)

**When to use each:**

- **ARQ (As-Reported Quarterly)**: Use for signals. This is the original reported data.
- **ARY (As-Reported Annual)**: Use for annual metrics.
- **MRQ (Most Recent Quarterly)**: Restated data, can introduce lookahead bias.
- **MRY (Most Recent Annual)**: Restated data, can introduce lookahead bias.

**Recommendation:** Stick with ARQ/ARY for backtesting to avoid restatement lookahead.

---

## Data Coverage

Based on v2 database:
- **Price data**: Daily from ~2000 to present
- **Fundamental data**: Quarterly from ~1999 to present
- **Insider data**: Transactions from ~2005 to present
- **Universe**: ~15,000+ tickers (active + delisted)

---

## Common Queries

### Get latest price for ticker
```sql
SELECT * FROM sharadar_prices
WHERE ticker = 'AAPL'
ORDER BY date DESC
LIMIT 1;
```

### Get quarterly fundamentals (point-in-time)
```sql
SELECT * FROM sharadar_sf1
WHERE ticker = 'AAPL'
  AND dimension = 'ARQ'
  AND calendardate <= '2023-12-31'  -- As of this date
ORDER BY calendardate DESC
LIMIT 4;  -- Last 4 quarters
```

### Get insider buying in last 90 days (point-in-time)
```sql
SELECT * FROM sharadar_insiders
WHERE ticker = 'AAPL'
  AND filingdate BETWEEN '2023-10-01' AND '2023-12-31'
  AND transactioncode = 'P'  -- Purchases only
ORDER BY filingdate DESC;
```

### Get active US stocks with market cap > $1B
```sql
SELECT t.* FROM sharadar_tickers t
JOIN sharadar_daily d ON t.ticker = d.ticker
WHERE t.isdelisted = 'N'
  AND t.category = 'Domestic'
  AND d.marketcap > 1e9
  AND d.date = (SELECT MAX(date) FROM sharadar_daily);
```

---

## Implementation Notes for v3

1. **Database Location:** The v2 database is at `/Users/samuelksherman/signaltide/data/signaltide.db` (7.6GB)

2. **Read-Only Access:** v3 should connect read-only to this database. Do not modify it.

3. **Schema Creation:** For v3, you can:
   - Connect directly to v2 database (read-only)
   - OR: Copy relevant tables to a new v3 database
   - OR: Use mock data generator for testing first

4. **Indexes:** The existing indexes are optimized for typical queries. Add more if needed.

5. **Data Types:** Note that dates are stored as TEXT or NUM. Convert to datetime in Python.

6. **NULL Handling:** Many fields can be NULL. Handle appropriately in calculations.

---

## Next Steps

1. ✓ Document schema (this file)
2. Create simplified DataManager for v3 (read-only, basic caching)
3. Test with mock data first
4. Connect to real database after testing

---

## References

- Sharadar Data Documentation: https://www.quandl.com/databases/SF1/documentation
- SEC Form 4 Guide: https://www.sec.gov/files/form4data.pdf
- SIC Codes: https://www.sec.gov/info/edgar/siccodes.htm
