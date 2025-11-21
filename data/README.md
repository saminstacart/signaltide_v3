# Data Layout

This directory contains data files and databases for the Signal Tide v3 trading system.

## Directory Structure

- **`data/databases/`** – SQLite databases (Sharadar / market data)
  - These are large files (multi-GB) and are **NOT** checked into git
  - Database path is configured via `SIGNALTIDE_DB_PATH` environment variable
  - See `docs/core/DATA_ARCHITECTURE.md` for schema details

- **`data/raw/`** – Raw downloaded files (CSV, ZIP, etc.)
  - Original data files before processing
  - Not checked into git

- **`data/intermediate/`** – Temporary or derived artifacts
  - Computed features, cached results
  - Not checked into git

## Database Setup

The main database is typically located at:
```
/Users/samuelksherman/signaltide/data/signaltide.db
```

Set the environment variable:
```bash
export SIGNALTIDE_DB_PATH=/Users/samuelksherman/signaltide/data/signaltide.db
```

Rebuild instructions and schema documentation are in `docs/core/DATA_ARCHITECTURE.md`.

## .gitignore

All data files are excluded from git via `.gitignore` patterns:
```gitignore
data/databases/*.db
data/raw/**
data/intermediate/**
```

This keeps the repository lightweight and fast while maintaining clean separation of code and data.
