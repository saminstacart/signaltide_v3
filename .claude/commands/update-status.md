---
description: Update STATUS.md with current project state
---

Please update STATUS.md to reflect the current state of the project. Follow these rules:

1. **Update "Last Updated"** timestamp to now
2. **Update "Phase"** to current phase
3. **Update "Just Completed"** section:
   - List only items completed in the last session
   - Include key metrics and evidence
4. **Update "What's Next"** section:
   - List the immediate next task
   - Include task breakdown and success criteria
5. **Update "Metrics"** if significant changes
6. **Update "Recent Git Commits"** with latest 4 commits

**Key Principles:**
- STATUS.md is the SINGLE SOURCE OF TRUTH
- Keep it concise (under 300 lines)
- Archive anything over 7 days old
- Include only actionable next steps
- Evidence over claims (git commits, test results, file paths)

**What to check:**
- Latest git commit hash and message
- Test pass rates (run `python3 -m pytest tests/ -v --tb=no`)
- File counts (`find . -name "*.py" -type f | wc -l`)
- Database size (`ls -lh data/databases/market_data.db`)

**Format:**
- Use emoji sparingly (✅ ❌ ⚠️ only)
- Dates: YYYY-MM-DD HH:MM format
- Metrics: Include numbers with context
- Links: Use relative paths to files

After updating, read the updated STATUS.md back to confirm accuracy.
