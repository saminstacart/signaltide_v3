---
description: Archive status documents older than 7 days
---

Automatically archive status-related markdown files that are stale or superseded by STATUS.md.

**Process:**
1. Check for status/update files in root with "Last Updated" older than 7 days
2. Move to `archive/status_reports_YYYY-MM/` with date suffix
3. Update DOCUMENTATION_MAP.md to reflect archived files
4. Confirm STATUS.md is up-to-date

**Files to check:**
- *STATE*.md
- *STATUS*.md
- *NEXT*.md
- *CURRENT*.md
- *UPDATE*.md
- *REPORT*.md (if not audit reports)

**Never archive:**
- README.md
- DOCUMENTATION_MAP.md
- RED_TEAM_AUDIT_REPORT.md
- PHASE_*_AUDIT_REPORT.md (audit reports are historical by design)
- docs/ directory files

**Archive naming:**
```
{FILENAME}_{YYYY-MM-DD}.md
```

Example: `CURRENT_STATE.md` → `archive/status_reports_2025-11/CURRENT_STATE_2025-11-19.md`

After archiving, list what was archived and confirm STATUS.md is the current source of truth.
