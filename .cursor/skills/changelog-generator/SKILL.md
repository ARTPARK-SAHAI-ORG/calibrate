---
name: changelog-generator
description: Generates changelogs from git history in Keep a Changelog format
---

# Changelog Generator

Generate changelogs following Keep a Changelog format.

## Instructions

1. Run `git log` to get recent commits
2. Group changes by type:
   - Added (new features)
   - Changed (changes in existing)
   - Deprecated
   - Removed
   - Fixed (bug fixes)
   - Security
3. Format in markdown with date headers
4. Write to CHANGELOG.md

## Format

```markdown
## [Version] - YYYY-MM-DD

### Added

- New feature description

### Fixed

- Bug fix description
```
