# Release Management Guide

## Creating a Release

1. **Update Version Numbers**
   ```bash
   # Update version in pyproject.toml
   vim pyproject.toml
   
   # Update version in __init__.py
   vim src/llm_ripper/__init__.py
   ```

2. **Update CHANGELOG.md**
   ```bash
   # Add new version section with changes
   vim CHANGELOG.md
   ```

3. **Create and Push Tag**
   ```bash
   git add .
   git commit -m "chore: bump version to v1.0.1"
   git tag -a v1.0.1 -m "Release version 1.0.1"
   git push origin main
   git push origin v1.0.1
   ```

4. **GitHub Release**
   - Go to: https://github.com/qrv0/LLM-Ripper/releases
   - Click "Create a new release"
   - Select the tag you just created
   - Fill in release notes
   - Upload any binary assets if needed

## Version Schema
We use Semantic Versioning (SemVer):
- MAJOR.MINOR.PATCH
- Example: 1.2.3
  - 1: Major version (breaking changes)
  - 2: Minor version (new features, backward compatible)
  - 3: Patch version (bug fixes)

## Release Notes Template
```markdown
## What's Changed
- Feature: Description of new feature
- Fix: Description of bug fix
- Docs: Documentation improvements
- Chore: Maintenance updates

## Breaking Changes
- List any breaking changes here

## New Contributors
- @username made their first contribution in #PR

**Full Changelog**: https://github.com/qrv0/LLM-Ripper/compare/v1.0.0...v1.0.1
```
