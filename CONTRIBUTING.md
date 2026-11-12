# Contributing to Doehlert Suite

Thank you for considering a contribution.

## Before opening a pull request

1. Open an issue for substantial changes so the design can be discussed first.
2. Keep statistical assumptions, formulas, numerical constraints, and units explicit.
3. Avoid changing the statistical engine and UI in the same patch unless necessary.
4. Add or update automated tests for behavior changes.
5. Update both English and Spanish documentation when user-visible behavior changes.

## Run checks

```bash
python -m compileall -q .
python -m unittest discover -s auditor/tests -v
```

## Pull request checklist

- [ ] The change has a clear description and rationale.
- [ ] Tests were added or updated where appropriate.
- [ ] English and Spanish user-facing documentation are synchronized.
- [ ] No private experimental data, generated exports, credentials, or local settings are included.
