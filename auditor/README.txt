INTERNAL AUDIT AND TEST COMPONENTS

- audit_doehlert.py: integrated audit module loaded by the application's audit tab. Keep this file in this folder.
- run_extended_audit.py: runs the automated test suite and writes a JSON summary.
- tests/: maintainer tests for selected domain, optimization, multiple-response, and publication-comparison behavior.

The desktop application does not need the tests to run at startup. Execute them from the repository root with: python -m unittest discover -s auditor/tests -v
