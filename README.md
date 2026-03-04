# SIP Analysis App

## Continuous Improvement Checklist

- [ ] Track load and fit timing and optimize slowest path.
- [ ] Add more parser fixtures from real instruments.
- [ ] Add cancellation support for long bootstrap/MCMC jobs.
- [ ] Add fit-bound warnings in UI diagnostics panel.
- [ ] Add packaging scripts for macOS/Windows.
- [ ] Expand tests for model layer behavior.

## Feedback Loop

1. Identify user friction (errors, confusion, long waits).
2. Add defaults/guardrails/tooltips.
3. Add tests for each bug fix.
4. Measure runtime before and after optimization.
5. Refactor only after behavior is covered by tests.

## Debug Bundle

Use `Help -> Export Debug Bundle` in the app to export:

- environment snapshot
- settings
- dataset and fit counts

Share this bundle when reporting issues.
