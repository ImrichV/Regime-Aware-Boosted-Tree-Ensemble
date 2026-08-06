# Validation Policy

## Core rules

- use expanding or rolling walk-forward evaluation;
- purge label horizons across fold boundaries;
- generate downstream decisions from out-of-fold upstream predictions;
- protect deterministic ticker holdouts;
- keep a separately labelled temporal pseudo-lockbox;
- record every tested variant and retain failures;
- require stable neighbouring parameters and repeated unseen improvement;
- use next-open timing and pessimistic daily-bar ambiguity resolution;
- include transaction-cost and delay sabotage tests;
- separate candidate-definition acceptance from profitability evidence.

## Contamination warning

The wider research project has examined many periods through 2026. No period is claimed to be perfectly pristine. The original Stooq panel is survivor-dominated, so even a passing model remains a candidate for further validation, not production-ready evidence.
