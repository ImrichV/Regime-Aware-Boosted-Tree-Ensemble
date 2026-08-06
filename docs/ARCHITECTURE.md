# Architecture

The system is intentionally modular. Each stage produces a versioned artifact with stable row keys, fingerprints, and timing rules.

## Flow

1. **Canonical data engine** — validates files, preserves source lineage, and quarantines invalid rows.
2. **Causal shared features** — calculates backward-looking stock, volatility, volume, path, and relative-context descriptors.
3. **Candidate framework** — creates permissive setup records without future labels.
4. **Outcome engine** — independently computes future returns, MFE/MAE, and barrier outcomes.
5. **Selection-safe research table** — aligns candidates, features, outcomes, and split metadata.
6. **Specialist boosted trees** — fit one model family per setup family.
7. **Regime gate** — estimates whether current market/context conditions are favorable.
8. **Fusion and abstention** — combine specialist and gate scores, calibrate confidence, and allow no-trade decisions.
9. **Later layers** — ranking, execution, exits, sizing, and portfolio controls.

## Non-destructive information flow

Candidate identity, raw features, family membership, outcomes, predictions, and portfolio decisions live in separate tables. A later model cannot rewrite the upstream record that generated it.

## Timing

Features and candidates may use the completed signal bar. Earliest entry is the next valid session open. Future outcome windows are stored only in the separate outcome layer. Training folds purge labels whose horizon crosses the test boundary.
