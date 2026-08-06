# Module Contracts

## Data

Canonical rows preserve instrument identity, ordered dates, OHLCV values, source member, and source-row lineage. Invalid rows are quarantined rather than silently repaired.

## Features

A feature row is keyed by instrument and signal date. Features use information available after the completed signal bar only. Rolling state may not cross continuity resets. Future labels and predictions are forbidden.

## Candidates

A candidate has a deterministic SHA-256-derived ID from schema version, setup family, setup version, instrument, signal date, and direction. Decision time is the signal close; earliest entry is the next valid session open. Candidate records may not contain future returns, labels, exits, ranks, or portfolio decisions.

## Outcomes

Outcome rows are stored separately and joined through candidate ID. They may contain multi-horizon returns, MFE, MAE, barrier outcomes, and split metadata. Outcome creation never mutates candidate records.

## Research

Research rows combine causal candidate lineage, signal-close features, and separately generated labels. Protected ticker partitions and protected temporal partitions are excluded from development research stores.

## Specialist ensemble

A specialist model is trained within one setup family. The regime gate consumes designated market/context columns separately. Predictions preserve candidate ID, family, model status, specialist probability, regime probability, and fused score. Unseen families are not silently mapped to another model.
