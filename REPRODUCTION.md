# Reproduction

## Public software verification

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e . --no-build-isolation
python -m pip install pytest
pytest -q
python -m compileall -q src
rabte doctor --project-root .
```

## Synthetic boosted-tree demonstration

```bash
python examples/synthetic_specialist_ensemble.py
```

## Real-data pipeline

The public configs assume a legally obtained Stooq daily-data ZIP one directory above the repository. The data archive and generated artifacts are not distributed here. Start with dry runs and inspect every manifest before publication.

```bash
rabte data audit --config configs/data.example.yaml --dry-run
rabte features build --config configs/features.example.yaml --dry-run
rabte candidates build --config configs/public_demo_candidate.yaml --dry-run
```

The public candidate generator is deliberately illustrative and must not be interpreted as a validated trading system.
