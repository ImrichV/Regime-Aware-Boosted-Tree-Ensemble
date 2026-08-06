from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


@dataclass(frozen=True)
class EnsembleConfig:
    """Fixed configuration for the public boosted-tree reference layer.

    The implementation intentionally exposes no hyperparameter search.  It is a
    reproducible architectural example that sits on top of the accepted research
    table contract.  Selection, tuning, portfolio construction, and live trading are
    outside its scope.
    """

    feature_columns: tuple[str, ...]
    regime_columns: tuple[str, ...]
    family_column: str = "setup_family"
    regression_target: str = "y_return"
    classification_target: str = "y_target"
    candidate_id_column: str = "candidate_id"
    minimum_family_rows: int = 200
    random_state: int = 260806
    learning_rate: float = 0.05
    max_iter: int = 150
    max_leaf_nodes: int = 15
    min_samples_leaf: int = 40
    l2_regularization: float = 1.0

    def validate(self) -> None:
        if not self.feature_columns:
            raise ValueError("feature_columns must not be empty")
        if not self.regime_columns:
            raise ValueError("regime_columns must not be empty")
        if len(set(self.feature_columns)) != len(self.feature_columns):
            raise ValueError("feature_columns contains duplicates")
        if len(set(self.regime_columns)) != len(self.regime_columns):
            raise ValueError("regime_columns contains duplicates")
        if self.minimum_family_rows < 20:
            raise ValueError("minimum_family_rows must be at least 20")


@dataclass
class _Specialist:
    regressor: Pipeline
    classifier: Pipeline
    row_count: int


def _make_regressor(config: EnsembleConfig) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            (
                "model",
                HistGradientBoostingRegressor(
                    learning_rate=config.learning_rate,
                    max_iter=config.max_iter,
                    max_leaf_nodes=config.max_leaf_nodes,
                    min_samples_leaf=config.min_samples_leaf,
                    l2_regularization=config.l2_regularization,
                    random_state=config.random_state,
                ),
            ),
        ]
    )


def _make_classifier(config: EnsembleConfig, seed_offset: int = 0) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            (
                "model",
                HistGradientBoostingClassifier(
                    learning_rate=config.learning_rate,
                    max_iter=config.max_iter,
                    max_leaf_nodes=config.max_leaf_nodes,
                    min_samples_leaf=config.min_samples_leaf,
                    l2_regularization=config.l2_regularization,
                    random_state=config.random_state + seed_offset,
                ),
            ),
        ]
    )


class RegimeAwareBoostedTreeEnsemble:
    """One boosted-tree specialist per setup family plus a global regime gate.

    The gate is trained only on the designated regime/context columns.  Each
    specialist is trained only on rows belonging to its setup family.  At inference
    time the final score is the specialist target probability multiplied by the
    probability that the current regime/context is favorable.

    This class is a public reference implementation.  It does not perform feature
    selection, hyperparameter search, cross-validation, threshold selection, trade
    execution, sizing, or portfolio optimization.
    """

    def __init__(self, config: EnsembleConfig) -> None:
        config.validate()
        self.config = config
        self._specialists: dict[str, _Specialist] = {}
        self._gate: Pipeline | None = None
        self._fitted = False

    @property
    def families(self) -> tuple[str, ...]:
        return tuple(sorted(self._specialists))

    def _required_columns(self) -> set[str]:
        c = self.config
        return {
            c.family_column,
            c.regression_target,
            c.classification_target,
            *c.feature_columns,
            *c.regime_columns,
        }

    def fit(self, frame: pd.DataFrame) -> "RegimeAwareBoostedTreeEnsemble":
        missing = self._required_columns().difference(frame.columns)
        if missing:
            raise ValueError(f"training frame missing columns: {sorted(missing)}")
        if len(frame) == 0:
            raise ValueError("training frame is empty")

        work = frame.copy()
        work = work[
            work[self.config.regression_target].notna()
            & work[self.config.classification_target].isin([0, 1])
            & work[self.config.family_column].notna()
        ]
        if len(work) == 0:
            raise ValueError("no rows remain after target validation")
        y_gate = work[self.config.classification_target].astype(int)
        if y_gate.nunique() < 2:
            raise ValueError("regime gate requires both target classes")

        self._gate = _make_classifier(self.config, seed_offset=1000)
        self._gate.fit(work.loc[:, self.config.regime_columns], y_gate)

        specialists: dict[str, _Specialist] = {}
        for family, family_frame in work.groupby(self.config.family_column, sort=True):
            if len(family_frame) < self.config.minimum_family_rows:
                continue
            y_class = family_frame[self.config.classification_target].astype(int)
            if y_class.nunique() < 2:
                continue
            regressor = _make_regressor(self.config)
            classifier = _make_classifier(self.config)
            x = family_frame.loc[:, self.config.feature_columns]
            regressor.fit(x, family_frame[self.config.regression_target].astype(float))
            classifier.fit(x, y_class)
            specialists[str(family)] = _Specialist(
                regressor=regressor,
                classifier=classifier,
                row_count=int(len(family_frame)),
            )

        if not specialists:
            raise ValueError("no setup family met the specialist fitting requirements")
        self._specialists = specialists
        self._fitted = True
        return self

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted or self._gate is None:
            raise RuntimeError("ensemble has not been fitted")
        c = self.config
        required = {c.family_column, *c.feature_columns, *c.regime_columns}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"prediction frame missing columns: {sorted(missing)}")

        gate_probability = self._gate.predict_proba(frame.loc[:, c.regime_columns])[:, 1]
        output = pd.DataFrame(index=frame.index)
        if c.candidate_id_column in frame:
            output[c.candidate_id_column] = frame[c.candidate_id_column].astype(str)
        output[c.family_column] = frame[c.family_column].astype(str)
        output["regime_gate_probability"] = gate_probability
        output["specialist_expected_return"] = np.nan
        output["specialist_target_probability"] = np.nan
        output["ensemble_score"] = np.nan
        output["model_status"] = "family_not_fitted"

        for family, specialist in self._specialists.items():
            mask = frame[c.family_column].astype(str).eq(family)
            if not mask.any():
                continue
            x = frame.loc[mask, c.feature_columns]
            expected_return = specialist.regressor.predict(x)
            target_probability = specialist.classifier.predict_proba(x)[:, 1]
            gated_score = target_probability * gate_probability[mask.to_numpy()]
            output.loc[mask, "specialist_expected_return"] = expected_return
            output.loc[mask, "specialist_target_probability"] = target_probability
            output.loc[mask, "ensemble_score"] = gated_score
            output.loc[mask, "model_status"] = "scored"

        return output.reset_index(drop=True)

    def specialist_metadata(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"setup_family": family, "training_rows": specialist.row_count}
                for family, specialist in sorted(self._specialists.items())
            ]
        )
