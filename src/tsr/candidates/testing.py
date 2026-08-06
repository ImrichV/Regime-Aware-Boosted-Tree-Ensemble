from __future__ import annotations

import numpy as np

from .base import CandidateContext, CandidateGenerator, CandidateSignals
from .schema import GeneratorMetadata


class FrameworkProbeGenerator(CandidateGenerator):
    """Deterministic causal test probe. It is not a trading playbook and cannot be published."""

    metadata = GeneratorMetadata(
        setup_family="framework_probe",
        setup_version="v1.0.0",
        direction="LONG",
        required_features=("return_20", "history_bars"),
        description="Acceptance-test probe selecting deterministic history positions; no alpha claim.",
        permissive=True,
    )

    def generate(self, context: CandidateContext) -> CandidateSignals:
        self.validate_context(context)
        modulus = int(self.config.get("modulus", 97))
        offset = int(self.config.get("offset", 0))
        if modulus < 2:
            raise ValueError("probe modulus must be at least 2")
        history = context.features["history_bars"].to_numpy(np.int64)
        ret20 = context.features["return_20"].to_numpy(np.float64)
        mask = (history >= 21) & np.isfinite(ret20) & (((history + offset) % modulus) == 0)
        strength = np.where(mask, ret20, np.nan)
        payload = {
            int(index): {"probe_modulus": modulus, "probe_offset": offset}
            for index in np.flatnonzero(mask)
        }
        return CandidateSignals(mask=mask.astype(bool), raw_setup_strength=strength, payload_by_row=payload)


class FutureLeakingProbeGenerator(CandidateGenerator):
    """Test-only intentionally invalid generator used to confirm prefix audits catch leakage."""

    metadata = GeneratorMetadata(
        setup_family="future_leaking_probe",
        setup_version="v1.0.0",
        direction="LONG",
        required_features=("return_1",),
        description="Intentionally invalid test fixture.",
    )

    def generate(self, context: CandidateContext) -> CandidateSignals:
        values = context.features["return_1"].to_numpy(np.float64)
        mask = np.zeros(len(values), dtype=bool)
        if len(values) > 1:
            mask[:-1] = np.nan_to_num(values[1:], nan=0.0) > 0
        return CandidateSignals(mask=mask)
