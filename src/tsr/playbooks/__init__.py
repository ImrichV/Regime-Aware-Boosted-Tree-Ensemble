"""Public demonstration playbooks.

The production research archive contains additional frozen setup families whose exact
thresholds are intentionally not published in this public repository.  The demo
playbook shows the causal generator interface without disclosing proprietary rules.
"""

from .public_demo import PublicTrendPullbackDemoGenerator

__all__ = ["PublicTrendPullbackDemoGenerator"]
