"""Classical / geometric pipelines. §8.3, §B.3.

Uses pyRiemann and scikit-learn; do not re-implement standard estimators.
Stage 1: CSP+LDA and Cov→TangentSpace→LR.
Stage 2 and 3 comparators (xDAWN+Tangent+LDA) are extension points — not implemented.
"""
from __future__ import annotations

from sklearn.pipeline import Pipeline


def build_mi_csp_pipeline(
    n_components: int = 6,
    classifier: str = "lda",
    tangent_space: bool = False,
) -> Pipeline:
    """Build a CSP+LDA or Cov→TangentSpace→LR sklearn Pipeline. §B.3.

    n_components: number of CSP spatial filters.
    classifier: 'lda' or 'logreg'.
    tangent_space: if True, use Cov→TangentSpace→LR instead of CSP.
    Returns a fitted-able sklearn Pipeline.
    """
    raise NotImplementedError("Milestone 4")


# Extension point for Stage 2/3 — not implemented in Stage 1.
# def build_p300_riemann_pipeline(...) -> Pipeline: ...
