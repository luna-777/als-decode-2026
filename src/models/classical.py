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
    """Build a CSP+LDA or Cov→TangentSpace→LR sklearn Pipeline for Stage 1. §B.3.

    n_components : number of CSP spatial filters (ignored when tangent_space=True).
    classifier   : 'lda' or 'logreg' (only used in CSP path).
    tangent_space: if True, use Cov→TangentSpace (Riemannian) → LogReg instead of CSP.

    The returned Pipeline accepts (n_epochs, n_channels, n_times) arrays and fits
    in a standard sklearn manner (fit / predict / predict_proba).
    """
    if tangent_space:
        from pyriemann.estimation import Covariances
        from pyriemann.tangentspace import TangentSpace
        from sklearn.linear_model import LogisticRegression

        return Pipeline(
            [
                ("cov", Covariances(estimator="lwf")),
                ("ts", TangentSpace(metric="riemann")),
                ("clf", LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)),
            ]
        )

    # CSP path — uses MNE's sklearn-compatible CSP
    from mne.decoding import CSP
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.linear_model import LogisticRegression

    csp = CSP(
        n_components=n_components,
        reg="ledoit_wolf",
        log=True,
        norm_trace=True,
    )
    if classifier == "logreg":
        clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    else:
        clf = LinearDiscriminantAnalysis()

    return Pipeline([("csp", csp), ("clf", clf)])


# Extension point for Stage 2/3 — not implemented in Stage 1.
# def build_p300_riemann_pipeline(...) -> Pipeline: ...
