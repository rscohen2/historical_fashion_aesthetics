import numpy as np


def monroe_logodds(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Monroe et al. (2008) Dirichlet-prior log-odds of each doc vs. all other docs.

    `mat` is a (docs × words) count matrix; the prior is its column sums.
    Returns (delta, sigma), both shaped like `mat`; the z-score is delta / sigma.
    """
    prior = mat.sum(axis=0)
    n_prior = float(prior.sum())
    g1 = mat
    g2 = prior - mat
    n1 = mat.sum(axis=1, keepdims=True)
    n2 = n_prior - n1

    odds1 = (g1 + prior) / ((n1 + n_prior) - (g1 + prior))
    odds2 = (g2 + prior) / ((n2 + n_prior) - (g2 + prior))
    delta = np.log(odds1) - np.log(odds2)
    sigma = np.sqrt(1.0 / (g1 + prior) + 1.0 / (g2 + prior))
    return delta, sigma
