"""
Helpers for the dynamics grid-search experiment.

Pure functions plus a single-cell runner. No plotting, no notebook-only logic.
"""

from abm_dynamics import (
    Mechanism,
    mechanism_homophily,
    mechanism_triadic_closure,
    mechanism_popularity,
)


def build_mechanisms(
    b_homophily: float,
    b_triadic: float,
    b_popularity: float,
) -> list[Mechanism]:
    """Return mechanism closures for non-zero coefficients only.

    Mechanisms with coefficient 0 are omitted entirely (cleaner than
    contributing zero log-odds). Closures bind the coefficient at
    definition time via default-argument capture to avoid late-binding bugs.
    """
    mechs: list[Mechanism] = []
    if b_homophily > 0:
        mechs.append(lambda s, b=b_homophily: mechanism_homophily(s, b))
    if b_triadic > 0:
        mechs.append(lambda s, b=b_triadic: mechanism_triadic_closure(s, b))
    if b_popularity > 0:
        mechs.append(lambda s, b=b_popularity: mechanism_popularity(s, b))
    return mechs
