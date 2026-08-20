#!/usr/bin/env python3
"""Rediscretized vs Galerkin coarse operator, across flux (design 3.6).

The paper coarsens by REDISCRETIZATION -- the coarse link is the ordered
product of the two fine links it spans -- rather than by the Galerkin triple
product P^H L P. Lemma lem:alias is a strong structural argument AGAINST that
choice: at the aliasing flux the rediscretized coarse operator becomes
SINGULAR (its composed connection is a pure gauge) where the Galerkin one
stays positive definite. The paper's defence has been historical -- the
Galerkin variant of the 1990s parallel-transported multigrid lost to the
rediscretized one in its own comparisons -- plus a single measured factor at
the aliasing point. That is an argument by analogy to 1990s Dirac work, and
it should be settled on THIS operator.

So: same L, same P, same smoother, same alpha line search; only the coarse
operator differs. Two-grid asymptotic factors across the flux range, including
the aliasing point where the rediscretized operator is singular by Lemma
lem:alias, plus the cost each choice implies.

COST is the other half, and it is not symmetric. The rediscretized coarse
operator is O(N) to build from the links and keeps the coarse stencil at 7
points. The Galerkin product must be assembled, and on a structured lattice
with this P it FILLS IN: the coarse operator acquires entries the fine stencil
never had, so both the assembly and every subsequent coarse apply cost more,
on every level, for every solve. On a hierarchy rebuilt per frame or per
configuration -- the paper's stated regime -- that setup is paid every time.

Reported per flux: two-grid rho for each coarse operator, and the coarse
stencil's nonzeros per row, which is what the recursion multiplies.
"""
import numpy as np
import os

here = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(here, "verify_lower_bound.py")).read().split("def check(")[0])


def twogrid_rho(L, P, Lc, sweeps=2, ncyc=80, seed=0):
    """Asymptotic two-grid factor of the alpha-stabilized cycle.

    Identical to frelaxation_twogrid.py's, restated here so this script stands
    alone: red-black GS pre/post, coarse solve, energy-optimal step.
    """
    n2 = L.shape[0]
    r = np.random.default_rng(seed)
    b = r.normal(size=n2) + 1j * r.normal(size=n2)
    x = np.zeros(n2, complex)
    n = int(np.sqrt(n2))
    red = [i * n + j for i in range(n) for j in range(n) if (i + j) % 2 == 0]
    blk = [i * n + j for i in range(n) for j in range(n) if (i + j) % 2 == 1]
    D = np.diag(L).real

    def gs(x):
        for grp in (red, blk):
            x[grp] += (b[grp] - (L[grp, :] @ x)) / D[grp]
        return x

    res = []
    for _ in range(ncyc):
        for _ in range(sweeps):
            x = gs(x)
        rr = b - L @ x
        # Galerkin Lc can be singular-ish too near aliasing; lstsq is the
        # neutral choice so neither variant is advantaged by the solve.
        p = P @ np.linalg.lstsq(Lc, P.conj().T @ rr, rcond=None)[0]
        den = np.real(np.vdot(p, L @ p))
        if den > 0:
            x += (np.real(np.vdot(p, rr)) / den) * p
        for _ in range(sweeps):
            x = gs(x)
        res.append(np.linalg.norm(b - L @ x))
    res = np.array(res)
    good = res[res > 1e-13]
    if len(good) < 5:
        return 0.0
    tail = good[len(good) // 2:]
    if len(tail) < 3:
        return 0.0
    return float(np.median(tail[1:] / tail[:-1]))


def nnz_per_row(M, tol=1e-12):
    return float(np.mean(np.sum(np.abs(M) > tol, axis=1)))


def main():
    n = 16
    print("=== Rediscretized vs Galerkin coarse operator, 2D n=%d, uniform weights ===" % n)
    print("  same L, same P, same smoother, same line search; only L_c differs")
    print("  %-22s  %-9s %-9s   %-9s %-9s" %
          ("flux phi per plaquette", "rho redis", "rho Galer", "nnz redis", "nnz Galer"))
    wx = np.ones((n, n))
    wy = np.ones((n, n))
    # flux sweep: 0 (flat) up to and past the aliasing point phi = pi/2, where
    # the composed coarse connection is a pure gauge and L_c is singular.
    # Zero flux is excluded deliberately: the trivial connection has a constant
    # kernel, so both variants stagnate on it and the factor is 1 for reasons
    # that have nothing to do with the coarse operator.
    for num, den, tag in [(1, 64, "2pi/64 (weak)"), (1, 16, "2pi/16"), (1, 8, "2pi/8"),
                          (3, 16, "3*2pi/16"), (1, 4, "2pi/4 = ALIASING"),
                          (5, 16, "5*2pi/16"), (3, 8, "3*2pi/8")]:
        phi = 2.0 * np.pi * num / den
        thx = np.zeros((n, n))
        thy = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                thx[i, j] = -phi * j
        L = build(n, thx, thy, wx, wy)
        P = build_P(n, thx, thy, wx, wy)
        Lc_redis = coarse(n, thx, thy, wx, wy)
        Lc_galer = P.conj().T @ L @ P
        r_redis = twogrid_rho(L, P, Lc_redis)
        r_galer = twogrid_rho(L, P, Lc_galer)
        print("  %-22s  %-9.4f %-9.4f   %-9.2f %-9.2f" %
              (tag, r_redis, r_galer, nnz_per_row(Lc_redis), nnz_per_row(Lc_galer)))


if __name__ == "__main__":
    main()
