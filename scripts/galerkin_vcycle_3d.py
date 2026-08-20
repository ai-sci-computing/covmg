#!/usr/bin/env python3
"""Rediscretized vs Galerkin as a MULTILEVEL V-CYCLE, not a two-grid factor.

The companion script galerkin_vs_rediscretized_3d.py answers the design
question with (a) measured two-grid convergence factors and (b) a MODEL for the
time, since it never runs a V-cycle and never times anything. A review objected
that the paper raises its own stakes here -- "if Galerkin also solved faster,
our choice would be indefensible in a paper about speed" -- and then settles it
with a model. This script removes the model.

WHAT IS COMPARED. The transfers are geometric, so both hierarchies use exactly
the same prolongations, exported from the library's own prolongGauge (never
rebuilt by hand: an earlier attempt implemented only pass 1 of the covariant
fill and silently produced a P with 7 nonzeros per coarse column instead of
27). The two hierarchies differ ONLY in the coarse operator:

    rediscretized:  L_{l+1} = the connection Laplacian of the coarsened lattice
    Galerkin:       L_{l+1} = P_l^H L_l P_l          (recursively, so the
                                                      Galerkin-of-Galerkin
                                                      fill-in is real, not
                                                      assumed away)

Same smoother (2+2 red-black Gauss-Seidel), same energy line search on the
coarse correction, same coarsest-level direct solve, same right-hand side.

The coarse flux bookkeeping: phi -> 4 phi per level while n -> n/2, so
n_Phi = phi n^2 / 2pi is INVARIANT under coarsening. That is why the coarse
rediscretized operator of (n=16, n_Phi) is exactly (n=8, n_Phi).

WHAT IS MEASURED HERE. Iterations to a relative residual of 1e-8; the nonzeros
per row at every level (so the Galerkin-of-Galerkin fill is visible rather than
assumed); and the per-apply cost of each hierarchy, timed on the actual sparse
matvecs. Work is then iterations x per-cycle apply cost, both measured. Setup
is still excluded, and that continues to favour Galerkin -- rediscretization is
one closed-form O(N) pass over the links, Galerkin is a triple product per
level, paid on every rebuild in the regime this paper targets.

Inputs (from the repository root, with build/tools on the path):
    for NP in 1 2 4 8; do
      export_operator u1      16 $NP L16_$NP.mtx
      export_operator u1       8 $NP L8_$NP.mtx
      export_operator u1       4 $NP L4_$NP.mtx
      export_operator prolong 16 $NP P16_$NP.mtx
      export_operator prolong  8 $NP P8_$NP.mtx
    done
"""
import sys
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl


def rd(path):
    """Read a complex MatrixMarket file written by tools/export_operator."""
    with open(path) as f:
        lines = [l for l in f if not l.startswith('%')]
    m, n, nnz = map(int, lines[0].split())
    r = np.empty(nnz, int)
    c = np.empty(nnz, int)
    v = np.empty(nnz, complex)
    for t, l in enumerate(lines[1:nnz + 1]):
        a = l.split()
        r[t] = int(a[0]) - 1
        c[t] = int(a[1]) - 1
        v[t] = float(a[2]) + 1j * float(a[3])
    return sp.csr_matrix(sp.coo_matrix((v, (r, c)), shape=(m, n)))


def colors(n):
    """Red-black index sets for an n^3 lattice in the export's ordering."""
    idx = np.arange(n * n * n)
    i, j, k = idx // (n * n), (idx // n) % n, idx % n
    par = (i + j + k) % 2
    return idx[par == 0], idx[par == 1]


def vcycle(levels, cols, b, x, nu=2):
    """One V-cycle. levels[l] is the operator, cols[l] its red-black split.

    The coarse correction carries the energy line search, exactly as the
    shipped solver does -- without it the non-Galerkin cycle has no descent
    guarantee, so omitting it here would compare the two coarse operators
    under a scheme neither method uses.
    """
    L = levels[l0 := 0] if False else levels[0]

    def rec(l, x, b):
        A = levels[l]
        red, blk = cols[l]
        D = A.diagonal().real
        if l + 1 == len(levels):
            return spl.spsolve(sp.csc_matrix(A), b)
        for _ in range(nu):
            for g in (red, blk):
                x[g] += (b[g] - A[g, :] @ x) / D[g]
        r = b - A @ x
        P = Ps[l]
        rc = P.conj().T @ r
        ec = rec(l + 1, np.zeros(A_shapes[l + 1], complex), rc)
        p = P @ ec
        den = np.real(np.vdot(p, A @ p))
        if den > 0:
            x = x + (np.real(np.vdot(p, r)) / den) * p
        for _ in range(nu):
            for g in (red, blk):
                x[g] += (b[g] - A[g, :] @ x) / D[g]
        return x

    return rec(0, x, b)


def solve(levels, cols, tol=1e-8, cap=200, seed=0):
    """V-cycles to `tol`; returns (iterations, seconds)."""
    N = levels[0].shape[0]
    rng = np.random.default_rng(seed)
    b = rng.normal(size=N) + 1j * rng.normal(size=N)
    x = np.zeros(N, complex)
    nb = np.linalg.norm(b)
    t0 = time.perf_counter()
    for it in range(1, cap + 1):
        x = vcycle(levels, cols, b, x)
        if np.linalg.norm(b - levels[0] @ x) / nb < tol:
            return it, time.perf_counter() - t0
    return -1, time.perf_counter() - t0


def apply_cost(levels):
    """Measured seconds for one operator apply at every level, summed.

    This is the per-cycle work that differs between the two hierarchies: same
    transfers, same smoother sweeps count, different operator densities.
    """
    tot = 0.0
    for A in levels:
        v = np.ones(A.shape[0], complex)
        t0 = time.perf_counter()
        for _ in range(20):
            A @ v
        tot += (time.perf_counter() - t0) / 20
    return tot


if __name__ == '__main__':
    print("=== 3D n=16: rediscretized vs Galerkin, MULTILEVEL V-cycle ===")
    print("  same transfers (exported prolongGauge), same smoother, same line search")
    print()
    print("  %-5s | %-22s | %-22s | %s"
          % ("nPhi", "rediscretized", "Galerkin", "verdict"))
    print("  %-5s | %-7s %-6s %-7s | %-7s %-6s %-7s | %s"
          % ("", "its", "nnz/row", "apply", "its", "nnz/row", "apply", "work G/R"))
    for NP in [1, 2, 4, 8, 12, 16]:
        L16, L8, L4 = (rd('L%d_%d.mtx' % (n, NP)) for n in (16, 8, 4))
        P16, P8 = rd('P16_%d.mtx' % NP), rd('P8_%d.mtx' % NP)
        Ps = [P16, P8]

        red = [L16, L8, L4]
        gal1 = (P16.conj().T @ L16 @ P16).tocsr()
        gal1.eliminate_zeros()
        gal2 = (P8.conj().T @ gal1 @ P8).tocsr()
        gal2.eliminate_zeros()
        gal = [L16, gal1, gal2]

        cols = [colors(16), colors(8), colors(4)]
        A_shapes = [A.shape[0] for A in red]

        itR, _ = solve(red, cols)
        itG, _ = solve(gal, cols)
        cR, cG = apply_cost(red), apply_cost(gal)
        dR = '/'.join('%.0f' % (A.nnz / A.shape[0]) for A in red)
        dG = '/'.join('%.0f' % (A.nnz / A.shape[0]) for A in gal)
        workR, workG = itR * cR, itG * cG
        verdict = ('%.2fx' % (workG / workR)) if itR > 0 and itG > 0 else 'n/a'
        print("  %-5d | %-7s %-6s %-7.1f | %-7s %-6s %-7.1f | %s"
              % (NP, itR, dR, cR * 1e6, itG, dG, cG * 1e6, verdict))
