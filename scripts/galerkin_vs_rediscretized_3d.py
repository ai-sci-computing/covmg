#!/usr/bin/env python3
"""Rediscretized vs Galerkin in THREE dimensions.

The 2D companion script (galerkin_vs_rediscretized.py) finds the Galerkin
coarse operator the better one where the two differ -- up to a factor 3 in the
two-grid factor. That result does NOT survive the change of dimension, and
since the paper's operator is 3D it is this script that decides the design
choice.

Two things move together going from 2D to 3D. The Galerkin fill-in grows from
5 -> 9 entries per row to 7 -> 27, a penalty of 3.9x rather than 1.8x; and the
factor advantage collapses to at most 1.2x, vanishing entirely at nPhi = 4.
Net, by the work model below, Galerkin is 1.27-1.38x SLOWER across the whole
flux range including the aliasing point.

The prolongation is NOT rebuilt here. It is exported from the library's own
prolongGauge, because reconstructing the multi-pass covariant fill by hand is
error-prone: a first attempt implemented only pass 1, leaving face-centre and
cell-centre rows empty, which silently produced a P with 7 nonzeros per coarse
column instead of 27 and a meaningless comparison.

Inputs (from the repository root, with build/tools on the path):
    for NP in 1 2 4 8 16; do
      export_operator prolong 16 $NP P_$NP.mtx      # verified covariant fill
      export_operator u1      16 $NP L_$NP.mtx      # fine operator
      export_operator u1       8 $NP Lc_$NP.mtx     # REdiscretized coarse operator
    done

WHAT IS MEASURED AND WHAT IS MODELLED. The two-grid factors and the operator
densities are measured. The wall-time ratio is a MODEL: work per V-cycle is
taken as sum_l 8^-l times the level's stencil density, which assumes the
Galerkin density stays at its level-1 value of 27. In reality Galerkin of
Galerkin fills further, so the model is optimistic FOR Galerkin. Setup is
excluded entirely, and that also favours rediscretization -- one closed-form
O(N) pass over the links against a triple product per level, paid on every
rebuild. So the direction of the conclusion is safe even though the number is
not a measurement.
"""
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spl


def rd(path):
    """Read a complex MatrixMarket file written by tools/export_operator."""
    with open(path) as f:
        lines = [l for l in f if not l.startswith('%')]
    m, n, nnz = map(int, lines[0].split())
    r = np.empty(nnz, int); c = np.empty(nnz, int); v = np.empty(nnz, complex)
    for t, l in enumerate(lines[1:nnz + 1]):
        a = l.split()
        r[t] = int(a[0]) - 1; c[t] = int(a[1]) - 1
        v[t] = float(a[2]) + 1j * float(a[3])
    return sp.csr_matrix(sp.coo_matrix((v, (r, c)), shape=(m, n)))
n=16
red=np.array([ (i*n+j)*n+k for i in range(n) for j in range(n) for k in range(n) if (i+j+k)%2==0])
blk=np.array([ (i*n+j)*n+k for i in range(n) for j in range(n) for k in range(n) if (i+j+k)%2==1])
print("=== 3D n=16, verified prolongGauge: rediscretized vs Galerkin ===")
print("  %-6s %-10s %-10s  %-8s %-8s  %s" % ("nPhi","rho redis","rho Galer","nnz R","nnz G","modelled time G/R"))
for NP in [1,2,4,8,16]:
    L = rd('L_%d.mtx' % NP)
    P = rd('P_%d.mtx' % NP)
    Lc = rd('Lc_%d.mtx' % NP)
    Lg=(P.conj().T@L@P).tocsr(); Lg.eliminate_zeros()
    D=L.diagonal().real
    def rho(Lcoarse,sweeps=2,ncyc=60,seed=0):
        N=L.shape[0]; r=np.random.default_rng(seed)
        b=r.normal(size=N)+1j*r.normal(size=N); x=np.zeros(N,complex)
        try: solve=spl.factorized(sp.csc_matrix(Lcoarse))
        except Exception: return float('nan')
        res=[]
        def gs(x):
            for g in (red,blk): x[g]+=(b[g]-L[g,:]@x)/D[g]
            return x
        for _ in range(ncyc):
            for _ in range(sweeps): x=gs(x)
            rr=b-L@x; p=P@solve(P.conj().T@rr)
            den=np.real(np.vdot(p,L@p))
            if den>0: x=x+(np.real(np.vdot(p,rr))/den)*p
            for _ in range(sweeps): x=gs(x)
            res.append(np.linalg.norm(b-L@x))
        res=np.array(res); good=res[res>1e-13]
        if len(good)<6: return 0.0
        t=good[len(good)//2:]; return float(np.median(t[1:]/t[:-1]))
    rR,rG=rho(Lc),rho(Lg); dR,dG=Lc.nnz/Lc.shape[0],Lg.nnz/Lg.shape[0]
    wR=sum(8.0**-l for l in range(6)); wG=1.0+(dG/dR)*sum(8.0**-l for l in range(1,6))
    try:
        ratio=(np.log(1e-8)/np.log(rG)*wG)/(np.log(1e-8)/np.log(rR)*wR)
    except Exception: ratio=float('nan')
    print("  %-6d %-10.4f %-10.4f  %-8.1f %-8.1f  %.2fx" % (NP,rR,rG,dR,dG,ratio))
