#!/usr/bin/env python3
"""Von Neumann stability of the D3Q7 advection-diffusion collisions in src/collision.

For a UNIFORM velocity u the one-step map of a Fourier mode exp(i k.x) is
A(k) = diag(exp(-i k.c_q)) C, where C is the 7x7 linear collision at fixed u. The
scheme is linearly stable when max_k |eig A(k)| <= 1. This prints, for each
operator and diffusivity, the largest stable |u| along an axis, a face diagonal
and a body diagonal -- the table in collision/ScalarTRT.hpp's banner.

The operators are members of one two-relaxation-time family: the flux (odd)
moments relax at omega, which sets D = cs2 (1/omega - 1/2) with cs2 = 1/4, and
the ghost (even) moments at omega_plus, with Lambda = (1/omega_plus - 1/2)(1/omega - 1/2):

    ScalarBGK          omega_plus = omega
    ScalarRegularised  omega_plus = 1
    ScalarTRT          omega_plus from Lambda (1/4 by default)

The equilibrium is the tree's: flux T u, every ghost cs2 T (no u^2 term).

Needs numpy, which the system python on the maintainers' Mac does not have; a
throwaway venv (python3 -m venv v && v/bin/pip install numpy) is enough.

    python3 tools/scalar_stability.py                   # the banner's table
    python3 tools/scalar_stability.py --D 1e-3 --lam 0.0833 0.1667 0.1875 0.25
"""
import argparse

import numpy as np

CS2 = 0.25
C7 = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], float)


def collision(u, w_minus, w_plus):
    """The 7x7 collision matrix at velocity u, in the slot order of D3Q7."""
    M = np.zeros((7, 7))
    for col in range(7):
        h = np.zeros(7)
        h[col] = 1.0
        T = h.sum()
        out = np.zeros(7)
        ghosts = 0.0
        for a in range(3):
            jp, jm = 1 + 2 * a, 2 + 2 * a
            j, m = h[jp] - h[jm], h[jp] + h[jm]
            jn = (1.0 - w_minus) * j + w_minus * T * u[a]
            mn = (1.0 - w_plus) * m + w_plus * CS2 * T
            out[jp], out[jm] = 0.5 * (mn + jn), 0.5 * (mn - jn)
            ghosts += mn
        out[0] = T - ghosts
        M[:, col] = out
    return M


def k_grid(n):
    ks = np.linspace(0.0, np.pi, n)
    ky = np.concatenate([-ks[::-1], ks[1:]])
    KX, KY, KZ = np.meshgrid(ks, ky, ks, indexing="ij")
    K = np.stack([KX.ravel(), KY.ravel(), KZ.ravel()], 1)
    return np.exp(-1j * K @ C7.T)          # (nk, 7): the streaming phases


def radius(phases, u, w_minus, w_plus):
    A = phases[:, :, None] * collision(u, w_minus, w_plus)[None]
    return np.abs(np.linalg.eigvals(A)).max()


def threshold(phases, direction, w_minus, w_plus, hi=0.6, iters=24):
    d = np.asarray(direction, float)
    d /= np.linalg.norm(d)
    if radius(phases, hi * d, w_minus, w_plus) <= 1.0 + 1e-10:
        return float("inf")
    lo = 0.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if radius(phases, mid * d, w_minus, w_plus) > 1.0 + 1e-10:
            hi = mid
        else:
            lo = mid
    return lo


def w_plus_of(w_minus, lam):
    return 1.0 / (lam / (1.0 / w_minus - 0.5) + 0.5)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--D", type=float, nargs="+", default=[1e-3, 5e-4, 2.5e-4],
                    help="diffusivities (lattice units)")
    ap.add_argument("--lam", type=float, nargs="*", default=[0.25],
                    help="TRT Lambda values to scan besides BGK and regularised")
    ap.add_argument("--n", type=int, default=12, help="k points per half axis (default 12)")
    args = ap.parse_args()
    phases = k_grid(args.n)
    dirs = {"x": (1, 0, 0), "xy": (1, 1, 0), "xyz": (1, 1, 1)}
    for D in args.D:
        wm = 1.0 / (D / CS2 + 0.5)
        print(f"\nD = {D:.3e}   omega = {wm:.6f}", flush=True)
        ops = [("ScalarRegularised", 1.0), ("ScalarBGK", wm)]
        ops += [(f"ScalarTRT Lambda = {lam:.4g}", w_plus_of(wm, lam)) for lam in args.lam]
        for name, wp in ops:
            th = {k: threshold(phases, v, wm, wp) for k, v in dirs.items()}
            print(f"  {name:26s} omega_plus = {wp:8.5f}   largest stable |u|:  "
                  f"x {th['x']:.4f}   xy {th['xy']:.4f}   xyz {th['xyz']:.4f}", flush=True)


if __name__ == "__main__":
    main()
