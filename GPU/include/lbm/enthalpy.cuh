#pragma once
//==============================================================================
//  ENTHALPY PHASE CHANGE -- the GPU twin of the parent's
//  src/collision/EnthalpyBGK.hpp + EnthalpyRegularised.hpp, and the pass that
//  couples a melting field to the fluid.
//
//  It shares NO header with the parent. The material law, its inversion, the
//  equilibrium, both collisions and the coupling are written again here, so
//  that the two trees agreeing on validation/ice_equilibrium.cpp is evidence
//  rather than an identity. What is deliberately the SAME is the algebra:
//
//    transported   H = e(T) + La f_l, the total enthalpy (the scalar's field is
//                  therefore an ENTHALPY, not a temperature -- invert it)
//    equilibrium   h_i = w_i (dE + E c_i.u / cs2) + delta_i0 (dH - dE),
//                  dH = H - T_ref, dE = E - T_ref, E the SENSIBLE enthalpy:
//                  the latent part is carried by the rest population and is not
//                  advected (Voller & Prakash; the parent's EnthalpyAdvect::
//                  Sensible, the only mode ported)
//    rate          the solver's omega where alpha = k / c_p is uniform, else
//                  per node, omega = c / (k / cs2 + c / 2), c = dE/dT, with the
//                  PARALLEL mixture k = k_s + f_l (k_l - k_s) (the parent's
//                  MushMix::Parallel, the only rule ported)
//
//  TWO CONSEQUENCES FOR A DRIVER, both measured in the parent tree:
//    * anti-bounce-back (ScalarDirichlet) imposes its value through the
//      NON-REST equilibrium w_i (E - T_ref), so it must be handed the SENSIBLE
//      E(T_w) = H - La f_l, not H;
//    * an isothermal front (T_s == T_l) has no unique discrete equilibrium --
//      a window of link positions about a cell wide is steady.
//==============================================================================
#include "core.cuh"
#include "streaming.cuh"   // LBM_CUDA_CHECK

#include <cstdio>
#include <cstdlib>

namespace lbm {

//------------------------------------------------------------------------------
// The material. A trivially copyable POD: it rides inside ScalarParams by value,
// so a kernel has it in registers without a pointer chase.
//------------------------------------------------------------------------------
struct PhaseChange {
  // Set these. All volumetric: cp is rho c_p, La is rho L.
  Real T_s = Real(0);      // solidus
  Real T_l = Real(0);      // liquidus; T_l == T_s is an isothermal front
  Real cp_s = Real(1), cp_l = Real(1);
  Real k_s = Real(1), k_l = Real(1);
  Real La = Real(1);
  Real E_datum = Real(0);  // gauge: E(T_s) = E_datum

  // Derived by normalise(). A PhaseChange that was never normalised inverts
  // through these defaults and returns a plausible WRONG temperature -- the
  // parent paid for that twice -- so the enthalpy ops refuse one (see
  // require_material below) rather than trusting the caller.
  Real dTm = Real(0), d_cp = Real(0), B = Real(1), H_l = Real(1), inv_B = Real(1);
  bool linear_band = true, uniform = true, ready = false;

  void normalise() {
    if (T_l < T_s || cp_s <= Real(0) || cp_l <= Real(0) || k_s <= Real(0) ||
        k_l <= Real(0) || La < Real(0)) {
      std::fprintf(stderr, "PhaseChange: need T_l >= T_s, cp > 0, k > 0, La >= 0 "
                           "(got T_s %g T_l %g cp %g/%g k %g/%g La %g)\n",
                   double(T_s), double(T_l), double(cp_s), double(cp_l),
                   double(k_s), double(k_l), double(La));
      std::exit(2);
    }
    dTm = T_l - T_s;
    d_cp = cp_l - cp_s;
    B = cp_s * dTm + La;
    H_l = Real(0.5) * (cp_s + cp_l) * dTm + La;
    if (B <= Real(0)) {
      std::fprintf(stderr, "PhaseChange: no band and no latent heat -- there is no "
                           "phase change to model. Use ScalarOp::BGK.\n");
      std::exit(2);
    }
    inv_B = Real(1) / B;
    linear_band = (d_cp * dTm == Real(0));
    uniform = (k_s * cp_l == k_l * cp_s);
    ready = true;
  }

  // H -> (f_l, T, E, dE/dT). H = 0 is the solidus with f_l = 0; the band holds
  // H = B f + (d_cp dTm / 2) f^2, solved by the citardauq root so that
  // cp_s == cp_l (the commonest case) is not a division by zero. f is solved
  // first and T = T_s + f dTm second, so an isothermal front is an ordinary
  // point. f_l is never clamped: a clamped fraction creates or destroys latent
  // heat.
  LBM_HD LBM_INLINE void invert(Real H, Real& fl, Real& T, Real& E, Real& dEdT) const {
    if (H <= Real(0)) {
      fl = Real(0);  T = T_s + H / cp_s;  dEdT = cp_s;
    } else if (H >= H_l) {
      fl = Real(1);  T = T_l + (H - H_l) / cp_l;  dEdT = cp_l;
    } else {
      fl = linear_band ? H * inv_B
                       : Real(2) * H / (B + enth_sqrt(B * B + Real(2) * d_cp * dTm * H));
      T = T_s + fl * dTm;
      dEdT = cp_s + d_cp * fl;
    }
    E = E_datum + H - La * fl;
  }

  // The forward map. TWO arguments: at an isothermal front H(T) is not a
  // function, and a one-argument form would pick a branch by accident.
  LBM_HD LBM_INLINE Real enthalpy_of(Real T, Real fl) const {
    Real e;
    if (T <= T_s)       e = cp_s * (T - T_s);
    else if (T >= T_l)  e = Real(0.5) * (cp_s + cp_l) * dTm + cp_l * (T - T_l);
    else {
      const Real f = (T - T_s) / dTm;
      e = cp_s * (T - T_s) + Real(0.5) * d_cp * dTm * f * f;
    }
    return e + La * fl;
  }

  LBM_HD LBM_INLINE Real conductivity(Real fl) const { return k_s + fl * (k_l - k_s); }

  LBM_HD LBM_INLINE Real temperature_of(Real H) const {
    Real f, T, E, c; invert(H, f, T, E, c); return T;
  }
  LBM_HD LBM_INLINE Real liquid_fraction(Real H) const {
    Real f, T, E, c; invert(H, f, T, E, c); return f;
  }

 private:
  // Explicit per precision, as colour.cuh argues, so nvcc never meets an
  // ambiguous overload in one build and not the other.
  static LBM_HD LBM_INLINE Real enth_sqrt(Real x) {
#if defined(LBM_DOUBLE)
    return ::sqrt(x);
#else
    return ::sqrtf(x);
#endif
  }
};

inline void require_material(ScalarOp op, const PhaseChange& pc) {
  if ((op == ScalarOp::EnthalpyBGK || op == ScalarOp::EnthalpyRegularised) && !pc.ready) {
    std::fprintf(stderr, "enthalpy scalar op without a normalised material: call "
                         "set_material() before the first step. NOTHING WAS RUN.\n");
    std::exit(2);
  }
}

//------------------------------------------------------------------------------
// The two collisions. dH = sum_i h_i (the deviation from T_ref, which is an
// ENTHALPY here: set it to the mean enthalpy of the problem).
//------------------------------------------------------------------------------
template <class L>
LBM_HD LBM_INLINE Real enthalpy_rate(const PhaseChange& pc, Real fl, Real dEdT, Real omega) {
  return pc.uniform ? omega
                    : dEdT / (pc.conductivity(fl) * L::inv_cs2() + Real(0.5) * dEdT);
}

template <class L>
LBM_HD LBM_INLINE void collide_enthalpy_bgk(Real h[L::Q], Real dH, Real T_ref,
                                            const PhaseChange& pc, Real ux, Real uy,
                                            Real uz, Real omega) {
  Real fl, T, E, dEdT;
  pc.invert(T_ref + dH, fl, T, E, dEdT);
  const Real dE = E - T_ref;
  const Real w  = enthalpy_rate<L>(pc, fl, dEdT, omega);
  for (int i = 0; i < L::Q; ++i) {
    const Real cu = Real(L::cx(i)) * ux + Real(L::cy(i)) * uy + Real(L::cz(i)) * uz;
    const Real eq = L::w(i) * (dE + E * L::inv_cs2() * cu) + (i == 0 ? dH - dE : Real(0));
    h[i] += w * (eq - h[i]);
  }
}

// D3Q7 only, like collide_scalar_regularised: the flux moments relax at the
// rate, the three ghosts go to their equilibrium cs2 dE outright, and h[0]
// is the residual that carries dH -- including the latent part.
LBM_HD LBM_INLINE void collide_enthalpy_regularised(Real h[7], Real dH, Real T_ref,
                                                    const PhaseChange& pc, Real ux,
                                                    Real uy, Real uz, Real omega) {
  Real fl, T, E, dEdT;
  pc.invert(T_ref + dH, fl, T, E, dEdT);
  const Real dE = E - T_ref;
  const Real w  = enthalpy_rate<D3Q7>(pc, fl, dEdT, omega);
  const Real jx = h[1] - h[2], jy = h[3] - h[4], jz = h[5] - h[6];
  const Real px = jx + w * (E * ux - jx);
  const Real py = jy + w * (E * uy - jy);
  const Real pz = jz + w * (E * uz - jz);
  const Real d  = D3Q7::cs2() * dE;
  h[1] = Real(0.5) * (d + px);  h[2] = Real(0.5) * (d - px);
  h[3] = Real(0.5) * (d + py);  h[4] = Real(0.5) * (d - py);
  h[5] = Real(0.5) * (d + pz);  h[6] = Real(0.5) * (d - pz);
  h[0] = dH - Real(3) * d;
}

//------------------------------------------------------------------------------
// THE SEED of a scalar node: the operator's OWN equilibrium at rest. For the
// enthalpy ops that is w_i dE + delta_i0 (dH - dE), the latent heat in the rest
// population only. Seeding the plain scalar form w_i dH instead puts La f / 8
// into every MOVING population. In the bulk the first collision undoes that,
// but a skipped (adiabatic) ghost keeps it in the slots its neighbour reads
// once -- and that injected 2.5 enthalpy units per wall link in the first GPU
// melting run: +0.07 C over the whole bulk and a 9.5 C spot beside an 8 C wall,
// where the parent (which seeds its operator's own equilibrium) read 8.000.
// Dirichlet ghosts hold the SENSIBLE wall value and keep the plain form, which
// is what anti-bounce-back imposes against.
//------------------------------------------------------------------------------
template <class L>
LBM_HD LBM_INLINE void scalar_seed(Real g[L::Q], Real value, Real T_ref, bool enthalpy,
                                   const PhaseChange& pc) {
  if (!enthalpy) {
    for (int i = 0; i < L::Q; ++i) g[i] = L::w(i) * (value - T_ref);
    return;
  }
  Real f, T, E, c;
  pc.invert(value, f, T, E, c);
  const Real dH = value - T_ref, dE = E - T_ref;
  for (int i = 0; i < L::Q; ++i) g[i] = L::w(i) * dE + (i == 0 ? dH - dE : Real(0));
}

LBM_HD LBM_INLINE bool is_enthalpy(ScalarOp op) {
  return op == ScalarOp::EnthalpyBGK || op == ScalarOp::EnthalpyRegularised;
}

//------------------------------------------------------------------------------
// THE MELTING COUPLING, once per step before the fluid: from the scalar's
// enthalpy field, the anomalous buoyancy the fluid's ForceDarcy takes as its
// external force, and the Carman-Kozeny drag it closes implicitly.
//
//   F_up = f_l w (|T - Tm|^q - |T0 - Tm|^q)      (q = 2: Weady et al.'s law;
//                                                  1.894816: Gebhart-Mollendorf)
//   A    = A_solid eps (1 - f_l)^2 / (f_l^3 + eps)
//
// f_l-weighted because the ice is held by the drag and |T - Tm|^q inside it
// would compress the lattice density hydrostatically (the parent's
// ice_equilibrium banner). Gravity is along -y, so the force is +y.
//------------------------------------------------------------------------------
struct MeltParams {
  const Real* H = nullptr;      // the scalar's field -- an ENTHALPY
  Real* Fy = nullptr;           // out: the fluid's per-node Fy
  Real* A = nullptr;            // out: the fluid's drag coefficient
  PhaseChange pc;
  Real w = Real(0), q = Real(2), Tm = Real(4), T0 = Real(0);
  Real A_solid = Real(100), eps = Real(1e-3);
  long N = 0;
};

LBM_HD LBM_INLINE Real melt_pow(Real x, Real q) {
#if defined(LBM_DOUBLE)
  return ::pow(x, q);
#else
  return ::powf(x, q);
#endif
}

LBM_HD LBM_INLINE void melt_node(const MeltParams& p, long n) {
  Real f, T, E, c;
  p.pc.invert(p.H[n], f, T, E, c);
  const Real a = T - p.Tm, b = p.T0 - p.Tm;
  const Real ta = (p.q == Real(2)) ? a * a : melt_pow(a < Real(0) ? -a : a, p.q);
  const Real tb = (p.q == Real(2)) ? b * b : melt_pow(b < Real(0) ? -b : b, p.q);
  p.Fy[n] = f * p.w * (ta - tb);
  p.A[n]  = p.A_solid * p.eps * (Real(1) - f) * (Real(1) - f) / (f * f * f + p.eps);
}

#if defined(__CUDACC__)
template <int Unused = 0>
__global__ void melt_kernel(MeltParams p) {
  const long n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= p.N) return;
  melt_node(p, n);
}
#endif

inline void melt_pass(const MeltParams& p) {
#if defined(__CUDACC__)
  const int B = 256, G = int((p.N + B - 1) / B);
  melt_kernel<0><<<G, B>>>(p);
  LBM_CUDA_CHECK(cudaGetLastError());
#else
  for (long n = 0; n < p.N; ++n) melt_node(p, n);
#endif
}

}  // namespace lbm
