#pragma once
//==============================================================================
//  Electrohydrodynamics: the coupling pass.
//
//  Patnaik, Skillen & De Rosis, Eng. Comput. 41:4977-5002 (2025). Four fields,
//  each feeding the next:
//
//      grad^2 phi = -q/eps                       the potential
//      E = -grad phi                             by finite difference (model C)
//      d_t q + div[q(u + K E)] = D grad^2 q      charge, at the DRIFT velocity
//      d_t u + u.grad u = -grad p/rho + nu grad^2 u + qE/rho
//
//  The fluid and both scalars are solvers this tree already has. What is left
//  is one per-node pass that reads phi and q and writes three things: the DRIFT
//  velocity the charge is advected by, the COULOMB FORCE the fluid is forced
//  by, and the SOURCE the potential is relaxed with. That pass is this file.
//
//  Structured like body.cuh: an LBM_HD node function, a kernel that wraps it,
//  and a host loop that runs the same function -- so a wrong stencil or a
//  wrong coupling order is found on a laptop, which is where the parent tree
//  found several of them.
//
//  ===================== THE STENCILS ARE THE PHYSICS =========================
//  E IS A DERIVATIVE OF THE FIELD CARRYING THE BOUNDARY VALUE, and that governs
//  every choice here. The parent tree measured halfway plates as a uniform
//  +3.3 % bias in the bulk charge at H = 40, halving to +1.3 % at H = 80 --
//  O(1/H), dragging the whole convergence rate to first order -- because the
//  one-sided stencil then works purely from interior nodes and never sees the
//  imposed phi. On-node plates (ScalarMoment) took C = 10 at H = 80 from
//  11.19 % to 1.49 %. So:
//
//    * the plates ARE nodes, and d_y phi at y = 0 and y = H uses the
//      second-order ONE-SIDED stencil, which starts from the imposed value;
//    * E_x at a side wall is set to ZERO rather than differenced. That IS the
//      boundary condition d_x phi = 0, and it is exact; a one-sided stencil
//      applied to a mirrored value gives the condition plus a truncation error
//      of the interior gradient.
//
//  ===================== THE FORCE IS ZEROED ON THE WALL LINES ================
//  Regularised walls are FLUID nodes: they collide, and they would be forced.
//  CLAUDE.md's zhou_thermal entry is the same trap wearing a different field.
//  Here q and E are both correct at the wall, so the force there would be the
//  RIGHT force -- but the node's velocity is PRESCRIBED to zero by the wall
//  closure, so it cannot do anything except fight that closure.
//
//  ===================== THE POISSON SOURCE ===================================
//  Eq. (26): the potential is relaxed in pseudo-time by a scalar lattice whose
//  "diffusivity" beta is a relaxation rate and not a physical constant. At the
//  fixed point beta grad^2 phi + (beta/eps) q = 0, i.e. grad^2 phi = -q/eps,
//  independent of beta. The source carries an Adams-Bashforth extrapolation,
//  (3q - q_prev)/2, which is what makes it second order in the coupling rather
//  than merely convergent.
//==============================================================================
#include "streaming.cuh"

#include <vector>

namespace lbm {

//------------------------------------------------------------------------------
// A plain N-element field, on whichever side of the bus this build uses. The
// driver owns several of these; nothing else in the tree needed one, because
// every other coupling writes into a solver's own arrays.
//------------------------------------------------------------------------------
class Field {
 public:
  explicit Field(long N) : n_(N) {
#if defined(__CUDACC__)
    LBM_CUDA_CHECK(cudaMalloc(&p_, sizeof(Real) * std::size_t(N)));
    LBM_CUDA_CHECK(cudaMemset(p_, 0, sizeof(Real) * std::size_t(N)));
#else
    v_.assign(std::size_t(N), Real(0));
    p_ = v_.data();
#endif
  }
  ~Field() {
#if defined(__CUDACC__)
    cudaFree(p_);
#endif
  }
  Field(const Field&) = delete;
  Field& operator=(const Field&) = delete;

  Real* data() { return p_; }
  const Real* data() const { return p_; }
  long size() const { return n_; }

  void to_host(std::vector<Real>& out) const {
    out.resize(std::size_t(n_));
#if defined(__CUDACC__)
    LBM_CUDA_CHECK(cudaMemcpy(out.data(), p_, sizeof(Real) * std::size_t(n_),
                              cudaMemcpyDeviceToHost));
#else
    std::copy(v_.begin(), v_.end(), out.begin());
#endif
  }
  void from_host(const std::vector<Real>& in) {
#if defined(__CUDACC__)
    LBM_CUDA_CHECK(cudaMemcpy(p_, in.data(), sizeof(Real) * std::size_t(n_),
                              cudaMemcpyHostToDevice));
#else
    std::copy(in.begin(), in.end(), v_.begin());
#endif
  }

 private:
  long n_ = 0;
  Real* p_ = nullptr;
#if !defined(__CUDACC__)
  std::vector<Real> v_;
#endif
};

struct EhdParams {
  const Real* phi = nullptr;
  const Real* q = nullptr;
  const Real* ux = nullptr;            // null in the hydrostatic reference
  const Real* uy = nullptr;
  Real* kx = nullptr;                  // drift velocity u + K E, out
  Real* ky = nullptr;
  Real* kz = nullptr;
  Real* Fx = nullptr;                  // Coulomb force q E, out
  Real* Fy = nullptr;
  Real* Fz = nullptr;
  Real* src = nullptr;                 // Poisson source, out
  Real* qprev = nullptr;               // previous charge, in and out
  int nx = 0, ny = 0, nz = 0, H = 0;
  Real K = Real(1), eps = Real(1), beta = Real(0.3);
  bool sidewalls = true;               // x = 0, nx-1 are walls: E_x = 0 there
};

//------------------------------------------------------------------------------
// One node: E, the drift, the Coulomb force and the Poisson source.
//------------------------------------------------------------------------------
LBM_HD LBM_INLINE void ehd_node(const EhdParams& p, long n) {
  int x, y, z;
  coords(n, p.nx, p.ny, x, y, z);
  const int H = p.H;

  auto at = [&](int xx, int yy) {
    return p.phi[node_id(wrap(xx, p.nx), wrap(yy, p.ny), z, p.nx, p.ny)];
  };

  Real Ey;
  if (y == 0)
    Ey = -(Real(-1.5) * at(x, 0) + Real(2) * at(x, 1) - Real(0.5) * at(x, 2));
  else if (y == H)
    Ey = -(Real(1.5) * at(x, H) - Real(2) * at(x, H - 1) + Real(0.5) * at(x, H - 2));
  else
    Ey = Real(-0.5) * (at(x, y + 1) - at(x, y - 1));

  // d_x phi = 0 IS the side-wall condition, so E_x there is exactly zero.
  const bool xwall = p.sidewalls && (x == 0 || x == p.nx - 1);
  const Real Ex = xwall ? Real(0)
                        : Real(-0.5) * (at(x + 1, y) - at(x - 1, y));

  const Real qn = p.q[n];
  const bool wall = xwall || y == 0 || y == H;
  const Real u = p.ux ? p.ux[n] : Real(0);
  const Real v = p.uy ? p.uy[n] : Real(0);

  p.Fx[n] = wall ? Real(0) : qn * Ex;
  p.Fy[n] = wall ? Real(0) : qn * Ey;
  p.Fz[n] = Real(0);
  p.kx[n] = p.K * Ex + u;
  p.ky[n] = p.K * Ey + v;
  p.kz[n] = Real(0);

  // Eq. (26), with the Adams-Bashforth extrapolation of the charge.
  p.src[n] = (p.beta / p.eps) * (Real(1.5) * qn - Real(0.5) * p.qprev[n]);
  p.qprev[n] = qn;
}

#if defined(__CUDACC__)
__global__ inline void ehd_kernel(EhdParams p, long N) {
  const long n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  ehd_node(p, n);
}
#endif

//------------------------------------------------------------------------------
// Run the pass over every node, on whichever backend this build uses.
//------------------------------------------------------------------------------
inline void ehd_pass(const EhdParams& p) {
  const long N = long(p.nx) * p.ny * p.nz;
#if defined(__CUDACC__)
  const int B = 256, G = int((N + B - 1) / B);
  ehd_kernel<<<G, B>>>(p, N);
  LBM_CUDA_CHECK(cudaGetLastError());
#else
  for (long n = 0; n < N; ++n) ehd_node(p, n);
#endif
}

}  // namespace lbm
