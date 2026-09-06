#pragma once
//==============================================================================
//  Passive scalar on D3Q7 -- temperature, or any advected-diffused quantity.
//
//  Structurally different from the fluid in one way: the velocity is an INPUT.
//  The scalar carries only its zeroth moment and is advected by whatever field
//  the fluid hands it, which is why it has its own solver rather than another
//  collision operator plugged into the fluid one.
//
//  BOUNDARY CONDITIONS, as alternative collisions on marked cells:
//
//    adiabatic  h_i^out = h_opp(i)^in                              (zero flux)
//    Dirichlet  h_i^out = -h_opp(i)^in + 2 w_i (T_wall - T_ref)    (anti-bounce-back)
//
//  The Dirichlet form carries T_ref because the arrays hold h = g - w_i T_ref;
//  since w_i == w_opp(i), the reference simply shifts the target value.
//
//  WHERE THE TWO WALLS PUT THEIR PLANE, which is the thing most easily got
//  wrong: anti-bounce-back places T_wall half-way between the Dirichlet node and
//  its fluid neighbour, exactly as halfway bounce-back places the no-slip plane.
//  So a conduction problem between two Dirichlet layers has its exact linear
//  solution measured between the PLANES, not between the nodes, and a slab of H
//  fluid nodes spans a gap of H + 1 lattice units. Getting this wrong shows up
//  as an O(1/H) error that looks like a convergence problem and is not one.
//
//  NOT INCLUDED, and absent rather than untested: the open (outflow) boundary
//  and Dellar's moment condition. Outflow needs a donor map and a second kernel
//  after a fence -- reading a donor inside the main kernel is a genuine race
//  under Esoteric Pull, because the two slots a node reads are the two it
//  writes. That is a self-contained piece of work and it is not here.
//==============================================================================
#include "specular.cuh"
#include "streaming.cuh"

namespace lbm {

using ScalarLattice = D3Q7;

struct ScalarParams {
  Real* h = nullptr;
  const std::uint8_t* flags = nullptr;
  const Real* wall = nullptr;                // Dirichlet values, per node
  const Real* ux = nullptr;                  // advecting velocity, owned by the fluid
  const Real* uy = nullptr;
  const Real* uz = nullptr;
  Real* T_out = nullptr;                     // the concentration field
  // Donor node for every ScalarOutflow cell: the interior cell whose value it
  // copies. Built once on the host at set_geometry; see build_donors.
  const long* donor = nullptr;
  // ScalarMoment: bitmask of the directions whose source node lies outside the
  // field, built once on the host. ScalarSpecular: that node's OUTWARD normal.
  const std::uint32_t* unk = nullptr;
  const std::uint8_t* spec = nullptr;
  const Real* src = nullptr;                 // per-node source, added as w_i S
  int nx = 0, ny = 0, nz = 0;
  Real omega = Real(1), T_ref = Real(0);
  // Runtime rather than a template parameter, unlike HasGeometry/HasOutflow.
  // Those gate a memory STREAM, which is what a bandwidth-bound kernel pays
  // for; this gates a branch that every thread in the grid takes the same way,
  // which costs nothing. Another template axis would double eight kernels.
  ScalarOp op = ScalarOp::BGK;
};

//------------------------------------------------------------------------------
// Dellar's moment condition, Eqs. (13a)-(13b).
//
// The field these lattices carry is a ZEROTH moment, T = sum_i h_i, so imposing
// its boundary value is ONE linear equation in the unknown populations. On a
// cross lattice a straight wall leaves exactly one unknown -- the direction
// pointing into the domain along the normal -- so that equation determines it
// uniquely and exactly, and the value is attained AT the node rather than half
// way to the next one. No closure assumption and no free parameter.
//
// A corner leaves more than one unknown and the single moment no longer closes
// the system; there the deficit is shared in proportion to the weights, which
// introduces no directional preference and still reproduces the moment exactly.
// That is a fallback, not part of the published method.
//
// WHY THIS IS WORTH ITS COST HERE. Anti-bounce-back (ScalarDirichlet) puts the
// wall half way between nodes, which is right for a temperature and wrong for a
// potential: the parent tree measured the halfway family as dragging the whole
// EHD convergence rate to first order, because E = -grad phi is a DERIVATIVE of
// the field carrying the boundary value and a one-sided stencil built from
// interior nodes never sees the imposed value at all.
//------------------------------------------------------------------------------
template <class L>
LBM_HD LBM_INLINE void impose_scalar_moment(Real* h, Real target, std::uint32_t unknown) {
  Real known = Real(0), wsum = Real(0);
  for (int i = 0; i < L::Q; ++i) {
    if (unknown & (1u << i)) wsum += L::w(i);
    else                     known += h[i];
  }
  if (!(wsum > Real(0))) return;            // nothing streamed in: leave it be
  const Real deficit = target - known;
  for (int i = 0; i < L::Q; ++i)
    if (unknown & (1u << i)) h[i] = deficit * L::w(i) / wsum;
}

//------------------------------------------------------------------------------
// One node, one step.
//
// An adiabatic cell returns immediately, for the same reason a Solid fluid cell
// does: bounce-back is the identity on Esoteric Pull's storage. A Dirichlet cell
// cannot be skipped -- anti-bounce-back flips a sign and adds a source -- but it
// still writes back into the same two slots it read, so the in-place scheme is
// undisturbed either way.
//------------------------------------------------------------------------------
template <int Parity, bool Advected, bool HasGeometry, bool HasOutflow = false,
          class L = ScalarLattice>
LBM_HD LBM_INLINE void scalar_node_update(const ScalarParams& p, long N, long n) {
  // With HasGeometry false every cell is bulk and the flags load is never
  // emitted -- see solver.cuh. `fl` is then a compile-time constant.
  const std::uint8_t fl = HasGeometry ? p.flags[n] : std::uint8_t(ScalarBulk);
  if (fl == ScalarExcluded || fl == ScalarAdiabatic) return;
  // Handled by the second pass, after the kernel boundary. See streaming.cuh.
  if (HasOutflow && fl == ScalarOutflow) return;

  int x, y, z;
  coords(n, p.nx, p.ny, x, y, z);

  constexpr int Q = L::Q;
  Real h[Q];
  gather<Parity, L>(p.h, N, x, y, z, p.nx, p.ny, p.nz, h);

  if (fl == ScalarDirichlet) {
    const Real dTw = p.wall[n] - p.T_ref;
    Real out[Q];
    for (int i = 0; i < Q; ++i)
      out[i] = -h[opp(i)] + Real(2) * L::w(i) * dTw;
    scatter<Parity, L>(p.h, N, x, y, z, p.nx, p.ny, p.nz, out);
    // The donor of an outflow node may be a Dirichlet cell's neighbour but is
    // never the Dirichlet cell itself, so this write is for completeness of the
    // field rather than for the second pass.
    if (HasOutflow) p.T_out[n] = p.wall[n];
    return;
  }

  // BOTH OF THESE FALL THROUGH TO THE COLLISION, and that is the difference
  // between them and the two boundary conditions above. A Dirichlet or an
  // adiabatic cell is a GHOST whose populations are prescribed outright; these
  // two are real nodes carrying a real value, whose streamed-in state is merely
  // incomplete. So they are repaired and then collided like any other node --
  // which is also why the source kernel has to visit them (see add_source).
  if (fl == ScalarMoment)
    impose_scalar_moment<L>(h, p.wall[n] - p.T_ref, p.unk[n]);
  else if (fl == ScalarSpecular)
    mirror_unknowns<L>(h, p.spec[n]);

  const Real dT = scalar_deviation<L>(h);
  Real vx = Real(0), vy = Real(0), vz = Real(0);
  if (Advected) { vx = p.ux[n]; vy = p.uy[n]; vz = p.uz[n]; }
  // THREE OPERATORS, ONE PATH. The charge carrier takes its central moments
  // about the DRIFT velocity it was handed, which is why it needs no separate
  // solver -- only a different collision at the same point in the same node
  // update. See collide_charge_cm in core.cuh.
  if (p.op == ScalarOp::ChargeCM)
    collide_charge_cm<L>(h, vx, vy, vz, p.omega);
  else if (p.op == ScalarOp::Regularised)
    collide_scalar_regularised(h, dT, p.T_ref, vx, vy, vz, p.omega);
  else
    collide_scalar<L>(h, dT, p.T_ref, vx, vy, vz, p.omega);
  scatter<Parity, L>(p.h, N, x, y, z, p.nx, p.ny, p.nz, h);

  // ONE EXTRA STORE, AND ONLY WHERE IT IS NEEDED. The second pass reads the
  // field at its donor, so the main pass has to publish it -- but a closed
  // problem should not pay 4 bytes per node for a boundary it does not have,
  // so the write is behind a template flag and is not emitted at all when no
  // cell is marked outflow. This is the same argument the flags load makes in
  // streaming.cuh, and the same reason it is a template parameter.
  if (HasOutflow) p.T_out[n] = p.T_ref + dT;
}

//------------------------------------------------------------------------------
// PASS 2. Outflow, after the kernel boundary.
//
// Equilibrium extrapolation: the DONOR's concentration, the node's OWN
// velocity, and no non-equilibrium part. Every population is overwritten, so
// whatever streamed in is irrelevant and the inward-pointing directions carry
// the interior's value back into the domain.
//
// Discarding the non-equilibrium part slightly damps the diffusive flux at the
// exit. That is negligible when the exit is advection-dominated, which is the
// only situation an open boundary belongs in, and it is a known approximation
// rather than an accident.
//
// A node whose donor is itself is INERT: it was found at setup to have no bulk
// neighbour, it was counted and reported there, and it keeps whatever streamed
// into it -- i.e. it falls back to bounce-back. Silently reading a neighbour's
// garbage would be the alternative.
//------------------------------------------------------------------------------
template <int Parity, bool Advected, class L = ScalarLattice>
LBM_HD LBM_INLINE void scalar_outflow_node(const ScalarParams& p, long N, long n) {
  if (p.flags[n] != ScalarOutflow) return;
  const long src = p.donor[n];
  if (src == n) return;                       // degenerate, reported at setup

  int x, y, z;
  coords(n, p.nx, p.ny, x, y, z);
  const Real dT = p.T_out[src] - p.T_ref;
  Real vx = Real(0), vy = Real(0), vz = Real(0);
  if (Advected) { vx = p.ux[n]; vy = p.uy[n]; vz = p.uz[n]; }

  constexpr int Q = L::Q;
  Real g[Q];
  for (int i = 0; i < Q; ++i)
    g[i] = scalar_eq<L>(i, dT, p.T_ref, vx, vy, vz);
  scatter<Parity, L>(p.h, N, x, y, z, p.nx, p.ny, p.nz, g);
  p.T_out[n] = p.T_ref + dT;
}

//------------------------------------------------------------------------------
// The field at one node, without advancing anything.
//
// This is the pass that makes the coupling simultaneous rather than lagged: the
// fluid must collide against T at the SAME time level, and the step kernel above
// cannot supply it, because the value it computes is consumed and overwritten in
// the same launch. Seven reads per node.
//------------------------------------------------------------------------------
template <int Parity, bool HasGeometry, class L = ScalarLattice>
LBM_HD LBM_INLINE void scalar_field_node(const ScalarParams& p, long N, long n) {
  const std::uint8_t fl = HasGeometry ? p.flags[n] : std::uint8_t(ScalarBulk);
  if (fl == ScalarExcluded) return;
  if (fl == ScalarDirichlet) { p.T_out[n] = p.wall[n]; return; }
  // A moment node attains its value AT the node, so that IS the field there;
  // the streamed populations still hold the unrepaired inward direction, so
  // their sum is not it.
  if (fl == ScalarMoment) { p.T_out[n] = p.wall[n]; return; }
  // AN OUTFLOW NODE IS ALREADY PUBLISHED, by the second pass, and it must NOT
  // be recomputed here. It does hold a real concentration -- unlike an
  // adiabatic node, which is a ghost -- but the pass that set it wrote
  // POST-COLLISION populations, and this kernel runs at the NEXT parity, so
  // what it would sum is the post-STREAMING state: everything that arrived
  // from outside, including whatever the periodic wrap delivered from the far
  // face. In ehd_cavity.cu's closed box that is the injector, and the collector
  // read 0.33 q0 against a neighbour at 0.075.
  if (fl == ScalarOutflow) return;

  int x, y, z;
  coords(n, p.nx, p.ny, x, y, z);
  Real h[L::Q];
  gather<Parity, L>(p.h, N, x, y, z, p.nx, p.ny, p.nz, h);
  // The unknown half of a specular node is whatever the periodic wrap delivered
  // from the far side of the box, so the sum is only the field once the mirror
  // has been reapplied -- the same repair the step makes, and it has to stay
  // the same one.
  if (fl == ScalarSpecular) mirror_unknowns<L>(h, p.spec[n]);
  p.T_out[n] = p.T_ref + scalar_deviation<L>(h);
}

//------------------------------------------------------------------------------
// Unknown directions at every ScalarMoment node, as a bitmask.
//
// A direction is unknown when the node its population would have STREAMED FROM
// lies outside the field: off a non-periodic edge, or on a cell excluded from
// the transport.
//
// THE PERIODICITY HAS TO BE PASSED IN, and that is not a detail here. This
// tree's streaming always wraps -- non-periodic walls are realised by marking
// cells, not by clipping the index -- so at a wall node the "missing" direction
// silently arrives from the FAR SIDE of the box instead. For bounce-back that
// never mattered, because a wall cell layer stands between the wrap and the
// fluid. For an on-node condition it matters completely: the node is itself in
// the fluid, so without this the injector at y = 0 would take its unknown
// direction from the collector at y = ny-1 and the mask would be empty.
//
// Plain host code, run once at set_geometry, shared by both backends.
//------------------------------------------------------------------------------
template <class L = ScalarLattice>
inline void build_scalar_unknowns(const std::vector<std::uint8_t>& flags,
                                  int nx, int ny, int nz, const bool periodic[3],
                                  std::vector<std::uint32_t>& unk) {
  const long N = long(nx) * ny * nz;
  unk.assign(std::size_t(N), 0u);
  auto in_field = [&](int x, int y, int z) {
    return flags[std::size_t(node_id(x, y, z, nx, ny))] != ScalarExcluded;
  };
  const int dim[3] = {nx, ny, nz};
  for (int z = 0; z < nz; ++z)
    for (int y = 0; y < ny; ++y)
      for (int x = 0; x < nx; ++x) {
        const long n = node_id(x, y, z, nx, ny);
        if (flags[std::size_t(n)] != ScalarMoment) continue;
        std::uint32_t m = 0;
        for (int i = 0; i < L::Q; ++i) {
          int sxyz[3] = {x - L::cx(i),
                         y - L::cy(i),
                         z - L::cz(i)};
          bool outside = false;
          for (int a = 0; a < 3; ++a) {
            if (periodic[a]) sxyz[a] = wrap(sxyz[a], dim[a]);
            else if (sxyz[a] < 0 || sxyz[a] >= dim[a]) outside = true;
          }
          if (!outside && !in_field(sxyz[0], sxyz[1], sxyz[2])) outside = true;
          if (outside) m |= (1u << i);
        }
        unk[std::size_t(n)] = m;
      }
}

//------------------------------------------------------------------------------
// A per-node source, added as w_i S.
//
// IT VISITS ScalarSpecular AS WELL AS ScalarBulk, and that is the whole reason
// this comment exists. The four prescribed cell types are overwritten
// downstream, so a source added to them is thrown away -- but a specular node
// is a BULK node carrying a mirror closure, and withholding the source there
// leaves its PDE unsolved in the wall column. In the parent tree that mistake
// read as a mediocre boundary condition rather than as a bug: I0 came out
// +8.87 % against +0.51 % for a boundary-free box, and the order fell from two
// to one. Nothing failed.
//------------------------------------------------------------------------------
template <int Parity, class L = ScalarLattice>
LBM_HD LBM_INLINE void scalar_source_node(const ScalarParams& p, long N, long n) {
  const std::uint8_t fl = p.flags ? p.flags[n] : std::uint8_t(ScalarBulk);
  if (fl != ScalarBulk && fl != ScalarSpecular) return;
  const Real S = p.src[n];
  if (S == Real(0)) return;
  int x, y, z;
  coords(n, p.nx, p.ny, x, y, z);
  constexpr int Q = L::Q;
  Real h[Q];
  gather<Parity, L>(p.h, N, x, y, z, p.nx, p.ny, p.nz, h);
  for (int i = 0; i < Q; ++i) h[i] += L::w(i) * S;
  // STORE WITH THE PAIR SWAPPED, and this is the whole subtlety of the file.
  // gather and scatter are a STREAMING pair, not a read-modify-write: scatter
  // puts in[i] into the slot gather took out[i+1] from. Handing the values
  // straight back therefore advances the field by one step every time a source
  // is added. Swapping the pair undoes the crossing -- it is the same identity
  // that lets an adiabatic cell be skipped entirely.
  //
  // A UNIFORM FIELD IS INVARIANT UNDER STREAMING, so the obvious test of a
  // source (constant S, flat field, does it climb at S per step?) passes either
  // way. It did. What caught this was the potential in ehd_cavity.cu coming out
  // with a first-cell gradient 40 % short of the interior one.
  Real out[Q];
  out[0] = h[0];
  for (int i = 1; i < Q; i += 2) { out[i] = h[i + 1]; out[i + 1] = h[i]; }
  scatter<Parity, L>(p.h, N, x, y, z, p.nx, p.ny, p.nz, out);
}

//------------------------------------------------------------------------------
// Donor for every ScalarOutflow node: the interior cell whose value it copies.
//
// The OUTWARD directions are the axes whose neighbour is outside the field, and
// the donor is the neighbour one step INWARD along all of them at once -- so a
// face node takes its axis neighbour, an edge node the diagonal and a corner
// node the body diagonal. Without that, a box edge has no purely axial interior
// neighbour at all and every edge and corner of an open box would be inert.
//
// A donor must be ScalarBulk, NEVER another outflow node: the second pass reads
// the field at the donor, and reading it at a node the same pass writes would
// put the race straight back. A node with no bulk neighbour is left pointing at
// itself -- the condition is then a no-op there and it falls back to
// bounce-back -- and such nodes are counted and reported rather than silently
// reading whatever is next door.
//
// Plain host code, run once at set_geometry, shared by the CUDA and host
// drivers. Returns the number of outflow nodes found; `degenerate` comes back
// with how many of them are inert.
//------------------------------------------------------------------------------
inline long build_scalar_donors(const std::vector<std::uint8_t>& flags,
                                int nx, int ny, int nz, const bool periodic[3],
                                std::vector<long>& donor, long& degenerate) {
  const long N = long(nx) * ny * nz;
  donor.resize(std::size_t(N));
  for (long n = 0; n < N; ++n) donor[std::size_t(n)] = n;

  auto flag_at = [&](int x, int y, int z) -> std::uint8_t {
    // Periodic indexing, matching neighbour() in streaming.cuh: "outside the
    // field" is a flag, not an index, so a periodic axis simply wraps.
    return flags[std::size_t(node_id(wrap(x, nx), wrap(y, ny), wrap(z, nz), nx, ny))];
  };

  const int dirs[6][3] = {{-1,0,0},{1,0,0},{0,-1,0},{0,1,0},{0,0,-1},{0,0,1}};
  long nout = 0;
  degenerate = 0;
  for (int z = 0; z < nz; ++z)
    for (int y = 0; y < ny; ++y)
      for (int x = 0; x < nx; ++x) {
        const long n = node_id(x, y, z, nx, ny);
        if (flags[std::size_t(n)] != ScalarOutflow) continue;
        ++nout;
        // OUTWARD MEANS "there is nothing usable that way", and there are TWO
        // ways for that to be true. The neighbour may be marked out of the
        // transport -- which is how a channel with bounce-back walls says it --
        // or it may simply be off a non-periodic edge, which is how a box with
        // ON-NODE walls says it, because such a box excludes nothing at all.
        // Only the first was tested, so the collector corners of
        // GPU/src/ehd_cavity.cu found no outward axis, fell through to the
        // axial fallback, and came out inert: their four axial neighbours are
        // two more outflow nodes, a specular wall column, and -- through the
        // wrap -- the injector.
        const int dim[3] = {nx, ny, nz};
        int ix = 0, iy = 0, iz = 0;
        for (int k = 0; k < 6; ++k) {
          const int sx = x + dirs[k][0], sy = y + dirs[k][1], sz = z + dirs[k][2];
          const int c[3] = {sx, sy, sz};
          bool out = false;
          for (int a = 0; a < 3; ++a)
            if (!periodic[a] && (c[a] < 0 || c[a] >= dim[a])) out = true;
          if (!out && flag_at(sx, sy, sz) != ScalarExcluded) continue;
          ix -= dirs[k][0];  iy -= dirs[k][1];  iz -= dirs[k][2];
        }
        long best = n;
        if ((ix || iy || iz) && flag_at(x + ix, y + iy, z + iz) == ScalarBulk)
          best = node_id(wrap(x + ix, nx), wrap(y + iy, ny), wrap(z + iz, nz), nx, ny);
        if (best == n)                        // fall back to any bulk neighbour
          for (int k = 0; k < 6; ++k)
            if (flag_at(x + dirs[k][0], y + dirs[k][1], z + dirs[k][2]) == ScalarBulk) {
              best = node_id(wrap(x + dirs[k][0], nx), wrap(y + dirs[k][1], ny),
                             wrap(z + dirs[k][2], nz), nx, ny);
              break;
            }
        donor[std::size_t(n)] = best;
        if (best == n) ++degenerate;
      }
  if (degenerate)
    std::fprintf(stderr,
                 "  [scalar] %ld of %ld outflow node(s) have no bulk neighbour "
                 "and are inert\n", degenerate, nout);
  return nout;
}

#if defined(__CUDACC__)

template <int Parity, bool Advected, bool HasGeometry, bool HasOutflow,
          class L = ScalarLattice>
__global__ void scalar_kernel(ScalarParams p, long N) {
  const long n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  scalar_node_update<Parity, Advected, HasGeometry, HasOutflow, L>(p, N, n);
}

template <int Parity, bool Advected, class L = ScalarLattice>
__global__ void scalar_outflow_kernel(ScalarParams p, long N) {
  const long n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  scalar_outflow_node<Parity, Advected, L>(p, N, n);
}

template <int Parity, class L = ScalarLattice>
__global__ void scalar_source_kernel(ScalarParams p, long N) {
  const long n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  scalar_source_node<Parity, L>(p, N, n);
}

template <int Parity, bool HasGeometry, class L = ScalarLattice>
__global__ void scalar_field_kernel(ScalarParams p, long N) {
  const long n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  scalar_field_node<Parity, HasGeometry, L>(p, N, n);
}

//------------------------------------------------------------------------------
// Seed at equilibrium with zero velocity. A Dirichlet node is seeded at its own
// wall value, so the boundary is consistent from step zero rather than relaxing
// into place over the first few steps.
//------------------------------------------------------------------------------
template <class Init>
__global__ void scalar_initialise(Real* __restrict__ h,
                                  const std::uint8_t* __restrict__ flags,
                                  const Real* __restrict__ wall,
                                  int nx, int ny, int nz, Real T_ref, Init init) {
  const long N = long(nx) * ny * nz;
  const long n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  int x, y, z;
  coords(n, nx, ny, x, y, z);
  const Real T = (flags[n] == ScalarDirichlet || flags[n] == ScalarMoment)
                 ? wall[n] : init(x, y, z);
  Real g[ScalarLattice::Q];
  for (int i = 0; i < ScalarLattice::Q; ++i)
    g[i] = scalar_eq<ScalarLattice>(i, T - T_ref, T_ref, Real(0), Real(0), Real(0));
  init_scatter<0, ScalarLattice>(h, N, x, y, z, nx, ny, nz, g);
}

//==============================================================================
//  Host-side driver.
//==============================================================================
//------------------------------------------------------------------------------
//  The solver is templated on its LATTICE, and that is the whole of the charge
//  port. A charge carrier is this same machinery on D3Q27 -- a product lattice,
//  so the full-order equilibrium is the discrete Maxwellian -- advected at the
//  DRIFT velocity u + K E and collided by ScalarOp::ChargeCM. Nothing else about
//  it differs, which is why it is not a separate solver.
//------------------------------------------------------------------------------
template <class L>
class ScalarSolverT {
 public:
  ScalarSolverT(int nx, int ny, int nz, Real diffusivity, Real T_ref = Real(0),
               ScalarOp op = ScalarOp::BGK)
      : nx_(nx), ny_(ny), nz_(nz), T_ref_(T_ref), op_(op) {
    omega_ = omega_from_diffusivity<L>(diffusivity);
    N_ = long(nx) * ny * nz;
    LBM_CUDA_CHECK(cudaMalloc(&h_, sizeof(Real) * L::Q * N_));
    LBM_CUDA_CHECK(cudaMalloc(&flags_, sizeof(std::uint8_t) * N_));
    LBM_CUDA_CHECK(cudaMemset(flags_, ScalarBulk, sizeof(std::uint8_t) * N_));
    LBM_CUDA_CHECK(cudaMalloc(&wall_, sizeof(Real) * N_));
    LBM_CUDA_CHECK(cudaMemset(wall_, 0, sizeof(Real) * N_));
    LBM_CUDA_CHECK(cudaMalloc(&T_, sizeof(Real) * N_));
    LBM_CUDA_CHECK(cudaMemset(T_, 0, sizeof(Real) * N_));
  }
  ~ScalarSolverT() {
    cudaFree(h_); cudaFree(flags_); cudaFree(wall_); cudaFree(T_);
    cudaFree(donor_); cudaFree(unk_); cudaFree(spec_);
  }

  ScalarSolverT(const ScalarSolverT&) = delete;
  ScalarSolverT& operator=(const ScalarSolverT&) = delete;

  // WHICH AXES WRAP. Call BEFORE set_geometry. It changes nothing about the
  // streaming, which always wraps; it tells the on-node conditions which of
  // their neighbours are real -- see build_scalar_unknowns.
  void set_periodicity(bool px, bool py, bool pz) {
    periodic_[0] = px; periodic_[1] = py; periodic_[2] = pz;
  }

  // Outward normals for ScalarSpecular cells, one NormalCode per node.
  void set_specular_walls(const std::vector<std::uint8_t>& nrm) {
    if (!spec_) LBM_CUDA_CHECK(cudaMalloc(&spec_, sizeof(std::uint8_t) * N_));
    LBM_CUDA_CHECK(cudaMemcpy(spec_, nrm.data(), sizeof(std::uint8_t) * N_,
                              cudaMemcpyHostToDevice));
  }

  // Add w_i S[n] to every bulk and specular node, at the CURRENT parity, so it
  // lands on the state the next step() will read.
  void add_source(const Real* S_device) {
    src_ = const_cast<Real*>(S_device);
    const int B = 256, G = int((N_ + B - 1) / B);
    if (t_ % 2 == 0) scalar_source_kernel<0, L><<<G, B>>>(params(), N_);
    else             scalar_source_kernel<1, L><<<G, B>>>(params(), N_);
    LBM_CUDA_CHECK(cudaGetLastError());
  }

  // Cell roles and, for Dirichlet cells, the value each one holds.
  void set_geometry(const std::vector<std::uint8_t>& flags,
                    const std::vector<Real>& wall) {
    LBM_CUDA_CHECK(cudaMemcpy(flags_, flags.data(), sizeof(std::uint8_t) * N_,
                              cudaMemcpyHostToDevice));
    LBM_CUDA_CHECK(cudaMemcpy(wall_, wall.data(), sizeof(Real) * N_,
                              cudaMemcpyHostToDevice));
    has_geometry_ = true;

    std::vector<std::uint32_t> unk;
    build_scalar_unknowns<L>(flags, nx_, ny_, nz_, periodic_, unk);
    if (!unk_) LBM_CUDA_CHECK(cudaMalloc(&unk_, sizeof(std::uint32_t) * N_));
    LBM_CUDA_CHECK(cudaMemcpy(unk_, unk.data(), sizeof(std::uint32_t) * N_,
                              cudaMemcpyHostToDevice));

    std::vector<long> donor;
    long degenerate = 0;
    const long nout = build_scalar_donors(flags, nx_, ny_, nz_, periodic_, donor,
                                         degenerate);
    has_outflow_ = nout > 0;
    if (has_outflow_) {
      if (!donor_) LBM_CUDA_CHECK(cudaMalloc(&donor_, sizeof(long) * N_));
      LBM_CUDA_CHECK(cudaMemcpy(donor_, donor.data(), sizeof(long) * N_,
                                cudaMemcpyHostToDevice));
    }
  }

  // Device pointers owned by the fluid solver. Without this the scalar diffuses
  // but does not advect, which is a legitimate mode (pure conduction).
  void advect_with(const Real* ux, const Real* uy, const Real* uz) {
    ux_ = ux; uy_ = uy; uz_ = uz;
  }

  template <class Init>
  void initialise_with(Init init) {
    const int B = 128;
    scalar_initialise<<<int((N_ + B - 1) / B), B>>>(h_, flags_, wall_,
                                                    nx_, ny_, nz_, T_ref_, init);
    LBM_CUDA_CHECK(cudaGetLastError());
    LBM_CUDA_CHECK(cudaDeviceSynchronize());
    t_ = 0;
    compute_field();
  }

  void step() {
    const int B = 128, G = int((N_ + B - 1) / B);
    // The outflow pass must see the main pass's field writes. A kernel boundary
    // on the same stream IS that fence, so no explicit synchronise is needed --
    // and the pass is not launched at all without an open boundary.
    if (t_ % 2 == 0) {
      launch_advect<0>(G, B);
      if (has_outflow_) launch_outflow<0>(G, B);
    } else {
      launch_advect<1>(G, B);
      if (has_outflow_) launch_outflow<1>(G, B);
    }
    LBM_CUDA_CHECK(cudaGetLastError());
    ++t_;
  }

  // Refresh T on the device. Call before the fluid step in a coupled run --
  // see the coupling-order note in solver.cuh.
  void compute_field() {
    const int B = 128, G = int((N_ + B - 1) / B);
    if (t_ % 2 == 0) {
      if (has_geometry_) scalar_field_kernel<0, true, L><<<G, B>>>(params(), N_);
      else               scalar_field_kernel<0, false, L><<<G, B>>>(params(), N_);
    } else {
      if (has_geometry_) scalar_field_kernel<1, true, L><<<G, B>>>(params(), N_);
      else               scalar_field_kernel<1, false, L><<<G, B>>>(params(), N_);
    }
    LBM_CUDA_CHECK(cudaGetLastError());
  }

  void field_to_host(std::vector<Real>& T) {
    compute_field();
    LBM_CUDA_CHECK(cudaDeviceSynchronize());
    T.resize(N_);
    LBM_CUDA_CHECK(cudaMemcpy(T.data(), T_, sizeof(Real) * N_, cudaMemcpyDeviceToHost));
  }

  const Real* field_device() const { return T_; }
  Real omega() const { return omega_; }
  Real diffusivity() const { return diffusivity_from_omega<L>(omega_); }
  std::size_t timestep() const { return t_; }

 private:
  template <int P> void launch_advect(int G, int B) {
    if (ux_) launch_geom<P, true>(G, B);
    else     launch_geom<P, false>(G, B);
  }
  template <int P, bool A> void launch_geom(int G, int B) {
    if (has_outflow_) {
      // has_outflow_ implies has_geometry_: outflow is a flag.
      scalar_kernel<P, A, true, true, L><<<G, B>>>(params(), N_);
    } else if (has_geometry_) {
      scalar_kernel<P, A, true, false, L><<<G, B>>>(params(), N_);
    } else {
      scalar_kernel<P, A, false, false, L><<<G, B>>>(params(), N_);
    }
  }
  template <int P> void launch_outflow(int G, int B) {
    if (ux_) scalar_outflow_kernel<P, true, L><<<G, B>>>(params(), N_);
    else     scalar_outflow_kernel<P, false, L><<<G, B>>>(params(), N_);
  }

  ScalarParams params() const {
    ScalarParams p;
    p.h = h_; p.flags = flags_; p.wall = wall_;
    p.ux = ux_; p.uy = uy_; p.uz = uz_;
    p.T_out = T_;
    p.donor = donor_;
    p.unk = unk_; p.spec = spec_; p.src = src_;
    p.nx = nx_; p.ny = ny_; p.nz = nz_;
    p.omega = omega_; p.T_ref = T_ref_;
    p.op = op_;
    return p;
  }

  int nx_, ny_, nz_;
  long N_;
  Real T_ref_, omega_;
  ScalarOp op_ = ScalarOp::BGK;
  Real* h_ = nullptr;
  std::uint32_t* unk_ = nullptr;
  std::uint8_t* spec_ = nullptr;
  Real* src_ = nullptr;
  bool periodic_[3] = {true, true, true};
  std::uint8_t* flags_ = nullptr;
  Real* wall_ = nullptr;
  Real* T_ = nullptr;
  long* donor_ = nullptr;
  const Real *ux_ = nullptr, *uy_ = nullptr, *uz_ = nullptr;
  bool has_geometry_ = false;
  bool has_outflow_ = false;
  std::size_t t_ = 0;
};

using ScalarSolver = ScalarSolverT<ScalarLattice>;
using ChargeSolver = ScalarSolverT<D3Q27>;

#endif  // __CUDACC__

}  // namespace lbm
