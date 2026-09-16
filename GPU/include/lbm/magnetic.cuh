#pragma once
//==============================================================================
//  Magnetohydrodynamics -- the induction half.
//
//  The magnetic field is carried by Dellar's vector-valued distribution: a
//  separate D3Q7 population set for each Cartesian component of B, so 3 x 7 = 21
//  populations per node against the fluid's 27. The equilibrium is in core.cuh;
//  what lives here is the streaming, the coupling to u, and the driver.
//
//  Storage is g[(a * Q + i) * N + n] -- component-major, so each component's
//  block is contiguous and the SAME streaming functions the fluid uses operate
//  on it unchanged, one component at a time, by offsetting the base pointer.
//
//  THE LORENTZ COUPLING DOES NOT LIVE HERE. It is applied by giving the FLUID
//  equilibrium the Maxwell stress in its second moment (see `maxwell` in
//  core.cuh and `couple_magnetic` in solver.cuh), not by computing div(BB) and
//  applying it as a body force. The force route needs derivatives of B and loses
//  an order; this route is exact at the level of the equilibrium.
//
//  ALL THREE COMPONENTS ARE GATHERED BEFORE ANY IS COLLIDED. The equilibrium of
//  component a depends on the whole vector B, so collide-as-you-go would feed
//  post-collision values of B_x into the equilibrium of B_y. That is a silent
//  error: the field still evolves, still looks like a field, and damps at the
//  wrong rate.
//
//  BOUNDARIES. Periodic only, in practice. A non-fluid cell is skipped, which on
//  Esoteric Pull's storage means bounce-back on g -- and bounce-back on the
//  induction distribution is NOT a physical magnetic wall. It is neither the
//  perfectly conducting nor the insulating condition; Dellar's moment-based
//  wall is what those need, and it is not implemented here. The skip exists so a
//  domain with geometry runs rather than reads uninitialised storage. Do not
//  read a wall-bounded MHD result off this code.
//==============================================================================
#include "streaming.cuh"

namespace lbm {

using MagneticLattice = D3Q7;

//------------------------------------------------------------------------------
// Per-node magnetic boundary kind. A port of MagWallCode in
// ../src/solver/MagneticSolver.hpp.
//
//   MagBulk       collide normally
//   MagDirichlet  the node takes an imposed external field, by Dellar's
//                 moment condition (core.cuh) -- attained AT the node
//   MagOutflow    zero gradient: the node takes B from its upstream neighbour
//
//   MagNeumann    dB/dn = 0, the PERFECTLY CONDUCTING wall
//
// MagNeumann IS MagOutflow WITH A DIRECTION. Mechanically both take the field
// from inward neighbours; the difference is that outflow is hard-coded to -x
// while Neumann reads its face's outward normal, which is what lets it sit on
// any face of a duct. Physically they are not the same thing at all: at an
// INSULATING wall the induced field has nowhere to go and vanishes, which is
// Dirichlet on the total field; at a PERFECTLY CONDUCTING wall the tangential
// electric field vanishes instead, and in the induced-field formulation that is
// a zero NORMAL DERIVATIVE. Shercliff's duct against Hunt's -- measurably
// different flows, not a refinement of one another.
//
// IT NEEDS A DIRICHLET BOUNDARY SOMEWHERE, and that is a restriction rather
// than a caution. The parent measured a CLOSED domain that is Neumann on every
// face going non-finite inside sixty steps at the grid scale -- magnetic energy
// up 263x in twenty steps, max|b| 0.14 -> 19.2 -- and ruled out both the
// staircase and omega -> 2 as causes. A pure Neumann problem does not determine
// the field LEVEL. Use it for a conducting wall PAIR whose remaining boundary
// is Dirichlet, which is what Hunt's duct is; do not close a domain with it.
//
// WHAT IS STILL NOT HERE: a real conducting wall that couples the interior
// field to a wall current, and an insulating one that matches onto an exterior
// vacuum field. Both are a separate piece of work and are NOT faked.
//
// MagOutflow IS NOT VALIDATED AND HAS A MEASURED DEFECT, inherited from the
// parent along with the condition: on an inlet-driven Hartmann flow whose exact
// solution is independent of x -- so any streamwise variation IS boundary error
// -- it drives B about 6% high over the last ten nodes, and moving the outlet
// from Lx = 121 to 241 cut the sampled error 3.4x. Keep the outlet far from
// anything being measured. The cause is not identified; the natural suspects
// are the interaction with div B = 0 and the fact that a pure Neumann condition
// does not constrain the field LEVEL at all.
//------------------------------------------------------------------------------
enum MagCell : std::uint8_t {
  MagBulk      = 0,
  MagDirichlet = 1,
  MagOutflow   = 2,
  MagNeumann   = 3,
};

// The OUTWARD normal of a Neumann node's face. The node reads its field one and
// two steps INWARD from it, so the face is what turns a hard-coded -x outflow
// into a condition that can sit on any face of a duct.
enum MagFace : std::uint8_t {
  MFaceXm = 0, MFaceXp = 1, MFaceYm = 2, MFaceYp = 3, MFaceZm = 4, MFaceZp = 5,
  MFaceNone = 255,
};

struct MagneticParams {
  Real* g = nullptr;                         // 3 * Q * N
  const std::uint8_t* flags = nullptr;       // the fluid's geometry
  const std::uint8_t* mface = nullptr;      // outward face of a Neumann node
  const Real* ux = nullptr;
  const Real* uy = nullptr;
  const Real* uz = nullptr;
  Real* Bx = nullptr;                        // written by compute_field only
  Real* By = nullptr;
  Real* Bz = nullptr;
  // Magnetic boundary kinds, and the imposed field at a Dirichlet node.
  //
  // THREE FULL ARRAYS RATHER THAN THE PARENT'S TAG TABLE. The parent stores one
  // byte per node indexing a table of at most 256 distinct wall fields, which
  // is thriftier and imposes a limit; here it is 12 bytes per node with no
  // limit, allocated only when walls are set at all. On a device where the
  // magnetic populations already cost 84 bytes per node that is a 14% surcharge
  // on runs that have walls and nothing on runs that do not.
  // A PER-NODE VECTOR SOURCE, added POST-collision and on the weights.
  //
  // What it is for: the reference's Eq. (2) penalisation, S = -chi (B.n) n / eps,
  // which damps the NORMAL component of B and leaves the tangential one free --
  // a perfect conductor coated in insulant. Anything per-node and vector-valued
  // fits the same hook.
  //
  // WHY THE WEIGHTS. sum_i w_i = 1 puts S_a straight into B, and sum_i w_i c_i = 0
  // leaves the induction flux u B - B u untouched, so the source moves the field
  // and not its transport. Added between collide and scatter, which is also why
  // it needs no pair-swap correction: gather and scatter are a STREAMING pair,
  // and a separate pass that gathered, added and scattered back would advance the
  // field by a step rather than update it in place -- the trap CLAUDE.md records
  // for the EHD Poisson source.
  //
  // Null-checked at runtime rather than templated: the pointer is uniform across
  // every node, so the branch predicts perfectly and costs nothing, where a
  // fourth template axis would double twelve kernel instantiations to twenty-four.
  const Real* sx = nullptr;
  const Real* sy = nullptr;
  const Real* sz = nullptr;
  const std::uint8_t* mwall = nullptr;
  const std::uint8_t* unknown = nullptr;     // directions that streamed in from outside
  const Real* wBx = nullptr;
  const Real* wBy = nullptr;
  const Real* wBz = nullptr;
  int nx = 0, ny = 0, nz = 0;
  Real omega = Real(1);
};

LBM_HD LBM_INLINE long magnetic_offset(int a, long N) {
  return long(a) * long(MagneticLattice::Q) * N;
}

// `Advected` is false for a motionless conductor -- resistive decay isolates the
// induction equation and its resistivity with no coupling to a flow at all, and
// it is the first thing to check when a magnetic result looks wrong.
template <int Parity, bool Advected, bool HasGeometry, bool HasWalls = false>
LBM_HD LBM_INLINE void magnetic_node_update(const MagneticParams& p, long N, long n) {
  if (HasGeometry && p.flags[n] != Fluid) return;

  int x, y, z;
  coords(n, p.nx, p.ny, x, y, z);

  constexpr int Q = MagneticLattice::Q;
  Real g[3][Q];
  for (int a = 0; a < 3; ++a)
    gather<Parity, MagneticLattice>(p.g + magnetic_offset(a, N), N,
                                    x, y, z, p.nx, p.ny, p.nz, g[a]);

  // THE FIELD COMES FROM THE FIELD ARRAY, NOT FROM THE RAW SUM, whenever walls
  // exist. At a moment wall the streamed populations are still missing their
  // inward direction, so their sum is NOT B -- feeding that into the induction
  // equilibrium is how a wall-bounded run diverges with nothing to point at.
  // The field pass has already put the right number there for every node,
  // imposed at a Dirichlet node and taken from upstream at an outflow one.
  Real B[3];
  if (HasWalls) {
    B[0] = p.Bx[n];  B[1] = p.By[n];  B[2] = p.Bz[n];
  } else {
    for (int a = 0; a < 3; ++a) {
      B[a] = Real(0);
      for (int i = 0; i < Q; ++i) B[a] += g[a][i];
    }
  }

  Real u[3] = {Real(0), Real(0), Real(0)};
  if (Advected) { u[0] = p.ux[n]; u[1] = p.uy[n]; u[2] = p.uz[n]; }

  const std::uint8_t code = HasWalls ? p.mwall[n] : std::uint8_t(MagBulk);
  for (int a = 0; a < 3; ++a) {
    // Eqs. (13a)-(13b): choose the inward-pointing populations so the zeroth
    // moment is the target, THEN collide as usual. Dirichlet and outflow differ
    // only in where the target number comes from.
    if (HasWalls && code != MagBulk)
      impose_moment<MagneticLattice>(g[a], B[a], p.unknown[n]);
    collide_magnetic<MagneticLattice>(g[a], a, B, u, p.omega);
    if (p.sx) {
      const Real sa = (a == 0) ? p.sx[n] : (a == 1) ? p.sy[n] : p.sz[n];
      for (int i = 0; i < Q; ++i) g[a][i] += MagneticLattice::w(i) * sa;
    }
    scatter<Parity, MagneticLattice>(p.g + magnetic_offset(a, N), N,
                                     x, y, z, p.nx, p.ny, p.nz, g[a]);
  }
}

template <int Parity, bool HasGeometry, bool HasWalls = false>
LBM_HD LBM_INLINE void magnetic_field_node(const MagneticParams& p, long N, long n) {
  if (HasGeometry && p.flags[n] != Fluid) { p.Bx[n] = p.By[n] = p.Bz[n] = Real(0); return; }
  int x, y, z;
  coords(n, p.nx, p.ny, x, y, z);

  if (HasWalls) {
    const std::uint8_t code = p.mwall[n];
    // A Dirichlet node's field is the imposed one BY CONSTRUCTION. Reading its
    // raw population sum instead gives whatever streamed in minus the direction
    // that has not been chosen yet, which is not a field.
    if (code == MagDirichlet) {
      p.Bx[n] = p.wBx[n];  p.By[n] = p.wBy[n];  p.Bz[n] = p.wBz[n];
      return;
    }
    // Zero-gradient outflow: same problem, different answer. Take the UPSTREAM
    // neighbour's populations. Reading a neighbour is safe in this kernel and
    // only in this kernel -- it reads populations and writes one B per thread,
    // so nothing it reads is written here. Reading B(upstream) instead would
    // race against whoever writes it.
    if (code == MagOutflow) x = wrap(x - 1, p.nx);

    // dB/dn = 0, to SECOND order, from TWO neighbours: B_wall = (4 B_1 - B_2)/3.
    //
    // Copying one neighbour imposes the condition only to FIRST order, and the
    // parent measured that as not a detail: on Hunt's duct the velocity error
    // grew with Hartmann number -- 3.0e-3 / 6.6e-2 / 1.7e-1 at Ha = 1/5/10 --
    // and the side jets came out at 1.579 against an analytic 1.853, because a
    // conducting wall carries the return current and the whole solution leans
    // on it. The field LEVEL was never the problem there (mean offset
    // -1.4e-18); the error was always SHAPE.
    //
    // Populations are read, never a neighbour's B, for the same reason as the
    // outflow branch above: B is being written by another thread right now.
    if (code == MagNeumann) {
      const std::uint8_t f = p.mface ? p.mface[n] : std::uint8_t(MFaceNone);
      const int ex = (f == MFaceXm) ? 1 : (f == MFaceXp) ? -1 : 0;
      const int ey = (f == MFaceYm) ? 1 : (f == MFaceYp) ? -1 : 0;
      const int ez = (f == MFaceZm) ? 1 : (f == MFaceZp) ? -1 : 0;
      Real gm[MagneticLattice::Q], Bn[3];
      for (int a = 0; a < 3; ++a) {
        Real bb[2];
        for (int m = 1; m <= 2; ++m) {
          gather<Parity, MagneticLattice>(
              p.g + magnetic_offset(a, N), N,
              wrap(x + m * ex, p.nx), wrap(y + m * ey, p.ny),
              wrap(z + m * ez, p.nz), p.nx, p.ny, p.nz, gm);
          bb[m - 1] = Real(0);
          for (int i = 0; i < MagneticLattice::Q; ++i) bb[m - 1] += gm[i];
        }
        Bn[a] = (Real(4) * bb[0] - bb[1]) / Real(3);
      }
      p.Bx[n] = Bn[0]; p.By[n] = Bn[1]; p.Bz[n] = Bn[2];
      return;
    }
  }

  Real g[MagneticLattice::Q], B[3];
  for (int a = 0; a < 3; ++a) {
    gather<Parity, MagneticLattice>(p.g + magnetic_offset(a, N), N,
                                    x, y, z, p.nx, p.ny, p.nz, g);
    B[a] = Real(0);
    for (int i = 0; i < MagneticLattice::Q; ++i) B[a] += g[i];
  }
  p.Bx[n] = B[0]; p.By[n] = B[1]; p.Bz[n] = B[2];
}

//------------------------------------------------------------------------------
// Every MagNeumann node must carry a face. Shared by the device solver and the
// host reference so the two cannot drift -- the whole value of having both is
// that they are the same rules twice, not two sets of rules.
//
// e = 0 is the failure worth refusing: with no inward direction the
// reconstruction (4 B - B)/3 collapses to B at the node itself, which is a
// plausible number and not a boundary condition.
//------------------------------------------------------------------------------
inline void validate_magnetic_faces(const std::vector<std::uint8_t>& kind,
                                    const std::vector<std::uint8_t>& face,
                                    long N) {
  for (long i = 0; i < N; ++i) {
    if (kind[std::size_t(i)] != MagNeumann) continue;
    const std::uint8_t f = face.empty() ? std::uint8_t(MFaceNone)
                                        : face[std::size_t(i)];
    if (f > MFaceZp) {
      std::fprintf(stderr,
          "set_walls: node %ld is MagNeumann with no face (got %u). A Neumann "
          "node reads its field one and two steps along its INWARD normal, so "
          "it cannot be placed without one.\n", i, unsigned(f));
      std::exit(1);
    }
  }
}

//------------------------------------------------------------------------------
// Build the unknown-direction mask for every magnetic wall node.
//
// A direction is unknown when the node it would have streamed FROM is not part
// of the transport. `geom` is the fluid's geometry; empty means every node is
// fluid, so only nodes on a solid face are walls and a fully periodic array has
// none. Returns the number of wall nodes; `blind` comes back with how many of
// them have no unknown direction at all.
//
// A BLIND WALL NODE IS ALMOST ALWAYS A MIS-SPECIFIED GEOMETRY -- typically the
// wall marked one cell INSIDE the solid rather than on the fluid face -- and
// impose_moment then leaves it alone, so the condition silently does nothing.
// Hence the count, and the report.
//
// Plain host code, run once at setup, shared by the CUDA and host drivers.
//------------------------------------------------------------------------------
inline long build_magnetic_walls(const std::vector<std::uint8_t>& kind,
                                 const std::vector<std::uint8_t>& geom,
                                 int nx, int ny, int nz,
                                 std::vector<std::uint8_t>& unk, long& blind) {
  const long N = long(nx) * ny * nz;
  unk.assign(std::size_t(N), 0);
  auto outside = [&](int x, int y, int z) {
    if (geom.empty()) return false;
    return geom[std::size_t(node_id(x, y, z, nx, ny))] != Fluid;
  };
  long nwall = 0;
  blind = 0;
  for (int z = 0; z < nz; ++z)
    for (int y = 0; y < ny; ++y)
      for (int x = 0; x < nx; ++x) {
        const long n = node_id(x, y, z, nx, ny);
        if (kind[std::size_t(n)] == MagBulk) continue;
        ++nwall;
        unk[std::size_t(n)] = unknown_mask<MagneticLattice>(x, y, z, nx, ny, nz, outside);
        if (unk[std::size_t(n)] == 0) ++blind;
      }
  if (blind)
    std::fprintf(stderr,
                 "  [magnetic] %ld of %ld wall node(s) have no inward unknown "
                 "direction and are inert -- check the wall is on the fluid face\n",
                 blind, nwall);
  return nwall;
}

#if defined(__CUDACC__)

template <int Parity, bool Advected, bool HasGeometry, bool HasWalls>
__global__ void magnetic_kernel(MagneticParams p, long N) {
  const long n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  magnetic_node_update<Parity, Advected, HasGeometry, HasWalls>(p, N, n);
}

template <int Parity, bool HasGeometry, bool HasWalls>
__global__ void magnetic_field_kernel(MagneticParams p, long N) {
  const long n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  magnetic_field_node<Parity, HasGeometry, HasWalls>(p, N, n);
}

// Seeded at equilibrium with the initial velocity, not at rest: the equilibrium
// is u-dependent, and seeding it with u = 0 puts a transient into the induction
// equation that a decay-rate measurement then has to outlive.
template <class InitB, class InitU>
__global__ void magnetic_initialise(Real* __restrict__ g, int nx, int ny, int nz,
                                    InitB initB, InitU initU) {
  const long N = long(nx) * ny * nz;
  const long n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  int x, y, z;
  coords(n, nx, ny, x, y, z);

  Real B[3], u[3];
  initB(x, y, z, B);
  initU(x, y, z, u);
  Real gl[MagneticLattice::Q];
  for (int a = 0; a < 3; ++a) {
    for (int i = 0; i < MagneticLattice::Q; ++i)
      gl[i] = magnetic_eq<MagneticLattice>(i, a, B, u);
    init_scatter<0, MagneticLattice>(g + magnetic_offset(a, N), N,
                                     x, y, z, nx, ny, nz, gl);
  }
}

//==============================================================================
//  Host-side driver.
//==============================================================================
class MagneticSolver {
 public:
  MagneticSolver(int nx, int ny, int nz, Real resistivity)
      : nx_(nx), ny_(ny), nz_(nz) {
    omega_ = omega_from_resistivity<MagneticLattice>(resistivity);
    N_ = long(nx) * ny * nz;
    LBM_CUDA_CHECK(cudaMalloc(&g_, sizeof(Real) * 3 * MagneticLattice::Q * N_));
    LBM_CUDA_CHECK(cudaMalloc(&flags_, sizeof(std::uint8_t) * N_));
    LBM_CUDA_CHECK(cudaMemset(flags_, Fluid, sizeof(std::uint8_t) * N_));
    LBM_CUDA_CHECK(cudaMalloc(&Bx_, sizeof(Real) * N_));
    LBM_CUDA_CHECK(cudaMalloc(&By_, sizeof(Real) * N_));
    LBM_CUDA_CHECK(cudaMalloc(&Bz_, sizeof(Real) * N_));
  }
  ~MagneticSolver() {
    cudaFree(g_); cudaFree(flags_); cudaFree(Bx_); cudaFree(By_); cudaFree(Bz_);
    cudaFree(mwall_); cudaFree(unk_); cudaFree(wBx_); cudaFree(wBy_); cudaFree(wBz_);
    cudaFree(mface_);
  }

  MagneticSolver(const MagneticSolver&) = delete;
  MagneticSolver& operator=(const MagneticSolver&) = delete;

  void set_geometry(const std::vector<std::uint8_t>& flags) {
    LBM_CUDA_CHECK(cudaMemcpy(flags_, flags.data(), sizeof(std::uint8_t) * N_,
                              cudaMemcpyHostToDevice));
    has_geometry_ = true;
    geom_ = flags;
  }

  //--------------------------------------------------------------------------
  // Magnetic walls. `kind` is one MagCell per node and `wall` the imposed field
  // at each Dirichlet node (three components, ignored elsewhere).
  //
  // CALL set_geometry FIRST if the run has any. The unknown-direction mask is
  // built from BOTH: a direction is unknown when its source is outside the
  // domain OR is a non-fluid cell, and this needs the fluid's geometry to know
  // the second. With no geometry set, every node is fluid and only the domain
  // edge counts -- which is right for a channel bounded by the array itself.
  //--------------------------------------------------------------------------
  // `face` is required only where `kind` is MagNeumann, and carries that node's
  // OUTWARD normal (MFaceXm..MFaceZp). Passing it empty is allowed and means
  // "no Neumann nodes"; a MagNeumann node with no face is refused rather than
  // silently reading its own cell, because e = 0 would make the reconstruction
  // (4B - B)/3 = B at the node itself -- a plausible number and not a boundary
  // condition.
  void set_walls(const std::vector<std::uint8_t>& kind,
                 const std::vector<Real>& wall_bx,
                 const std::vector<Real>& wall_by,
                 const std::vector<Real>& wall_bz,
                 const std::vector<std::uint8_t>& face = {}) {
    if (long(kind.size()) != N_) {
      std::fprintf(stderr, "set_walls: %zu flags for %ld nodes\n", kind.size(), N_);
      std::exit(1);
    }
    validate_magnetic_faces(kind, face, N_);

    std::vector<std::uint8_t> unk;
    long blind = 0;
    const long nwall = build_magnetic_walls(kind, geom_, nx_, ny_, nz_, unk, blind);

    if (!mwall_) {
      LBM_CUDA_CHECK(cudaMalloc(&mwall_, sizeof(std::uint8_t) * N_));
      LBM_CUDA_CHECK(cudaMalloc(&unk_,   sizeof(std::uint8_t) * N_));
      LBM_CUDA_CHECK(cudaMalloc(&wBx_,   sizeof(Real) * N_));
      LBM_CUDA_CHECK(cudaMalloc(&wBy_,   sizeof(Real) * N_));
      LBM_CUDA_CHECK(cudaMalloc(&wBz_,   sizeof(Real) * N_));
      LBM_CUDA_CHECK(cudaMalloc(&mface_, sizeof(std::uint8_t) * N_));
    }
    {
      // static_cast, NOT std::size_t(N_): the latter is the most vexing parse
      // -- `hf` becomes a FUNCTION declaration taking (size_t, uint8_t) -- and
      // this tree has now hit that five times. nvcc caught it; the host build
      // structurally could not, because MagneticSolver::set_walls in this file
      // is only instantiated on the device path.
      std::vector<std::uint8_t> hf(static_cast<std::size_t>(N_),
                                   static_cast<std::uint8_t>(MFaceNone));
      if (!face.empty()) hf = face;
      LBM_CUDA_CHECK(cudaMemcpy(mface_, hf.data(), sizeof(std::uint8_t) * N_,
                                cudaMemcpyHostToDevice));
    }
    LBM_CUDA_CHECK(cudaMemcpy(mwall_, kind.data(), sizeof(std::uint8_t) * N_,
                              cudaMemcpyHostToDevice));
    LBM_CUDA_CHECK(cudaMemcpy(unk_, unk.data(), sizeof(std::uint8_t) * N_,
                              cudaMemcpyHostToDevice));
    LBM_CUDA_CHECK(cudaMemcpy(wBx_, wall_bx.data(), sizeof(Real) * N_, cudaMemcpyHostToDevice));
    LBM_CUDA_CHECK(cudaMemcpy(wBy_, wall_by.data(), sizeof(Real) * N_, cudaMemcpyHostToDevice));
    LBM_CUDA_CHECK(cudaMemcpy(wBz_, wall_bz.data(), sizeof(Real) * N_, cudaMemcpyHostToDevice));
    has_walls_ = nwall > 0;
  }
  // All three arrays or none: a partially-null triple would silently apply the
  // source to some components and not others, which is the accident the parent's
  // set_source guard exists to refuse.
  void set_source(const Real* sx, const Real* sy, const Real* sz) {
    sx_ = sx; sy_ = sy; sz_ = sz;
  }
  void advect_with(const Real* ux, const Real* uy, const Real* uz) {
    ux_ = ux; uy_ = uy; uz_ = uz;
  }

  template <class InitB, class InitU>
  void initialise_with(InitB initB, InitU initU) {
    const int B = 128;
    magnetic_initialise<<<int((N_ + B - 1) / B), B>>>(g_, nx_, ny_, nz_, initB, initU);
    LBM_CUDA_CHECK(cudaGetLastError());
    LBM_CUDA_CHECK(cudaDeviceSynchronize());
    t_ = 0;
    compute_field();
  }

  void step() {
    // WITH WALLS THE COLLISION READS B FROM THE FIELD ARRAY, not from the raw
    // population sum -- at a moment wall the sum is not the field. So the field
    // pass must have run since the last step. A coupled driver calls
    // compute_field() itself before stepping the fluid, and the flag stops that
    // from being paid for twice; an uncoupled one gets it here and does not have
    // to know. A periodic run takes neither branch.
    if (has_walls_ && !field_current_) compute_field();
    const int B = 128, G = int((N_ + B - 1) / B);
    if (t_ % 2 == 0) launch_advect<0>(G, B);
    else             launch_advect<1>(G, B);
    LBM_CUDA_CHECK(cudaGetLastError());
    field_current_ = false;
    ++t_;
  }

  void compute_field() {
    const int B = 128, G = int((N_ + B - 1) / B);
    if (t_ % 2 == 0) launch_field<0>(G, B);
    else             launch_field<1>(G, B);
    LBM_CUDA_CHECK(cudaGetLastError());
    field_current_ = true;
  }

  void field_to_host(std::vector<Real>& bx, std::vector<Real>& by, std::vector<Real>& bz) {
    compute_field();
    LBM_CUDA_CHECK(cudaDeviceSynchronize());
    bx.resize(N_); by.resize(N_); bz.resize(N_);
    LBM_CUDA_CHECK(cudaMemcpy(bx.data(), Bx_, sizeof(Real) * N_, cudaMemcpyDeviceToHost));
    LBM_CUDA_CHECK(cudaMemcpy(by.data(), By_, sizeof(Real) * N_, cudaMemcpyDeviceToHost));
    LBM_CUDA_CHECK(cudaMemcpy(bz.data(), Bz_, sizeof(Real) * N_, cudaMemcpyDeviceToHost));
  }

  const Real* Bx_device() const { return Bx_; }
  const Real* By_device() const { return By_; }
  const Real* Bz_device() const { return Bz_; }
  Real omega() const { return omega_; }
  Real resistivity() const { return diffusivity_from_omega<MagneticLattice>(omega_); }
  std::size_t timestep() const { return t_; }

 private:
  template <int P> void launch_advect(int G, int B) {
    if (ux_) launch_geom<P, true>(G, B);
    else     launch_geom<P, false>(G, B);
  }
  template <int P, bool A> void launch_geom(int G, int B) {
    if (has_walls_)         magnetic_kernel<P, A, true,  true ><<<G, B>>>(params(), N_);
    else if (has_geometry_) magnetic_kernel<P, A, true,  false><<<G, B>>>(params(), N_);
    else                    magnetic_kernel<P, A, false, false><<<G, B>>>(params(), N_);
  }
  template <int P> void launch_field(int G, int B) {
    if (has_walls_)         magnetic_field_kernel<P, true,  true ><<<G, B>>>(params(), N_);
    else if (has_geometry_) magnetic_field_kernel<P, true,  false><<<G, B>>>(params(), N_);
    else                    magnetic_field_kernel<P, false, false><<<G, B>>>(params(), N_);
  }

  MagneticParams params() const {
    MagneticParams p;
    p.g = g_; p.flags = flags_;
    p.ux = ux_; p.uy = uy_; p.uz = uz_;
    p.sx = sx_; p.sy = sy_; p.sz = sz_;
    p.Bx = Bx_; p.By = By_; p.Bz = Bz_;
    p.mwall = mwall_; p.unknown = unk_; p.mface = mface_;
    p.wBx = wBx_; p.wBy = wBy_; p.wBz = wBz_;
    p.nx = nx_; p.ny = ny_; p.nz = nz_;
    p.omega = omega_;
    return p;
  }

  int nx_, ny_, nz_;
  long N_;
  Real omega_;
  Real* g_ = nullptr;
  std::uint8_t* flags_ = nullptr;
  Real *Bx_ = nullptr, *By_ = nullptr, *Bz_ = nullptr;
  std::uint8_t *mwall_ = nullptr, *unk_ = nullptr, *mface_ = nullptr;
  Real *wBx_ = nullptr, *wBy_ = nullptr, *wBz_ = nullptr;
  std::vector<std::uint8_t> geom_;            // host copy, for the unknown mask
  const Real *ux_ = nullptr, *uy_ = nullptr, *uz_ = nullptr;
  const Real *sx_ = nullptr, *sy_ = nullptr, *sz_ = nullptr;
  bool has_geometry_ = false;
  bool has_walls_ = false;
  bool field_current_ = false;
  std::size_t t_ = 0;
};

#endif  // __CUDACC__

}  // namespace lbm
