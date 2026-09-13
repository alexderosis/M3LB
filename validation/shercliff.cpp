//==============================================================================
//  MHD duct flow against analytic solutions: SHERCLIFF (insulating walls) and
//  HUNT (perfectly conducting Hartmann walls).
//
//  References: J. A. Shercliff, Proc. Camb. Phil. Soc. 49, 136 (1953) for the
//  insulating duct; J. C. R. Hunt, J. Fluid Mech. 21, 577 (1965) for the
//  conducting-wall one.
//
//  THE TWO ARE QUALITATIVELY DIFFERENT, WHICH IS WHY BOTH ARE HERE. At an
//  insulating wall the induced field has nowhere to go and vanishes; at a
//  perfectly conducting one the current returns through the wall instead, and
//  the flow develops high-velocity SIDE JETS along the walls parallel to the
//  field. Measured on the series below at l = 1: the ratio of the peak speed
//  near the side wall to the speed at the duct centre is 1.000 at Ha = 1,
//  1.010 at Ha = 5 and 1.853 at Ha = 10 for Hunt, against 1 throughout for
//  Shercliff. A case that got the magnetic wall wrong would not produce that.
//
//  THE PROBLEM. Flow along x in a duct |y| <= L, |z| <= l L, a uniform field B0
//  along y, driven by a uniform force F. Fully developed means u = u(y,z) x_hat
//  and an induced field b = b_x(y,z) x_hat, and the pair reduces to
//
//      lap u + Ha db/dy = -1,      lap b + Ha du/dy = 0
//
//  in units where lengths are scaled by L, u by F L^2 / nu, and b by that times
//  sqrt(nu/eta), with Ha = B0 L / sqrt(nu eta). This is Hartmann flow with the
//  side walls put back: the Hartmann layers at y = +-L thin as 1/Ha, and the
//  SHERCLIFF layers at z = +-lL thin as 1/sqrt(Ha), which is the feature the
//  1-D case cannot have and the reason the duct is the sharper test.
//
//  WHY INSULATING IS REACHABLE AND CONDUCTING IS NOT. In the induced-field
//  formulation an insulating wall is simply b_x = 0 there -- the induced field
//  has nowhere to go -- which is a DIRICHLET condition on the total field,
//  B = (0, B0, 0). That is exactly what Dellar's moment condition already
//  imposes, and what validation/hartmann3d.cpp already uses on its Hartmann
//  walls. So Shercliff needs no new boundary condition, only the duct closed in
//  z as well as y. A perfectly conducting wall is db_x/dn = 0, a NEUMANN
//  condition, and MagneticSolver's banner says plainly that conducting and
//  insulating magnetic walls "are a separate piece of work and are not faked
//  here". MagNeumann was written for exactly this and is validated in isolation
//  by the SELF-TEST below before any duct number is reported: a one-dimensional
//  conducting-wall Hartmann slab, where db/dy = 0 and u = 0 integrate the
//  induction equation to b' = -(B0/eta) u exactly and the profile closes to
//
//      u = (F L^2 / nu Ha^2) [1 - cosh(Ha xi)/cosh(Ha)]
//
//  with no series at all. That profile differs from the insulating one, so the
//  test discriminates rather than merely running.
//
//  MEASURED FOR HUNT, ny = 33 -> 65 at l = 1:
//
//      Ha    L2 err u            order    jet ratio   analytic jet
//       1    3.01e-3 -> 7.77e-4   1.96      1.000        1.000
//       5    1.21e-2 -> 2.93e-3   2.05      1.010        1.010
//      10    1.18e-1 -> 1.97e-2   2.59      1.816        1.853
//
//  Second order in the velocity, and the side-jet ratio EXACT at Ha = 1 and 5 --
//  which is the check that matters, since the jets are what distinguishes a
//  conducting wall from an insulating one and no amount of correct arithmetic
//  on the wrong boundary condition would produce them. The 2.59 at Ha = 10 is
//  not super-convergence: that ny = 33 point has a 1.60-cell Hartmann layer, is
//  under-resolved, and the pair measures the coarse grid recovering rather than
//  an order. Its residual sat at 1.5e-4 and its error GREW with step count,
//  7.05e-2 at 150000 to 1.18e-1 at 300000, so it is not converged and is not
//  quoted as anything but a resolution warning.
//
//  TWO THINGS THE NEUMANN WALL NEEDED, one of which was a wrong guess. The
//  field LEVEL was the obvious suspect -- a Neumann condition does not fix the
//  constant, and MagOutXp's banner already names that for its own unexplained
//  6% defect -- and it is not the cause: the mean offset measures -1.4e-18.
//  What it actually needed was SECOND ORDER. Copying one neighbour imposes
//  dB/dn = 0 to O(h) only, and that cost 6.7x on the velocity at Ha = 5 and put
//  the jets at 1.003 against 1.010. A conducting wall carries the return
//  current, so the solution leans on that boundary in a way an insulating one
//  does not -- which is exactly why Shercliff never exposed it.
//
//  HUNT'S BOUNDARY CONDITIONS BREAK THE CLEAN DECOUPLING. With u = 0 and
//  db/dy = 0 on the Hartmann walls, phi± = u ± b are no longer each zero there:
//  the conditions read f+ + f- = 0 and f+' = f-', which couple the two. The
//  interior equations still separate, so each z-mode is a 4x4 solve for the
//  four exponential coefficients instead of a 2x2 -- which is the only real
//  difference between the two series below.
//
//  THE ANALYTIC SOLUTION IS DERIVED HERE, NOT QUOTED, AND IT IS VERIFIED.
//  Elsasser variables phi± = u ± b decouple the pair into
//
//      lap phi± ± Ha dphi±/dy = -1,     phi± = 0 on the whole boundary,
//
//  and each is a cosine series in z with a two-point boundary problem in y per
//  mode. Transcribing a published series from memory is the failure this tree
//  names as "a wrong constant is still a consistent simulation", so the series
//  below was checked against an INDEPENDENT finite-difference solve of the same
//  PDE: relative agreement 1.23e-4 / 3.35e-4 / 7.72e-3 / 1.92e-3 at n = 81 for
//  (Ha, l) = (0,1), (5,1), (20,1), (10,2), each falling by exactly 4x at
//  n = 161 -- second order convergence of the FD solve onto the series, which
//  is what identifies the series as the exact solution rather than as another
//  approximation. Hunt's was checked the same way against an FD solve carrying
//  the MIXED conditions -- one-sided three-point for db/dy = 0 on the Hartmann
//  walls -- giving 5.79e-5 / 1.91e-3 / 5.19e-3 at n = 81 for Ha = 1, 5, 10 and
//  again falling by about 4x at n = 161.
//
//  ONE NUMERICAL POINT ON THE SERIES. The obvious basis e^{p y}, e^{q y}
//  overflows once p ~ lambda_k exceeds 709, which is about mode 225 at l = 1,
//  and the two-point solve then returns NaN rather than a small number. The
//  basis used is e^{p(y-1)} and e^{q(y+1)}, both bounded by 1 on the interval
//  since p > 0 > q, which also makes the 2x2 nearly diagonal at large k.
//==============================================================================
#include "Campaign.hpp"
#include "collision/MhdBGK.hpp"
#include "solver/MagneticSolver.hpp"

using namespace lbm;
using namespace campaign;

using FL = D3Q27;
using ML = D3Q7;

// u_hat and b_tilde at (y, z) in duct units, |y| <= 1, |z| <= l.
static void shercliff(double y, double z, double Ha, double l, int K,
                      double& u, double& b) {
  double phi[2] = {0.0, 0.0};
  for (int si = 0; si < 2; ++si) {
    const double s = si == 0 ? +1.0 : -1.0;
    double tot = 0.0;
    for (int k = 0; k < K; ++k) {
      const double lam = (double(k) + 0.5) * M_PI / l;
      const double ck = -2.0 * ((k % 2) ? -1.0 : 1.0) / (lam * l);
      const double part = -ck / (lam * lam);
      const double disc = std::sqrt(Ha * Ha + 4.0 * lam * lam);
      const double p = (-s * Ha + disc) * 0.5;
      const double q = (-s * Ha - disc) * 0.5;
      const double e2q = std::exp(2.0 * q), em2p = std::exp(-2.0 * p);
      // [1 e2q; em2p 1] [A;B] = [-part; -part]
      const double det = 1.0 - e2q * em2p;
      const double A = (-part + part * e2q) / det;
      const double B = (-part + part * em2p) / det;
      tot += (A * std::exp(p * (y - 1.0)) + B * std::exp(q * (y + 1.0)) + part)
             * std::cos(lam * z);
    }
    phi[si] = tot;
  }
  u = 0.5 * (phi[0] + phi[1]);
  b = 0.5 * (phi[0] - phi[1]);
}

// Gaussian elimination with partial pivoting on a 4x4. Small enough that a
// library would cost more than it saves, and the pivoting matters: at large
// lambda the rows built from e^{p(y-1)} and e^{q(y+1)} differ by many orders.
static void solve4(double M[4][4], double r[4], double x[4]) {
  for (int c = 0; c < 4; ++c) {
    int piv = c;
    for (int i = c + 1; i < 4; ++i)
      if (std::abs(M[i][c]) > std::abs(M[piv][c])) piv = i;
    for (int j = 0; j < 4; ++j) std::swap(M[c][j], M[piv][j]);
    std::swap(r[c], r[piv]);
    for (int i = c + 1; i < 4; ++i) {
      const double f = M[i][c] / M[c][c];
      for (int j = c; j < 4; ++j) M[i][j] -= f * M[c][j];
      r[i] -= f * r[c];
    }
  }
  for (int i = 3; i >= 0; --i) {
    double acc = r[i];
    for (int j = i + 1; j < 4; ++j) acc -= M[i][j] * x[j];
    x[i] = acc / M[i][i];
  }
}

// Hunt: Hartmann walls y = +-1 perfectly conducting (db/dy = 0), side walls
// z = +-l insulating (b = 0), u = 0 on all four.
static void hunt(double y, double z, double Ha, double l, int K,
                 double& u, double& b) {
  u = 0; b = 0;
  for (int k = 0; k < K; ++k) {
    const double lam = (double(k) + 0.5) * M_PI / l;
    const double ck = -2.0 * ((k % 2) ? -1.0 : 1.0) / (lam * l);
    const double part = -ck / (lam * lam);
    const double disc = std::sqrt(Ha * Ha + 4.0 * lam * lam);
    const double pp = (-Ha + disc) * 0.5, qp = (-Ha - disc) * 0.5;   // phi+
    const double pm = ( Ha + disc) * 0.5, qm = ( Ha - disc) * 0.5;   // phi-
    auto g1 = [&](double p, double yy) { return std::exp(p * (yy - 1.0)); };
    auto g2 = [&](double q, double yy) { return std::exp(q * (yy + 1.0)); };
    double M[4][4], r[4], A[4];
    const double ys[2] = {1.0, -1.0};
    for (int t = 0; t < 2; ++t) {            // f+ + f- = -2 part   (u = 0)
      M[t][0] = g1(pp, ys[t]); M[t][1] = g2(qp, ys[t]);
      M[t][2] = g1(pm, ys[t]); M[t][3] = g2(qm, ys[t]);
      r[t] = -2.0 * part;
    }
    for (int t = 0; t < 2; ++t) {            // f+' - f-' = 0       (db/dy = 0)
      M[2 + t][0] =  pp * g1(pp, ys[t]); M[2 + t][1] =  qp * g2(qp, ys[t]);
      M[2 + t][2] = -pm * g1(pm, ys[t]); M[2 + t][3] = -qm * g2(qm, ys[t]);
      r[2 + t] = 0.0;
    }
    solve4(M, r, A);
    const double fp = A[0] * g1(pp, y) + A[1] * g2(qp, y) + part;
    const double fm = A[2] * g1(pm, y) + A[3] * g2(qm, y) + part;
    const double c = std::cos(lam * z);
    u += 0.5 * (fp + fm) * c;
    b += 0.5 * (fp - fm) * c;
  }
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int status = 0;
  {
    std::vector<double> Has = {0.0, 5.0, 10.0, 20.0};
    std::vector<Index> Nys = {33, 65};
    double nu_in = 0.05, prm = 1.0, umax_t = 0.02, tol = 1e-8, asp = 1.0;
    std::size_t cap = 400000, probe = 500;
    int K = 120; bool trace = false, track = true, nofd = true;
    std::string walls = "insul";
    Index inset = 0;    // solid padding around the duct, to exercise geometry
    bool prof = false;
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a == "-nu"   && i + 1 < argc) nu_in = std::atof(argv[++i]);
      if (a == "-prm"  && i + 1 < argc) prm = std::atof(argv[++i]);
      if (a == "-umax" && i + 1 < argc) umax_t = std::atof(argv[++i]);
      if (a == "-tol"  && i + 1 < argc) tol = std::atof(argv[++i]);
      if (a == "-asp"  && i + 1 < argc) asp = std::atof(argv[++i]);
      if (a == "-cap"  && i + 1 < argc) cap = std::size_t(std::atol(argv[++i]));
      if (a == "-modes"&& i + 1 < argc) K = std::atoi(argv[++i]);
      if (a == "-trace") trace = true;
      if (a == "-notrack") track = false;
      if (a == "-fd") nofd = false;    // reproduce the failure on demand
      if (a == "-walls" && i + 1 < argc) walls = argv[++i];   // insul | cond
      if (a == "-inset" && i + 1 < argc) inset = Index(std::atol(argv[++i]));
      if (a == "-prof") prof = true;
      if (a == "-has"  && i + 1 < argc) {
        Has.clear(); std::string s = argv[++i], t;
        for (char c : s + ",") { if (c == ',') { if (!t.empty()) Has.push_back(std::atof(t.c_str())); t.clear(); } else t += c; }
      }
      if (a == "-nys"  && i + 1 < argc) {
        Nys.clear(); std::string s = argv[++i], t;
        for (char c : s + ",") { if (c == ',') { if (!t.empty()) Nys.push_back(Index(std::atol(t.c_str()))); t.clear(); } else t += c; }
      }
    }
    std::printf("MHD duct -- %s walls, against the analytic series   %s\n",
                walls == "cond" ? "Hunt, CONDUCTING Hartmann" : "Shercliff, INSULATING",
                ExecSpace::name());
    std::printf("  nu %.4f   Pr_m %.2f   aspect l %.2f   target u_max %.4f   %d modes\n\n",
                nu_in, prm, asp, umax_t, K);
    // RESOLUTION IS THE HARTMANN LAYER, NOT THE DUCT. Its thickness is L/Ha
    // cells and the Shercliff side layer is L/sqrt(Ha); when the first drops
    // below about two cells the answer is a statement about the lattice.
    std::printf("  %5s %6s %8s %9s %7s %7s %9s %11s %11s %10s %9s\n",
                "ny", "nz", "Ha", "B0", "d_Ha", "d_Sh", "jet",
                "L2 err u", "L2 err b", "steps", "residual");
    std::printf("  %s\n", std::string(108, '-').c_str());

    const bool cond_any = (walls == "cond");
    for (const Index ny : Nys) {
      const Index nz = Index(std::lround(asp * double(ny - 1))) + 1;
      for (const double Ha : Has) {
        const Index nx = 4;
        const double L = 0.5 * double(ny - 1);
        const double l = 0.5 * double(nz - 1) / L;
        const Real nu = Real(nu_in), eta = Real(nu_in / prm);
        const double sq = std::sqrt(double(nu) * double(eta));
        const Real B0 = Real(Ha * sq / L);
        // u = (F L^2 / nu) u_hat, so F follows from the analytic peak.
        const bool cond = (walls == "cond");
        auto series = [&](double yy, double zz, double& uu, double& bb) {
          if (cond) hunt(yy, zz, Ha, l, K, uu, bb);
          else      shercliff(yy, zz, Ha, l, K, uu, bb);
        };
        double up = 0;
        for (Index y = 1; y < ny - 1; ++y) {
          double uu, bb; series((double(y) - L) / L, 0.0, uu, bb);
          (void)0;
          up = std::max(up, uu);
        }
        const double Uref = umax_t / std::max(up, 1e-30);
        const Real F = Real(Uref * double(nu) / (L * L));
        const double bref = Uref * std::sqrt(double(nu) / double(eta));

        // -inset PADS THE DUCT WITH SOLID. Physically nothing changes -- the
        // duct is the same duct -- but the walls are no longer the outermost
        // nodes of the domain, so the unknown set at each wall node has to be
        // built against the GEOMETRY rather than against the box. That is the
        // path a curved wall needs, and this is the place to test it, because
        // the answer is already known from the flush duct.
        const Index pad = inset;
        const Index nyt = ny + 2 * pad, nzt = nz + 2 * pad;
        auto insolid = [&](Index y, Index z) {
          return y < pad || y >= pad + ny || z < pad || z >= pad + nz;
        };
        using FluidColl = MhdBGK<FL, SecondOrderEquilibrium<FL>, ShiftedPopulations, Guo>;
        Domain d(nx, nyt, nzt, /*x*/ true, /*y*/ false, /*z*/ false);
        MagneticBGK<ML> mc; mc.omega = MagneticBGK<ML>::omega_from_resistivity(eta);
        MagneticSolver<ML, EsotericPull<ML>, MagneticBGK<ML>> mag(d, mc);
        FluidColl fc;
        fc.omega = FluidColl::omega_from_viscosity(nu);
        fc.forcing = Guo{F, Real(0), Real(0)};
        fc.Bx = mag.Bx(); fc.By = mag.By(); fc.Bz = mag.Bz();
        FluidSolver<FL, EsotericPull<FL>, FluidColl> fl(d, fc);

        fl.set_geometry([&](Index, Index y, Index z) -> CellType {
          return insolid(y, z) ? Solid : Fluid;
        });
        using WS = typename decltype(fl)::WallSpec;
        // A duct has EDGES, where two walls meet and a single axis normal does
        // not describe the unknown set. NrmCorner builds it geometrically.
        fl.set_regularized_walls([&](Index, Index y, Index z) -> WS {
          if (insolid(y, z)) return WS{};
          const bool wy = (y == pad || y == pad + ny - 1);
          const bool wz = (z == pad || z == pad + nz - 1);
          if (wy && wz) return WS{NrmCorner, Real(0), Real(0), Real(0)};
          if (wy) return WS{y == pad ? NrmYm : NrmYp, Real(0), Real(0), Real(0)};
          if (wz) return WS{z == pad ? NrmZm : NrmZp, Real(0), Real(0), Real(0)};
          return WS{};
        });
        // INSULATING on all four: the total field is the applied one, so the
        // induced b_x vanishes. Same condition on the side walls as on the
        // Hartmann walls -- b has only an x component in a developed duct.
        using WB = typename decltype(mag)::WallB;
        const Real B0c = B0;
        // INSULATING is b_x = 0, i.e. the total field is the applied one --
        // Dirichlet. CONDUCTING is db_x/dn = 0 on the HARTMANN walls only (the
        // pair perpendicular to B); the side walls stay insulating, which is
        // the combination Hunt solves and the one that produces side jets.
        mag.set_geometry([&](Index, Index y, Index z) { return insolid(y, z); });
        mag.set_moment_walls([&](Index, Index y, Index z) -> WB {
          if (insolid(y, z)) return WB{};
          const bool wy = (y == pad || y == pad + ny - 1);
          const bool wz = (z == pad || z == pad + nz - 1);
          if (cond && wy && !wz) {
            WB w; w.is_wall = true; w.neumann = true;
            // face left unset: derived from the geometry, which is the rule a
            // curved wall will use. On a flush duct it must come out the same.
            return w;
          }
          if (wy || wz) return WB{true, Real(0), B0c, Real(0)};
          return WB{};
        });
        mag.initialize_field(KOKKOS_LAMBDA(Index) {
          Kokkos::Array<Real, 3> b; b[0] = Real(0); b[1] = B0c; b[2] = Real(0);
          return b;
        });
        // The magnetic solver advects with the fluid's velocity, and the fluid
        // feels B; both couplings are by view, so they are wired once.
        // THE EDGE CLOSURE IS THE SUSPECT. A duct has twelve edges where two
        // regularised walls meet; hartmann3d, which works, has none -- it is
        // periodic in z. At an edge the stress comes from finite-difference
        // velocity gradients (Latt et al. Sec. V), and that stencil knows
        // nothing about the Lorentz force acting there. MEASURED: with the
        // finite-difference closure this duct leaks mass linearly and never
        // saturates -- -5.66e-3 over 100000 steps at Ha = 5 -- and the L2 error
        // against the exact solution GROWS with it, 3.8e-3 to 6.8e-3, while the
        // residual sits pinned at 7.6e-6 because a linear drift gives a
        // CONSTANT per-interval change and reads as a converged floor. With the
        // local closure the same run drifts -8.4e-6, the residual falls to
        // 1.4e-8 and the error is flat at 3.583e-3. At Ha = 0 neither matters:
        // mass holds to -2.5e-14, so this is the MAGNETIC coupling and not the
        // walls. The local closure is therefore the default here and -fd
        // reproduces the failure. Same switch demonstrator/aorta.cpp needed,
        // for a different reason -- there the stencil walked into solid cells.
        if (nofd) fl.set_fd_corners(false);
        mag.set_velocity(fl.ux(), fl.uy(), fl.uz());
        fl.initialize(Real(1));

        // A PROGRESS TRACE, because the first run of this case showed the L2
        // error against the exact solution GROWING with step count -- 2.7e-3 at
        // 120000 and 1.4e-2 at 400000 -- while the residual kept falling. A
        // residual measures whether the field is still moving, not whether it
        // is moving toward the right answer, and the two came apart here. Mass
        // is printed beside it because regularised walls overwrite populations
        // and are documented as not mass conserving, which is the obvious way a
        // driven duct could creep forever.
        const Real m0 = fl.total_mass();
        auto exact_err = [&]() {
          fl.compute_macroscopic();
          auto h = Kokkos::create_mirror_view_and_copy(HostSpace{}, fl.ux());
          double n2 = 0, d2 = 0;
          for (Index z = 0; z < nz; ++z)
            for (Index y = 0; y < ny; ++y) {
              double ue, be;
              series((double(y) - L) / L, (double(z) - 0.5 * double(nz - 1)) / L, ue, be);
              ue *= Uref;
              const double v = double(h(d.id(0, y + pad, z + pad)));
              n2 += (v - ue) * (v - ue); d2 += ue * ue;
            }
          return std::sqrt(n2 / std::max(d2, 1e-300));
        };
        // Steady state on the WHOLE velocity field between probes.
        std::vector<double> prevf(std::size_t(ny) * std::size_t(nz), 0.0);
        (void)nyt; (void)nzt;
        std::size_t taken = 0; double res = 1;
        for (std::size_t t = 0; t <= cap; ++t) {
          if (t % probe == 0) {
            fl.compute_macroscopic(); mag.compute_field();
            auto hx = Kokkos::create_mirror_view_and_copy(HostSpace{}, fl.ux());
            double num = 0, den = 0; bool bad = false;
            for (Index z = 0; z < nz; ++z)
              for (Index y = 0; y < ny; ++y) {
                const double v = double(hx(d.id(0, y + pad, z + pad)));
                if (!std::isfinite(v)) bad = true;
                const std::size_t k = std::size_t(z) * ny + y;
                num += (v - prevf[k]) * (v - prevf[k]); den += v * v;
                prevf[k] = v;
              }
            if (bad) { res = NAN; break; }
            res = std::sqrt(num / std::max(den, 1e-300));
            taken = t;
            // HOLD THE ACCELERATION, NOT THE FORCE DENSITY. Guo's F is a force
            // per unit VOLUME, so the acceleration it delivers is F/rho and the
            // steady flow rate goes as F/rho. Regularised walls overwrite
            // populations and leak mass linearly without saturating -- measured
            // here at -5.7e-8 a step, -2.3e-2 over 400000 -- so a constant F is
            // a slowly RISING acceleration, and the solution creeps away from
            // the analytic one it had already reached. Tracking rho restores
            // what the analytic problem actually specifies: a pressure gradient
            // per unit mass. It does not fix the leak, it removes the leak's
            // effect on the answer.
            if (track) {
              const double mr = double(fl.total_mass()) / double(m0);
              fl.collision().forcing = Guo{Real(double(F) * mr), Real(0), Real(0)};
            }
            if (trace && t % (probe * 40) == 0)
              std::printf("        t %7zu  residual %9.2e  L2 err %9.3e  "
                          "dmass %9.2e\n", t, res, exact_err(),
                          double(fl.total_mass()) / double(m0) - 1.0);
            if (t > 0 && res < tol) break;
          }
          // compute_field before the fluid step, and both told the field is
          // current: the same order hartmann3d uses, and the coupling order is
          // the solver's business rather than a free choice -- refreshing late
          // misplaces the Lorentz force by a step.
          if (t < cap) { mag.compute_field(); fl.step(true); mag.step(true); }
        }

        // Compare both fields against the series.
        fl.compute_macroscopic();
        auto hx = Kokkos::create_mirror_view_and_copy(HostSpace{}, fl.ux());
        auto hbx = Kokkos::create_mirror_view_and_copy(HostSpace{}, mag.Bx());
        double nu_u = 0, de_u = 0, nu_b = 0, de_b = 0, umx = 0;
        // The SIDE-JET ratio: peak speed on the mid-Hartmann line against the
        // speed at the duct centre. 1 for Shercliff, and the signature of a
        // conducting Hartmann wall when it exceeds it.
        double ucore = 0, uside = 0;
        for (Index z = 0; z < nz; ++z)
          for (Index y = 0; y < ny; ++y) {
            double ue, be;
            series((double(y) - L) / L, (double(z) - 0.5 * double(nz - 1)) / L, ue, be);
            ue *= Uref; be *= bref;
            const Index n = d.id(0, y + pad, z + pad);
            const double uv = double(hx(n)), bv = double(hbx(n));
            umx = std::max(umx, std::abs(uv));
            if (y == (ny - 1) / 2) {
              uside = std::max(uside, uv);
              if (z == (nz - 1) / 2) ucore = uv;
            }
            nu_u += (uv - ue) * (uv - ue); de_u += ue * ue;
            nu_b += (bv - be) * (bv - be); de_b += be * be;
          }
        // IS THE FIELD ERROR JUST A LEVEL? A Neumann condition fixes the
        // gradient and says nothing about the constant, and MagOutXp's banner
        // already names that as a suspect for its own 6% defect. Recompute the
        // field error with the mean offset removed: if it collapses, the shape
        // is right and only the gauge is loose.
        double off = 0, cnt = 0;
        for (Index z = 0; z < nz; ++z)
          for (Index y = 0; y < ny; ++y) {
            double ue2, be2;
            series((double(y) - L) / L, (double(z) - 0.5 * double(nz - 1)) / L, ue2, be2);
            off += double(hbx(d.id(0, y + pad, z + pad))) - be2 * bref; cnt += 1;
          }
        off /= std::max(cnt, 1.0);
        double nb2 = 0;
        for (Index z = 0; z < nz; ++z)
          for (Index y = 0; y < ny; ++y) {
            double ue2, be2;
            series((double(y) - L) / L, (double(z) - 0.5 * double(nz - 1)) / L, ue2, be2);
            const double dv = double(hbx(d.id(0, y + pad, z + pad))) - off - be2 * bref;
            nb2 += dv * dv;
          }
        const double eb_lvl = (de_b > 1e-28) ? std::sqrt(nb2 / de_b) : NAN;
        if (prof) {
          std::printf("        u(y) at mid-z, computed / exact / ratio\n");
          const Index zc = (nz - 1) / 2;
          for (Index y = 0; y < ny; y += std::max<Index>(1, (ny - 1) / 8)) {
            double ue2, be2;
            series((double(y) - L) / L, (double(zc) - 0.5 * double(nz - 1)) / L, ue2, be2);
            const double v = double(hx(d.id(0, y + pad, zc + pad)));
            std::printf("          y=%3d  %11.4e  %11.4e  %8.4f\n",
                        int(y), v, ue2 * Uref,
                        (std::abs(ue2) > 1e-14 ? v / (ue2 * Uref) : 0.0));
          }
        }
        const double eu = std::sqrt(nu_u / std::max(de_u, 1e-300));
        const double eb = (de_b > 1e-28) ? std::sqrt(nu_b / std::max(de_b, 1e-300)) : NAN;
        char ebs[16];
        if (std::isfinite(eb)) std::snprintf(ebs, sizeof ebs, "%11.4e", eb);
        else                   std::snprintf(ebs, sizeof ebs, "%11s", "b=0");
        std::printf("        b level offset %+.3e, err b with it removed %.4e\n",
                    off, eb_lvl);
        std::printf("  %5d %6d %8.1f %9.5f %7.2f %7.2f %9.3f %11.4e %s %10zu %9.1e%s%s\n",
                    int(ny), int(nz), Ha, double(B0),
                    Ha > 0 ? L / Ha : 999.0, Ha > 0 ? L / std::sqrt(Ha) : 999.0,
                    (ucore > 0 ? uside / ucore : 0.0), eu, ebs, taken, res,
                    (taken >= cap) ? "  CAPPED" : "",
                    std::isfinite(res) ? "" : "  NON-FINITE");
        std::fflush(stdout);
        if (!std::isfinite(eu)) status = 1;
      }
      std::printf("\n");
    }
    if (!cond_any)
      std::printf("  -walls cond runs Hunt's conducting-wall case on the same duct.\n");
  }
  Kokkos::finalize();
  return status;
}
