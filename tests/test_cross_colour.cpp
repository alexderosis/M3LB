//==============================================================================
//  THE TWO COLOUR-GRADIENT IMPLEMENTATIONS, DRIVEN THROUGH THE SAME STATES.
//
//  src/collision/ColourGradient.hpp and GPU/include/lbm/colour.cuh implement
//  Appendix D of De Rosis, Huang & Coreixas (Phys. Fluids 31, 117102, 2019) from
//  two separate code bases that share NO HEADERS. CLAUDE.md's standing rule is
//  that they agree where they overlap and that disagreement is a bug in one of
//  them. Until 2026-09-19 that rule was enforced by running a case on each and
//  comparing printed diagnostics, which is slow, noisy, and only ever exercised
//  whatever states the case happened to visit.
//
//  This drives both operators DIRECTLY, on identical randomised states, and
//  diffs all 27 populations. It needs no simulation and runs in milliseconds.
//
//  WHY IT LIVES IN THE PARENT AND NOT IN GPU/. The parent already links Kokkos,
//  which ColourGradient.hpp needs for its node-field Views; GPU/ deliberately
//  does not, and making its CMake acquire Kokkos to run a test would trade the
//  independence that makes the comparison worth anything. So the parent reaches
//  ACROSS into GPU/include, and GPU/ is left knowing nothing about this.
//
//  WHAT IT MEASURED WHEN WRITTEN: worst |src - gpu| = 0 exactly, over 320 states
//  at gamma = 1, 10, 100 and 1000 under both rest-term readings, at BOTH
//  precisions -- FP64 and FP32 alike, which is a sharper statement than it
//  sounds, since it means the two also round identically. The
//  assertion below is a tolerance and not equality on purpose -- the two are
//  independently written, so their agreement to the last bit is a property of
//  the arithmetic happening to fall in the same order, which a different
//  compiler or optimisation level may not preserve. The number is PRINTED, so a
//  reader sees whether it is still zero; the tolerance only catches a real
//  divergence.
//
//  IT CAN FAIL, AND THAT WAS MEASURED RATHER THAN ASSUMED. Perturbing ONE
//  coefficient in ONE of cg_source_high's seventeen machine-generated slots on
//  the GPU side by 1e-7 RELATIVE -- 1/3 written as 1/3.0000001, far smaller than
//  any transcription error a human makes -- turns seven of the nine checks red,
//  at a worst relative population difference of 4.1e-11. So the tolerance sits
//  about four orders below the smallest plausible real defect. A test that
//  passes and has never been shown to fail is a test of nothing; this tree
//  records that lesson under the GPU gather/scatter bug and it applies here.
//
//  THE LATTICE ORDERINGS ARE CHECKED, NOT ASSUMED. Both are D3Q27 and both
//  currently number their velocities identically, but the harness matches slot
//  to slot by integer velocity first and reports what it found. Comparing two
//  differently-ordered sets would produce a large, structured, entirely
//  spurious disagreement.
//==============================================================================
#include "Check.hpp"
#include "collision/ColourGradient.hpp"
#include "core/Types.hpp"

#include <cmath>

using namespace lbm;
using L  = D3Q27;
using CG = ColourGradient<L>;

extern "C" {
void xc_gpu_collide(double*, double, const double*, double, const double*,
                    const double*, double, double, double, double, double,
                    double, double, double, double, double, double, double, int);
void xc_gpu_velocities(int*);
}

//------------------------------------------------------------------------------
// The parent operator behind the same interface. Its gradients are node FIELDS,
// so it gets a one-node domain; everything else is passed by value.
//------------------------------------------------------------------------------
static void src_collide(double* f, double rho, const double* u, double p,
                        const double* g, const double* dr,
                        double ar, double ab, double nur, double nub, double A,
                        double ombulk, double bx, double by, double bz,
                        double rref, double rr0, double rb0, int per_colour) {
  CG cg;
  cg.alpha_r = Real(ar);  cg.alpha_b = Real(ab);
  cg.nu_r = Real(nur);    cg.nu_b = Real(nub);
  cg.A = Real(A);         cg.omega_bulk = Real(ombulk);
  cg.bx = Real(bx);  cg.by = Real(by);  cg.bz = Real(bz);
  cg.rho_ref = Real(rref);
  cg.rho_r0 = Real(rr0);  cg.rho_b0 = Real(rb0);
  cg.rest = per_colour ? CG::RestTerm::PerColour : CG::RestTerm::AlphaBar;

  View1D<Real> gx("gx", 1), gy("gy", 1), gz("gz", 1);
  View1D<Real> rx("rx", 1), ry("ry", 1), rz("rz", 1);
  auto set = [](View1D<Real> v, double val) {
    auto h = Kokkos::create_mirror_view(v);
    h(0) = Real(val);
    Kokkos::deep_copy(v, h);
  };
  set(gx, g[0]);  set(gy, g[1]);  set(gz, g[2]);
  set(rx, dr[0]); set(ry, dr[1]); set(rz, dr[2]);
  cg.Gx = gx; cg.Gy = gy; cg.Gz = gz;
  cg.Rx = rx; cg.Ry = ry; cg.Rz = rz;

  Real ff[L::Q], uu[3];
  for (int i = 0; i < L::Q; ++i) ff[i] = Real(f[i]);
  for (int a = 0; a < 3; ++a) uu[a] = Real(u[a]);
  cg.collide(ff, Real(rho), uu, Real(p), 0);
  for (int i = 0; i < L::Q; ++i) f[i] = double(ff[i]);
}

// A fixed stream rather than <random>, so a failure here is the same failure on
// every machine and can be bisected.
static double urand(unsigned long long& st) {
  st = st * 6364136223846793005ULL + 1442695040888963407ULL;
  return double((st >> 11) & ((1ULL << 53) - 1)) / double(1ULL << 53);
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    // ---- match the two velocity sets before comparing anything -------------
    int cs[81], cg_[81], map[27];
    for (int i = 0; i < L::Q; ++i) {
      cs[3 * i + 0] = cvel<L>(i, 0);
      cs[3 * i + 1] = cvel<L>(i, 1);
      cs[3 * i + 2] = cvel<L>(i, 2);
    }
    xc_gpu_velocities(cg_);
    int unmatched = 0, reordered = 0;
    for (int i = 0; i < 27; ++i) {
      map[i] = -1;
      for (int j = 0; j < 27; ++j)
        if (cs[3*i] == cg_[3*j] && cs[3*i+1] == cg_[3*j+1] && cs[3*i+2] == cg_[3*j+2])
          map[i] = j;
      if (map[i] < 0) ++unmatched;
      else if (map[i] != i) ++reordered;
    }
    std::printf("\n1. the two D3Q27 velocity sets\n");
    check::ok(unmatched == 0, "every parent velocity exists in the GPU set");
    std::printf("        (%d of 27 slots are numbered differently; the harness remaps)\n",
                reordered);

    //--------------------------------------------------------------------------
    std::printf("\n2. the operators, population by population\n");
    // Tight, but not equality: see the banner. FP32 cannot hold the parent's own
    // interface arithmetic to better than this, and that is a property of the
    // precision rather than of either implementation.
    const double rtol = (sizeof(Real) == 4) ? 1e-5 : 1e-13;
    double worst_all = 0;
    for (int per_colour = 1; per_colour >= 0; --per_colour) {
      for (double gamma : {1.0, 10.0, 100.0, 1000.0}) {
        const double ab = 8.0 / 27.0, ar = 1.0 - (1.0 - ab) / gamma;
        unsigned long long st = 0x5124719ULL + (unsigned long long)gamma;
        double wrel = 0;
        const int N = 40;
        for (int t = 0; t < N; ++t) {
          const double nur = 0.02 + 0.3 * urand(st), nub = 0.02 + 0.3 * urand(st);
          const double A   = (t % 4 == 0) ? 0.0 : 0.002 + 0.02 * urand(st);
          const double omb = 0.4 + 1.4 * urand(st);
          const double bx = 1e-5 * (2 * urand(st) - 1);
          const double by = 1e-5 * (2 * urand(st) - 1);
          const double bz = 1e-5 * (2 * urand(st) - 1);
          const double rref = (t % 3 == 0) ? 0.0 : 0.5 + urand(st);
          const bool flat = (t % 5 == 1);          // the |grad phi| = 0 branch
          const double g[3] = {flat ? 0.0 : 0.4 * (2 * urand(st) - 1),
                               flat ? 0.0 : 0.4 * (2 * urand(st) - 1),
                               flat ? 0.0 : 0.4 * (2 * urand(st) - 1)};
          const double dr[3] = {0.3 * (2 * urand(st) - 1),
                                0.3 * (2 * urand(st) - 1),
                                0.3 * (2 * urand(st) - 1)};
          const double rr = 0.2 + gamma * urand(st), rb = 0.2 + urand(st);
          const double rho = rr + rb;
          const double a_ = rr / gamma, b_ = rb;
          const double p = (a_ + b_ > 0) ? (a_ - b_) / (a_ + b_) : 0.0;
          const double u[3] = {0.1 * (2 * urand(st) - 1),
                               0.1 * (2 * urand(st) - 1),
                               0.1 * (2 * urand(st) - 1)};

          double fs[27], fg[27];
          for (int i = 0; i < 27; ++i) {
            const double v = 0.05 * rho * (1.0 + 0.3 * (2 * urand(st) - 1));
            fs[i] = v;  fg[map[i]] = v;          // the SAME physical state
          }
          src_collide(fs, rho, u, p, g, dr, ar, ab, nur, nub, A, omb, bx, by, bz,
                      rref, gamma, 1.0, per_colour);
          xc_gpu_collide(fg, rho, u, p, g, dr, ar, ab, nur, nub, A, omb, bx, by,
                         bz, rref, gamma, 1.0, per_colour);
          double scale = 0;
          for (int i = 0; i < 27; ++i) scale = std::fmax(scale, std::fabs(fs[i]));
          if (scale <= 0) continue;
          for (int i = 0; i < 27; ++i)
            wrel = std::fmax(wrel, std::fabs(fs[i] - fg[map[i]]) / scale);
        }
        worst_all = std::fmax(worst_all, wrel);
        char b[160];
        std::snprintf(b, sizeof b,
                      "RestTerm::%-9s gamma = %-6g  %d states",
                      per_colour ? "PerColour" : "AlphaBar", gamma, N);
        check::near(wrel, 0.0, rtol, b);
      }
    }
    std::printf("\n        worst relative difference over all 320 states: %.3e\n",
                worst_all);
    std::printf("        (it was EXACTLY ZERO when this test was written, at both precisions)\n");
  }
  const int rc = check::report("cross_colour");
  Kokkos::finalize();
  return rc;
}
