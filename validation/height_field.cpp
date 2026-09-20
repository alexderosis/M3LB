//==============================================================================
//  Height-field geometry reader: round trip, solid rule, and rejection.
//
//  A reader is not obviously worth a test until you notice how it fails. None of
//  the ways this one can go wrong produce an error: a transposed index order
//  rotates the city, a mis-parsed dx rescales it, an off-by-one in the solid
//  rule shaves a layer off every roof. Each of those runs to completion and
//  produces a plausible plume. So the checks below are all of the form "compare
//  against something independently known", never "did it crash".
//
//  Everything here is self-contained: the test writes its own .npy and
//  meta.json, so it needs no data files and runs anywhere. $LBM_CITY adds a
//  field written by something else -- a real city, or the fixture from
//  tools/osm_city.py --emit-fixture -- and where that file's metadata carries
//  the writer's own answers they are checked rather than printed, which is the
//  only part of this test that can catch the two implementations disagreeing.
//  Its absence is not a failure.
//==============================================================================
#include "core/Types.hpp"
#include "io/HeightField.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

using namespace lbm;

static int failures = 0;

static void check(bool ok, const char* what) {
  std::printf("   %-58s %s\n", what, ok ? "ok" : "FAIL");
  if (!ok) ++failures;
}

template <class F>
static bool throws(F f) {
  try { f(); } catch (const std::exception&) { return true; }
  return false;
}

//------------------------------------------------------------------------------
// A deliberately ASYMMETRIC ramp: h = 10*i + j metres. Any transpose, flip or
// axis swap changes the values read back, which a symmetric test pattern would
// hide completely.
//------------------------------------------------------------------------------
static void write_case(const std::string& dir, Index nx, Index ny, Index nz,
                       double dx, const char* extra = "") {
  std::vector<float> h(std::size_t(nx) * std::size_t(ny));
  for (Index i = 0; i < nx; ++i)
    for (Index j = 0; j < ny; ++j)
      h[std::size_t(i) * std::size_t(ny) + std::size_t(j)] = float(10 * i + j);
  npy::write_f32(dir + "/t_heights.npy", h, nx, ny, 1);

  std::ofstream m(dir + "/t_meta.json");
  m << "{\n  \"place\": \"synthetic ramp\",\n"
    << "  \"nx\": " << int(nx) << ",\n  \"ny\": " << int(ny) << ",\n"
    << "  \"nz\": " << int(nz) << ",\n  \"dx\": " << dx << extra << "\n}\n";
}

int main(int argc, char** argv) {
  // The first argument that is NOT a flag. Kokkos parses its own --kokkos-*
  // options but leaves them in argv, so taking argv[1] blindly makes a scratch
  // directory out of "--kokkos-num-threads=4" and the run dies trying to write
  // into it. This is the only registered test that takes a positional argument,
  // and it broke the moment the suite started passing a thread count.
  std::string dir = ".";
  for (int i = 1; i < argc; ++i)
    if (argv[i][0] != '-') { dir = argv[i]; break; }
  std::printf("\nHeight-field geometry reader\n%s\n\n", std::string(70, '=').c_str());

  const Index nx = 7, ny = 5, nz = 9;
  const double dx = 4.0;
  write_case(dir, nx, ny, nz, dx);

  std::printf("1. ROUND TRIP  (asymmetric ramp h = 10i + j, so a transpose shows)\n\n");
  const HeightField g = load_height_field(dir + "/t_heights.npy", dir + "/t_meta.json");
  check(g.nx == nx && g.ny == ny && g.nz == nz, "dimensions survive meta.json");
  check(std::abs(g.dx - dx) < 1e-12, "dx survives meta.json");
  check(g.place == "synthetic ramp", "place string is parsed");

  bool values = true, ordering = true;
  for (Index i = 0; i < nx; ++i)
    for (Index j = 0; j < ny; ++j)
      if (std::abs(g.at(i, j) - double(10 * i + j)) > 1e-6) values = false;
  // The distinguishing pair: at(1,0) and at(0,1) differ by 9 m, and a
  // transposed reader swaps them.
  if (std::abs(g.at(1, 0) - 10.0) > 1e-6 || std::abs(g.at(0, 1) - 1.0) > 1e-6)
    ordering = false;
  check(values,   "every height is the value that was written");
  check(ordering, "index order is i*ny + j, not transposed");

  std::printf("\n2. SOLID RULE  (cell centre at (k + 1/2) dx lies below the column)\n\n");
  // Column (0,0) has h = 0: no solid cell at all, because the first centre is
  // at dx/2 > 0. Column (2,0) has h = 20 = 5 dx: centres at 2, 6, 10, 14, 18
  // are below it and the one at 22 is not, so exactly 5 cells.
  check(!g.solid(0, 0, 0), "a zero-height column has no solid cell");
  int n20 = 0;
  for (Index k = 0; k < nz; ++k) if (g.solid(2, 0, k)) ++n20;
  check(n20 == 5, "h = 20 m at dx = 4 m fills exactly 5 cells");
  // The boundary case the closed form gets wrong: h/dx - 1/2 exactly integral.
  write_case(dir, 3, 3, 9, 2.0);
  const HeightField gb = load_height_field(dir + "/t_heights.npy", dir + "/t_meta.json");
  int nb = 0;
  for (Index k = 0; k < 9; ++k) if (gb.solid(0, 1, k)) ++nb;   // h = 1, dx = 2
  check(nb == 0, "h = 1 m at dx = 2 m fills 0 cells (centre 1.0 not below 1.0)");
  std::size_t direct = 0;
  for (Index i = 0; i < 3; ++i)
    for (Index j = 0; j < 3; ++j)
      for (Index k = 0; k < 9; ++k) direct += gb.solid(i, j, k) ? 1 : 0;
  check(direct == gb.solid_count(), "solid_count agrees with solid() cell by cell");

  std::printf("\n3. REJECTION  (each of these is silently wrong if accepted)\n\n");
  {
    std::ofstream m(dir + "/t_bad.json");
    m << "{ \"nx\": 7, \"ny\": 5, \"dx\": 4.0 }\n";       // no nz
  }
  check(throws([&]{ load_height_field(dir + "/t_heights.npy", dir + "/t_bad.json"); }),
        "a missing key is an error, not a default");
  {
    std::ofstream m(dir + "/t_bad.json");
    m << "{ \"nx\": 3, \"ny\": 3, \"nz\": 9, \"dx\": 2.0, \"dz\": 5.0 }\n";
  }
  check(throws([&]{ load_height_field(dir + "/t_heights.npy", dir + "/t_bad.json"); }),
        "dz differing from dx is rejected, not silently stretched");
  {
    std::ofstream m(dir + "/t_bad.json");
    m << "{ \"nx\": 40, \"ny\": 40, \"nz\": 9, \"dx\": 2.0 }\n";
  }
  check(throws([&]{ load_height_field(dir + "/t_heights.npy", dir + "/t_bad.json"); }),
        "a shape that disagrees with meta.json is rejected");
  check(throws([&]{ npy::read_f32(dir + "/t_meta.json", 0); }),
        "a file that is not a .npy is rejected");

  //----------------------------------------------------------------------------
  //  4. A FIELD THIS PROGRAM DID NOT WRITE.
  //
  //  Everything above is a round trip through one implementation: this file
  //  writes a .npy and reads it back, so a reader and a writer that are wrong
  //  in the same way agree perfectly. The seam that matters in practice is the
  //  other one -- `tools/osm_city.py` writes the .npy, this reads it -- and the
  //  interesting way for THAT to fail is not an exception but a transpose,
  //  which every total in `report()` is blind to. So when the metadata carries
  //  the writer's own answers (which `--emit-fixture` puts there, and a real
  //  city's metadata does not) they are CHECKED rather than printed.
  if (const char* p = std::getenv("LBM_CITY")) {
    std::printf("\n4. A FIELD WRITTEN ELSEWHERE  ($LBM_CITY = %s)\n\n", p);
    try {
      const HeightField c = load_height_field(std::string(p) + "_heights.npy",
                                              std::string(p) + "_meta.json");
      report(c, "city");

      std::ifstream mf(std::string(p) + "_meta.json");
      const std::string all((std::istreambuf_iterator<char>(mf)),
                            std::istreambuf_iterator<char>());
      auto opt = [&](const char* key, double miss) {
        const std::string k = std::string("\"") + key + "\"";
        const std::size_t q = all.find(k);
        if (q == std::string::npos) return miss;
        return std::atof(all.c_str() + all.find(':', q + k.size()) + 1);
      };
      const double ci = opt("check_i", -1);
      if (ci >= 0) {
        const Index i = Index(ci), j = Index(opt("check_j", -1));
        std::printf("   writer says the first built column is i=%d j=%d at %.1f m\n",
                    int(i), int(j), opt("check_h", 0));
        check(std::abs(c.at(i, j) - opt("check_h", 0)) < 1e-6,
              "that column reads back at that height (a transpose fails here)");
        // The SAME height at the transposed index would let a square fixture
        // pass while rotated; the fixture is 24x16 so the index does not exist.
        check(i >= c.ny || j >= c.nx || std::abs(c.at(j, i) - opt("check_h", 0)) > 1e-6,
              "and the transposed index does not also read that height");
        std::size_t built = 0;
        for (Index a = 0; a < c.nx; ++a)
          for (Index b = 0; b < c.ny; ++b) built += (c.at(a, b) > 0);
        check(std::abs(double(built) / double(c.columns()) -
                       opt("built_fraction", -1)) < 1e-9,
              "built fraction agrees with the writer's own count");
        check(std::abs(c.max_height() - opt("check_max_h", -1)) < 1e-6,
              "tallest column agrees with the writer's own maximum");
      }
    } catch (const std::exception& e) {
      std::printf("   could not load: %s\n", e.what());
      ++failures;
    }
  }

  std::remove((dir + "/t_heights.npy").c_str());
  std::remove((dir + "/t_meta.json").c_str());
  std::remove((dir + "/t_bad.json").c_str());

  std::printf("\n%s\n\n", failures ? "FAILURES ABOVE" : "all checks passed");
  return failures ? 1 : 0;
}
