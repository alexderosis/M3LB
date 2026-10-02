// tg_mhd_vol -- 3-D volumes of the local dissipation from tg_mhd's raw dumps,
// for the paper's 3-D views.
//
//     g++ -O2 -std=c++17 -pthread tools/tg_mhd_vol.cpp -o tg_mhd_vol
//     tg_mhd_vol RUN_DIR [--stride 2] [--times 2.3,4.6] [--threads 4]
//
// RUN_DIR is a run directory of demonstrator/tg_mhd or GPU/src/tg_mhd.cu run
// with -raw (GPU/csf3/tg_mhd_snap.sub writes them): series.dat names the setup,
// Re and Pm, and raw/fields_NNNN.raw hold the full fields (write_raw_fields:
// "TGMHDRAW", int32 nx ny nz nv, double t, double h, then nv float32 blocks
// u_x u_y u_z b_x b_y b_z rho, x fastest, in the paper's units). For each dump
// nearest a requested time (all of them by default) it writes, under RUN_DIR/vol/,
//
//     visc_t<t>.f32, ohm_t<t>.f32   nu|w|^2 and eta|j|^2, reduced by the
//                                   MAXIMUM over stride^3 blocks to
//                                   M = ceil(N/stride) a side, x fastest
//     index.json                    one record per file: field, time, N, M,
//                                   stride, Re, h, the box mean eps at that time
//                                   (series.dat, interpolated), and the box
//                                   means of the two fields
//
// A block MAXIMUM, not a mean, for the reason demonstrator/tg_mhd.cpp gives for
// its -dumpvol: a maximum-intensity projection of a mean would dim the one-cell
// sheets the picture is of.
//
// THE DERIVATIVES ARE THE DRIVERS' -- HostFields::d in demonstrator/tg_mhd.cpp,
// and tools/tg_mhd_slices.py's, which reproduces series.dat's eps to 1.7e-8:
// second-order central differences, and at a wall node the wall's own
// extension, the mirror (odd or even by the parity) for a free-slip velocity and
// for B, a one-sided second-order stencil for a no-slip velocity. The check is
// built in: every dump's trapezoid-weighted box mean of nu|w|^2 + eta|j|^2 is
// printed against series.dat's eps -- gated (relative 1e-5) when the dump falls
// on a probe, reported when it falls between two and eps is interpolated.
//
// Standalone C++17, the standard library and three POSIX calls (opendir, stat,
// mkdir) -- not <filesystem>, which needs -lstdc++fs on the GCC 8 that RHEL 8
// ships -- so it builds with the system g++ on a CSF3 node next to the dumps and only the volumes travel. It holds the
// six velocity and field components of one dump in memory: 3.2 GB at N = 512.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <dirent.h>
#include <sys/stat.h>

namespace {

enum Ext { OneSided, Even, Odd };

struct Run {
  std::string tag;
  double re = 0, pm = 1;
  bool noslip = false, conducting = false;
  std::vector<double> ts, eps;
};

Run read_run(const std::string& dir) {
  std::ifstream f(dir + "/series.dat");
  if (!f) { std::fprintf(stderr, "no series.dat in %s\n", dir.c_str()); std::exit(1); }
  Run r;
  std::string head, line;
  std::getline(f, head);
  std::smatch m;
  if (std::regex_search(head, m, std::regex("\\b((tgc|tgi|tga|hydro)_[A-Za-z0-9_]+)"))) r.tag = m[1];
  if (std::regex_search(head, m, std::regex("\\bRe=([0-9.eE+-]+)"))) r.re = std::stod(m[1]);
  if (std::regex_search(head, m, std::regex("\\bPm=([0-9.eE+-]+)"))) r.pm = std::stod(m[1]);
  if (r.tag.empty() || r.re <= 0) { std::fprintf(stderr, "%s: series.dat header names no setup/Re\n", dir.c_str()); std::exit(1); }
  if (r.tag.find("_periodic_") != std::string::npos) { std::fprintf(stderr, "a periodic run has no walls\n"); std::exit(1); }
  r.noslip = r.tag.find("_noslip_") != std::string::npos;
  r.conducting = r.tag.find("_cond_") != std::string::npos;
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream is(line);
    std::vector<double> v;
    double x;
    while (is >> x) v.push_back(x);
    if (v.size() > 7) { r.ts.push_back(v[0]); r.eps.push_back(v[7]); }
  }
  return r;
}

// series.dat's eps at t, interpolated; *on_probe says whether t is a probe time
double eps_at(const Run& r, double t, bool* on_probe) {
  *on_probe = false;
  for (std::size_t i = 0; i < r.ts.size(); ++i)
    if (std::fabs(r.ts[i] - t) < 1e-5) { *on_probe = true; return r.eps[i]; }   // series.dat prints t to 6 decimals
  for (std::size_t i = 0; i + 1 < r.ts.size(); ++i)
    if (r.ts[i] <= t && t <= r.ts[i + 1]) {
      const double w = (t - r.ts[i]) / (r.ts[i + 1] - r.ts[i]);
      return r.eps[i] + w * (r.eps[i + 1] - r.eps[i]);
    }
  return r.eps.empty() ? 0.0 : r.eps.back();
}

struct Dump {
  std::int64_t L = 0;
  double t = 0, h = 0;
  std::vector<float> f[6];                          // u_x u_y u_z b_x b_y b_z
};

Dump read_dump(const std::string& p) {
  std::ifstream f(p, std::ios::binary);
  char magic[8];
  f.read(magic, 8);
  if (!f || std::memcmp(magic, "TGMHDRAW", 8) != 0) { std::fprintf(stderr, "%s: not a tg_mhd raw dump\n", p.c_str()); std::exit(1); }
  std::int32_t n[4];
  f.read(reinterpret_cast<char*>(n), sizeof n);
  Dump d;
  f.read(reinterpret_cast<char*>(&d.t), sizeof d.t);
  f.read(reinterpret_cast<char*>(&d.h), sizeof d.h);
  if (n[0] != n[1] || n[1] != n[2] || n[3] != 7) { std::fprintf(stderr, "%s: expected a cube with 7 fields\n", p.c_str()); std::exit(1); }
  d.L = n[0];
  const std::size_t np = std::size_t(d.L) * d.L * d.L;
  for (int c = 0; c < 6; ++c) {
    d.f[c].resize(np);
    f.read(reinterpret_cast<char*>(d.f[c].data()), std::streamsize(np * sizeof(float)));
    if (!f) { std::fprintf(stderr, "%s: truncated\n", p.c_str()); std::exit(1); }
  }
  return d;
}

// HostFields::ext: the extension of component c across a wall normal to axis k
Ext ext(const Run& r, int c, int k) {
  if (c < 3) return r.noslip ? OneSided : (c == k ? Odd : Even);
  const bool normal = (c - 3 == k);
  return (normal == r.conducting) ? Odd : Even;
}

// HostFields::d: d f_c / d x_k at a node, per cell
double deriv(const Dump& d, const Run& r, int c, int k, std::int64_t x, std::int64_t y, std::int64_t z) {
  const std::int64_t L = d.L, p[3] = {x, y, z};
  const std::int64_t stride = (k == 0) ? 1 : (k == 1) ? L : L * L;
  const float* f = d.f[c].data();
  const std::int64_t at0 = (z * L + y) * L + x;
  if (p[k] > 0 && p[k] < L - 1) return 0.5 * (double(f[at0 + stride]) - double(f[at0 - stride]));
  const int s = (p[k] == 0) ? 1 : -1;                // step INTO the box
  const Ext e = ext(r, c, k);
  if (e == OneSided)
    return s * (-1.5 * f[at0] + 2.0 * f[at0 + s * stride] - 0.5 * f[at0 + 2 * s * stride]);
  if (e == Even) return 0.0;
  return s * double(f[at0 + s * stride]);           // odd mirror ghost: f(-1) = -f(+1)
}

void write_f32(const std::string& p, const std::vector<float>& v) {
  std::ofstream o(p, std::ios::binary);
  o.write(reinterpret_cast<const char*>(v.data()), std::streamsize(v.size() * sizeof(float)));
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) { std::fprintf(stderr, "usage: tg_mhd_vol RUN_DIR [--stride S] [--times t1,t2] [--threads T]\n"); return 2; }
  const std::string dir = argv[1];
  int S = 2, nthreads = int(std::max(1u, std::thread::hardware_concurrency()));
  std::vector<double> want;
  for (int a = 2; a < argc; ++a) {
    const std::string k = argv[a];
    if (k == "--stride" && a + 1 < argc) S = std::atoi(argv[++a]);
    else if (k == "--threads" && a + 1 < argc) nthreads = std::atoi(argv[++a]);
    else if (k == "--times" && a + 1 < argc) {
      std::stringstream ss(argv[++a]);
      std::string tok;
      while (std::getline(ss, tok, ',')) want.push_back(std::stod(tok));
    } else { std::fprintf(stderr, "unknown argument %s\n", k.c_str()); return 2; }
  }
  if (S < 1 || nthreads < 1) { std::fprintf(stderr, "--stride and --threads must be >= 1\n"); return 2; }
  const Run run = read_run(dir);
  const double nu = 1.0 / run.re, eta = nu / run.pm;

  std::vector<std::string> raw;
  if (DIR* dp = opendir((dir + "/raw").c_str())) {
    while (const dirent* e = readdir(dp)) {
      const std::string n = e->d_name;
      if (n.rfind("fields_", 0) == 0 && n.size() > 4 && n.compare(n.size() - 4, 4, ".raw") == 0)
        raw.push_back(dir + "/raw/" + n);
    }
    closedir(dp);
  }
  std::sort(raw.begin(), raw.end());
  if (raw.empty()) { std::fprintf(stderr, "no raw/fields_*.raw under %s -- run the driver with -raw\n", dir.c_str()); return 1; }
  struct stat st;
  if (stat((dir + "/vol").c_str(), &st) != 0 && mkdir((dir + "/vol").c_str(), 0755) != 0) {
    std::fprintf(stderr, "cannot create %s/vol\n", dir.c_str());
    return 1;
  }

  std::vector<std::string> records;
  const std::string idx = dir + "/vol/index.json";
  int bad = 0;
  for (const auto& path : raw) {
    // the dump time is in the header; skip the dumps that are not wanted before reading 3 GB
    double tdump;
    {
      std::ifstream f(path, std::ios::binary);
      f.seekg(8 + 16);
      f.read(reinterpret_cast<char*>(&tdump), sizeof tdump);
    }
    if (!want.empty()) {
      bool hit = false;
      for (double w : want) hit = hit || std::fabs(w - tdump) < 0.06;
      if (!hit) continue;
    }
    const Dump d = read_dump(path);
    const std::int64_t L = d.L, M = (L + S - 1) / S;
    const double ih = 1.0 / d.h;
    std::vector<float> vv(std::size_t(M) * M * M, 0.0f), vo(vv.size(), 0.0f);
    std::vector<double> sv(nthreads, 0.0), so(nthreads, 0.0), sw(nthreads, 0.0);
    // threads take whole z-blocks of the output, so no two write one block
    auto work = [&](int tid) {
      for (std::int64_t bz = tid; bz < M; bz += nthreads)
        for (std::int64_t z = bz * S; z < std::min(L, (bz + 1) * S); ++z) {
          const double wz = (z == 0 || z == L - 1) ? 0.5 : 1.0;
          for (std::int64_t y = 0; y < L; ++y) {
            const double wy = (y == 0 || y == L - 1) ? 0.5 : 1.0;
            for (std::int64_t x = 0; x < L; ++x) {
              double du[3][3], db[3][3];
              for (int c = 0; c < 3; ++c)
                for (int k = 0; k < 3; ++k) {
                  du[c][k] = deriv(d, run, c, k, x, y, z) * ih;
                  db[c][k] = deriv(d, run, 3 + c, k, x, y, z) * ih;
                }
              const double w0 = du[2][1] - du[1][2], w1 = du[0][2] - du[2][0], w2 = du[1][0] - du[0][1];
              const double j0 = db[2][1] - db[1][2], j1 = db[0][2] - db[2][0], j2 = db[1][0] - db[0][1];
              const double ev = nu * (w0 * w0 + w1 * w1 + w2 * w2), eo = eta * (j0 * j0 + j1 * j1 + j2 * j2);
              const double w = wz * wy * ((x == 0 || x == L - 1) ? 0.5 : 1.0);
              sv[tid] += w * ev;
              so[tid] += w * eo;
              sw[tid] += w;
              const std::size_t m = (std::size_t(bz) * M + std::size_t(y / S)) * M + std::size_t(x / S);
              vv[m] = std::max(vv[m], float(ev));
              vo[m] = std::max(vo[m], float(eo));
            }
          }
        }
    };
    std::vector<std::thread> th;
    for (int i = 0; i < nthreads; ++i) th.emplace_back(work, i);
    for (auto& t : th) t.join();
    double tv = 0, to = 0, tw = 0;
    for (int i = 0; i < nthreads; ++i) { tv += sv[i]; to += so[i]; tw += sw[i]; }
    const double mv = tv / tw, mo = to / tw;
    bool on_probe;
    const double ref = eps_at(run, d.t, &on_probe);
    const double rel = std::fabs(mv + mo - ref) / std::max(std::fabs(ref), 1e-300);
    const bool ok = !on_probe || rel < 1e-5;
    bad += ok ? 0 : 1;
    std::printf("%s  t = %.4f: box mean of nu|w|^2 + eta|j|^2 = %.8e, series eps = %.8e%s, rel %.1e  %s\n",
                path.substr(path.rfind('/') + 1).c_str(), d.t, mv + mo, ref, on_probe ? "" : " (interpolated)", rel,
                on_probe ? (ok ? "PASS" : "FAIL") : "not gated");
    char tbuf[32];
    std::snprintf(tbuf, sizeof tbuf, "%.3f", d.t);
    const struct { const char* name; const std::vector<float>* v; double mean; } out[2] = {
        {"visc", &vv, mv}, {"ohm", &vo, mo}};
    for (const auto& o : out) {
      const std::string file = std::string(o.name) + "_t" + tbuf + ".f32";
      write_f32(dir + "/vol/" + file, *o.v);
      char rec[1024];
      std::snprintf(rec, sizeof rec,
                    "{\"file\": \"%s\", \"field\": \"%s\", \"t\": %.10g, \"N\": %lld, \"M\": %lld, \"stride\": %d, "
                    "\"reduce\": \"max\", \"Re\": %.10g, \"Pm\": %.10g, \"h\": %.17g, \"tag\": \"%s\", "
                    "\"eps_mean\": %.10g, \"box_mean\": %.10g, \"layout\": \"x fastest, then y, then z\"}",
                    file.c_str(), o.name, d.t, (long long)L, (long long)M, S, run.re, run.pm, d.h,
                    run.tag.c_str(), ref, o.mean);
      records.push_back(rec);
      std::printf("  wrote vol/%s  (%lld^3, block max over %d^3)\n", file.c_str(), (long long)M, S);
    }
  }
  // Merge with an earlier index: keep its records for files this call did not
  // rewrite (one record per line, as written below), as tg_mhd_slices.py does.
  std::vector<std::string> keep;
  {
    std::ifstream old(idx);
    std::string line;
    const std::regex fre("\"file\": \"([^\"]+)\"");
    while (std::getline(old, line)) {
      std::smatch m;
      if (!std::regex_search(line, m, fre)) continue;
      bool rewritten = false;
      for (const auto& r : records) rewritten = rewritten || r.find("\"file\": \"" + m[1].str() + "\"") != std::string::npos;
      if (rewritten) continue;
      std::string rec = line.substr(line.find('{'));
      while (!rec.empty() && (rec.back() == ',' || rec.back() == ' ')) rec.pop_back();
      keep.push_back(rec);
    }
  }
  keep.insert(keep.end(), records.begin(), records.end());
  std::ofstream j(idx);
  j << "[\n";
  for (std::size_t i = 0; i < keep.size(); ++i) j << " " << keep[i] << (i + 1 < keep.size() ? ",\n" : "\n");
  j << "]\n";
  return bad ? 1 : 0;
}
