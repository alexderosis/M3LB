// Shell-averaged energy spectra (and optionally |j|) from a .vti frame of M3LB's GPU orszag_tang driver.
// The frame holds Jmag, rho, u, b as Float32, appended raw, little-endian, UInt32 byte counts (lbm::write_vti_bin).
// u and b are in lattice units; physical u = u_lat / v0 and b = b_lat / v0, with v0 = Ma / (2 sqrt 2 sqrt 3).
//   E(k) = 1/2 sum over the shell k-1/2 <= |k| < k+1/2 of |F(q)|^2 / N^6   (so that sum_k E(k) = <|q|^2>/2)
//   usage: spectra_vti frame.vti out_prefix [-ma 0.034] [-threads 8] [-jraw stride]
//   writes out_prefix_spec.txt (k E_u E_b) and, with -jraw s, out_prefix_jspec_s.raw: |curl b| (physical) by
//   spectral differentiation with the 2/3 cube mask |k_a| <= N/3, sampled every s nodes (int32 n,n,n then float32).
#include "fft3.hpp"
#include <cstdint>
#include <fstream>
#include <sstream>
#include <map>

int main(int argc, char** argv) {
  if (argc < 3) { fprintf(stderr, "usage: spectra_vti frame.vti out_prefix [-ma Ma] [-threads n] [-jraw stride]\n"); return 1; }
  const std::string in = argv[1], out = argv[2];
  double Ma = 0.034; int jstride = 0;
  for (int i = 3; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "-ma" && i + 1 < argc) Ma = atof(argv[++i]);
    if (a == "-threads" && i + 1 < argc) NT = atoi(argv[++i]);
    if (a == "-jraw" && i + 1 < argc) jstride = atoi(argv[++i]);
  }
  const double v0 = Ma / (2. * std::sqrt(2.) * std::sqrt(3.));
  std::ifstream f(in, std::ios::binary);
  if (!f) { fprintf(stderr, "cannot open %s\n", in.c_str()); return 1; }
  // header up to the '_' that starts the appended block
  std::string hdr; char c;
  while (f.get(c)) { hdr.push_back(c); if (hdr.size() > 8 && hdr.compare(hdr.size() - 2, 2, "\n_") == 0 && hdr.find("<AppendedData") != std::string::npos) break; }
  int N = 0;
  { size_t p = hdr.find("WholeExtent=\""); int a, b, cc, d, e, g; sscanf(hdr.c_str() + p + 13, "%d %d %d %d %d %d", &a, &b, &cc, &d, &e, &g); N = b + 1; if (d + 1 != N || g + 1 != N) { fprintf(stderr, "not a cube\n"); return 1; } }
  // arrays in file order
  std::vector<std::pair<std::string, int>> arrs;
  for (size_t p = hdr.find("<DataArray"); p != std::string::npos; p = hdr.find("<DataArray", p + 1)) {
    size_t q = hdr.find("Name=\"", p) + 6, r = hdr.find('"', q);
    size_t s2 = hdr.find("NumberOfComponents=\"", p) + 20;
    arrs.push_back({hdr.substr(q, r - q), atoi(hdr.c_str() + s2)});
  }
  const size_t n3 = (size_t)N * N * N;
  std::map<std::string, std::vector<float>> data;
  for (auto& a : arrs) {
    uint32_t nb; f.read(reinterpret_cast<char*>(&nb), 4);
    if (a.first == "u" || a.first == "b") {
      std::vector<float> v(n3 * a.second); f.read(reinterpret_cast<char*>(v.data()), nb);
      data[a.first] = std::move(v);
    } else f.seekg(nb, std::ios::cur);
  }
  if (!data.count("u") || !data.count("b")) { fprintf(stderr, "u or b missing\n"); return 1; }
  FFT3 fft(N);
  std::vector<double> kv(N); for (int i = 0; i < N; ++i) kv[i] = (i <= N / 2) ? i : i - N;
  const int nsh = int(std::ceil(std::sqrt(3.) * N / 2)) + 2;
  std::vector<double> Eu(nsh, 0.), Eb(nsh, 0.);
  const double norm = 1. / ((double)n3 * (double)n3);
  vec F(n3), Bh[3];
  for (int which = 0; which < 2; ++which) {
    const std::vector<float>& v = data[which == 0 ? "u" : "b"];
    std::vector<double>& E = (which == 0) ? Eu : Eb;
    for (int comp = 0; comp < 3; ++comp) {
      for (size_t i = 0; i < n3; ++i) F[i] = v[3 * i + comp] / v0;
      fft(F, false);
      for (int iz = 0; iz < N; ++iz) for (int iy = 0; iy < N; ++iy) for (int ix = 0; ix < N; ++ix) {
        const size_t id = ((size_t)iz * N + iy) * N + ix;
        const double k = std::sqrt(kv[ix] * kv[ix] + kv[iy] * kv[iy] + kv[iz] * kv[iz]);
        E[(int)(k + 0.5)] += 0.5 * std::norm(F[id]) * norm;
      }
      if (which == 1 && jstride > 0) Bh[comp] = F;
    }
  }
  FILE* fo = fopen((out + "_spec.txt").c_str(), "w");
  fprintf(fo, "# %s  N=%d  v0=%.8e  physical units;  columns: k  E_u(k)  E_b(k)\n", in.c_str(), N, v0);
  double su = 0, sb = 0;
  for (int k = 0; k < nsh; ++k) { fprintf(fo, "%d %.10e %.10e\n", k, Eu[k], Eb[k]); su += Eu[k]; sb += Eb[k]; }
  fclose(fo);
  fprintf(stderr, "%s: N = %d, E_u = %.8f, E_b = %.8f (physical)\n", in.c_str(), N, su, sb);
  if (jstride > 0) {
    const int kmax = N / 3; const cd I(0, 1);
    vec J[3]; for (int c2 = 0; c2 < 3; ++c2) J[c2].assign(n3, 0);
    for (int iz = 0; iz < N; ++iz) for (int iy = 0; iy < N; ++iy) for (int ix = 0; ix < N; ++ix) {
      const size_t id = ((size_t)iz * N + iy) * N + ix;
      const double a = kv[ix], b = kv[iy], cc = kv[iz];
      if (std::abs(a) > kmax || std::abs(b) > kmax || std::abs(cc) > kmax) continue;
      J[0][id] = I * (b * Bh[2][id] - cc * Bh[1][id]);
      J[1][id] = I * (cc * Bh[0][id] - a * Bh[2][id]);
      J[2][id] = I * (a * Bh[1][id] - b * Bh[0][id]);
    }
    for (int c2 = 0; c2 < 3; ++c2) fft(J[c2], true);
    const int Mr = N / jstride;
    std::vector<float> o((size_t)Mr * Mr * Mr);
    for (int z = 0; z < Mr; ++z) for (int y = 0; y < Mr; ++y) for (int x = 0; x < Mr; ++x) {
      const size_t id = ((size_t)(z * jstride) * N + y * jstride) * N + x * jstride;
      o[((size_t)z * Mr + y) * Mr + x] = (float)std::sqrt(std::norm(J[0][id].real()) + std::norm(J[1][id].real()) + std::norm(J[2][id].real()));
    }
    const std::string jn = out + "_jspec_s" + std::to_string(jstride) + ".raw";
    std::ofstream jo(jn, std::ios::binary);
    const int32_t m = Mr; jo.write(reinterpret_cast<const char*>(&m), 4); jo.write(reinterpret_cast<const char*>(&m), 4); jo.write(reinterpret_cast<const char*>(&m), 4);
    jo.write(reinterpret_cast<const char*>(o.data()), std::streamsize(o.size() * 4));
    fprintf(stderr, "  wrote %s (%d^3)\n", jn.c_str(), Mr);
  }
  return 0;
}
