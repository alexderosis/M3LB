#pragma once
//==============================================================================
//  VTK PolyData (.vtp) writer -- triangle surfaces, ASCII.
//
//  WHY THIS EXISTS AT ALL, given VtiWriter. A .vti is vtkImageData: a uniform
//  grid over the WHOLE domain. A volume-penalised case has no geometry in it --
//  the sphere is a REGION marked by chi, because that is what penalisation is --
//  so ParaView opens the file and correctly draws a BOX. Getting the sphere
//  needs a Threshold or a Contour, and a reader who does not already know the
//  file has no geometry has no reason to guess that. This writes the surface as
//  an actual triangle mesh so it loads as a sphere with no filter at all.
//
//  IT IS CONTEXT, NOT DATA. The mesh carries no field values and is not what
//  anything is measured from: the physics lives on the .vti, and the effective
//  wall of a penalised boundary is not the nominal radius anyway -- the disc
//  measures it at R - 0.431 cells (demonstrator/mhd_decay.cpp's banner). So this
//  surface is drawn at the NOMINAL R, it is a reference frame for the eye, and
//  a measurement taken off it would be wrong by about half a cell.
//
//  ASCII on purpose. An icosphere at the default subdivision is 642 points and
//  1280 triangles -- about 40 kB as text, written once per run rather than once
//  per frame, so the binary path that VtiWriter needs for 12.6 MB frames buys
//  nothing here and costs the ability to read the file.
//
//  WRITTEN ONCE, NOT PER FRAME. The geometry does not move. A per-frame copy
//  would be 17 identical files and would invite someone to animate it.
//==============================================================================
#include "core/Types.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace lbm {

//------------------------------------------------------------------------------
// A geodesic sphere: an icosahedron with each triangle split into four, `subdiv`
// times, every new vertex pushed back onto the sphere. 20 * 4^subdiv triangles.
//
// An icosahedron rather than a lat-long grid because the latter crowds its
// vertices at the poles and leaves degenerate slivers there, which show up as a
// seam the moment the surface is lit or made translucent -- and translucent is
// exactly how a container surface gets used.
//------------------------------------------------------------------------------
inline void icosphere(int subdiv, double cx, double cy, double cz, double r,
                      std::vector<float>& pts, std::vector<std::int32_t>& tris) {
  const double t = (1.0 + std::sqrt(5.0)) / 2.0;
  std::vector<std::array<double, 3>> v = {
      {-1, t, 0}, {1, t, 0}, {-1, -t, 0}, {1, -t, 0},
      {0, -1, t}, {0, 1, t}, {0, -1, -t}, {0, 1, -t},
      {t, 0, -1}, {t, 0, 1}, {-t, 0, -1}, {-t, 0, 1}};
  for (auto& p : v) {
    const double n = std::sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
    p[0] /= n; p[1] /= n; p[2] /= n;
  }
  std::vector<std::array<int, 3>> f = {
      {0, 11, 5}, {0, 5, 1}, {0, 1, 7}, {0, 7, 10}, {0, 10, 11},
      {1, 5, 9}, {5, 11, 4}, {11, 10, 2}, {10, 7, 6}, {7, 1, 8},
      {3, 9, 4}, {3, 4, 2}, {3, 2, 6}, {3, 6, 8}, {3, 8, 9},
      {4, 9, 5}, {2, 4, 11}, {6, 2, 10}, {8, 6, 7}, {9, 8, 1}};

  for (int s = 0; s < subdiv; ++s) {
    // The midpoint cache is what keeps the mesh WATERTIGHT: two triangles share
    // an edge, and without it each would create its own copy of the midpoint,
    // giving a surface with cracks that a contour or a clip then leaks through.
    std::map<std::pair<int, int>, int> mid;
    auto midpoint = [&](int a, int b) {
      const auto key = std::minmax(a, b);
      auto it = mid.find({key.first, key.second});
      if (it != mid.end()) return it->second;
      std::array<double, 3> m{0.5 * (v[a][0] + v[b][0]), 0.5 * (v[a][1] + v[b][1]),
                              0.5 * (v[a][2] + v[b][2])};
      const double n = std::sqrt(m[0] * m[0] + m[1] * m[1] + m[2] * m[2]);
      m[0] /= n; m[1] /= n; m[2] /= n;
      v.push_back(m);
      const int id = int(v.size()) - 1;
      mid[{key.first, key.second}] = id;
      return id;
    };
    std::vector<std::array<int, 3>> nf;
    nf.reserve(f.size() * 4);
    for (const auto& tri : f) {
      const int a = midpoint(tri[0], tri[1]);
      const int b = midpoint(tri[1], tri[2]);
      const int c = midpoint(tri[2], tri[0]);
      nf.push_back({tri[0], a, c});
      nf.push_back({tri[1], b, a});
      nf.push_back({tri[2], c, b});
      nf.push_back({a, b, c});
    }
    f.swap(nf);
  }

  pts.clear(); pts.reserve(v.size() * 3);
  for (const auto& p : v) {
    pts.push_back(float(cx + r * p[0]));
    pts.push_back(float(cy + r * p[1]));
    pts.push_back(float(cz + r * p[2]));
  }
  tris.clear(); tris.reserve(f.size() * 3);
  for (const auto& tri : f) {
    tris.push_back(tri[0]); tris.push_back(tri[1]); tris.push_back(tri[2]);
  }
}

//------------------------------------------------------------------------------
inline void write_vtp_triangles(const std::string& path,
                                const std::vector<float>& pts,
                                const std::vector<std::int32_t>& tris) {
  const std::size_t np = pts.size() / 3, nt = tris.size() / 3;
  std::ofstream f(path);
  f.precision(7);
  f << "<?xml version=\"1.0\"?>\n"
    << "<VTKFile type=\"PolyData\" version=\"0.1\" byte_order=\"LittleEndian\">\n"
    << "  <PolyData>\n    <Piece NumberOfPoints=\"" << np
    << "\" NumberOfVerts=\"0\" NumberOfLines=\"0\" NumberOfStrips=\"0\""
       " NumberOfPolys=\"" << nt << "\">\n"
    << "      <Points>\n"
    << "        <DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">\n";
  for (std::size_t i = 0; i < pts.size(); i += 3)
    f << pts[i] << ' ' << pts[i + 1] << ' ' << pts[i + 2] << '\n';
  f << "        </DataArray>\n      </Points>\n      <Polys>\n"
    << "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n";
  for (std::size_t i = 0; i < tris.size(); i += 3)
    f << tris[i] << ' ' << tris[i + 1] << ' ' << tris[i + 2] << '\n';
  f << "        </DataArray>\n"
    << "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n";
  for (std::size_t k = 1; k <= nt; ++k) f << 3 * k << (k % 20 ? ' ' : '\n');
  f << "\n        </DataArray>\n      </Polys>\n    </Piece>\n"
    << "  </PolyData>\n</VTKFile>\n";
}

}  // namespace lbm
