#pragma once
//==============================================================================
//  VTK ImageData (.vti) and a .pvd time series, for ParaView. Binary, raw
//  appended, plain host C++ -- no CUDA, no solver headers.
//
//  WRITTEN FRESH RATHER THAN INCLUDED FROM ../src/io/VtiWriter.hpp, and that is
//  the tree's rule rather than an oversight: GPU/ shares no headers with the
//  Kokkos side, because the whole value of keeping two implementations is that
//  they agree where they overlap. A shared writer would make agreement on the
//  OUTPUT vacuous.
//
//  BINARY, because ASCII is not close. Six arrays over 128^3 is 12 floats per
//  point: 100 MB per frame raw against about 1.3 GB as text, and ParaView
//  spends the difference parsing.
//
//  THE APPENDED FORMAT, since an off-by-one here gives a file ParaView rejects
//  with a message that names nothing. Offsets count from the byte AFTER the '_'
//  that opens AppendedData, and with header_type="UInt32" every block is
//  <uint32 nbytes><nbytes of raw data> -- so array i's offset is the running
//  sum of (4 + bytes) over the arrays before it, and the count is part of the
//  block rather than a separate table.
//
//  encoding="raw" IS NOT OPTIONAL. VTK takes raw data only on an explicit
//  encoding attribute; without it the payload is base64-decoded into garbage
//  with no error raised.
//
//  X IS THE FASTEST AXIS, then y, then z -- what ImageData assumes. A cubic box
//  with a centred sphere is symmetric under swapping x and z, so a transposed
//  writer renders an IDENTICAL picture and passes every framing check; only a
//  non-cubic grid can see it. The Kokkos side measured its own loop that way
//  (7 x 11 x 13, 0 of 1001 points disagreeing) and this one uses the same order.
//==============================================================================
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

namespace lbm {

struct VtiArray {
  std::string name;
  int ncomp = 1;
  std::vector<float> data;          // npoints * ncomp, x fastest
};

// `spacing`: the physical size of one cell on every axis. The default 1 is grid
// index units and writes the same bytes as before it existed; src/tg_mhd.cu
// passes its h so ParaView shows the real [0, pi] box. Same as the parent's.
inline bool write_vti_bin(const std::string& path, int nx, int ny, int nz,
                          const std::vector<VtiArray>& arrays, double spacing = 1.0) {
  const std::size_t np = std::size_t(nx) * std::size_t(ny) * std::size_t(nz);
  for (const auto& a : arrays)
    if (a.data.size() != np * std::size_t(a.ncomp)) {
      std::fprintf(stderr, "write_vti_bin: array '%s' has %zu values, expected %zu\n",
                   a.name.c_str(), a.data.size(), np * std::size_t(a.ncomp));
      return false;
    }

  // Name the first 1- and 3-component arrays so ParaView colours by something
  // on load instead of opening grey.
  std::string scal, vec;
  for (const auto& a : arrays) {
    if (a.ncomp == 1 && scal.empty()) scal = a.name;
    if (a.ncomp == 3 && vec.empty())  vec  = a.name;
  }

  std::ofstream f(path, std::ios::binary);
  if (!f) { std::fprintf(stderr, "write_vti_bin: cannot open %s\n", path.c_str()); return false; }
  f << "<?xml version=\"1.0\"?>\n"
    << "<VTKFile type=\"ImageData\" version=\"1.0\" byte_order=\"LittleEndian\""
       " header_type=\"UInt32\">\n"
    << "  <ImageData WholeExtent=\"0 " << nx - 1 << " 0 " << ny - 1 << " 0 " << nz - 1
    << "\" Origin=\"0 0 0\" Spacing=\"" << spacing << " " << spacing << " "
    << spacing << "\">\n"
    << "    <Piece Extent=\"0 " << nx - 1 << " 0 " << ny - 1 << " 0 " << nz - 1
    << "\">\n      <PointData";
  if (!scal.empty()) f << " Scalars=\"" << scal << "\"";
  if (!vec.empty())  f << " Vectors=\"" << vec << "\"";
  f << ">\n";

  std::size_t off = 0;
  for (const auto& a : arrays) {
    f << "        <DataArray type=\"Float32\" Name=\"" << a.name
      << "\" NumberOfComponents=\"" << a.ncomp
      << "\" format=\"appended\" offset=\"" << off << "\"/>\n";
    off += sizeof(std::uint32_t) + a.data.size() * sizeof(float);
  }
  f << "      </PointData>\n    </Piece>\n  </ImageData>\n"
    << "  <AppendedData encoding=\"raw\">\n_";
  for (const auto& a : arrays) {
    const std::uint32_t nb = std::uint32_t(a.data.size() * sizeof(float));
    f.write(reinterpret_cast<const char*>(&nb), sizeof nb);
    f.write(reinterpret_cast<const char*>(a.data.data()), std::streamsize(nb));
  }
  f << "\n  </AppendedData>\n</VTKFile>\n";
  return bool(f);
}

//------------------------------------------------------------------------------
// The time series. Without it ParaView loads the frames as a numbered file
// GROUP, which animates but puts the frame INDEX on the clock -- so an unevenly
// sampled run plays back at the wrong rate and the time annotation is a lie.
//
// `file` is written as given; keep it relative to the .pvd so the directory can
// be copied to another machine and still open.
//------------------------------------------------------------------------------
inline bool write_pvd(const std::string& path,
                      const std::vector<std::pair<double, std::string>>& frames) {
  std::ofstream f(path);
  if (!f) return false;
  f.precision(9);
  f << "<?xml version=\"1.0\"?>\n"
    << "<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\">\n"
    << "  <Collection>\n";
  for (const auto& fr : frames)
    f << "    <DataSet timestep=\"" << fr.first
      << "\" group=\"\" part=\"0\" file=\"" << fr.second << "\"/>\n";
  f << "  </Collection>\n</VTKFile>\n";
  return bool(f);
}

}  // namespace lbm
