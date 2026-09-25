#pragma once
//==============================================================================
//  VTK ImageData (.vti) writers for ParaView. TWO of them, and the choice is
//  not stylistic.
//
//  write_vti      -- ASCII, one solver, rho + u + flag. Adequate for
//                    validation-sized grids and unchanged since Milestone 3.
//  write_vti_bin  -- BINARY (raw appended), an arbitrary list of named arrays
//                    from wherever the caller likes, plus write_pvd for a time
//                    series.
//
//  WHY THE SECOND ONE EXISTS. ASCII is about 13x larger and it is not close:
//  a 64^3 grid with u, b, J, |J|, rho and chi is 12 floats per point, which is
//  12.6 MB per frame as raw Float32 and roughly 160 MB as ASCII Float64. Over
//  the 129 frames of a decaying-MHD run that is 1.6 GB against 21 GB, and
//  ParaView spends the difference parsing text. The ASCII writer is kept
//  because several cases use it and it is easier to eyeball when a single frame
//  looks wrong.
//
//  AND WHY IT IS NOT TEMPLATED ON A SOLVER. write_vti takes a Solver and asks
//  it for rho/u/flags, which is exactly why it cannot write a MAGNETIC field:
//  that lives in a different solver, and J lives in neither -- it is a curl the
//  case computes. A coupled case has no single object to hand over. So this one
//  takes arrays, and the case decides what is worth writing.
//
//  THE APPENDED FORMAT, since an off-by-one here produces a file ParaView
//  rejects with a message that names nothing useful. Offsets are relative to
//  the byte AFTER the '_' that opens AppendedData, and with header_type="UInt32"
//  every block is <uint32 nbytes><nbytes of raw data> -- so the offset of array
//  i is the running sum of (4 + bytes) over the arrays before it, and the byte
//  count is part of the block rather than a separate table.
//
//  AND THE AXIS ORDER IS MEASURED, NOT READ. A cubic box with a centred sphere
//  is symmetric under swapping x and z, so a d.id(z,y,x)-style transposition
//  renders an IDENTICAL picture, has the identical value count, and passes every
//  framing check there is -- the file is well formed and the physics is mirrored.
//  Only a non-cubic grid can see it. Checked on 7 x 11 x 13 with the point's own
//  coordinates written as the data: 0 of 1001 points disagreed with VTK's
//  x-fastest order. Re-run that if this writer's loops or Domain::id ever move;
//  a 64^3 frame will not tell you.
//
//  Milestone 6 replaces both with XDMF + HDF5 so that output is parallel and
//  does not scale with rank count.
//==============================================================================
#include "core/Types.hpp"
#include "grid/Domain.hpp"

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace lbm {

template <class Solver>
void write_vti(const std::string& path, Solver& s) {
  s.compute_macroscopic();
  const Domain& d = s.domain();

  auto h_rho = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.rho());
  auto h_ux  = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.ux());
  auto h_uy  = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.uy());
  auto h_uz  = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.uz());
  auto h_fl  = Kokkos::create_mirror_view_and_copy(HostSpace{}, s.flags());

  std::ofstream f(path);
  f.precision(9);
  f << "<?xml version=\"1.0\"?>\n"
    << "<VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"LittleEndian\">\n"
    << "  <ImageData WholeExtent=\"0 " << d.nx - 1 << " 0 " << d.ny - 1 << " 0 "
    << d.nz - 1 << "\" Origin=\"0 0 0\" Spacing=\"1 1 1\">\n"
    << "    <Piece Extent=\"0 " << d.nx - 1 << " 0 " << d.ny - 1 << " 0 "
    << d.nz - 1 << "\">\n      <PointData Scalars=\"rho\" Vectors=\"u\">\n";

  f << "        <DataArray type=\"Float64\" Name=\"rho\" format=\"ascii\">\n";
  for (Index z = 0; z < d.nz; ++z)
    for (Index y = 0; y < d.ny; ++y)
      for (Index x = 0; x < d.nx; ++x) f << h_rho(d.id(x, y, z)) << ' ';
  f << "\n        </DataArray>\n";

  f << "        <DataArray type=\"Float64\" Name=\"u\" NumberOfComponents=\"3\" format=\"ascii\">\n";
  for (Index z = 0; z < d.nz; ++z)
    for (Index y = 0; y < d.ny; ++y)
      for (Index x = 0; x < d.nx; ++x) {
        const Index n = d.id(x, y, z);
        f << h_ux(n) << ' ' << h_uy(n) << ' ' << h_uz(n) << ' ';
      }
  f << "\n        </DataArray>\n";

  f << "        <DataArray type=\"UInt8\" Name=\"flag\" format=\"ascii\">\n";
  for (Index z = 0; z < d.nz; ++z)
    for (Index y = 0; y < d.ny; ++y)
      for (Index x = 0; x < d.nx; ++x) f << int(h_fl(d.id(x, y, z))) << ' ';
  f << "\n        </DataArray>\n"
    << "      </PointData>\n    </Piece>\n  </ImageData>\n</VTKFile>\n";
}

//------------------------------------------------------------------------------
// One named point array. `data` is npoints * ncomp floats with X FASTEST, then
// y, then z -- the order ImageData assumes and the order every loop in this
// tree already writes.
//------------------------------------------------------------------------------
struct VtiArray {
  std::string name;
  int ncomp = 1;
  std::vector<float> data;
};

// `spacing` is the physical size of one cell, the same on every axis. It
// defaults to 1 -- grid-index units -- so every caller written before it existed
// produces the same bytes; a case that wants ParaView to show its real box
// (demonstrator/tg_mhd: [0, pi]^3) passes its h.
inline void write_vti_bin(const std::string& path, Index nx, Index ny, Index nz,
                          const std::vector<VtiArray>& arrays, double spacing = 1.0) {
  const std::size_t np = std::size_t(nx) * std::size_t(ny) * std::size_t(nz);
  for (const auto& a : arrays)
    if (a.data.size() != np * std::size_t(a.ncomp))
      throw std::runtime_error("write_vti_bin: array '" + a.name + "' has " +
                               std::to_string(a.data.size()) + " values, expected " +
                               std::to_string(np * std::size_t(a.ncomp)));

  // Declare the first 1- and 3-component arrays so ParaView colours by
  // something sensible on load rather than opening grey.
  std::string scal, vec;
  for (const auto& a : arrays) {
    if (a.ncomp == 1 && scal.empty()) scal = a.name;
    if (a.ncomp == 3 && vec.empty())  vec  = a.name;
  }

  std::ofstream f(path, std::ios::binary);
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
}

//------------------------------------------------------------------------------
// The time series. Without this ParaView loads the frames as a numbered file
// GROUP, which animates but puts the frame INDEX on the clock -- so a run whose
// samples are not equally spaced in time plays back at the wrong rate and the
// time annotation is a lie. The .pvd carries the real times.
//
// Paths are written as given; keep them relative to the .pvd so the directory
// can be moved or copied to another machine and still open.
//------------------------------------------------------------------------------
inline void write_pvd(const std::string& path,
                      const std::vector<std::pair<double, std::string>>& frames) {
  std::ofstream f(path);
  f.precision(9);
  f << "<?xml version=\"1.0\"?>\n"
    << "<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\">\n"
    << "  <Collection>\n";
  for (const auto& fr : frames)
    f << "    <DataSet timestep=\"" << fr.first
      << "\" group=\"\" part=\"0\" file=\"" << fr.second << "\"/>\n";
  f << "  </Collection>\n</VTKFile>\n";
}

}  // namespace lbm
