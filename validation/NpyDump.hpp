#pragma once
//==============================================================================
//  A TINY .npy WRITER, shared by the cases that dump animation frames.
//
//  Why a header and not a copy in each case: validation/keyhole.cpp and
//  validation/melt_pool.cpp both dump frames for tools/render_*.py, and the
//  format is a CONTRACT with those scripts -- a magic, a length and an ASCII
//  dict that np.load parses. Two copies of a contract drift. (This is not the
//  src/ vs GPU/ duplication, which is deliberate and exists so the two can
//  disagree; here a disagreement would just be a bug.)
//
//  float32 throughout: these are frames for a picture, not a restart file, and
//  it halves the dump.
//==============================================================================
#include <cstdio>
#include <string>
#include <vector>

namespace lbm {

inline void write_npy(const std::string& path, const std::vector<float>& a,
               const std::vector<std::size_t>& shape) {
  std::size_t want = 1;
  for (std::size_t d : shape) want *= d;
  if (want != a.size()) {
    std::printf("write_npy: shape %zu does not match %zu values for %s -- "
                "NOT WRITTEN\n", want, a.size(), path.c_str());
    return;
  }
  std::string dict = "{'descr': '<f4', 'fortran_order': False, 'shape': (";
  for (std::size_t i = 0; i < shape.size(); ++i) {
    dict += std::to_string(shape[i]);
    if (i + 1 < shape.size() || shape.size() == 1) dict += ",";
    if (i + 1 < shape.size()) dict += " ";
  }
  dict += "), }";
  // the header (magic 6 + version 2 + len 2 + dict) must be a multiple of 64
  std::size_t pad = 64 - ((10 + dict.size() + 1) % 64);
  if (pad == 64) pad = 0;
  dict.append(pad, ' ');
  dict += "\n";
  std::FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) { std::printf("keyhole: cannot write %s\n", path.c_str()); return; }
  const unsigned char magic[8] = {0x93,'N','U','M','P','Y',1,0};
  std::fwrite(magic, 1, 8, f);
  const unsigned short hl = static_cast<unsigned short>(dict.size());
  std::fwrite(&hl, 2, 1, f);
  std::fwrite(dict.data(), 1, dict.size(), f);
  if (!a.empty()) std::fwrite(a.data(), sizeof(float), a.size(), f);
  std::fclose(f);
}

}  // namespace lbm
