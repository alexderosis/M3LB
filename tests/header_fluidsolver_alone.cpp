//==============================================================================
//  solver/FluidSolver.hpp, INCLUDED FIRST AND ALONE.
//
//  Not a test that runs: an object that has to COMPILE. FluidSolver.hpp was the
//  one header in src/ that did not compile on its own -- it names `Macro`,
//  which lived in collision/BGK.hpp, so every case had to include a collision
//  header first and "alphabetical include order fails to compile" was written
//  into CLAUDE.md, ONBOARDING.md and the new-case skill as a rule to remember.
//  Macro moved to core/Types.hpp on 2026-11-10 and this file is what keeps the
//  rule from coming back: if a later edit makes the solver lean on something a
//  collision header declares, the build stops here and says which name.
//
//  WHAT IT DOES NOT DO: check the other forty-eight headers. They compiled alone
//  when all forty-nine were tried that day; tests/headers_alone.sh re-tries
//  them, by hand, since it costs two minutes. Instantiating
//  the solver is also out of scope -- that needs an operator, and every other
//  target in the tree does it.
//==============================================================================
#include "solver/FluidSolver.hpp"

namespace lbm_header_check {
// Something for the object file to hold, and a use of the name that used to be
// missing, so a reader sees what the header is expected to bring with it.
inline constexpr std::size_t macro_size = sizeof(lbm::Macro);
}  // namespace lbm_header_check
