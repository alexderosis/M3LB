#pragma once
//==============================================================================
//  Which lattice a gradient stencil runs on, given the lattice of the field.
//
//  A pure lattice trait, and it lives here rather than beside its first consumer
//  because it has THREE of them -- PhaseFieldSolver, ScalarGradient and
//  ViscousInterfaceForce -- and the other two were reaching it by including the
//  whole phase-field solver. That was free until PhaseFieldSolver.hpp acquired a
//  default collision operator, at which point every thermal translation unit in
//  the tree started compiling two central-moment collision headers to learn one
//  typedef. Splitting it out costs nothing and stops that.
//
//  THE RULE: the richest velocity set of the same dimension, not the field's
//  own. Isotropy of the gradient stencil is what sets the spurious-current floor
//  around a static droplet, and the axis-only stencil of a reduced lattice is
//  markedly worse. So a D3Q7 field still pays a 27-neighbour gather for its
//  gradient -- which is also why choosing D3Q7 for a phase field buys less than
//  its population count suggests.
//==============================================================================
#include "lattice/Lattices.hpp"

namespace lbm {

template <class L> struct GradientLatticeOf;
template <> struct GradientLatticeOf<D2Q5>  { using type = D2Q9;  };
template <> struct GradientLatticeOf<D2Q9>  { using type = D2Q9;  };
template <> struct GradientLatticeOf<D3Q7>  { using type = D3Q27; };
template <> struct GradientLatticeOf<D3Q27> { using type = D3Q27; };

template <class L> using GradientLattice = typename GradientLatticeOf<L>::type;

}  // namespace lbm
