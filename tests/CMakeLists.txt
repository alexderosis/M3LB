foreach(t lattice streaming equilibrium collision moments boundary multiphase phase_field body rigid3d colour_gradient free_surface)
  add_executable(test_${t} test_${t}.cpp)
  target_link_libraries(test_${t} PRIVATE lbm)
  target_include_directories(test_${t} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  add_test(NAME ${t} COMMAND test_${t} ${LBM_TEST_ARGS})
endforeach()

#------------------------------------------------------------------------------
#  THE CROSS-CHECK: the parent's colour-gradient operator against GPU/'s.
#
#  Two objects, because both code bases put their lattice and their operator in
#  `namespace lbm` and including both headers in one translation unit collides.
#  Joined by extern "C", they never meet.
#
#  THE PARENT REACHES INTO GPU/include AND NOT THE OTHER WAY. GPU/ is a separate
#  CMake project that deliberately does not link Kokkos; making it do so to run
#  this test would spend the independence that makes the comparison mean
#  anything. So only this target knows about both.
#
#  PRECISION HAS TO BE MATCHED BY HAND, and the two defaults are OPPOSITE:
#  GPU/'s Real is float unless LBM_DOUBLE is defined, the parent's is double
#  unless LBM_SINGLE_PRECISION is. Mismatching them still COMPILES and reports a
#  disagreement of about 1e-7 -- FP32 epsilon, indistinguishable at a glance from
#  a physics bug. That happened while this was being written.
#
#  Skipped, with a message, if GPU/ is not present: this is one repository today,
#  but the test is the only thing in the parent that assumes so.
#------------------------------------------------------------------------------
if(EXISTS ${CMAKE_SOURCE_DIR}/GPU/include/lbm/colour.cuh)
  add_executable(test_cross_colour test_cross_colour.cpp cross_colour_gpu.cpp)
  target_link_libraries(test_cross_colour PRIVATE lbm)
  target_include_directories(test_cross_colour PRIVATE
                             ${CMAKE_CURRENT_SOURCE_DIR}
                             ${CMAKE_SOURCE_DIR}/GPU/include)
  if(NOT LBM_PRECISION STREQUAL "float")
    set_source_files_properties(cross_colour_gpu.cpp PROPERTIES
                                COMPILE_DEFINITIONS LBM_DOUBLE)
  endif()
  add_test(NAME cross_colour COMMAND test_cross_colour ${LBM_TEST_ARGS})
else()
  message(STATUS "LBM: GPU/ not present, skipping the colour-gradient cross-check")
endif()
