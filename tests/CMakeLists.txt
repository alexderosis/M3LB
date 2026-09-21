foreach(t lattice streaming equilibrium collision moments boundary multiphase phase_field body rigid3d colour_gradient free_surface enthalpy)
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

#------------------------------------------------------------------------------
#  THE OSM IMPORTER'S OWN CHECKS, and the seam between it and the solver.
#
#  `tools/osm_city.py` is stdlib-only by design, so ctest needs an interpreter
#  and nothing else -- no venv, no pip, no network. Two tests, because they
#  check different things and only one of them can fail for an interesting
#  reason (the second lives next door, for the reason given against it):
#
#    osm_city           the importer against itself: projection against
#                       hard-coded pyproj values, polygon fill against areas
#                       known in closed form, the height ladder, .npy round
#                       trip, and both OSM response shapes.
#    osm_city_fixture   the importer against THE SOLVER -- registered in
#                       validation/CMakeLists.txt, because `height_field` is
#                       defined there and this directory is configured FIRST,
#                       so an `if(TARGET height_field)` here is simply false
#                       and the test would register nowhere, in silence.
#
#  What is NOT here is the Manchester cross-check against the Pollutant
#  project's osmnx raster (160,000 of 160,000 cells identical). It needs a
#  4 MB frozen Overpass snapshot that lives outside this repository, so it is a
#  recorded measurement in doc/m3lb.tex with the command to repeat it, not a
#  test. Run it when the rasteriser changes.
#------------------------------------------------------------------------------
find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
  add_test(NAME osm_city COMMAND ${Python3_EXECUTABLE}
           ${CMAKE_SOURCE_DIR}/tools/osm_city.py --selftest)
else()
  message(STATUS "LBM: no python3, skipping the OSM importer checks")
endif()
