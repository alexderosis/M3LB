foreach(v claudemd_skeleton_check density_anomaly ice_equilibrium mhd_cm_shifted mhd_parity_wall specular specular_node scalar_specular ehd_hydrostatic phase_wall poiseuille decaying_flows galilean thermal mhd regularized hartmann scalar_walls scalar_mass gaussian_plume height_field laplace layered_poiseuille floating_body gravity_wave stefan melt_pool marangoni mushy_sink recoil surface_tension)
  add_executable(${v} ${v}.cpp)
  target_link_libraries(${v} PRIVATE lbm)
  if(LBM_BUILD_TESTS)
    add_test(NAME ${v} COMMAND ${v} ${LBM_TEST_ARGS})
  endif()
endforeach()

# mixer_forward (doc/mixer_adjoint_plan.md, P1) is registered ONCE PER ANCHOR, so
# an anchor that fails by registration -- 2, 3 at L = 16 and 5 do, for measured
# reasons recorded in results/mixer_forward/registration/ -- cannot hide a
# regression in another. Anchor 4 (-band) takes hours and is not a test. Its
# gates are FP64 gates: on an FP32 build every entry exits 77, which
# SKIP_RETURN_CODE makes ctest report as SKIPPED rather than as a pass of gates
# that never ran.
add_executable(mixer_forward mixer_forward.cpp)
target_link_libraries(mixer_forward PRIVATE lbm)
if(LBM_BUILD_TESTS)
  foreach(k 1 2 3 5 6)
    add_test(NAME mixer_forward_${k} COMMAND mixer_forward -only ${k} ${LBM_TEST_ARGS})
    set_tests_properties(mixer_forward_${k} PROPERTIES SKIP_RETURN_CODE 77)
  endforeach()
endif()

#------------------------------------------------------------------------------
#  THE IMPORTER AGAINST THE SOLVER.  tools/osm_city.py writes a small
#  non-square city; height_field reads it back and checks the writer's own
#  answers rather than printing them. The seam is a .npy written by one
#  language and parsed by another, and the interesting way for it to fail is
#  not an exception but a TRANSPOSE -- which leaves every cell count, built
#  fraction and mass budget unchanged. That happened while this was written,
#  and only a diff against an independently built raster caught it; this is the
#  cheap standing guard against it coming back. The importer's own checks are
#  `osm_city` in tests/. Both are skipped, with a message, without python3.
#------------------------------------------------------------------------------
find_package(Python3 COMPONENTS Interpreter QUIET)
if(LBM_BUILD_TESTS AND Python3_Interpreter_FOUND)
  set(LBM_OSM_FIXTURE ${CMAKE_CURRENT_BINARY_DIR}/osm_fixture/fix)
  add_test(NAME osm_city_fixture COMMAND ${Python3_EXECUTABLE}
           ${CMAKE_SOURCE_DIR}/tools/osm_city.py --emit-fixture ${LBM_OSM_FIXTURE})
  set_tests_properties(osm_city_fixture PROPERTIES FIXTURES_SETUP osm_fixture)
  add_test(NAME height_field_osm COMMAND height_field ${LBM_TEST_ARGS})
  set_tests_properties(height_field_osm PROPERTIES
                       FIXTURES_REQUIRED osm_fixture
                       ENVIRONMENT "LBM_CITY=${LBM_OSM_FIXTURE}")
endif()

# Analysis tool, not a regression test: a full grid-convergence sweep over every
# lattice x operator combination, emitting CSV.
foreach(a specwall_unsteady mhd_pipe shercliff stokes_disc square_cylinder colour_flat ehd_cavity ehd_electroconvection enan_rt_cg phase_flat enan_wedge enan_rt enan_interface hartmann3d zhou_thermal channel3d static_droplet tgv3d_bench grid_convergence natural_convection orszag_tang cavity rayleigh_benard shear_layer tgv2d tgv3d dipole cavity_cm poiseuille_inlet hartmann_inlet orszag_tang_3d cmcheck forcing_cm forcing_transverse forcing_3d cmbench keyhole)
  add_executable(${a} ${a}.cpp)
  target_link_libraries(${a} PRIVATE lbm)
endforeach()
