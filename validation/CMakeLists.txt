foreach(v claudemd_skeleton_check density_anomaly ice_equilibrium mhd_cm_shifted specular specular_node scalar_specular ehd_hydrostatic phase_wall poiseuille decaying_flows galilean thermal mhd regularized hartmann scalar_walls scalar_mass gaussian_plume height_field laplace layered_poiseuille floating_body gravity_wave stefan melt_pool marangoni mushy_sink recoil surface_tension)
  add_executable(${v} ${v}.cpp)
  target_link_libraries(${v} PRIVATE lbm)
  if(LBM_BUILD_TESTS)
    add_test(NAME ${v} COMMAND ${v} ${LBM_TEST_ARGS})
  endif()
endforeach()

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
foreach(a mhd_pipe shercliff stokes_disc square_cylinder colour_flat ehd_cavity ehd_electroconvection enan_rt_cg phase_flat enan_wedge enan_rt enan_interface hartmann3d zhou_thermal channel3d static_droplet tgv3d_bench grid_convergence natural_convection orszag_tang cavity rayleigh_benard shear_layer tgv2d tgv3d dipole cavity_cm poiseuille_inlet hartmann_inlet orszag_tang_3d cmcheck forcing_cm forcing_transverse forcing_3d cmbench keyhole)
  add_executable(${a} ${a}.cpp)
  target_link_libraries(${a} PRIVATE lbm)
endforeach()
