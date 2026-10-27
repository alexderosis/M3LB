# Nianzhi's mixer1.stl, run in M3LB (2026-10-23)

`demonstrator/mixer_stl.cpp` on the voxels `tools/stl_voxels.py` recovers from
Nianzhi's `mixer1.stl` (OpenLB driver `mixer3d.cpp`): square duct W = 200 um,
900 um long, obstacles on the floor half from 200 to 700 um; U = 0.01 m/s,
nu = 1e-6, D = 1e-9 m^2/s, so Re_W = 2, Pe_W = 2000. tau = 0.875, FP64,
4 threads. The STL itself is NOT in the repository (Nianzhi's geometry; the
repository is public).

    n20fix.*     N = 20 (dx = 5 um), uniform correction on   -- the result
    n20nofix.*   N = 20, correction off  (M shifted by the 2% compressibility dilution)
    n20empty.*   N = 20, all-fluid voxels -- against empty_duct_reference.py
    n30fix.*     N = 30 (dx = 3.33 um), correction on -- the grid check

Field dumps (`*_c.npy`, `*_u?.npy`) are not kept: rerun with `-out`. The figures
(`tools/plot_mixer_stl.py`) are not kept either: their plan views show the obstacle
layout, which is Nianzhi's unpublished design.
