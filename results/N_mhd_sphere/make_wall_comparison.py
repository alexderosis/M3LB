#!/usr/bin/env python3
"""
Compose the insulating-vs-conducting comparison animation for mhd_sphere.

    python3 make_wall_comparison.py [out.mp4]

WHY THIS FILE EXISTS. The .mp4 is not tracked -- .gitignore's rule is that a
rendered animation is regenerable from a dump and the dump is itself ignored --
so the composition has to be tracked instead, or it lives only in whoever's
shell history built it. This is that composition.

THE WHOLE PIPELINE, because the composition is the last step of four and the
earlier three are what actually cost the time:

    # 1. the two runs. IDENTICAL except -mwall; the seed is fixed, so this is a
    #    one-variable comparison and not two similar-looking runs.
    b=./build/demonstrator/mhd_sphere
    $b --kokkos-num-threads=4 -n 96 -re 500 -kmax 6 -k0 4 -steps 3072 \
       -dump 16 -mwall cond
    # 2. render that case, then MOVE the outputs aside -- both runs write to
    #    results/N_mhd_sphere/anim_frames/ and the second overwrites the first
    python3 render_slices.py && python3 render_volume.py
    ffmpeg -y -framerate 20 -i png/frame_%04d.png -c:v libx264 -pix_fmt yuv420p \
       -crf 20 sphere_n96re500_slices.mp4
    ffmpeg -y -framerate 20 -i png/jvol_%04d.png -c:v libx264 -pix_fmt yuv420p \
       -crf 18 -vf "scale=iw*2:ih*2:flags=neighbor" sphere_n96re500_jvol.mp4
    # 3. repeat 1-2 with -mwall insul, naming the outputs *_insul_*
    # 4. this script

LAYOUT. Two rows, insulating above conducting, each row the three-panel slice
view beside the |J| volume projection, with a colour-keyed label strip above.
The strips are drawn with render_slices.py's own bitmap font, imported rather
than reimplemented, because THIS ffmpeg HAS NO drawtext -- a stacked comparison
with no key is unreadable, and there is no other way to put text on it.

THE VOLUME IS CROPPED TO THE SPHERE, for two reasons and one of them is a bug.
The good reason: its caption duplicates the slice panel's, which already carries
t/Te, the frame, N, R and both energies. The bug: render_volume.py sizes its
canvas DOWN as N grows -- 462 px wide at N = 64, 348 at N = 96 -- while the
caption length does not follow, so at N = 96 the text overruns and is cut
mid-word at both top and bottom ("MAX-INTENSITY PROJ", "(global P9",
"= 0.457 x vre"). The numbers survive; the trailing labels do not. Cropping
sidesteps it here, and the standalone volume .mp4s still carry it. The real fix
is render_volume.py --size (default 440), which has NOT been applied because
re-rendering the conducting case needs its dumps and those were overwritten by
the insulating run.

WIDTHS ARE PINNED, NOT DERIVED. scale=-2:534 on the cropped volume rounds to
620 where the label strip was built for 618, and vstack then fails with a width
mismatch rather than letting it slide. So the volume width is stated explicitly
and LBL_W is its sum with the slice width; changing either means changing both.
"""
import importlib.util
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

SLICE_W, ROW_H = 924, 534      # the slice render at N = 96
VOL_W = 618                    # the cropped volume, scaled to ROW_H
LBL_W = SLICE_W + VOL_W        # every vstack input must be exactly this wide
LBL_H = 34
CROP = "crop=696:600:0:130"    # the sphere alone, dropping both captions
FPS, FRAMES = 20, 193

CASES = [
    ("insul", "INSULATING", "B = 0   flux leaves the wall"
     "            N=96  Re=500  Pm=1  |J| volume at right", (120, 190, 255)),
    ("cond", "CONDUCTING", "B.n = 0   flux stays in the volume"
     "      same seed, same parameters, one flag apart", (255, 176, 96)),
]


def load_font():
    """render_slices.py is a script, not a module, but importing it is still the
    right call: it owns the font, the Canvas and the PNG encoder, and a second
    copy of any of them would drift."""
    spec = importlib.util.spec_from_file_location(
        "render_slices", os.path.join(HERE, "render_slices.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def strip(rs, path, label, sub, accent):
    cv = rs.Canvas(LBL_W, LBL_H, (12, 12, 16))
    cv.rect(0, 0, 6, LBL_H, accent)            # colour key down the left edge
    x = cv.text(14, 9, label, accent, 2)
    cv.text(x + 16, 13, sub, (150, 150, 160), 1)
    rs.write_png(path, LBL_W, LBL_H, cv.px)


def main(argv):
    out = argv[1] if len(argv) > 1 else os.path.join(
        HERE, "mhd_sphere_wall_comparison.mp4")
    rs = load_font()

    src, missing = {}, []
    for key, _, _, _ in CASES:
        tag = "_insul" if key == "insul" else ""
        for kind in ("slices", "jvol"):
            p = os.path.join(HERE, "sphere_n96re500%s_%s.mp4" % (tag, kind))
            src[(key, kind)] = p
            if not os.path.exists(p):
                missing.append(os.path.basename(p))
    if missing:
        sys.stderr.write("missing input(s): %s\n"
                         "Run steps 1-3 from this file's docstring first.\n"
                         % ", ".join(missing))
        return 1

    labs = {}
    for key, label, sub, accent in CASES:
        labs[key] = os.path.join(HERE, ".lab_%s.png" % key)
        strip(rs, labs[key], label, sub, accent)

    # inputs: 0 insul slices, 1 insul jvol, 2 cond slices, 3 cond jvol, 4/5 labels
    cmd = ["ffmpeg", "-y",
           "-i", src[("insul", "slices")], "-i", src[("insul", "jvol")],
           "-i", src[("cond", "slices")], "-i", src[("cond", "jvol")],
           "-loop", "1", "-framerate", str(FPS), "-i", labs["insul"],
           "-loop", "1", "-framerate", str(FPS), "-i", labs["cond"],
           "-filter_complex",
           "[1:v]%s,scale=%d:%d[iv];"
           "[3:v]%s,scale=%d:%d[cv];"
           "[0:v][iv]hstack=inputs=2[top];"
           "[2:v][cv]hstack=inputs=2[bot];"
           "[4:v][top][5:v][bot]vstack=inputs=4[v]"
           % (CROP, VOL_W, ROW_H, CROP, VOL_W, ROW_H),
           "-map", "[v]",
           # -frames:v is load-bearing: the label inputs are -loop 1, i.e.
           # INFINITE, and -shortest does not terminate a filter_complex. Without
           # this the encode never ends.
           "-frames:v", str(FRAMES),
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
           "-movflags", "+faststart", out]
    r = subprocess.run(cmd, capture_output=True, text=True)
    for p in labs.values():
        if os.path.exists(p):
            os.remove(p)
    if r.returncode != 0:
        sys.stderr.write(r.stderr[-2000:])
        return r.returncode
    print("wrote %s  (%d x %d, %d frames)"
          % (out, LBL_W, LBL_H * 2 + ROW_H * 2, FRAMES))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
