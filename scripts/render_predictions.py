#!/usr/bin/env python
"""Render FastFusion-IF predictions onto molecular surfaces (PyMOL).

Turns the per-residue prediction CSV written by ``evaluate.py`` /
``evaluate_ensemble.py`` into publication figures of the kind used throughout
the interface-prediction literature:

  * **probability view** -- the solvent-excluded surface coloured by predicted
    interface score on a blue -> white -> red scale, with the experimentally
    determined interface drawn on top as a green contour (MaSIF-site, Fig. 4 style).
  * **class view** -- true positives green, false positives red, false negatives
    yellow, correct rejections grey (AGAT-PPIS / GraphPPIS style).

Nothing is re-predicted here: the probabilities come straight from the CSV that
produced your reported metrics, so the figure and the table cannot disagree.

The ground-truth contour is computed geometrically: an interface residue is on
the contour when at least one non-interface residue lies within
``--contour-radius`` Angstrom of it. That traces the rim of the true interface
patch rather than shading the whole patch, which is what makes the overlay
readable on top of the heat map.

Requires PyMOL only at render time::

    conda install -c conda-forge pymol-open-source

This script itself needs only numpy + pandas: it writes the annotated PDB files
and the .pml scripts, then you run PyMOL on them.

Examples
--------
Automatically pick a strong, a typical and a hard example::

    python scripts/render_predictions.py \
        --predictions eval/bench_evo_ens_test315_tuned/test_ensemble_per_residue_predictions.csv \
        --per-protein eval/bench_evo_ens_test315_tuned/test_ensemble_per_protein_metrics.csv \
        --out-dir figures/renders --auto 3

Render named chains::

    python scripts/render_predictions.py \
        --predictions eval/bench_evo_ens_ubtest_tuned/test_ensemble_per_residue_predictions.csv \
        --out-dir figures/renders --protein 1cdbA 2c0mC

Then::

    cd figures/renders && pymol -cq render_all.pml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# PDB fixed-column layout (0-indexed python slices)
COL_CHAIN = 21
COL_RESSEQ = slice(22, 26)
COL_ICODE = 26
COL_OCC = slice(54, 60)
COL_BFAC = slice(60, 66)
COL_XYZ = (slice(30, 38), slice(38, 46), slice(46, 54))

CLASS_CODE = {"TN": 0.0, "TP": 1.0, "FP": 2.0, "FN": 3.0}


def res_key(chain: str, seq, icode) -> tuple[str, int, str]:
    """Normalise a residue identifier so CSV rows and PDB lines agree."""
    if icode is None or (isinstance(icode, float) and np.isnan(icode)):
        icode = ""
    return (str(chain).strip(), int(seq), str(icode).strip())


def load_pdb(path: Path) -> list[str]:
    return path.read_text().splitlines()


def residue_coords(lines: list[str], chain: str) -> dict[tuple, np.ndarray]:
    """Representative coordinate per residue (CA if present, else centroid)."""
    acc: dict[tuple, list] = {}
    ca: dict[tuple, np.ndarray] = {}
    for ln in lines:
        if not ln.startswith(("ATOM", "HETATM")) or len(ln) < 54:
            continue
        if ln[COL_CHAIN].strip() != chain:
            continue
        try:
            key = res_key(ln[COL_CHAIN], ln[COL_RESSEQ], ln[COL_ICODE])
            xyz = np.array([float(ln[s]) for s in COL_XYZ])
        except ValueError:
            continue
        acc.setdefault(key, []).append(xyz)
        if ln[12:16].strip() == "CA":
            ca[key] = xyz
    return {k: ca.get(k, np.mean(v, axis=0)) for k, v in acc.items()}


def contour_residues(coords: dict, labels: dict, radius: float) -> set:
    """Interface residues with a non-interface residue within `radius` A."""
    keys = [k for k in coords if k in labels]
    if not keys:
        return set()
    pos = np.array([coords[k] for k in keys])
    lab = np.array([labels[k] for k in keys])
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    near = d <= radius
    out = set()
    for i, k in enumerate(keys):
        if lab[i] == 1 and np.any(lab[near[i]] == 0):
            out.add(k)
    return out


def annotate_pdb(lines: list[str], chain: str, prob: dict, cls: dict) -> list[str]:
    """Write probability*100 into B-factor and the TP/FP/FN/TN code into occupancy."""
    out = []
    for ln in lines:
        if not ln.startswith(("ATOM", "HETATM")) or len(ln) < 66:
            out.append(ln)
            continue
        if ln[COL_CHAIN].strip() != chain:
            continue  # drop other chains: these are single-chain benchmark PDBs
        key = res_key(ln[COL_CHAIN], ln[COL_RESSEQ], ln[COL_ICODE])
        if key not in prob:
            continue  # residue not scored (e.g. a ligand) -- omit rather than fake a value
        b = f"{prob[key] * 100.0:6.2f}"
        o = f"{CLASS_CODE[cls[key]]:6.2f}"
        out.append(ln[:54] + o + b + ln[66:])
    out.append("END")
    return out


def sel(keys) -> str:
    """PyMOL residue selection string, insertion codes included."""
    if not keys:
        return "none"
    parts = [f"{s}{i}" if i else f"{s}" for _, s, i in sorted(keys, key=lambda k: k[1])]
    return "resi " + "+".join(parts)


PML_HEADER = """# {title}
# generated by scripts/render_predictions.py -- do not edit by hand
load {pdb}, prot
hide everything
bg_color white
set ray_opaque_background, 0
set opaque_background, 0
set antialias, 2
set ray_shadows, 0
set specular, 0.15
set surface_quality, 1
set transparency_mode, 1
set ray_trace_mode, 0
remove hydrogens
"""

PML_FOOTER = """
orient prot
zoom prot, {pad}
ray {w}, {h}
png {png}, dpi={dpi}
"""

# AGAT-PPIS / GraphPPIS show each case study twice, rotated 180 degrees, so the
# whole surface is visible. Appended when --two-view is given.
PML_BACKVIEW = """
turn y, 180
ray {w}, {h}
png {png2}, dpi={dpi}
"""


def write_prob_pml(path: Path, pdb: str, png: str, contour, args) -> None:
    body = PML_HEADER.format(title="interface probability (blue=low, red=high)", pdb=pdb)
    body += """
show surface, prot
spectrum b, blue_white_red, prot, minimum=0, maximum=100
"""
    if contour and not args.no_contour:
        body += f"""
# ground-truth interface rim, drawn as a green contour over the heat map
select gt_rim, prot and ({sel(contour)})
create gt_obj, gt_rim
show mesh, gt_obj
color green, gt_obj
set mesh_width, {args.mesh_width}
set mesh_quality, 0
disable gt_rim
"""
    body += PML_FOOTER.format(pad=args.pad, w=args.width, h=args.height,
                              png=png, dpi=args.dpi)
    if args.two_view:
        body += PML_BACKVIEW.format(w=args.width, h=args.height, dpi=args.dpi,
                                    png2=png.replace(".png", "_back.png"))
    path.write_text(body)


def write_class_pml(path: Path, pdb: str, png: str, args) -> None:
    body = PML_HEADER.format(title="TP green / FP red / FN yellow / TN grey", pdb=pdb)
    body += """
show surface, prot
color grey80, prot
color green,  prot and q > 0.5 and q < 1.5
color red,    prot and q > 1.5 and q < 2.5
color yellow, prot and q > 2.5
"""
    body += PML_FOOTER.format(pad=args.pad, w=args.width, h=args.height,
                              png=png, dpi=args.dpi)
    if args.two_view:
        body += PML_BACKVIEW.format(w=args.width, h=args.height, dpi=args.dpi,
                                    png2=png.replace(".png", "_back.png"))
    path.write_text(body)


def write_colorbar(path: Path) -> bool:
    """Matching blue-white-red colour bar for the figure legend."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
    except ImportError:
        return False
    cmap = LinearSegmentedColormap.from_list("bwr_pymol", ["#0000ff", "#ffffff", "#ff0000"])
    fig, ax = plt.subplots(figsize=(1.05, 2.5))
    grad = np.linspace(1, 0, 256).reshape(-1, 1)
    ax.imshow(grad, aspect="auto", cmap=cmap, extent=(0, 1, 0, 1))
    ax.set_xticks([])
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_yticklabels(["0", "0.5", "1.0"], fontsize=8)
    ax.yaxis.tick_right()
    ax.set_title("Interface\nscore", fontsize=8, pad=6)
    for s in ax.spines.values():
        s.set_linewidth(0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight", transparent=True)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", transparent=True)
    plt.close(fig)
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", required=True,
                    help="*_per_residue_predictions.csv from evaluate[_ensemble].py")
    ap.add_argument("--per-protein", default=None,
                    help="*_per_protein_metrics.csv, enables --auto selection")
    ap.add_argument("--out-dir", default="figures/renders")
    ap.add_argument("--protein", nargs="*", default=None,
                    help="chain ids to render, e.g. 4H3KB 1CDBA")
    ap.add_argument("--auto", type=int, default=0, metavar="N",
                    help="auto-select N examples spanning best -> hardest by per-protein PR-AUC")
    ap.add_argument("--pdb-dir", default=None,
                    help="override the directory of source_path in the CSV")
    ap.add_argument("--threshold", type=float, default=0.63,
                    help="decision threshold for the TP/FP/FN/TN view (default 0.63)")
    ap.add_argument("--contour-radius", type=float, default=8.0,
                    help="neighbour radius defining the ground-truth rim (default 8 A)")
    ap.add_argument("--no-contour", action="store_true",
                    help="omit the green ground-truth overlay")
    ap.add_argument("--two-view", action="store_true",
                    help="also render the 180-degree rotated view, as AGAT-PPIS and "
                         "GraphPPIS do for their case studies")
    ap.add_argument("--mesh-width", type=float, default=0.4)
    ap.add_argument("--width", type=int, default=2000)
    ap.add_argument("--height", type=int, default=1500)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--pad", type=float, default=3.0)
    args = ap.parse_args()

    df = pd.read_csv(args.predictions)
    need = {"group_id", "source_path", "res_chain", "res_seq", "label", "probability"}
    if not need.issubset(df.columns):
        sys.exit(f"ERROR: {args.predictions} is missing columns: {sorted(need - set(df.columns))}")
    if "insertion" not in df.columns:
        df["insertion"] = ""

    ids = {g: Path(p).stem for g, p in zip(df.group_id, df.source_path)}

    # ---- choose which chains to render --------------------------------------
    if args.protein:
        want = {p.upper() for p in args.protein}
        groups = [g for g, s in ids.items() if s.upper() in want]
        found = {ids[g].upper() for g in groups}
        for miss in sorted(want - found):
            print(f"  [warn] {miss} not present in the prediction file")
    elif args.auto:
        if not args.per_protein:
            sys.exit("ERROR: --auto needs --per-protein <..._per_protein_metrics.csv>")
        pp = pd.read_csv(args.per_protein).dropna(subset=["pr_auc"]).sort_values("pr_auc",
                                                                                 ascending=False)
        n = min(args.auto, len(pp))
        picks = np.linspace(0, len(pp) - 1, n).round().astype(int)  # best .. hardest
        groups = [pp.iloc[i]["group_id"] for i in picks]
        print("auto-selected by per-protein PR-AUC:")
        for i in picks:
            r = pp.iloc[i]
            print(f"  {ids.get(r['group_id'], r['group_id']):10s} PR-AUC={r['pr_auc']:.3f}")
    else:
        sys.exit("ERROR: pass --protein <ids> or --auto N")

    if not groups:
        sys.exit("ERROR: nothing selected -- check the chain ids against the CSV")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    for g in groups:
        sub = df[df.group_id == g]
        stem = ids[g]
        src = Path(sub.iloc[0]["source_path"])
        if args.pdb_dir:
            src = Path(args.pdb_dir) / src.name
        if not src.exists():
            print(f"  [skip] {stem}: PDB not found at {src}  (use --pdb-dir)")
            continue

        chain = str(sub.iloc[0]["res_chain"]).strip()
        prob, lab, cls = {}, {}, {}
        for _, r in sub.iterrows():
            k = res_key(r["res_chain"], r["res_seq"], r["insertion"])
            p = float(r["probability"])
            y = int(r["label"])
            prob[k] = p
            lab[k] = y
            pred = p >= args.threshold
            cls[k] = "TP" if (y and pred) else "FP" if (not y and pred) else \
                     "FN" if (y and not pred) else "TN"

        lines = load_pdb(src)
        pdb_out = out / f"{stem}_pred.pdb"
        pdb_out.write_text("\n".join(annotate_pdb(lines, chain, prob, cls)) + "\n")

        rim = contour_residues(residue_coords(lines, chain), lab, args.contour_radius)
        write_prob_pml(out / f"{stem}_prob.pml", pdb_out.name, f"{stem}_prob.png", rim, args)
        write_class_pml(out / f"{stem}_class.pml", pdb_out.name, f"{stem}_class.png", args)
        written += [f"{stem}_prob.pml", f"{stem}_class.pml"]

        n_pos = sum(lab.values())
        tp = sum(v == "TP" for v in cls.values())
        fp = sum(v == "FP" for v in cls.values())
        fn = sum(v == "FN" for v in cls.values())
        print(f"  {stem}: {len(prob)} residues, {n_pos} interface, "
              f"TP={tp} FP={fp} FN={fn}, contour={len(rim)}")

    if not written:
        sys.exit("\nNothing was written.")

    # Collect EVERY .pml in the directory, not just this batch, so calling the
    # script twice into the same --out-dir (e.g. a bound/unbound pair from two
    # different prediction files) does not silently drop the earlier renders.
    master = out / "render_all.pml"
    all_pml = sorted(p.name for p in out.glob("*.pml") if p.name != "render_all.pml")
    master.write_text("\n".join(f"@{w}\ndelete all\n" for w in all_pml))

    if write_colorbar(out / "colorbar.png"):
        print(f"\nwrote colorbar.png / colorbar.pdf (matches the blue_white_red scale)")

    print(f"\nwrote {len(written)} PyMOL scripts ({len(all_pml)} total in {out}/)")
    print("\nRender them all:")
    print(f"  cd {out} && pymol -cq render_all.pml")
    print("Or open one interactively to choose the viewpoint:")
    print(f"  cd {out} && pymol {written[0]}")
    print("\nIn interactive PyMOL, after rotating to the view you want:")
    print("  ray 2000, 1500; png my_view.png, dpi=300")


if __name__ == "__main__":
    main()
