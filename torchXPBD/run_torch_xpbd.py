#!/usr/bin/env python
"""Forward-drape pipeline: torch XPBD on a GarmentCode garment.

Reproduces XPBD (Macklin et al. 2016, "XPBD: Position-Based Simulation of
Compliant Constrained Dynamics") -- the compliant-constraint formulation
implemented in ``torch_xpbd.py`` -- using GarmentCode data as input:
``inputs/generated_rand_2E2EL4UZUS`` draped on its own source body
(``01709_straight``). Pure forward pass: no retargeting, no optimization,
no pattern scaling, no gradient.

Inputs (all from ``inputs/generated_rand_<sample>/``):
  * ``*_panels_2d.npz``      -- per-panel 2D pattern meshes (UNWELDED: seam
                                duplicates kept separate) + ``unwelded_to_welded``.
                                This is torch_xpbd's native input -- panels are
                                held together by the stitch constraint, not by
                                shared vertices (see torch_xpbd.py's docstring).
  * ``*_specification.json``  -- per-panel 3D placement (translation/rotation)
                                used to lift the flat panels into 3D.
  * ``*_vertex_labels.yaml``  -- ``lower_interface`` (the waistband's top loop),
                                GarmentCode's own attachment ("warm-up pin") label.
  * ``*_body_measurements.yaml`` -- ``_waist_level`` (cm) + the source body name.
  * ``5000_body_shapes_and_measures/meshes/<body>.obj`` -- collision body (METERS,
                                converted to this pipeline's cm-native units here).
  * ``*_sim.obj``             -- GarmentCode's own drape, used only as a
                                ground-truth reference for metrics.json.

The rest mesh handed to the simulator is the *flat* lift (``--init flat``,
default): each panel's raw 2D pattern vertices placed in 3D by its own
specification.json translation/rotation, seams left open. That keeps
``rest_vertices`` self-consistent for everything ``simulate_drape`` derives
from it -- particle mass from true flat-pattern triangle areas, and a zero
bending rest angle everywhere -- and leaves the sewing itself to the stitch
constraint. ``--init box`` instead starts from ``*_boxmesh.obj``, GarmentCode's
own assembled-but-undraped layout (seams already closed, but seam-adjacent
edges stretched up to ~65x); it is better conditioned as an initial state but
pollutes the derived mass/rest-angle with that stretch, so it is not the default.

No ``target_shape`` is passed -- the fitted silhouette has to come from
torch_xpbd's own physics: attachment (the waistband pin) + gravity + frictional
body collision, not from borrowing GarmentCode's own reference drape.

Outputs (``torchXPBD/outputs/``):
    torch_xpbd_forward.usd  -- animated drape + static body, for usdview/Blender
    final_drape.obj         -- final garment, welded back to *_sim.obj's numbering
    metrics.json            -- per-vertex distance to *_sim.obj

Run with the ``garmentcode`` conda env (the only one here with torch + igl + pxr):
    /oscar/data/ssrinath/users/rzhou52/.conda/envs/garmentcode/bin/python \
        torchXPBD/run_torch_xpbd.py
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import trimesh
import yaml
from scipy.spatial.transform import Rotation

torch.set_num_threads(1)  # default thread count causes MKL contention on this
                          # cluster's many-core nodes -- ~100x slowdown on the
                          # per-substep torch ops (observed: unset -> single
                          # frame didn't finish in 90s; set -> ~0.7s/frame)

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))  # torch_xpbd + utils.data_class

from torch_xpbd import XPBDConfig, simulate_drape  # noqa: E402
from utils.data_class import Mesh  # noqa: E402

SAMPLE = "rand_2E2EL4UZUS"
GARMENT = f"generated_{SAMPLE}"

REPO = Path(__file__).resolve().parent.parent
GARMENT_DIR = REPO / "inputs" / GARMENT
BODIES_DIR = REPO / "inputs" / "5000_body_shapes_and_measures" / "meshes"
OUT_DIR = Path(__file__).resolve().parent / "outputs"

PANELS_NPZ = GARMENT_DIR / f"{GARMENT}_panels_2d.npz"
SPEC_JSON = GARMENT_DIR / f"{SAMPLE}_specification.json"
VERTEX_LABELS = GARMENT_DIR / f"{GARMENT}_vertex_labels.yaml"
BODY_MEASUREMENTS = GARMENT_DIR / f"{SAMPLE}_body_measurements.yaml"
BOXMESH_OBJ = GARMENT_DIR / f"{GARMENT}_boxmesh.obj"
SIM_OBJ = GARMENT_DIR / f"{GARMENT}_sim.obj"

FLAT_SEWING_FRAMES = 80  # default --zero-gravity-steps for --init flat; see that flag's help


# ── input loading ─────────────────────────────────────────────────────────────

def load_obj(path):
    mesh = trimesh.load(path, process=False, maintain_order=True, force="mesh")
    return np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int64)


def write_obj(path, vertices, faces):
    with open(path, "w") as fh:
        for v in vertices:
            fh.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for f in faces:
            fh.write(f"f {f[0] + 1} {f[1] + 1} {f[2] + 1}\n")


def load_panels(npz_path):
    """Concatenate ``*_panels_2d.npz``'s per-panel 2D meshes into global UNWELDED
    arrays, keeping each panel's vertex coordinates and face index order exactly
    as stored (that order already matches ``*_sim.obj``'s winding -- see
    ``SemiImplicit/drape_semi_implicit_unwelded.py``; do not mirror or reverse it).

    Returns ``(verts_2d (N,2), faces (F,3), panel_slices {name: (start, stop)},
    unwelded_to_welded (N,), n_welded)``.
    """
    d = np.load(npz_path, allow_pickle=True)
    panel_order = [str(n) for n in d["__panel_order__"]]

    verts, faces, panel_slices, offset = [], [], {}, 0
    for name in panel_order:
        v, f = d[f"{name}::v"], d[f"{name}::f"]
        verts.append(np.asarray(v, dtype=np.float64))
        faces.append(np.asarray(f, dtype=np.int64) + offset)
        panel_slices[name] = (offset, offset + len(v))
        offset += len(v)

    return (
        np.concatenate(verts, axis=0),
        np.concatenate(faces, axis=0),
        panel_slices,
        np.asarray(d["unwelded_to_welded"], dtype=np.int64),
        int(d["n_welded"]),
    )


def lift_panels_to_3d(verts_2d, panel_slices, spec_path):
    """Place each flat panel in 3D by its specification.json rigid placement:
    ``x3 = R_xyz(rotation_deg) @ [u, v, 0] + translation`` (GarmentCode's own
    ``_point_in_3D`` convention). Verified against ``*_boxmesh.obj``: with this
    lift the two agree exactly away from the seams, and differ only where the
    box mesh's own seam-welding pulled boundary vertices together (up to 22.5 cm
    in z, 3.9 cm in x/y for this garment) -- i.e. exactly the gaps the stitch
    constraint is here to close.
    """
    panels = json.loads(Path(spec_path).read_text())["pattern"]["panels"]
    out = np.zeros((len(verts_2d), 3), dtype=np.float64)
    for name, (start, stop) in panel_slices.items():
        p = panels[name]
        rot = Rotation.from_euler("xyz", np.asarray(p["rotation"], dtype=np.float64), degrees=True)
        flat = np.c_[verts_2d[start:stop], np.zeros(stop - start)]
        out[start:stop] = rot.apply(flat) + np.asarray(p["translation"], dtype=np.float64)
    return out


def stitch_pairs_from_weld_map(unwelded_to_welded):
    """``(K, 2)`` unwelded index pairs that the stitch constraint holds together.

    Every group of unwelded vertices sharing one welded id is a seam
    correspondence from GarmentCode's own ``verts_loc_glob`` bookkeeping (exact,
    not a 3D nearest-neighbour guess). A group of size ``n`` contributes ``n-1``
    pairs in a star around its first member -- enough to make the group coincide,
    without the redundant constraints a fully-connected clique would add.
    """
    order = np.argsort(unwelded_to_welded, kind="stable")
    welded_sorted = unwelded_to_welded[order]
    group_starts = np.flatnonzero(np.r_[True, welded_sorted[1:] != welded_sorted[:-1]])
    pairs = []
    for start, stop in zip(group_starts, np.r_[group_starts[1:], len(order)]):
        if stop - start > 1:
            anchor = order[start]
            pairs.extend((anchor, other) for other in order[start + 1:stop])
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)


def unwelded_indices_for_welded_ids(unwelded_to_welded, welded_ids):
    """Every unwelded vertex mapping onto one of ``welded_ids`` (a
    ``vertex_labels.yaml`` label, which is stated in welded numbering)."""
    return np.flatnonzero(np.isin(unwelded_to_welded, np.asarray(welded_ids, dtype=np.int64)))


def weld(vertices_unwelded, unwelded_to_welded, n_welded):
    """Average the unwelded positions of each welded vertex -- puts a drape back
    into ``*_sim.obj``'s vertex numbering so it can be compared/written directly.
    The stitch constraint keeps each group near-coincident, so this is a
    formality, not a real reconciliation (the residual spread is reported)."""
    summed = np.zeros((n_welded, 3), dtype=np.float64)
    np.add.at(summed, unwelded_to_welded, vertices_unwelded)
    counts = np.bincount(unwelded_to_welded, minlength=n_welded)[:, None]
    return summed / counts


# ── main ──────────────────────────────────────────────────────────────────────

def build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--init", choices=("flat", "box"), default="flat",
                   help="where the drape STARTS (the rest state is the flat specification.json "
                        "lift either way): 'flat' = start from that same flat lift, seams wide "
                        "open, and let the stitch constraint sew the garment shut -- needs a long "
                        "--zero-gravity-steps sewing window, since seam closure is capped at "
                        "XPBDConfig.max_velocity. 'box' = start from *_boxmesh.obj, GarmentCode's "
                        "own assembled-but-undraped layout, with the seams already closed. "
                        "Default: flat.")
    p.add_argument("--frames", type=int, default=XPBDConfig.max_frames, help="max drape frames")
    p.add_argument("--substeps", type=int, default=XPBDConfig.substeps)
    p.add_argument("--solver-iters", type=int, default=XPBDConfig.solver_iters)
    p.add_argument("--zero-gravity-steps", type=int, default=None,
                   help="frames of gravity-free warm-up. Default: XPBDConfig's own "
                        f"{XPBDConfig.zero_gravity_steps} for --init box, but {FLAT_SEWING_FRAMES} for "
                        "--init flat, where this window is the sewing phase the stitch constraint "
                        "needs to pull the wide-open seams shut (measured: a 30 cm mean / 45 cm max "
                        "gap closes to <0.01 cm by about frame 60, seam closure being rate-limited "
                        "by XPBDConfig.max_velocity) -- letting gravity in before that drops an "
                        "unsewn set of flat panels past the body.")
    p.add_argument("--attachment-frames", type=int, default=XPBDConfig.attachment_frames)
    p.add_argument("--no-attachment", action="store_true",
                   help="drop the waistband pin entirely (gravity + collision only)")
    p.add_argument("--no-usd", action="store_true", help="skip the USD animation (faster)")
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--device", default="cpu", help="torch device (the igl collision query is CPU-only either way)")
    return p


def main(argv=None):
    args = build_arg_parser().parse_args(argv)

    print("=" * 70)
    print(f"torch XPBD forward drape: sample={SAMPLE}  init={args.init}")
    print("=" * 70)

    verts_2d, faces, panel_slices, unwelded_to_welded, n_welded = load_panels(PANELS_NPZ)
    print(f"panels: {list(panel_slices)}  ->  {len(verts_2d)} unwelded verts "
          f"({n_welded} welded), {len(faces)} faces")

    # Rest state is ALWAYS the flat specification.json lift -- that is what makes
    # the derived particle masses (true flat-pattern triangle areas) and bending
    # rest angles (zero, a flat panel) correct. --init only chooses where the
    # drape STARTS from (see simulate_drape's `initial_vertices`).
    rest_3d = lift_panels_to_3d(verts_2d, panel_slices, SPEC_JSON)
    initial_3d = None
    if args.init == "box":
        box_verts_cm, _ = load_obj(BOXMESH_OBJ)
        if len(box_verts_cm) != n_welded:
            raise RuntimeError(f"{BOXMESH_OBJ.name} has {len(box_verts_cm)} verts, expected {n_welded}")
        initial_3d = box_verts_cm[unwelded_to_welded]

    device = torch.device(args.device)
    rest_vertices = torch.as_tensor(rest_3d, dtype=torch.float64, device=device)
    initial_vertices = (None if initial_3d is None
                        else torch.as_tensor(initial_3d, dtype=torch.float64, device=device))
    start_3d = rest_3d if initial_3d is None else initial_3d
    verts_2d_t = torch.as_tensor(verts_2d, dtype=torch.float64, device=device)
    faces_t = torch.as_tensor(faces, dtype=torch.int64, device=device)
    material_uv = verts_2d_t[faces_t]  # (F, 3, 2) -- the triangle-strain rest frame

    stitch_pairs = stitch_pairs_from_weld_map(unwelded_to_welded)
    seam_gap = np.linalg.norm(start_3d[stitch_pairs[:, 0]] - start_3d[stitch_pairs[:, 1]], axis=1)
    print(f"stitch: {len(stitch_pairs)} pairs, initial gap max={seam_gap.max():.2f} cm "
          f"mean={seam_gap.mean():.2f} cm")

    # ── collision body: the garment's own source body, OBJ is in METERS while
    # this pipeline (and GarmentCode's cm-native pattern data) is in cm.
    measurements = yaml.safe_load(BODY_MEASUREMENTS.read_text())["body"]
    body_name = measurements["body_sample"]
    body_verts_m, body_faces = load_obj(BODIES_DIR / f"{body_name}.obj")
    target_body_cm = Mesh(vertices=body_verts_m * 100.0, faces=body_faces)
    print(f"body: {body_name}  {len(body_verts_m)} verts, "
          f"height={target_body_cm.vertices[:, 1].max():.1f} cm")

    # ── attachment (GarmentCode's warm-up pin, see torch_xpbd.py's docstring):
    # hold the 'lower_interface' waistband loop up at the body's waist height for
    # the drape's first attachment_frames -- otherwise gravity slides the garment
    # past the body before collision alone catches it. Measurement YAML values
    # (unlike the body OBJ) are already cm, matching torch_xpbd's cm-native
    # geometry -- same fallback formula GarmentCode's own Cloth._add_attachment_labels
    # uses when '_waist_level' isn't precomputed in the measurements file.
    attachment_indices = attachment_target_point = None
    if not args.no_attachment:
        lower_interface = yaml.safe_load(VERTEX_LABELS.read_text()).get("lower_interface", [])
        if lower_interface:
            waist_level = measurements.get(
                "_waist_level", measurements["height"] - measurements["head_l"] - measurements["waist_line"])
            attachment_indices = unwelded_indices_for_welded_ids(unwelded_to_welded, lower_interface)
            attachment_target_point = (0.0, float(waist_level), 0.0)
            print(f"attachment: 'lower_interface' -> {len(attachment_indices)} unwelded verts, "
                  f"waist_level={waist_level:.2f} cm")

    zero_gravity_steps = args.zero_gravity_steps
    if zero_gravity_steps is None:
        zero_gravity_steps = FLAT_SEWING_FRAMES if args.init == "flat" else XPBDConfig.zero_gravity_steps
    print(f"gravity-free warm-up: {zero_gravity_steps} frames")

    config = XPBDConfig(
        substeps=args.substeps,
        solver_iters=args.solver_iters,
        zero_gravity_steps=zero_gravity_steps,
        max_frames=args.frames,
        attachment_frames=args.attachment_frames,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    usd_path = None if args.no_usd else str(args.out_dir / "torch_xpbd_forward.usd")
    if usd_path and Path(usd_path).exists():
        Path(usd_path).unlink()  # Usd.Stage.CreateNew refuses to overwrite

    t0 = time.time()
    with torch.no_grad():  # forward-only script -- no adjoint pass here
        final_x = simulate_drape(
            rest_vertices, faces, target_body_cm,
            material_uv=material_uv, stitch_pairs=stitch_pairs,
            attachment_indices=attachment_indices, attachment_target_point=attachment_target_point,
            initial_vertices=initial_vertices, config=config, usd_path=usd_path,
        )
    elapsed = time.time() - t0
    assert torch.isfinite(final_x).all(), "torch_xpbd drape diverged"
    print(f"torch_xpbd done in {elapsed:.1f}s")

    # ── outputs ───────────────────────────────────────────────────────────────
    final_np = final_x.detach().cpu().numpy()
    welded_final = weld(final_np, unwelded_to_welded, n_welded)
    seam_residual = np.linalg.norm(final_np[stitch_pairs[:, 0]] - final_np[stitch_pairs[:, 1]], axis=1)
    welded_faces = unwelded_to_welded[faces]
    write_obj(args.out_dir / "final_drape.obj", welded_final, welded_faces)

    gt_verts, _ = load_obj(SIM_OBJ)
    dist = np.linalg.norm(welded_final - gt_verts, axis=1)
    metrics = {
        "sample": SAMPLE,
        "body": body_name,
        "init": args.init,
        "frames_max": args.frames,
        "substeps": args.substeps,
        "solver_iters": args.solver_iters,
        "zero_gravity_steps": zero_gravity_steps,
        "seconds": round(elapsed, 1),
        "n_unwelded": int(len(final_np)),
        "n_welded": int(n_welded),
        "n_faces": int(len(faces)),
        "n_stitch_pairs": int(len(stitch_pairs)),
        "seam_gap_initial_cm": {"mean": float(seam_gap.mean()), "max": float(seam_gap.max())},
        "seam_gap_final_cm": {"mean": float(seam_residual.mean()), "max": float(seam_residual.max())},
        "vs_sim_obj_cm": {
            "mean": float(dist.mean()), "median": float(np.median(dist)),
            "p95": float(np.percentile(dist, 95)), "max": float(dist.max()),
        },
        "bbox_cm": {"min": welded_final.min(0).tolist(), "max": welded_final.max(0).tolist()},
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"seam residual after drape: mean={seam_residual.mean():.3f} cm max={seam_residual.max():.3f} cm")
    print(f"vs {SIM_OBJ.name}: mean={dist.mean():.2f} cm median={np.median(dist):.2f} cm max={dist.max():.2f} cm")
    print(f"wrote {args.out_dir}/final_drape.obj, metrics.json"
          + (f", {Path(usd_path).name}" if usd_path else ""))
    return metrics


if __name__ == "__main__":
    main()
