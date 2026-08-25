#!/usr/bin/env python
"""Framework-A garment retargeting drape with Newton's SolverSemiImplicit.

Goal: retarget the *default* 2E2EL4UZUS garment sample (authored on the mean
body, mean_all.obj) onto the target body 01709_straight.obj, and validate the
forward simulation by draping a geometric-refit estimate on the target body.

FRAMEWORK A (chosen): everything on the garment side stays in the *default*
mesh family (11829 verts / 22851 faces), so q0, faces and rest state all share
one topology. This simulates the physically-honest object -- the default
garment, whose flat pattern is ~10-15 cm wider than the natively-01709-drafted
garment -- settling on 01709. The GarmentCode ground truth (rand_2E2EL4UZUS,
a *re-drafted*, 01709-fitted garment) is therefore only a size-different
reference: compare with point-to-surface distance and expect a systematic
looseness offset, NOT a per-vertex mm match.

Hard constraint honored: NO simulated drape of the garment on the *target*
body 01709 is used to seed any point. Only the *source* (mean-body) drape,
via the geometric refit, provides q0. rand_2E2EL4UZUS_sim.ply is read solely
for the final comparison.

Input decomposition (all default family unless noted):
  - Initial 3D positions q0:  default_2E2EL4UZUS_to_01709_straight.obj -- the
                              geometric-refit estimate of the default garment
                              on 01709. Numerically well-conditioned, already
                              near the body, and edge-length-wise ~0.99x the
                              default flat pattern (refit preserved size), so
                              it is consistent with the default-pattern rest.
  - Topology / faces:         default_2E2EL4UZUS_boxmesh.ply's faces. Verified
                              byte-identical to q0's faces; already welded, so
                              no unwelded->welded remap (no panels_2d.npz).
  - Stretch rest state:       each panel is laid perfectly flat at a distinct
                              constant Z in the boxmesh, so a panel-interior
                              triangle's boxmesh (x,y) ARE its flat-pattern
                              coordinates -> write inv_D / area straight into
                              ModelBuilder.tri_poses / tri_areas (cm->m), and
                              recompute particle mass from that flat area.
                              Cross-panel STITCH triangles (verts spanning >1
                              Z-plane) have no clean flat frame in a welded
                              mesh, so they KEEP the q0-derived rest that
                              add_cloth_mesh() computed (policy b): near the
                              body, well-conditioned, low residual stress.
  - Panel membership:         derived purely from the boxmesh Z-planes (no
                              panels_2d.npz, no sim_segmentation.txt -- both
                              belong to a different mesh family and would
                              misalign).
  - Bending rest state:       flat (rest angle 0) for panel-interior edges;
                              cross-panel seam edges get a low --seam-edge-ke.
  - Material defaults:        sim_props.yaml's sim.config.material (this
                              garment's only sim_props; config, body-agnostic).
  - Collision body:           01709_straight.obj, in METERS, static (body=-1).
  - Attachment (optional):    default vertex_labels.yaml groups intersected
                              with sim_props attachment_label_names, as a
                              warm-up spring-to-q0 force.
  - Ground truth (compare):   rand_2E2EL4UZUS_sim.ply (01709-fitted, 9330 v).
                              Point-to-surface only; never an input.

newton / warp are imported lazily inside build_model()/main() so the pure
data-prep helpers can be unit-tested without a Newton install.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import trimesh
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DIR = REPO_ROOT / "inputs/default_2E2EL4UZUS"
GARMENT_DIR = REPO_ROOT / "inputs/generated_rand_2E2EL4UZUS"

DEFAULT_Q0 = REPO_ROOT / "Refitting/outputs_refitting/default_2E2EL4UZUS_to_01709_straight.obj"
DEFAULT_BOXMESH = DEFAULT_DIR / "rand_2E2EL4UZUS_boxmesh.ply"
DEFAULT_SIM_PROPS = GARMENT_DIR / "sim_props.yaml"
DEFAULT_VERTEX_LABELS = DEFAULT_DIR / "rand_2E2EL4UZUS_vertex_labels.yaml"
DEFAULT_BODY = REPO_ROOT / "inputs/5000_body_shapes_and_measures/meshes/01709_straight.obj"
DEFAULT_GT = REPO_ROOT / "inputs/rand_2E2EL4UZUS/rand_2E2EL4UZUS_sim.ply"

CM_TO_M = 0.01


# --------------------------------------------------------------------------- #
# Pure data-prep helpers (no newton / warp dependency)                        #
# --------------------------------------------------------------------------- #
def load_obj(path):
    mesh = trimesh.load(path, process=False, maintain_order=True, force="mesh")
    return np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int64)


def load_boxmesh(path):
    """Boxmesh vertices (cm, panels flat at distinct Z) and welded faces."""
    mesh = trimesh.load(path, process=False, maintain_order=True, force="mesh")
    return np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int64)


def detect_panel_planes(box_z, min_count_frac=0.03):
    """The garment's panels are laid perfectly flat at distinct constant Z in
    the boxmesh; return those dominant Z values. Thin stitch/bridge bands have
    far fewer vertices and are excluded by the count threshold."""
    uz, cnt = np.unique(np.round(box_z, 2), return_counts=True)
    thresh = max(200, int(min_count_frac * len(box_z)))
    planes = np.sort(uz[cnt >= thresh])
    return planes, thresh


def classify_faces(box_vertices, faces, planes, z_tol=1.0):
    """Assign each vertex to the nearest panel Z-plane (if within z_tol), then
    label each face: interior (all 3 verts on the same plane) with that panel
    id, or stitch (-1, spanning >1 plane / touching a bridge band).

    Returns (face_panel_id [-1 for stitch], interior_face_mask, vertex_pid).
    """
    z = box_vertices[:, 2]
    dz = np.abs(z[:, None] - planes[None, :])
    pid = dz.argmin(1).astype(np.int32)
    near = dz.min(1) < z_tol

    fp = pid[faces]
    fnear = near[faces].all(1)
    interior = fnear & (fp[:, 0] == fp[:, 1]) & (fp[:, 1] == fp[:, 2])
    face_panel_id = np.where(interior, fp[:, 0], -1).astype(np.int32)
    return face_panel_id, interior, pid


def flat_pattern_xy(box_xy, faces, vertex_pid, interior_mask):
    """Return per-vertex flat-pattern 2D coordinates for stretch rest.

    Front/back panels are mirror images, so their boxmesh (x,y) come out with
    opposite signed winding -- writing those straight into tri_poses would give
    negative rest areas (bad mass, broken area term). For each panel whose
    interior triangles have net-negative signed area, mirror that panel's x
    coordinate: an isometry (edge lengths and |area| preserved) that flips the
    winding to positive. Interior faces only ever reference vertices of a
    single panel, so a per-panel mirror is self-consistent for them.
    """
    xy = box_xy.copy()
    _, areas = triangle_rest_2d(xy, faces)
    for p in np.unique(vertex_pid):
        fmask = interior_mask & (vertex_pid[faces[:, 0]] == p)
        if fmask.any() and areas[fmask].sum() < 0:
            xy[vertex_pid == p, 0] *= -1.0
    return xy


def triangle_rest_2d(verts2d, faces):
    """Per-triangle (inv_D, signed_area) in a flat 2D frame. inv_D is the
    inverse of the [edge1, edge2] rest matrix -- the same quantity Newton's
    add_triangles() derives, so it can be written straight into tri_poses."""
    p = verts2d[faces[:, 0]]
    q = verts2d[faces[:, 1]]
    r = verts2d[faces[:, 2]]
    D = np.stack([q - p, r - p], axis=-1)  # (N, 2, 2)
    areas = np.linalg.det(D) / 2.0
    inv_D = np.linalg.inv(D)
    return inv_D, areas


def triangle_area_3d(verts3d, faces):
    p = verts3d[faces[:, 0]]
    q = verts3d[faces[:, 1]]
    r = verts3d[faces[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(q - p, r - p), axis=1)


def average_edge_length(vertices, faces):
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    return float(lengths.mean())


def push_outside_body(vertices, body_vertices, body_faces, clearance):
    """Push every garment vertex to >= clearance outside the body surface.
    q0 is a geometric estimate; a small fraction starts inside/too close, and
    resolving that via first-substep contact forces destabilizes this
    tiny-mass mesh, so pre-correct it geometrically instead."""
    body = trimesh.Trimesh(vertices=body_vertices, faces=body_faces, process=False)
    signed_dist = trimesh.proximity.signed_distance(body, vertices)  # positive = inside
    violating = signed_dist > -clearance
    if not violating.any():
        return vertices, 0
    closest_points, _, triangle_ids = trimesh.proximity.closest_point(body, vertices[violating])
    normals = body.face_normals[triangle_ids]
    corrected = vertices.copy()
    corrected[violating] = closest_points + normals * clearance
    return corrected, int(violating.sum())


def write_obj(path, vertices, faces):
    with open(path, "w") as f:
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")


def load_sim_props(path):
    with open(path) as f:
        d = yaml.safe_load(f)
    cfg = d["sim"]["config"]
    return cfg["material"], cfg["options"]


def load_attachment_groups(vertex_labels_path, options):
    if not options.get("enable_attachment_constraint", False):
        return {}
    with open(vertex_labels_path) as f:
        labels = yaml.safe_load(f) or {}
    names = options.get("attachment_label_names", [])
    ke_list = options.get("attachment_stiffness", [])
    kd_list = options.get("attachment_damping", [])
    groups = {}
    for idx, name in enumerate(names):
        if name in labels and labels[name]:
            ke = ke_list[idx] if idx < len(ke_list) else ke_list[0]
            kd = kd_list[idx] if idx < len(kd_list) else kd_list[0]
            groups[name] = (np.array(labels[name], dtype=np.int64), float(ke), float(kd))
    return groups


def compare_to_gt(pred_vertices_m, pred_faces, gt_path):
    """Symmetric point-to-surface distance (cm) between the (11829-topology)
    prediction and the GT drape (rand_2E2EL4UZUS_sim.ply, different topology).
    Reported directions:
      pred->gt : each predicted vertex to the GT surface
      gt->pred : each GT vertex to the predicted surface
    Also reports the centroid offset as a coordinate-frame sanity check.
    """
    gt = trimesh.load(gt_path, process=False, force="mesh")
    gt_v = np.asarray(gt.vertices, dtype=np.float64)  # cm
    pred_cm = pred_vertices_m * 100.0
    pred = trimesh.Trimesh(vertices=pred_cm, faces=pred_faces, process=False)

    _, d_pg, _ = trimesh.proximity.closest_point(gt, pred_cm)      # pred -> gt surface
    _, d_gp, _ = trimesh.proximity.closest_point(pred, gt_v)       # gt   -> pred surface

    def stats(d):
        d = np.asarray(d, dtype=np.float64)
        return {
            "mean_cm": float(d.mean()),
            "median_cm": float(np.median(d)),
            "p90_cm": float(np.percentile(d, 90)),
            "max_cm": float(d.max()),
            "rmse_cm": float(np.sqrt((d ** 2).mean())),
        }

    return {
        "pred_to_gt": stats(d_pg),
        "gt_to_pred": stats(d_gp),
        "symmetric_mean_cm": float(0.5 * (d_pg.mean() + d_gp.mean())),
        "centroid_offset_cm": float(np.linalg.norm(pred_cm.mean(0) - gt_v.mean(0))),
        "note": "Framework A: default garment is ~10-15cm wider than the 01709-fitted GT; "
        "a systematic looseness offset is expected and is not solver error.",
    }


# --------------------------------------------------------------------------- #
# Model construction (needs newton / warp)                                    #
# --------------------------------------------------------------------------- #
def build_model(args, init_vertices_m, faces, flat_xy_m, interior_mask, face_panel_id, body_vertices, body_faces, particle_radius):
    import warp as wp
    import newton
    from newton.utils import MeshAdjacency

    builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=wp.vec3(0.0, -9.81, 0.0))

    body_cfg = builder.default_shape_cfg.copy()
    body_cfg.mu = args.friction
    body_cfg.margin = args.body_margin
    body_mesh = newton.Mesh(body_vertices, body_faces.flatten(), compute_inertia=False)
    builder.add_shape_mesh(body=-1, mesh=body_mesh, cfg=body_cfg)

    tri_start = len(builder.tri_indices)
    builder.add_cloth_mesh(
        pos=wp.vec3(0.0, 0.0, 0.0),
        rot=wp.quat_identity(),
        scale=1.0,
        vel=wp.vec3(0.0, 0.0, 0.0),
        vertices=init_vertices_m.tolist(),
        indices=faces.flatten().tolist(),
        density=args.density,
        tri_ke=args.tri_ke,
        tri_ka=args.tri_ka,
        tri_kd=args.tri_kd,
        edge_ke=args.edge_ke,
        edge_kd=args.edge_kd,
        particle_radius=particle_radius,
    )
    tri_end = len(builder.tri_indices)
    if tri_end - tri_start != len(faces):
        raise RuntimeError(
            f"add_cloth_mesh kept {tri_end - tri_start} of {len(faces)} triangles (some dropped "
            "as 3D-degenerate) -- the interior/stitch masks assume a 1:1, in-order match."
        )

    # Stretch rest: flat-pattern (boxmesh XY, cm->m) for interior faces; leave
    # add_cloth_mesh's q0-derived rest untouched for stitch faces (policy b).
    inv_D_flat, area_flat = triangle_rest_2d(flat_xy_m, faces)
    n_bad = int((area_flat[interior_mask] <= 0).sum())
    if n_bad:
        raise RuntimeError(f"{n_bad} interior panel triangles have non-positive flat area after winding fix -- check Z-plane classification")

    tri_poses = np.asarray(builder.tri_poses, dtype=np.float64)   # q0-derived, (T,2,2)
    tri_areas = np.asarray(builder.tri_areas, dtype=np.float64)   # q0-derived, (T,)
    tri_poses[tri_start:tri_end][interior_mask] = inv_D_flat[interior_mask]
    tri_areas[tri_start:tri_end][interior_mask] = area_flat[interior_mask]
    builder.tri_poses = tri_poses.tolist()
    builder.tri_areas = tri_areas.tolist()

    # Mass from rest area: flat area for interior faces, q0-3D area for stitch faces.
    area_q0 = triangle_area_3d(init_vertices_m, faces)
    rest_area = np.where(interior_mask, area_flat, area_q0)
    n_stitch = int((~interior_mask).sum())
    for i in range(len(builder.particle_mass)):
        builder.particle_mass[i] = 0.0
    for t, (i, j, k) in enumerate(faces):
        m = args.density * rest_area[t] / 3.0
        builder.particle_mass[i] += m
        builder.particle_mass[j] += m
        builder.particle_mass[k] += m

    # Bending rest: flat for panel-interior edges; low ke at cross-panel seams.
    adjacency = MeshAdjacency(faces)
    f0, f1 = adjacency.edge_tri_indices[:, 0], adjacency.edge_tri_indices[:, 1]
    interior_edge = f1 >= 0
    seam_mask = np.zeros(len(f0), dtype=bool)
    seam_mask[interior_edge] = face_panel_id[f0[interior_edge]] != face_panel_id[f1[interior_edge]]

    edge_rest_angle = np.asarray(builder.edge_rest_angle, dtype=np.float64)
    bending_props = np.asarray(builder.edge_bending_properties, dtype=np.float64)  # (E, 2) = (ke, kd)
    if not (len(edge_rest_angle) == len(bending_props) == len(seam_mask)):
        raise RuntimeError("edge count mismatch between builder and recomputed adjacency -- ordering assumption broke")

    n_seam = int(seam_mask.sum())
    print(f"{n_stitch} stitch faces (kept q0 rest) | {n_seam}/{len(seam_mask)} seam edges -> edge_ke={args.seam_edge_ke}")
    edge_rest_angle[interior_edge & ~seam_mask] = 0.0
    bending_props[seam_mask, 0] = args.seam_edge_ke
    builder.edge_rest_angle = edge_rest_angle.tolist()
    builder.edge_bending_properties = [tuple(row) for row in bending_props.tolist()]

    model = builder.finalize(device=args.device, requires_grad=args.requires_grad)
    model.soft_contact_ke = args.contact_ke
    model.soft_contact_kd = args.contact_kd
    model.soft_contact_mu = args.friction
    model.particle_max_velocity = args.max_velocity
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--q0", default=str(DEFAULT_Q0), help="Initial 3D positions (default garment refit onto 01709).")
    parser.add_argument("--boxmesh", default=str(DEFAULT_BOXMESH), help="Default-family boxmesh -> faces + flat-pattern rest.")
    parser.add_argument("--sim-props", default=str(DEFAULT_SIM_PROPS))
    parser.add_argument("--vertex-labels", default=str(DEFAULT_VERTEX_LABELS))
    parser.add_argument("--body", default=str(DEFAULT_BODY), help="Target collision body, in meters.")
    parser.add_argument("--gt", default=str(DEFAULT_GT), help="Ground-truth drape for comparison only (never an input).")
    parser.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "outputs_semi_implicit_default"))
    parser.add_argument("--device", default="cuda:0", help="Warp device, e.g. cuda:0 or cpu.")
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--substeps", type=int, default=32)
    parser.add_argument("--num-frames", type=int, default=180)
    parser.add_argument("--z-tol", type=float, default=1.0, help="Tolerance (cm) for assigning a boxmesh vertex to a panel Z-plane.")
    parser.add_argument("--requires-grad", action="store_true")
    parser.add_argument("--skip-push-outside", action="store_true")
    parser.add_argument("--no-usd", action="store_true", help="Skip the USD animation (still writes final OBJ).")
    parser.add_argument("--no-compare", action="store_true", help="Skip the GT comparison.")

    # sim_props.yaml material defaults, overridable.
    parser.add_argument("--friction", type=float, default=None)
    parser.add_argument("--density", type=float, default=None, help="Areal density [kg/m^2].")
    parser.add_argument("--tri-ke", type=float, default=None)
    parser.add_argument("--tri-ka", type=float, default=None)
    parser.add_argument("--tri-kd", type=float, default=None)
    parser.add_argument("--edge-ke", type=float, default=None, help="Panel-interior bending stiffness.")
    parser.add_argument("--edge-kd", type=float, default=None)
    parser.add_argument("--seam-edge-ke", type=float, default=0.0, help="Bending stiffness at cross-panel seam edges.")
    parser.add_argument("--body-margin", type=float, default=None)
    parser.add_argument("--contact-ke", type=float, default=1.0e4)
    parser.add_argument("--contact-kd", type=float, default=1.0e2)
    parser.add_argument("--particle-radius", type=float, default=None)
    parser.add_argument("--max-velocity", type=float, default=None, help="Per-particle speed clamp [m/s].")
    parser.add_argument("--use-attachments", action="store_true")
    parser.add_argument("--timing-only", type=int, default=0)
    args = parser.parse_args()

    import warp as wp
    import newton
    from newton.viewer import ViewerUSD

    wp.init()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mat, options = load_sim_props(args.sim_props)
    if args.friction is None:
        args.friction = mat["fabric_friction"]
    if args.density is None:
        args.density = mat["fabric_density"]
    if args.tri_ke is None:
        args.tri_ke = mat["garment_tri_ke"]
    if args.tri_ka is None:
        args.tri_ka = mat["garment_tri_ka"]
    if args.tri_kd is None:
        args.tri_kd = mat["garment_tri_kd"]
    if args.edge_ke is None:
        args.edge_ke = mat["garment_edge_ke"]
    if args.edge_kd is None:
        args.edge_kd = mat["garment_edge_kd"]
    if args.body_margin is None:
        args.body_margin = options.get("body_collision_thickness", 0.0) * CM_TO_M
    if args.particle_radius is None and "fabric_thickness" in mat:
        args.particle_radius = mat["fabric_thickness"] * CM_TO_M
    if args.max_velocity is None:
        args.max_velocity = options["global_max_velocity"] * CM_TO_M if "global_max_velocity" in options else 1.0e5

    # --- Load garment (default family) ---
    q0_cm, q0_faces = load_obj(args.q0)
    box_cm, faces = load_boxmesh(args.boxmesh)
    body_vertices, body_faces = load_obj(args.body)

    assert box_cm.shape[0] == q0_cm.shape[0], (
        f"vertex-count mismatch: boxmesh {box_cm.shape[0]} vs q0 {q0_cm.shape[0]} -- these must be the SAME mesh family."
    )
    assert np.array_equal(faces, q0_faces), "boxmesh faces != q0 faces -- q0 and boxmesh are not the same topology."

    # --- Panel membership from boxmesh Z-planes ---
    planes, thresh = detect_panel_planes(box_cm[:, 2])
    face_panel_id, interior_mask, vertex_pid = classify_faces(box_cm, faces, planes, z_tol=args.z_tol)
    flat_xy_m = flat_pattern_xy(box_cm[:, :2], faces, vertex_pid, interior_mask) * CM_TO_M

    init_vertices = q0_cm * CM_TO_M

    particle_radius = args.particle_radius
    if particle_radius is None:
        particle_radius = float(np.clip(average_edge_length(init_vertices, faces) * 0.5, 0.003, 0.01))

    if not args.skip_push_outside:
        clearance = particle_radius + args.body_margin
        init_vertices, n_pushed = push_outside_body(init_vertices, body_vertices, body_faces, clearance)
        print(f"Pushed {n_pushed}/{len(init_vertices)} q0 vertices out to >= {clearance * 1000:.2f}mm from the body.")

    # --- Attachments ---
    attachment_groups = {}
    attachment_time = 0.0
    if args.use_attachments:
        attachment_groups = load_attachment_groups(args.vertex_labels, options)
        spf = options.get("attachment_frames_spf", None) or (1.0 / 60.0)
        attachment_time = options.get("attachment_frames", 0) * spf
        if attachment_groups:
            print(f"Attachment groups: { {k: len(v[0]) for k, v in attachment_groups.items()} }, active {attachment_time:.2f}s")
        else:
            print("--use-attachments set, but no matching vertex_labels groups for this garment.")

    print(f"Device: {args.device}")
    print(f"Panel Z-planes (cm, count>={thresh}): {planes.tolist()}")
    print(f"Garment: {len(q0_cm)} verts, {len(faces)} faces "
          f"({int(interior_mask.sum())} interior + {int((~interior_mask).sum())} stitch)")
    print(f"Body:    {len(body_vertices)} verts, {len(body_faces)} tris (static collider)")
    print(f"radius={particle_radius:.5f}m density={args.density} friction={args.friction} vmax={args.max_velocity}")
    print(f"tri_ke={args.tri_ke} tri_ka={args.tri_ka} tri_kd={args.tri_kd} edge_ke={args.edge_ke} edge_kd={args.edge_kd}")

    model = build_model(args, init_vertices, faces, flat_xy_m, interior_mask, face_panel_id, body_vertices, body_faces, particle_radius)
    solver = newton.solvers.SolverSemiImplicit(model=model)

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    collision_pipeline = newton.CollisionPipeline(model)
    contacts = collision_pipeline.contacts()

    frame_dt = 1.0 / args.fps
    sim_dt = frame_dt / args.substeps
    n_frames = args.timing_only if args.timing_only > 0 else args.num_frames

    usd_viewer = None
    if args.timing_only <= 0 and not args.no_usd:
        usd_viewer = ViewerUSD(str(out_dir / "drape.usd"), fps=args.fps, up_axis="Y", num_frames=n_frames)
        usd_viewer.set_model(model)

    anchor_indices = np.concatenate([idx for idx, _, _ in attachment_groups.values()]) if attachment_groups else None
    anchor_positions = init_vertices[anchor_indices] if anchor_indices is not None else None
    anchor_ke = np.concatenate([np.full(len(idx), ke) for idx, ke, _ in attachment_groups.values()]) if attachment_groups else None
    anchor_kd = np.concatenate([np.full(len(idx), kd) for idx, _, kd in attachment_groups.values()]) if attachment_groups else None

    sim_time = 0.0
    wp.synchronize()
    t_start = time.time()
    for frame in range(1, n_frames + 1):
        for _ in range(args.substeps):
            state_0.clear_forces()
            if anchor_indices is not None and sim_time < attachment_time:
                q = state_0.particle_q.numpy()
                qd = state_0.particle_qd.numpy()
                f = state_0.particle_f.numpy()
                f[anchor_indices] += (anchor_ke[:, None] * (anchor_positions - q[anchor_indices])) - (anchor_kd[:, None] * qd[anchor_indices])
                state_0.particle_f.assign(f)
            collision_pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, sim_dt)
            state_0, state_1 = state_1, state_0
            sim_time += sim_dt
        if usd_viewer is not None:
            usd_viewer.begin_frame(frame * frame_dt)
            usd_viewer.log_state(state_0)
            usd_viewer.end_frame()

    wp.synchronize()
    elapsed = time.time() - t_start
    if usd_viewer is not None:
        usd_viewer.close()
        print(f"Drape animation -> {out_dir / 'drape.usd'}")

    print(f"Ran {n_frames} frames ({n_frames * args.substeps} substeps) in {elapsed:.2f}s "
          f"({elapsed / n_frames * 1000:.1f} ms/frame)")

    if args.timing_only > 0:
        print(f"Estimated full --num-frames {args.num_frames}: {elapsed / n_frames * args.num_frames:.1f}s")
        return

    final_q = state_0.particle_q.numpy()
    final_path = out_dir / "final_drape.obj"
    write_obj(final_path, final_q, faces)
    print(f"Final drape -> {final_path}")

    if not args.no_compare:
        metrics = compare_to_gt(final_q, faces, args.gt)
        with open(out_dir / "metrics.json", "w") as fp:
            json.dump(metrics, fp, indent=2)
        pg, gp = metrics["pred_to_gt"], metrics["gt_to_pred"]
        print(f"GT point-to-surface (cm) | pred->gt mean={pg['mean_cm']:.2f} med={pg['median_cm']:.2f} p90={pg['p90_cm']:.2f} "
              f"| gt->pred mean={gp['mean_cm']:.2f} | centroid_off={metrics['centroid_offset_cm']:.2f}")
        print(f"Metrics -> {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
