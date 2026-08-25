#!/usr/bin/env python
"""Panel-correct garment draping with Newton's SolverSemiImplicit.

SolverSemiImplicit, unlike SolverStyle3D, supports wp.Tape / gradients, so
it's the solver this pipeline commits to even though it's a plain explicit
integrator. That constraint rules out feeding it *_boxmesh.obj directly as
the rest shape: boxmesh lays garment panels out flat at small, distinct Z
offsets to avoid self-overlap, which creates extreme-aspect-ratio "bridge"
triangles (up to ~32:1, edges up to 31cm vs an ~1.8cm mesh average) at panel
seams. Using those 3D positions as the FEM rest shape gives some triangles
enormous, ill-conditioned local stiffness -- confirmed unstable even with
gravity + internal cloth forces alone, body collision disabled entirely.

Input decomposition (mirrors the reference sketch this script implements):
  - Topology / faces:         panels_2d.npz's per-panel faces, mapped through
                               `unwelded_to_welded` -- identical to
                               *_boxmesh.obj / *_sim.obj topology, just
                               reconstructed from the pattern data directly
                               instead of trusted from the box-mesh file.
  - Initial 3D positions q0:  target_shape.obj -- a geometric estimate of
                               this garment (2E2EL4UZUS) draped on this body
                               (00614_apart), numerically well-conditioned
                               and already close to the body.
  - Stretch rest state:       each panel's 2D pattern triangles
                               (panels_2d.npz `<panel>::v` / `<panel>::f`),
                               written directly into ModelBuilder.tri_poses /
                               tri_areas (and used to recompute particle
                               mass from flat-pattern area, not draped area)
                               after the generic add_cloth_mesh() call, which
                               otherwise derives rest state from the curved
                               3D q0 positions.
  - Bending rest state:       flat (rest angle = 0) for edges interior to a
                               single panel, overriding the geometric dihedral
                               angle add_cloth_mesh() would otherwise compute
                               from the curved q0 shape. Edges that cross a
                               panel boundary (seams) have no valid
                               panel-space bending geometry, so this script
                               detects them via panel provenance and gives
                               them a separately low bending stiffness
                               (--seam-edge-ke) instead of the flat override.
  - Material defaults:        sim_props.yaml's `sim.config.material` block
                               (GarmentCode's own warp-tuned values); override
                               via CLI if they don't suit SolverSemiImplicit.
  - Collision body:            01709_straight.obj, static shape (body=-1).
  - Attachment (optional):     vertex_labels.yaml groups that also appear in
                               sim_props.yaml's attachment_label_names get an
                               external spring-to-q0 force
                               (state.particle_f) for attachment_frames *
                               spf seconds, approximating GarmentCode's own
                               attachment-constraint warm-up.

Still forward-only: builder.finalize(requires_grad=False) by default. Set
--requires-grad to leave the model ready for a wp.Tape pass; this script
itself does not compute gradients.

Outputs an animated USD (via newton.viewer.ViewerUSD) of the drape process
plus a final OBJ.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import trimesh
import warp as wp
import yaml

import newton
from newton.utils import MeshAdjacency
from newton.viewer import ViewerUSD

REPO_ROOT = Path(__file__).resolve().parent.parent
GARMENT_DIR = REPO_ROOT / "inputs/generated_rand_2E2EL4UZUS"
DEFAULT_TARGET_SHAPE = REPO_ROOT / "Refitting/outputs_refitting/default_2E2EL4UZUS_to_01709_straight.obj"
DEFAULT_PANELS = GARMENT_DIR / "generated_rand_2E2EL4UZUS_panels_2d.npz"
DEFAULT_SIM_PROPS = GARMENT_DIR / "sim_props.yaml"
DEFAULT_VERTEX_LABELS = GARMENT_DIR / "generated_rand_2E2EL4UZUS_vertex_labels.yaml"
DEFAULT_BODY = REPO_ROOT / "inputs/5000_body_shapes_and_measures/meshes/01709_straight.obj"
DEFAULT_GROUND_TRUTH_SIM = GARMENT_DIR / "generated_rand_2E2EL4UZUS_sim.obj"


def load_obj(path):
    mesh = trimesh.load(path, process=False, maintain_order=True, force="mesh")
    return np.array(mesh.vertices, dtype=np.float64), np.array(mesh.faces, dtype=np.int64)


def load_panels(npz_path):
    """Concatenate per-panel 2D pattern data into global unwelded arrays.

    Returns panel_verts (unwelded, 2D, centimeters as stored), panel_indices
    (unwelded face triples), face_panel_id (which panel each face belongs
    to), and the unwelded->welded vertex map + welded vertex count.
    """
    d = np.load(npz_path, allow_pickle=True)
    panel_order = list(d["__panel_order__"])
    unwelded_to_welded = d["unwelded_to_welded"]
    n_welded = int(d["n_welded"])

    offset = 0
    panel_indices_all, panel_verts_all, face_panel_id = [], [], []
    for panel_id, name in enumerate(panel_order):
        v = d[f"{name}::v"]
        f = d[f"{name}::f"]
        # Front/back panels are mirror images and so have opposite 2D signed
        # area under the raw shoelace formula. Fix this by mirroring the
        # panel's own 2D coordinates (an isometry: preserves edge lengths and
        # area magnitude) rather than reversing face index order -- reversing
        # indices instead would also flip these faces' 3D winding (since
        # welded_faces is later derived from this same panel_indices array),
        # inverting their outward normals relative to the rest of the mesh.
        net_area = panel_triangle_data(v, f)[1].sum()
        if net_area < 0:
            v = v.copy()
            v[:, 0] = -v[:, 0]
        panel_indices_all.append(f + offset)
        panel_verts_all.append(v)
        face_panel_id.append(np.full(len(f), panel_id, dtype=np.int32))
        offset += len(v)

    return (
        np.concatenate(panel_verts_all, axis=0),
        np.concatenate(panel_indices_all, axis=0),
        np.concatenate(face_panel_id, axis=0),
        unwelded_to_welded,
        n_welded,
        panel_order,
    )


def panel_triangle_data(panel_verts, panel_indices):
    """Per-triangle (inv_D, signed_area) in panel (2D pattern) space.

    inv_D is the inverse of the [edge1, edge2] rest-frame matrix, the same
    quantity ModelBuilder.add_triangles() derives from 3D positions -- this
    is what makes it valid to write directly into ModelBuilder.tri_poses.
    """
    p = panel_verts[panel_indices[:, 0]]
    q = panel_verts[panel_indices[:, 1]]
    r = panel_verts[panel_indices[:, 2]]
    qp, rp = q - p, r - p
    D = np.stack([qp, rp], axis=-1)  # (N, 2, 2)
    areas = np.linalg.det(D) / 2.0
    inv_D = np.linalg.inv(D)
    return inv_D, areas


def average_edge_length(vertices, faces):
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    return float(lengths.mean())


def resample_via_nearest_point(guide_vertices, mismatched_vertices, mismatched_faces):
    """Map `guide_vertices` (our panel topology's vertex positions) onto the nearest point
    on a differently-topologized mesh's surface.

    Used when --target-shape has a different vertex/face count than panels_2d.npz's welded
    topology -- e.g. a target shape built from a different remeshing/generation of the same
    garment design, so there's no 1:1 index correspondence to exploit. `guide_vertices`
    should be a known-good, topology-matching mesh (e.g. this garment's own ground-truth
    sim.obj on the same body) so each query point starts already close to its true target,
    keeping the nearest-point correspondence unambiguous.
    """
    mesh = trimesh.Trimesh(vertices=mismatched_vertices, faces=mismatched_faces, process=False)
    closest_points, distances, _ = trimesh.proximity.closest_point(mesh, guide_vertices)
    return closest_points, distances


def push_outside_body(vertices, body_vertices, body_faces, clearance):
    """Guarantee every garment vertex is at least `clearance` outside the body surface.

    target_shape.obj is a geometric estimate, not a physical sim result, so a small
    fraction of its vertices start inside (or too close to) the body -- confirmed at
    up to ~2mm penetration on ~2% of vertices for this garment/body pair. Rather than
    relying on first-substep contact forces to resolve that (which, combined with the
    tiny per-particle mass here, is exactly what caused the earlier instability), push
    violating vertices out along the closest body surface normal before simulating.
    Vertices already satisfying the clearance are left untouched.
    """
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
    """Intersect vertex_labels.yaml groups with sim_props.yaml's
    attachment_label_names; returns {name: (indices, ke, kd)}. Empty dict if
    the vertex_labels file has none of the requested groups (e.g. this
    garment is a skirt with no collar) or attachments are disabled upstream."""
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


def build_model(args, mat, init_vertices, welded_faces, panel_verts, panel_indices, face_panel_id, body_vertices, body_faces, particle_radius):
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
        vertices=init_vertices.tolist(),
        indices=welded_faces.flatten().tolist(),
        density=args.density,
        tri_ke=args.tri_ke,
        tri_ka=args.tri_ka,
        tri_kd=args.tri_kd,
        edge_ke=args.edge_ke,
        edge_kd=args.edge_kd,
        particle_radius=particle_radius,
    )
    tri_end = len(builder.tri_indices)

    inv_D, panel_areas = panel_triangle_data(panel_verts, panel_indices)
    n_degenerate = int((panel_areas <= 0).sum())
    if n_degenerate:
        raise RuntimeError(f"{n_degenerate} degenerate panel triangles after winding normalization -- check panels_2d.npz")
    if tri_end - tri_start != len(inv_D):
        raise RuntimeError(
            f"triangle count mismatch: add_cloth_mesh kept {tri_end - tri_start} of {len(inv_D)} triangles "
            "(some were dropped as 3D-degenerate) -- panel/edge overrides below assume a 1:1, in-order match."
        )

    # Stretch rest state: replace add_cloth_mesh's 3D-derived (curved q0) rest
    # matrices with the flat 2D pattern's, and mass with flat-pattern-area-based.
    builder.tri_poses[tri_start:tri_end] = inv_D.tolist()
    builder.tri_areas[tri_start:tri_end] = panel_areas.tolist()
    for i in range(len(builder.particle_mass)):
        builder.particle_mass[i] = 0.0
    for t, (i, j, k) in enumerate(welded_faces):
        m = args.density * panel_areas[t] / 3.0
        builder.particle_mass[i] += m
        builder.particle_mass[j] += m
        builder.particle_mass[k] += m

    # Bending rest state: flat for panel-interior edges, low stiffness at seams
    # (edges whose two triangles come from different panels have no valid
    # panel-space bending geometry -- a real seam is free to fold anyway).
    adjacency = MeshAdjacency(welded_faces)
    f0, f1 = adjacency.edge_tri_indices[:, 0], adjacency.edge_tri_indices[:, 1]
    interior = f1 >= 0  # boundary edges (f1 == -1) can't be seams
    seam_mask = np.zeros(len(f0), dtype=bool)
    seam_mask[interior] = face_panel_id[f0[interior]] != face_panel_id[f1[interior]]

    edge_rest_angle = np.array(builder.edge_rest_angle, dtype=np.float64)
    bending_props = np.array(builder.edge_bending_properties, dtype=np.float64)  # (E, 2) = (ke, kd)
    if not (len(edge_rest_angle) == len(bending_props) == len(seam_mask)):
        raise RuntimeError("edge count mismatch between builder and recomputed adjacency -- ordering assumption broke")

    n_seam = int(seam_mask.sum())
    print(f"{n_seam} / {len(seam_mask)} edges are cross-panel seams -> edge_ke = {args.seam_edge_ke}")
    edge_rest_angle[interior & ~seam_mask] = 0.0
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
    parser.add_argument("--target-shape", default=str(DEFAULT_TARGET_SHAPE), help="Provides initial 3D particle positions q0.")
    parser.add_argument("--panels", default=str(DEFAULT_PANELS))
    parser.add_argument("--sim-props", default=str(DEFAULT_SIM_PROPS))
    parser.add_argument("--vertex-labels", default=str(DEFAULT_VERTEX_LABELS))
    parser.add_argument("--body", default=str(DEFAULT_BODY))
    parser.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "outputs_semi_implicit"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--substeps", type=int, default=32)
    parser.add_argument("--num-frames", type=int, default=180)
    parser.add_argument("--requires-grad", action="store_true", help="Finalize with requires_grad=True (still no wp.Tape here).")
    parser.add_argument(
        "--skip-push-outside",
        action="store_true",
        help="Skip pre-pushing q0 vertices to >= particle_radius + body_margin outside the body surface. "
        "On by default: target_shape.obj is a geometric estimate and starts a small fraction of vertices "
        "inside/too close to the body, which otherwise gets resolved by first-substep contact forces -- "
        "exactly the kind of large single-step correction that destabilizes this tiny-mass mesh.",
    )

    # sim_props.yaml material defaults are read at runtime and used unless overridden here.
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
    parser.add_argument(
        "--max-velocity",
        type=float,
        default=None,
        help="Per-particle speed clamp [m/s] (Model.particle_max_velocity, default 1e5 i.e. effectively "
        "unbounded). target_shape.obj locally mismatches the panel rest lengths by up to ~7x in places "
        "(it's a geometric estimate, not a physical sim result), which otherwise detonates on the first "
        "substep regardless of tri_ke/dt -- confirmed by disabling body collision entirely and still seeing "
        "unbounded growth. Clamping velocity (matching sim_props.yaml's own global_max_velocity, which this "
        "script's default converts from cm/s) tames it without needing to lower GarmentCode's tuned stiffness.",
    )

    parser.add_argument(
        "--ground-truth-sim",
        default=str(DEFAULT_GROUND_TRUTH_SIM),
        help="This garment's own ground-truth drape (same body, panels_2d.npz-matching topology). Used two ways: "
        "(1) if --target-shape has a different topology than panels_2d.npz (e.g. a refit estimate built from a "
        "different remeshing of the same design), its vertices become the nearest-point-on-surface query anchors "
        "for resampling --target-shape onto our topology; (2) after the run, if its topology matches ours, the "
        "final drape is compared against it as a quality check. Pass '' to disable both.",
    )
    parser.add_argument("--use-attachments", action="store_true", help="Apply vertex_labels.yaml attachment groups as a warm-up spring-to-q0 force.")
    parser.add_argument("--timing-only", type=int, default=0)
    args = parser.parse_args()

    wp.init()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mat, options = load_sim_props(args.sim_props)
    # sim_props.yaml values are Warp-tuned by GarmentCode's own pipeline; use them as
    # defaults for SolverSemiImplicit, overridable via CLI.
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
        args.body_margin = options.get("body_collision_thickness", 0.0) * 0.01  # cm -> m
    if args.particle_radius is None and "fabric_thickness" in mat:
        args.particle_radius = mat["fabric_thickness"] * 0.01  # cm -> m
    if args.max_velocity is None:
        args.max_velocity = options["global_max_velocity"] * 0.01 if "global_max_velocity" in options else 1.0e5  # cm/s -> m/s

    target_vertices_cm, target_faces = load_obj(args.target_shape)
    body_vertices, body_faces = load_obj(args.body)
    panel_verts, panel_indices, face_panel_id, unwelded_to_welded, n_welded, panel_order = load_panels(args.panels)
    panel_verts = panel_verts * 0.01  # panels_2d.npz is in the same centimeters as target_shape.obj

    welded_faces = unwelded_to_welded[panel_indices]
    # Exact match, winding included: load_panels() only mirrors panel *coordinates* for
    # negative-area panels, never reorders face indices, so 3D winding is untouched.
    topology_matches = welded_faces.shape == target_faces.shape and n_welded == len(target_vertices_cm) and np.array_equal(
        welded_faces, target_faces
    )
    ground_truth_vertices_cm = None  # loaded below if needed for resampling; reused for the final comparison

    if topology_matches:
        init_vertices = target_vertices_cm * 0.01  # cm -> m, this is q0
    else:
        # --target-shape wasn't built from this exact topology (e.g. a refit estimate from a
        # different remeshing/generation of the same garment design) -- there's no 1:1 vertex
        # correspondence to exploit. Resample it onto our topology by nearest-point-on-surface,
        # using the ground-truth sim.obj (already topology-matched) as query anchors.
        if not args.ground_truth_sim:
            raise RuntimeError(
                f"--target-shape ({args.target_shape}) has {target_faces.shape[0]} faces / {len(target_vertices_cm)} "
                f"verts, but panels_2d.npz's welded topology has {welded_faces.shape[0]} faces / {n_welded} verts. "
                "Pass --ground-truth-sim (a topology-matching mesh) to resample the mismatched target shape onto "
                "our topology, or fix --target-shape to match panels_2d.npz directly."
            )
        ground_truth_vertices_cm, guide_faces = load_obj(args.ground_truth_sim)
        assert n_welded == len(ground_truth_vertices_cm) and np.array_equal(
            np.sort(welded_faces, axis=1), np.sort(guide_faces, axis=1)
        ), f"--ground-truth-sim ({args.ground_truth_sim}) does not match panels_2d.npz's welded topology either."
        resampled_cm, resample_dist_cm = resample_via_nearest_point(ground_truth_vertices_cm, target_vertices_cm, target_faces)
        print(
            f"--target-shape topology ({target_faces.shape[0]} faces) != panels_2d.npz topology ({welded_faces.shape[0]} "
            f"faces); resampled q0 via nearest-point-on-surface using {args.ground_truth_sim} as query anchors "
            f"(resample distance: mean={resample_dist_cm.mean():.3f}cm, max={resample_dist_cm.max():.3f}cm)."
        )
        init_vertices = resampled_cm * 0.01  # cm -> m, this is q0

    particle_radius = args.particle_radius
    if particle_radius is None:
        particle_radius = float(np.clip(average_edge_length(init_vertices, welded_faces) * 0.5, 0.003, 0.01))

    if not args.skip_push_outside:
        clearance = particle_radius + args.body_margin
        init_vertices, n_pushed = push_outside_body(init_vertices, body_vertices, body_faces, clearance)
        print(f"Pushed {n_pushed}/{len(init_vertices)} q0 vertices out to >= {clearance * 1000:.2f}mm from the body surface.")

    attachment_groups = {}
    attachment_time = 0.0
    if args.use_attachments:
        attachment_groups = load_attachment_groups(args.vertex_labels, options)
        spf = options.get("attachment_frames_spf", None)
        stats_spf = None
        try:
            with open(args.sim_props) as f:
                stats_spf = list(yaml.safe_load(f)["sim"]["stats"]["spf"].values())[0]
        except (KeyError, IndexError, TypeError):
            stats_spf = 1.0 / 60.0
        attachment_time = options.get("attachment_frames", 0) * (spf or stats_spf)
        if attachment_groups:
            print(f"Attachment groups: { {k: len(v[0]) for k, v in attachment_groups.items()} }, active for {attachment_time:.2f}s")
        else:
            print("--use-attachments set, but no matching vertex_labels groups found for this garment.")

    print(f"Device: {args.device}")
    print(f"Panels ({len(panel_order)}): {panel_order}")
    print(f"Garment: {n_welded} welded verts, {len(welded_faces)} tris")
    print(f"Body:    {len(body_vertices)} verts, {len(body_faces)} tris (static collider)")
    print(f"particle_radius = {particle_radius:.5f} m, density={args.density}, friction={args.friction}")
    print(f"tri_ke={args.tri_ke} tri_ka={args.tri_ka} tri_kd={args.tri_kd} edge_ke={args.edge_ke} edge_kd={args.edge_kd}")

    model = build_model(args, mat, init_vertices, welded_faces, panel_verts, panel_indices, face_panel_id, body_vertices, body_faces, particle_radius)
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
    if args.timing_only <= 0:
        usd_path = out_dir / "drape.usd"
        usd_viewer = ViewerUSD(str(usd_path), fps=args.fps, up_axis="Y", num_frames=n_frames)
        usd_viewer.set_model(model)

    anchor_indices = np.concatenate([idx for idx, _, _ in attachment_groups.values()]) if attachment_groups else None
    anchor_positions = init_vertices[anchor_indices] if anchor_indices is not None else None
    anchor_ke = (
        np.concatenate([np.full(len(idx), ke) for idx, ke, _ in attachment_groups.values()]) if attachment_groups else None
    )
    anchor_kd = (
        np.concatenate([np.full(len(idx), kd) for idx, _, kd in attachment_groups.values()]) if attachment_groups else None
    )

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
                f[anchor_indices] += (anchor_ke[:, None] * (anchor_positions - q[anchor_indices])) - (
                    anchor_kd[:, None] * qd[anchor_indices]
                )
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
        print(f"Drape animation written to {out_dir / 'drape.usd'}")

    ms_per_frame = elapsed / n_frames * 1000
    ms_per_substep = elapsed / (n_frames * args.substeps) * 1000
    print(
        f"Ran {n_frames} frames ({n_frames * args.substeps} substeps) in {elapsed:.2f}s "
        f"({ms_per_frame:.1f} ms/frame, {ms_per_substep:.2f} ms/substep)"
    )

    if args.timing_only > 0:
        est_full_s = elapsed / n_frames * args.num_frames
        print(f"Estimated wall time for the full --num-frames {args.num_frames} run: {est_full_s:.1f}s")
    else:
        final_path = out_dir / "final_drape.obj"
        final_vertices = state_0.particle_q.numpy()
        write_obj(final_path, final_vertices, welded_faces)
        print(f"Final drape written to {final_path}")

        if args.ground_truth_sim:
            if ground_truth_vertices_cm is None:
                try:
                    gt_v_cm, gt_faces = load_obj(args.ground_truth_sim)
                    if n_welded == len(gt_v_cm) and np.array_equal(np.sort(welded_faces, axis=1), np.sort(gt_faces, axis=1)):
                        ground_truth_vertices_cm = gt_v_cm
                except (OSError, ValueError):
                    pass
            if ground_truth_vertices_cm is not None:
                dist = np.linalg.norm(final_vertices - ground_truth_vertices_cm * 0.01, axis=1)
                print(
                    f"vs ground truth ({args.ground_truth_sim}): mean={dist.mean() * 100:.2f}cm "
                    f"median={np.median(dist) * 100:.2f}cm max={dist.max() * 100:.2f}cm"
                )
            else:
                print(f"--ground-truth-sim ({args.ground_truth_sim}) topology doesn't match ours; skipped comparison.")


if __name__ == "__main__":
    main()
