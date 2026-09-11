#!/usr/bin/env python
"""Box-mesh structural-spring garment sewing with Newton's SolverSemiImplicit.

This replaces this experiment's earlier approach (specification.json-lifted
flat panels + seam-pair-only springs) with GarmentCode's own sewing
formulation, adapted to SolverSemiImplicit's explicit integration:

  - Topology + initial placement: *_boxmesh.obj directly -- GarmentCode's own
    assembled-but-undraped layout, already WELDED at seams (unlike this
    garment's raw panels_2d.npz data, which keeps seam-duplicate vertices
    separate). Seam vertices in a welded mesh are the *same* particle, so
    there is no seam gap left to close with a spring in the first place --
    but reconciling two independently-authored panel boundaries into one
    shared 3D seam stretches the mesh edges nearest that seam well past
    their true flat-pattern length: verified up to ~65x for this garment's
    most extreme edge (22.5cm box length vs. a 0.35cm true length; 571 of
    the mesh's 27288 edges need a real correction at all, of which 320 need
    more than 10x). *_orig_lens.pickle's own values top out around 1.4cm --
    that's the true target *length*, not the box/target *ratio*, which is
    what actually determines how violent naively snapping to it would be.
  - Structural springs: one per welded-mesh edge (not just seam pairs), the
    same "structural spring network" GarmentCode's own solver uses for
    in-plane stretch. Each spring's rest length starts at the box mesh's own
    (possibly seam-stretched) current edge length and anneals toward
    *_orig_lens.pickle's recorded true flat-pattern length -- present only
    for the minority of edges the box-mesh layout actually stretched; every
    other edge's "target" is just its own already-correct current length
    (see build_structural_springs), so most springs are a no-op rest-length
    hold, not a real anneal.
  - No separate FEM stretch stiffness by default (--tri-ke/--tri-ka/--tri-kd
    default to 0): GarmentCode's own solver leans on the structural-spring
    network for stretch, not a redundant triangle FEM term, and combining
    both here risks double-counting stiffness on an already-distorted
    box-mesh rest state. Triangles are still built (for topology, mesh
    collision, and rendering) and can be given nonzero FEM stiffness via
    CLI once the spring-only version is confirmed stable.
  - Bending: rest angle 0 everywhere (including cross-panel seam edges --
    matching GarmentCode's own builder, which does not special-case seam
    rest angle either), reusing panels_2d.npz's face_panel_id to find those
    seam edges exactly as drape_semi_implicit.py does (*_sim_segmentation.txt's
    per-vertex stitch labels would identify the same edges, so this avoids
    parsing a second, redundant seam source). Stiffness (edge_ke/edge_kd)
    defaults to small values (0.01 / 0.0001), *not* sim_props.yaml's
    garment_edge_ke/kd (1.0 / 10.0): those are GarmentCode's own XPBD
    constraint-projection parameters, not force parameters, and are not
    safe to reuse directly as a SolverSemiImplicit force's ke/kd -- edge_kd
    in particular, applied with no stability cap, is large enough relative
    to this mesh's ~1e-5 kg particles to blow up on its own. Bending is also
    held at 0 through the sewing/settle phases below and only ramped in
    once the box mesh has finished shrinking (see build_model / main()).
  - Structural-spring and body-contact stiffness/damping are still capped
    for explicit-integration stability (see derive_stable_ke_kd), but the
    structural-spring cap is now *network-aware*: a single edge's stability
    bound (mu = min_mass/2) badly overestimates how stiff it's safe to make
    every edge, because a real particle in this mesh is shared by up to 9
    springs pulling at once, not 1. The cap instead uses
    mu = min_mass / (2 * max_spring_degree).

The sewing sequence is not one smoothstep anneal (rest length and stiffness
ramping together) but five substep-scheduled stages -- switching sewing
formulations surfaced two problems a single combined ramp doesn't handle:
  1. Box-mesh sewing: gravity and body contact both off; bending held at 0;
     structural-spring stiffness held at a low constant `ke_sewing`
     (min(0.5, 0.1 * the stability-capped spring_ke_end)) throughout. Each
     edge's rest length anneals from its box length to its target length on
     its *own* smoothstep schedule, timed by --shrink-speed [m/s] rather
     than a single global fraction of the run: an edge needing the full
     ~22cm of contraction takes far longer to safely finish than one needing
     none, and a shared global fraction would either rush the worst edges or
     needlessly stretch out the trivial ones. This stage ends once every
     edge's own schedule completes (not a fixed duration).
  2. Settle: rest lengths hold at target, stiffness holds at `ke_sewing`,
     gravity/contact/bending still off (--settle-frames) -- lets the
     particles' actual positions, which lag the rest-length schedule, catch
     up before stiffness increases; ramping rest length and stiffness up
     together risks the worst spike right at the end of stage 1, when rest
     length is already short but position hasn't caught up and stiffness is
     already near its final value.
  3. Stiffen: structural-spring stiffness ramps `ke_sewing` -> its final
     (network-aware, stability-capped) value, and bending ramps 0 -> its
     final value, together (--stiffness-ramp-frames). Gravity/contact still off.
  4. Contact discovery: gravity still off; body-contact stiffness ramps
     0 -> its stability-capped value (--contact-ramp-frames). Springs and
     bending held at their final values throughout.
  5. Forward drape (the remainder): gravity restored, everything else held
     at final values -- an ordinary forward SolverSemiImplicit simulation.
Gravity turns on only once stage 4 finishes, not on a fixed frame count
(sim_props.yaml's zero_gravity_steps == 10 is tuned for GarmentCode's own
solver/timestep, not this one) -- it needs to wait for the *actual* box-mesh
shrink to finish, which --shrink-speed, not a frame count, determines.
Body contact stays off through stages 1-3 for the same reason it did in this
script's previous version: turning it on before the box mesh has finished
shrinking would let it resist the shrinking motion exactly where that motion
happens near the body's surface, and/or discover deep penetration built up
during a contact-blind shrink and have to push it out from a shock-inducing
starting depth. state_0.particle_q is written to `phase1_sewn.obj` /
`phase2_stiffened.obj` / `phase3_contact.obj` at the boundaries between
stages 1/2, 3, and 4 respectively, so a still-unstable run can be localized
to a specific stage instead of only ever seeing the final blowup.

SolverSemiImplicit usage (state/control/contacts allocation, the
clear_forces() -> collide() -> solver.step() -> swap substep loop, and
finalize(requires_grad=...) for a wp.Tape-ready model) follows the
conventions in Newton's own diffsim cloth example:
https://github.com/newton-physics/newton/blob/main/newton/examples/diffsim/example_diffsim_cloth.py
This script itself remains forward-only (no wp.Tape pass).

Since *_boxmesh.obj is already welded, the simulated mesh's particle count
equals *_sim.obj's vertex count throughout -- no post-hoc weld/average step
is needed before writing final_drape.obj or comparing to ground truth (unlike
this script's previous, unwelded-topology version).

Outputs an animated USD (via newton.viewer.ViewerUSD) of the drape process,
a final OBJ, and a metrics.json comparing the result to *_sim.obj.
"""

import argparse
import json
import time
from pathlib import Path
import pickle

import numpy as np
import trimesh
import warp as wp
import yaml

import newton
from newton.utils import MeshAdjacency
from newton.viewer import ViewerUSD

REPO_ROOT = Path(__file__).resolve().parent.parent
GARMENT_NAME = "generated_rand_2E2EL4UZUS"
SPEC_NAME = "rand_2E2EL4UZUS"
GARMENT_DIR = REPO_ROOT / "inputs" / GARMENT_NAME
DEFAULT_PANELS = GARMENT_DIR / f"{GARMENT_NAME}_panels_2d.npz"
DEFAULT_BOXMESH = GARMENT_DIR / f"{GARMENT_NAME}_boxmesh.obj"
DEFAULT_ORIG_LENS = GARMENT_DIR / f"{GARMENT_NAME}_orig_lens.pickle"
DEFAULT_SIM_PROPS = GARMENT_DIR / "sim_props.yaml"
DEFAULT_BODY_MEASUREMENTS = GARMENT_DIR / f"{SPEC_NAME}_body_measurements.yaml"
DEFAULT_SIM_OBJ = GARMENT_DIR / f"{GARMENT_NAME}_sim.obj"
DEFAULT_BODY_DIR = REPO_ROOT / "inputs/5000_body_shapes_and_measures/meshes"


def load_obj(path):
    mesh = trimesh.load(path, process=False, maintain_order=True, force="mesh")
    return np.array(mesh.vertices, dtype=np.float64), np.array(mesh.faces, dtype=np.int64)


def write_obj(path, vertices, faces):
    with open(path, "w") as f:
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")


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


def load_panels(npz_path):
    """Concatenate per-panel 2D pattern data into global unwelded arrays.

    Vertex coordinates and face index order are both kept exactly as
    panels_2d.npz stores them. Face index order already matches *_sim.obj's
    winding exactly (verified against ground truth), even though some
    panels' shoelace area comes out negative in their own local 2D frame --
    see build_model's handling of that.

    Returns panel_verts (unwelded, 2D, centimeters as stored), panel_indices
    (unwelded face triples, raw winding), face_panel_id (which panel each
    face belongs to -- used to find seam-crossing bending edges once welded),
    the unwelded->welded vertex map, and welded vertex count.
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


def normalize_face_cycle(row):
    """Rotate a face's 3 vertex ids so the smallest is first -- a winding-
    preserving canonical form, used to compare face sets independent of
    array row order (but not independent of winding direction)."""
    i = int(np.argmin(row))
    return tuple(np.roll(row, -i).tolist())


def load_orig_lens(path):
    """*_orig_lens.pickle: {(welded_i, welded_j): length_cm} for the subset
    of welded-mesh edges *_boxmesh.obj's flat-layout Z offsets stretched away
    from their true flat-pattern length (see module docstring). Converts to
    meters and normalizes keys to (min, max) welded-index order."""
    with open(path, "rb") as f:
        raw = pickle.load(f)
    return {(int(min(i, j)), int(max(i, j))): float(v) * 0.01 for (i, j), v in raw.items()}


def build_structural_springs(welded_faces, q_box_welded_m, orig_lens_m):
    """One structural spring per welded-mesh edge -- GarmentCode's own
    stretch network, not a seam-only correspondence. `box_lengths` is each
    edge's actual current length in the (already welded, but seam-adjacent
    edges stretched) box mesh; `target_lengths` is *_orig_lens.pickle's
    recorded true length where available, and otherwise just box_lengths
    again (most edges are ordinary, undistorted panel-interior edges --
    their "target" anneal is a no-op).

    Pairs are sorted to (min, max) and de-duplicated before the orig_lens
    lookup, matching how load_orig_lens normalizes its own keys -- this
    experiment's box mesh has edges needing up to ~65x contraction (not the
    ~1.4x an earlier version of this script's docstring incorrectly claimed;
    that number was orig_lens.pickle's own max *length*, not the box/target
    *ratio*), so getting every one of these lookups right matters far more
    here than it would for a mild correction.
    """
    adjacency = MeshAdjacency(welded_faces)
    pairs = np.unique(np.sort(adjacency.edge_indices[:, 2:4].astype(np.int64), axis=1), axis=0)
    box_lengths = np.linalg.norm(q_box_welded_m[pairs[:, 0]] - q_box_welded_m[pairs[:, 1]], axis=1)
    target_lengths = np.array(
        [orig_lens_m.get((int(i), int(j)), box_lengths[e]) for e, (i, j) in enumerate(pairs)]
    )

    pair_set = {(int(i), int(j)) for i, j in pairs}
    covered = sum(1 for k in orig_lens_m if k in pair_set)
    if covered != len(orig_lens_m):
        raise RuntimeError(
            f"orig_lens coverage {covered}/{len(orig_lens_m)} -- some *_orig_lens.pickle keys were not matched "
            "to a structural-spring edge; key ordering or topology mapping is wrong."
        )
    return pairs, box_lengths, target_lengths


def smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def compute_rest_state_and_mass(panel_verts, panel_indices, mass_faces, n_particles, density, mass_floor_frac=0.2):
    """FEM rest state (inv_D, |area|) in flat-pattern space (row-ordered to
    match panel_indices, hence also `mass_faces` -- see caller), and
    per-particle mass (density * area/3 per corner, accumulated onto
    `mass_faces`' -- welded -- vertex ids, floored to `mass_floor_frac` of
    the *mean* particle mass).

    A handful of sliver triangles from panels_2d.npz's triangulation would
    otherwise leave a few particles with mass orders of magnitude below the
    rest of the mesh. That's harmless for the FEM/mass itself, but those
    outlier-light particles set the worst-case explicit-integration
    stability bound for the structural springs (see derive_stable_ke_kd) --
    flooring them is a standard, physically defensible step (real fabric
    doesn't have near-zero-mass regions) that avoids a few slivers dictating
    how stiff every spring in the whole garment is allowed to be.
    """
    inv_D, areas_signed = panel_triangle_data(panel_verts, panel_indices)
    areas_abs = np.abs(areas_signed)
    mass = np.zeros(n_particles, dtype=np.float64)
    for t, (i, j, k) in enumerate(mass_faces):
        m = density * areas_abs[t] / 3.0
        mass[i] += m
        mass[j] += m
        mass[k] += m
    floor = mass_floor_frac * mass.mean()
    n_floored = int((mass < floor).sum())
    mass = np.maximum(mass, floor)
    return inv_D, areas_abs, mass, n_floored, floor


def derive_stable_ke_kd(mu, dt, requested_ke, requested_kd, safety=0.5):
    """Largest point-mass-vs-point-mass stiffness/damping numerically stable
    under SolverSemiImplicit's *explicit* substep integration, at substep
    size `dt`, for a force pair with reduced mass `mu`.

    Unlike the FEM triangle/bending forces (whose stiffness is spread over
    several coupled DOFs, giving a much lower effective per-DOF resonant
    frequency), a single point-mass force pair (a structural spring between
    two particles, or a particle pressed against the -- effectively
    infinite-mass -- static body collider) has no such slack:
      - stiffness: symplectic/semi-implicit Euler needs dt*sqrt(k/mu) <= 2,
        i.e. k <= 4*mu/dt**2; `safety` (< 1) pulls back from that edge.
      - damping: explicit damping diverges on its own past c*dt/mu > 2,
        independent of how it compares to the continuous critical-damping
        value, so it needs the same kind of cap.

    Returns (ke, kd) each capped at the requested value (never raised above
    what was asked, only pulled down if it would be unstable) plus whether
    capping happened and the two stability bounds, for logging.
    """
    ke_stable = safety * 4.0 * mu / dt**2
    kd_stable = safety * 2.0 * mu / dt
    ke = min(requested_ke, ke_stable)
    kd = min(requested_kd, kd_stable)
    capped = ke < requested_ke or kd < requested_kd
    return ke, kd, capped, ke_stable, kd_stable


def average_edge_length(vertices, faces):
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    return float(lengths.mean())


def push_outside_body(vertices, body_vertices, body_faces, clearance):
    """Guarantee every garment vertex is at least `clearance` outside the body surface.

    A safety net: *_boxmesh.obj's own placement is already close to the body's
    coordinate frame (it's GarmentCode's own pre-drape layout), so this is not
    expected to fire much, but is kept for robustness on other garments.
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


def load_sim_props(path):
    with open(path) as f:
        d = yaml.safe_load(f)
    cfg = d["sim"]["config"]
    return cfg["material"], cfg["options"]


def load_source_body_name(body_measurements_path):
    with open(body_measurements_path) as f:
        d = yaml.safe_load(f)
    return d["body"]["body_sample"]


def compare_to_ground_truth(positions_m, sim_obj_path):
    gt_vertices_cm, gt_faces = load_obj(sim_obj_path)
    gt_m = gt_vertices_cm * 0.01
    if gt_m.shape != positions_m.shape:
        raise RuntimeError(
            f"result has {positions_m.shape} vertices, ground truth {sim_obj_path} has {gt_m.shape} "
            "-- topology mismatch, aborting comparison."
        )
    dist_m = np.linalg.norm(positions_m - gt_m, axis=1)
    metrics = {
        "mean_cm": float(dist_m.mean() * 100.0),
        "median_cm": float(np.median(dist_m) * 100.0),
        "p90_cm": float(np.percentile(dist_m, 90) * 100.0),
        "max_cm": float(dist_m.max() * 100.0),
        "rmse_cm": float(np.sqrt((dist_m**2).mean()) * 100.0),
    }
    return metrics, gt_faces


def build_model(
    args, q_box_m, welded_faces, inv_D, panel_areas_abs, particle_mass, face_panel_id,
    body_vertices, body_faces, particle_radius, spring_pairs, spring_ke_start, spring_kd,
):
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
        vertices=q_box_m.tolist(),
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

    # inv_D/panel_areas_abs/particle_mass are precomputed by the caller (see
    # compute_rest_state_and_mass) from the flat 2D pattern, row-ordered to
    # match welded_faces -- inv_D's sign reflects each panel's own
    # local-2D-frame winding (some panels are negative there, and that's
    # fine: since args.tri_ke/tri_ka default to 0 for this experiment the
    # FEM triangle term contributes nothing regardless; kept correctly
    # signed anyway in case a caller re-enables it). particle_mass is
    # already floored and welded-length.
    if tri_end - tri_start != len(inv_D):
        raise RuntimeError(
            f"triangle count mismatch: add_cloth_mesh kept {tri_end - tri_start} of {len(inv_D)} triangles "
            "(some were dropped as 3D-degenerate) -- rest-state overrides below assume a 1:1, in-order match."
        )
    if len(particle_mass) != len(builder.particle_mass):
        raise RuntimeError("particle_mass length mismatch against builder's particle count.")

    builder.tri_poses[tri_start:tri_end] = inv_D.tolist()
    builder.tri_areas[tri_start:tri_end] = panel_areas_abs.tolist()
    builder.particle_mass = particle_mass.tolist()

    # Bending rest state: flat (0) everywhere, including cross-panel seam edges --
    # matching GarmentCode's own builder, which does not special-case seam rest
    # angle. Stiffness (ke, kd) is set to *zero* here regardless of --edge-ke/
    # --seam-edge-ke: the sim loop's stages 1-2 hold it at 0 (bending shouldn't
    # fight the box-mesh shrink) and ramps it to `final_bending_props` (returned
    # below) only in stage 3, once stiffening begins; see main().
    adjacency = MeshAdjacency(welded_faces)
    f0, f1 = adjacency.edge_tri_indices[:, 0], adjacency.edge_tri_indices[:, 1]
    interior = f1 >= 0  # boundary edges (f1 == -1) can't be seams
    seam_mask = np.zeros(len(f0), dtype=bool)
    seam_mask[interior] = face_panel_id[f0[interior]] != face_panel_id[f1[interior]]

    edge_rest_angle = np.array(builder.edge_rest_angle, dtype=np.float64)
    if len(edge_rest_angle) != len(seam_mask):
        raise RuntimeError("edge count mismatch between builder and recomputed adjacency -- ordering assumption broke")

    n_seam = int(seam_mask.sum())
    print(f"{n_seam} / {len(seam_mask)} edges are cross-panel seams -> edge_ke = {args.seam_edge_ke} (once ramped in)")
    edge_rest_angle[interior] = 0.0
    builder.edge_rest_angle = edge_rest_angle.tolist()

    final_bending_props = np.zeros((len(seam_mask), 2), dtype=np.float64)
    final_bending_props[:, 0] = np.where(seam_mask, args.seam_edge_ke, args.edge_ke)
    final_bending_props[:, 1] = args.edge_kd  # no separate seam damping parameter
    builder.edge_bending_properties = [(0.0, 0.0)] * len(seam_mask)

    # Structural springs: rest length starts at each edge's current box-mesh
    # length (matches what add_spring would auto-compute from builder.particle_q,
    # since that was populated by add_cloth_mesh from these same q_box_m
    # positions) and is annealed toward its target (true flat-pattern) length
    # over the sim loop's phase 1; see main().
    spring_start = len(builder.spring_indices) // 2
    for i, j in spring_pairs:
        builder.add_spring(int(i), int(j), spring_ke_start, spring_kd, control=0.0)
    spring_end = len(builder.spring_indices) // 2
    if spring_end - spring_start != len(spring_pairs):
        raise RuntimeError("spring count mismatch after add_spring calls.")

    model = builder.finalize(device=args.device, requires_grad=args.requires_grad)
    # gravity/soft_contact_ke/kd all start at (effectively) zero -- the sim loop's
    # phase 1/2 ramp them in; see main(). args.contact_ke/kd are their final values.
    model.gravity.assign([wp.vec3(0.0, 0.0, 0.0)])
    model.soft_contact_ke = 0.0
    model.soft_contact_kd = 0.0
    model.soft_contact_mu = args.friction
    model.particle_max_velocity = args.max_velocity
    return model, final_bending_props


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--panels", default=str(DEFAULT_PANELS), help="Source of unwelded_to_welded, face_panel_id, and flat-pattern rest-state data.")
    parser.add_argument("--boxmesh", default=str(DEFAULT_BOXMESH), help="Welded initial placement + structural-spring topology.")
    parser.add_argument("--orig-lens", default=str(DEFAULT_ORIG_LENS), help="Target lengths for box-mesh-stretched edges.")
    parser.add_argument("--sim-props", default=str(DEFAULT_SIM_PROPS))
    parser.add_argument("--body-measurements", default=str(DEFAULT_BODY_MEASUREMENTS))
    parser.add_argument("--sim-obj", default=str(DEFAULT_SIM_OBJ), help="Ground truth for the final per-vertex comparison.")
    parser.add_argument("--body", default=None, help="Override the source body mesh; default is resolved from --body-measurements' body_sample.")
    parser.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "outputs_semi_implicit_unwelded"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--substeps", type=int, default=64)
    parser.add_argument("--num-frames", type=int, default=1200, help="Must exceed stage 1 (data-dependent, see --shrink-speed -- ~665 frames at the default 0.03 m/s for this garment's worst edge) + --settle-frames + --stiffness-ramp-frames + --contact-ramp-frames, or there is no forward-drape stage left.")
    parser.add_argument("--shrink-speed", type=float, default=0.03, help="Stage 1: max rest-length contraction speed [m/s] for the most-stretched edge; each edge's own anneal duration scales with how far it has to shrink.")
    parser.add_argument("--settle-frames", type=int, default=60, help="Stage 2 duration: rest lengths held at target, stiffness held at ke_sewing, gravity/contact/bending still off.")
    parser.add_argument("--stiffness-ramp-frames", type=int, default=120, help="Stage 3 duration: structural-spring stiffness ke_sewing -> final, and bending 0 -> final, ramp together.")
    parser.add_argument("--contact-ramp-frames", type=int, default=120, help="Stage 4 duration: body-contact stiffness ramps 0 -> final. Gravity turns on once this stage ends.")
    parser.add_argument("--requires-grad", action="store_true", help="Finalize with requires_grad=True (still no wp.Tape here).")
    parser.add_argument(
        "--skip-push-outside",
        action="store_true",
        help="Skip pre-pushing box-mesh vertices to >= particle_radius + body_margin outside the body surface.",
    )

    # sim_props.yaml material defaults are read at runtime and used unless overridden here.
    parser.add_argument("--friction", type=float, default=None)
    parser.add_argument("--density", type=float, default=None, help="Areal density [kg/m^2].")
    parser.add_argument("--tri-ke", type=float, default=0.0, help="FEM triangle stretch stiffness. Default 0: stretch comes from structural springs, not a redundant FEM term (see module docstring).")
    parser.add_argument("--tri-ka", type=float, default=0.0, help="FEM triangle area-preservation stiffness. Default 0, same rationale as --tri-ke.")
    parser.add_argument("--tri-kd", type=float, default=0.0, help="FEM triangle damping. Default 0, same rationale as --tri-ke.")
    parser.add_argument("--edge-ke", type=float, default=0.01, help="Panel-interior bending stiffness. NOT sim_props.yaml's garment_edge_ke=1.0 -- that's an XPBD constraint-projection parameter, not a force parameter, and isn't safe to reuse directly here.")
    parser.add_argument("--edge-kd", type=float, default=0.0001, help="Panel-interior bending damping. NOT sim_props.yaml's garment_edge_kd=10.0, same rationale as --edge-ke (and large enough relative to this mesh's ~1e-5 kg particles, with no stability cap applied to it, to blow up on its own).")
    parser.add_argument(
        "--seam-edge-ke", type=float, default=None,
        help="Bending stiffness at cross-panel seam edges. Default: same as --edge-ke, i.e. no special-casing "
        "-- a welded seam is a continuous surface, not a free-folding boundary between disjoint pieces, so "
        "(unlike drape_semi_implicit.py's unwelded topology, where 0 is the right default) forcing it toward "
        "0 here creates an artificial zero-stiffness hinge line exactly where the structural springs' "
        "box-length -> target-length anneal is also concentrated, which visibly buckles/zippers along the seam.",
    )
    parser.add_argument("--body-margin", type=float, default=None)
    parser.add_argument("--contact-ke", type=float, default=1.0e4)
    parser.add_argument("--contact-kd", type=float, default=1.0e2)
    parser.add_argument("--particle-radius", type=float, default=None)
    parser.add_argument("--max-velocity", type=float, default=None, help="Per-particle speed clamp [m/s].")
    parser.add_argument("--spring-ke-start", type=float, default=None, help="Structural spring stiffness at sewing-phase start. Default: 1%% of sim_props.yaml's spring_ke.")
    parser.add_argument("--spring-ke-end", type=float, default=None, help="Structural spring stiffness once annealed. Default: sim_props.yaml's spring_ke.")
    parser.add_argument("--spring-kd", type=float, default=None, help="Structural spring damping, held constant throughout. Default: sim_props.yaml's spring_kd.")
    parser.add_argument("--timing-only", type=int, default=0)
    args = parser.parse_args()

    wp.init()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mat, options = load_sim_props(args.sim_props)
    if args.friction is None:
        args.friction = mat["fabric_friction"]
    if args.density is None:
        args.density = mat["fabric_density"]
    if args.seam_edge_ke is None:
        args.seam_edge_ke = args.edge_ke
    if args.body_margin is None:
        args.body_margin = options.get("body_collision_thickness", 0.0) * 0.01  # cm -> m
    if args.particle_radius is None and "fabric_thickness" in mat:
        args.particle_radius = mat["fabric_thickness"] * 0.01  # cm -> m
    if args.max_velocity is None:
        args.max_velocity = options["global_max_velocity"] * 0.01 if "global_max_velocity" in options else 1.0e5  # cm/s -> m/s
    if args.spring_ke_end is None:
        args.spring_ke_end = mat.get("spring_ke", 50000.0)
    if args.spring_ke_start is None:
        args.spring_ke_start = args.spring_ke_end * 0.01
    if args.spring_kd is None:
        args.spring_kd = mat.get("spring_kd", 10.0)

    panel_verts_cm, panel_indices, face_panel_id, unwelded_to_welded, n_welded, panel_order = load_panels(args.panels)
    panel_verts_m = panel_verts_cm * 0.01

    body_name = load_source_body_name(args.body_measurements)
    body_path = Path(args.body) if args.body else (DEFAULT_BODY_DIR / f"{body_name}.obj")
    body_vertices, body_faces = load_obj(body_path)

    gt_vertices_cm, gt_faces = load_obj(args.sim_obj)
    if n_welded != len(gt_vertices_cm):
        raise RuntimeError(f"welded vertex count {n_welded} != ground truth {args.sim_obj} vertex count {len(gt_vertices_cm)}")

    # panels_2d.npz's own face index order already matches *_sim.obj's winding
    # exactly for every panel in this garment (verified against ground truth) --
    # keep it completely as-authored (see panel_triangle_data/compute_rest_state_and_mass
    # for how a per-panel negative shoelace area is handled without needing to touch
    # index order at all).
    welded_faces = unwelded_to_welded[panel_indices]
    gt_cycles = {normalize_face_cycle(g) for g in gt_faces}
    n_winding_mismatch = sum(1 for r in welded_faces if normalize_face_cycle(r) not in gt_cycles)
    if n_winding_mismatch:
        raise RuntimeError(
            f"{n_winding_mismatch}/{len(welded_faces)} welded faces from panels_2d.npz do not match *_sim.obj's "
            "winding -- aborting (expected an exact match)."
        )
    print(f"Topology and winding match *_sim.obj exactly ({len(welded_faces)} faces).")

    # load_obj's maintain_order=True dedups *_boxmesh.obj's literal (seam-duplicated)
    # vertex list by position while preserving first-occurrence order -- confirmed
    # this lands on exactly panels_2d.npz's own welded numbering: box_faces comes out
    # array-equal to welded_faces, not just topologically equivalent. So no manual
    # welding/averaging is needed here at all, unlike this script's earlier version.
    box_vertices_cm, box_faces = load_obj(args.boxmesh)
    if len(box_vertices_cm) != n_welded:
        raise RuntimeError(f"*_boxmesh.obj welded vertex count {len(box_vertices_cm)} != panels_2d.npz's n_welded {n_welded}")
    if not np.array_equal(box_faces, welded_faces):
        raise RuntimeError(
            "*_boxmesh.obj's (welded) face indices do not exactly match unwelded_to_welded[panel_indices] -- "
            "aborting rather than silently trusting a mismatched topology or vertex numbering."
        )
    q_box_welded_m = box_vertices_cm * 0.01

    orig_lens_m = load_orig_lens(args.orig_lens)

    particle_radius = args.particle_radius
    if particle_radius is None:
        particle_radius = float(np.clip(average_edge_length(q_box_welded_m, welded_faces) * 0.5, 0.003, 0.01))

    if not args.skip_push_outside:
        clearance = particle_radius + args.body_margin
        q_box_welded_m, n_pushed = push_outside_body(q_box_welded_m, body_vertices, body_faces, clearance)
        print(f"Pushed {n_pushed}/{len(q_box_welded_m)} box-mesh vertices out to >= {clearance * 1000:.2f}mm from the body surface.")

    spring_pairs, spring_box_lengths_m, spring_target_lengths_m = build_structural_springs(welded_faces, q_box_welded_m, orig_lens_m)
    n_stretched = int((np.abs(spring_target_lengths_m - spring_box_lengths_m) > 1e-6).sum())
    print(
        f"Structural springs: {len(spring_pairs)} (one per welded-mesh edge); {n_stretched} have a real "
        f"box-length -> target-length anneal (the rest hold their already-correct box-mesh length)."
    )

    inv_D, panel_areas_abs, particle_mass, n_floored, mass_floor = compute_rest_state_and_mass(
        panel_verts_m, panel_indices, welded_faces, n_welded, args.density
    )
    if n_floored:
        print(f"Floored {n_floored}/{len(particle_mass)} particle masses up to {mass_floor * 1000:.4f} g for structural-spring stability.")

    frame_dt = 1.0 / args.fps
    sim_dt = frame_dt / args.substeps
    min_mass = particle_mass.min()

    # Structural springs: a single edge's stability bound (mu = min_mass/2, i.e. two
    # dynamic endpoints) badly overestimates safety here -- a real particle in this
    # mesh is shared by multiple springs pulling on it simultaneously (up to
    # max_degree at once), not just one, so the *network's* stability bound is
    # roughly max_degree times tighter. Dividing mu by max_degree is a conservative
    # way to account for that without modeling the full coupled system.
    degree = np.bincount(spring_pairs.reshape(-1), minlength=n_welded)
    max_degree = max(1, int(degree.max()))
    ke_end, kd, capped, ke_stable, kd_stable = derive_stable_ke_kd(min_mass / (2.0 * max_degree), sim_dt, args.spring_ke_end, args.spring_kd)
    if capped:
        print(
            f"Capped structural spring params for explicit-integration stability (given sim_dt={sim_dt:.2e}s, the "
            f"floored minimum particle mass, and max spring degree {max_degree}): ke {args.spring_ke_end:.1f} -> "
            f"{ke_end:.2f} (stable bound {ke_stable:.2f}), kd {args.spring_kd:.1f} -> {kd:.4f} (stable bound "
            f"{kd_stable:.4f}). sim_props.yaml's own spring_ke/spring_kd are tuned for GarmentCode's native "
            "solver/timestep, not this one."
        )
    args.spring_ke_end = ke_end
    args.spring_kd = kd
    args.spring_ke_start = min(args.spring_ke_start, args.spring_ke_end * 0.01)
    # Stage 1/2's constant stiffness: low enough that the box-mesh shrink itself
    # can't be the spike, regardless of how far any single edge still has to travel.
    ke_sewing = min(0.5, 0.1 * args.spring_ke_end)

    # Body contact: the body collider is static (effectively infinite mass), so the
    # reduced mass is just the (floored) particle's own mass mu = min_mass.
    contact_ke, contact_kd, contact_capped, contact_ke_stable, contact_kd_stable = derive_stable_ke_kd(
        min_mass, sim_dt, args.contact_ke, args.contact_kd
    )
    if contact_capped:
        print(
            f"Capped body-contact params for explicit-integration stability: contact_ke {args.contact_ke:.1f} -> "
            f"{contact_ke:.1f} (stable bound {contact_ke_stable:.1f}), contact_kd {args.contact_kd:.1f} -> {contact_kd:.1f} "
            f"(stable bound {contact_kd_stable:.3f})."
        )
    args.contact_ke = contact_ke
    args.contact_kd = contact_kd

    print(f"Device: {args.device}")
    print(f"Source body: {body_name} ({body_path})")
    print(f"Panels ({len(panel_order)}): {panel_order}")
    print(f"Garment: {n_welded} welded verts, {len(welded_faces)} tris, {len(spring_pairs)} structural springs, max spring degree {max_degree}")
    print(f"particle_radius = {particle_radius:.5f} m, density={args.density}, friction={args.friction}")
    print(f"tri_ke={args.tri_ke} tri_ka={args.tri_ka} tri_kd={args.tri_kd} edge_ke={args.edge_ke} edge_kd={args.edge_kd}")
    print(f"spring_ke {args.spring_ke_start:.2f} -> {args.spring_ke_end:.2f} (ke_sewing={ke_sewing:.3f}), spring_kd={args.spring_kd:.4f}")
    print(f"contact_ke={args.contact_ke:.2f}, contact_kd={args.contact_kd:.3f}")

    model, final_bending_props = build_model(
        args, q_box_welded_m, welded_faces, inv_D, panel_areas_abs, particle_mass, face_panel_id,
        body_vertices, body_faces, particle_radius, spring_pairs, args.spring_ke_start, args.spring_kd,
    )
    solver = newton.solvers.SolverSemiImplicit(model=model)

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()

    collision_pipeline = newton.CollisionPipeline(model)
    contacts = collision_pipeline.contacts()

    # Stage 1: each edge anneals on its own schedule, timed by how far it has to
    # shrink -- the smoothstep curve's peak slope is ~1.5x its average, so a 1.5x
    # margin on top of the naive distance/speed keeps the instantaneous contraction
    # rate under --shrink-speed even at that peak.
    contraction_m = np.maximum(spring_box_lengths_m - spring_target_lengths_m, 0.0)
    edge_duration_s = np.maximum(1.5 * contraction_m / args.shrink_speed, sim_dt)
    sewing_end_substep = max(1, int(np.ceil(edge_duration_s.max() / sim_dt)))
    settle_end_substep = sewing_end_substep + args.settle_frames * args.substeps
    stiffness_end_substep = settle_end_substep + args.stiffness_ramp_frames * args.substeps
    contact_end_substep = stiffness_end_substep + args.contact_ramp_frames * args.substeps
    print(
        f"Stage 1 (sewing): {sewing_end_substep} substeps ({sewing_end_substep / args.substeps:.1f} frames) for the "
        f"slowest edge ({contraction_m.max() * 100:.2f}cm contraction @ --shrink-speed {args.shrink_speed} m/s). "
        f"Stage boundaries (substeps): settle={settle_end_substep}, stiffen={stiffness_end_substep}, contact={contact_end_substep}."
    )

    n_frames = args.timing_only if args.timing_only > 0 else args.num_frames
    total_substeps = n_frames * args.substeps
    if total_substeps <= contact_end_substep:
        raise RuntimeError(
            f"--num-frames {n_frames} (* --substeps {args.substeps} = {total_substeps} substeps) does not leave room "
            f"for stage 5 (forward drape) after stage 4 ends at substep {contact_end_substep} -- increase --num-frames "
            "or, if stage 1 is the bottleneck, --shrink-speed."
        )
    gravity_vec = wp.vec3(0.0, -9.81, 0.0)
    gravity_on = False
    stiffness_ramp_span = max(1, stiffness_end_substep - settle_end_substep)
    contact_ramp_span = max(1, contact_end_substep - stiffness_end_substep)

    usd_viewer = None
    if args.timing_only <= 0:
        usd_path = out_dir / "drape.usd"
        usd_viewer = ViewerUSD(str(usd_path), fps=args.fps, up_axis="Y", num_frames=n_frames)
        usd_viewer.set_model(model)

    sim_time = 0.0
    global_substep = 0
    wp.synchronize()
    t_start = time.time()
    for frame in range(1, n_frames + 1):
        for _ in range(args.substeps):
            # Stage 1: box-mesh sewing -- each edge's rest length anneals on its own
            # shrink-speed-timed schedule; stiffness held at the low ke_sewing constant.
            if global_substep <= sewing_end_substep:
                edge_frac = smoothstep((global_substep * sim_dt) / edge_duration_s)
                rest_length_now = (spring_box_lengths_m * (1.0 - edge_frac) + spring_target_lengths_m * edge_frac).astype(np.float32)
                model.spring_rest_length.assign(rest_length_now)
                model.spring_stiffness.assign(np.full(len(spring_pairs), ke_sewing, dtype=np.float32))
            elif global_substep == sewing_end_substep + 1:
                write_obj(out_dir / "phase1_sewn.obj", state_0.particle_q.numpy(), welded_faces)

            # Stage 3: structural-spring stiffness ke_sewing -> final, bending 0 -> final, together.
            if settle_end_substep <= global_substep <= stiffness_end_substep:
                stiffness_frac = smoothstep((global_substep - settle_end_substep) / stiffness_ramp_span)
                model.spring_stiffness.assign(np.full(len(spring_pairs), ke_sewing + (args.spring_ke_end - ke_sewing) * stiffness_frac, dtype=np.float32))
                model.edge_bending_properties.assign((final_bending_props * stiffness_frac).astype(np.float32))
            elif global_substep == stiffness_end_substep + 1:
                write_obj(out_dir / "phase2_stiffened.obj", state_0.particle_q.numpy(), welded_faces)

            # Stage 4: contact discovery -- body-contact stiffness ramps in, everything else held.
            if stiffness_end_substep <= global_substep <= contact_end_substep:
                contact_frac = smoothstep((global_substep - stiffness_end_substep) / contact_ramp_span)
                model.soft_contact_ke = args.contact_ke * contact_frac
                model.soft_contact_kd = args.contact_kd * contact_frac
            elif global_substep == contact_end_substep + 1:
                write_obj(out_dir / "phase3_contact.obj", state_0.particle_q.numpy(), welded_faces)

            # Stage 5: forward drape -- gravity switches on once, exactly when stage 4 ends.
            if not gravity_on and global_substep >= contact_end_substep:
                model.gravity.assign([gravity_vec])
                gravity_on = True

            state_0.clear_forces()
            collision_pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, sim_dt)
            state_0, state_1 = state_1, state_0
            sim_time += sim_dt
            global_substep += 1

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
        return

    final_m = state_0.particle_q.numpy()

    final_path = out_dir / "final_drape.obj"
    write_obj(final_path, final_m, welded_faces)
    print(f"Final drape written to {final_path}")

    metrics, _ = compare_to_ground_truth(final_m, args.sim_obj)
    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Per-vertex comparison against {args.sim_obj}: {metrics}")
    print(f"Metrics written to {metrics_path}")


if __name__ == "__main__":
    main()
