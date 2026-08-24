#!/usr/bin/env python
"""Sew-from-flat-panels garment draping with Newton's SolverSemiImplicit.

Unlike drape_semi_implicit.py (which starts from a pre-draped, already-curved
target_shape.obj), this experiment starts the simulation from the garment's
*pattern-native* 3D placement: each panel is lifted from its 2D pattern
straight into 3D using the rigid transform (rotation, translation) that
*_specification.json stores for it -- the same placement GarmentCode itself
uses for the assembled-but-undraped garment, before any physical draping.

That placement is only a per-panel rigid transform, so it does not close the
seams: for this garment (2E2EL4UZUS, a skirt + waistband) the front and back
panels come out as two flat, parallel planes ~20-45cm apart at the side
seams (verified: 242 seam vertex groups from panels_2d.npz's
unwelded_to_welded map, gap range 7-45cm, median 35cm). Feeding zero-length,
full-stiffness sewing springs between seam vertices on frame 1 would apply a
huge instantaneous force to a near-massless cloth mesh -- exactly the kind of
single-step blowup drape_semi_implicit.py's own docstring warns about.

This script resolves that by annealing the sewing springs over the first
portion of the simulation (see --anneal-frac): each spring's rest length
ramps from its initial 3D gap down to 0, while its stiffness ramps from a
low starting value up to sim_props.yaml's tuned `spring_ke`, both on a
smoothstep curve (zero velocity at both ends -- no force discontinuity at
the start or end of the ramp). Body collision (source body = whichever
5000_body_shapes_and_measures mesh this garment's own
rand_*_body_measurements.yaml `body_sample` names, e.g. 01709_straight, not
an arbitrary default body) is active throughout, so as the springs pull the
side seams shut the panels are also pushed around the body's silhouette --
physically, this *is* wrapping a flat cut pattern around a body and sewing
it, not just a numerical trick.

Input decomposition:
  - Topology:               panels_2d.npz's per-panel faces, kept UNWELDED
                             (each panel is a topologically separate piece of
                             cloth -- add_cloth_mesh() does not deduplicate
                             vertices by position, so this is the literal mesh
                             SolverSemiImplicit simulates). Every interior
                             mesh edge is therefore panel-interior by
                             construction (no vertex is shared across
                             panels), so bending rest angles are flat (0)
                             everywhere with no separate seam case, unlike
                             drape_semi_implicit.py's welded-topology version.
  - Initial placement +     panels_2d.npz's 2D pattern vertices, rigidly
    stretch rest state:     transformed per-panel by *_specification.json's
                             rotation/translation for q0, and used directly
                             (pre-transform, in 2D) for tri_poses/tri_areas --
                             a rigid transform preserves edge lengths, angles
                             and areas, so both quantities come from the same
                             undistorted flat pattern.
  - Sewing correspondences: panels_2d.npz's unwelded_to_welded map, grouped
                             by shared welded id. Every pair of unwelded
                             vertices inside a group (GarmentCode's own record
                             of which panel-boundary vertices coincide once
                             welded) gets an annealed spring, per above.
  - Stitch diagnostics:     *_specification.json's `pattern.stitches` list
                             (named panel/edge correspondences) is cross-
                             checked against the panel-pairs the spring
                             groups above actually connect, purely as a sanity
                             report -- unwelded_to_welded is what builds the
                             springs, since it is already vertex-exact at the
                             simulated mesh's resolution, while `stitches`
                             only names coarse pattern-polygon edges.
  - Collision body:         the source body this garment was designed against
                             (body_measurements.yaml's `body_sample`), static
                             (body=-1), not an arbitrary default body.
  - Material defaults:      sim_props.yaml's `sim.config.material` block,
                             including `spring_ke` / `spring_kd` for the
                             sewing springs' post-anneal values; overridable
                             via CLI.

SolverSemiImplicit usage (state/control/contacts allocation, the
clear_forces() -> collide() -> solver.step() -> swap substep loop, and
finalize(requires_grad=...) for a wp.Tape-ready model) follows the
conventions in Newton's own diffsim cloth example:
https://github.com/newton-physics/newton/blob/main/newton/examples/diffsim/example_diffsim_cloth.py
This script itself remains forward-only (no wp.Tape pass).

Since the simulated mesh is unwelded, the final state has (possibly still
slightly separated) duplicate vertices at every seam. Those are averaged
back into GarmentCode's true welded topology via unwelded_to_welded before
(a) writing final_drape.obj and (b) computing per-vertex comparison metrics
against the ground-truth *_sim.obj. drape.usd, by contrast, animates the raw
unwelded mesh throughout -- watching the seams visibly close over the
anneal is itself a useful diagnostic of this experiment.

Outputs an animated USD (via newton.viewer.ViewerUSD) of the drape process,
a final welded OBJ, and a metrics.json comparing the welded result to
*_sim.obj.
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

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
DEFAULT_SPECIFICATION = GARMENT_DIR / f"{SPEC_NAME}_specification.json"
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


def euler_xyz_to_R(euler_deg):
    """Maya-convention intrinsic xyz Euler angles (degrees) -> 3x3 rotation matrix.

    Matches GarmentCode's own pygarment.pattern.rotation.euler_xyz_to_R
    (R = Rz(gamma) @ Ry(beta) @ Rx(alpha)) -- the convention
    *_specification.json's panel `rotation` fields are stored in.
    """
    a, b, g = np.deg2rad(np.asarray(euler_deg, dtype=np.float64))
    Rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    Ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    Rz = np.array([[np.cos(g), -np.sin(g), 0], [np.sin(g), np.cos(g), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


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
    panels_2d.npz stores them: coordinates are lifted to 3D by a rigid
    transform of *_specification.json's own choosing (see lift_panels_to_3d),
    and mirroring a panel's 2D coordinates is *not* an isometry of that 3D
    placement -- it changes which physical shape gets embedded, breaking
    alignment with the neighboring panels specification.json's translations
    were tuned against. Face index order is likewise left untouched: it
    already matches *_sim.obj's winding exactly (verified against ground
    truth), even though some panels' shoelace area comes out negative in
    their own local 2D frame -- see build_model's handling of that.

    Returns panel_verts (unwelded, 2D, centimeters as stored), panel_indices
    (unwelded face triples, raw winding), face_panel_id, the
    unwelded->welded vertex map, welded vertex count, panel name order, and
    each panel's (start, end) index range into the concatenated unwelded
    arrays (needed to look up that panel's own rotation/translation from
    *_specification.json).
    """
    d = np.load(npz_path, allow_pickle=True)
    panel_order = list(d["__panel_order__"])
    unwelded_to_welded = d["unwelded_to_welded"]
    n_welded = int(d["n_welded"])

    offset = 0
    panel_indices_all, panel_verts_all, face_panel_id = [], [], []
    panel_ranges = {}
    for panel_id, name in enumerate(panel_order):
        v = d[f"{name}::v"]
        f = d[f"{name}::f"]
        panel_ranges[name] = (offset, offset + len(v))
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
        panel_ranges,
    )


def normalize_face_cycle(row):
    """Rotate a face's 3 vertex ids so the smallest is first -- a winding-
    preserving canonical form, used to compare face sets independent of
    array row order (but not independent of winding direction)."""
    i = int(np.argmin(row))
    return tuple(np.roll(row, -i).tolist())


def load_specification(json_path):
    with open(json_path) as f:
        spec = json.load(f)
    return spec["pattern"]["panels"], spec["pattern"]["stitches"]


def lift_panels_to_3d(panel_verts, panel_ranges, panel_order, spec_panels):
    """Rigidly transform each panel's flat 2D pattern into 3D (centimeters),
    per *_specification.json's per-panel rotation + translation. This is
    both the initial placement q0 and (since a rigid transform is applied)
    equivalent to the flat pattern for stretch-rest purposes.
    """
    q0 = np.zeros((len(panel_verts), 3), dtype=np.float64)
    for name in panel_order:
        start, end = panel_ranges[name]
        panel = spec_panels[name]
        R = euler_xyz_to_R(panel["rotation"])
        t = np.array(panel["translation"], dtype=np.float64)
        v2 = panel_verts[start:end]
        v3_local = np.concatenate([v2, np.zeros((len(v2), 1))], axis=1)
        q0[start:end] = v3_local @ R.T + t
    return q0


def build_seam_springs(unwelded_to_welded, q0_m):
    """Group unwelded vertex indices by shared welded id; every pair within
    a group (size >= 2) is a sewing correspondence GarmentCode's own boxmesh
    construction recorded. Returns (pairs (N,2) int64, initial 3D gap per
    pair (N,) float64 meters) to anneal each spring's rest length from.
    """
    groups = defaultdict(list)
    for i, w in enumerate(unwelded_to_welded):
        groups[int(w)].append(i)

    pairs = []
    for idxs in groups.values():
        if len(idxs) < 2:
            continue
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                pairs.append((idxs[a], idxs[b]))
    pairs = np.array(pairs, dtype=np.int64)
    gap0 = np.linalg.norm(q0_m[pairs[:, 0]] - q0_m[pairs[:, 1]], axis=1)
    return pairs, gap0


def summarize_stitches(stitches, pairs, gap0_m, panel_ranges):
    """Diagnostic only: cross-check named stitches (specification.json,
    coarse panel/edge pairs) against the panel-pairs the fine-mesh seam
    springs (unwelded_to_welded groups) actually connect."""

    def vertex_panel(idx):
        for name, (start, end) in panel_ranges.items():
            if start <= idx < end:
                return name
        raise ValueError(f"unwelded vertex {idx} not covered by any panel range")

    spring_panel_pairs = defaultdict(list)
    for (i, j), gap in zip(pairs, gap0_m):
        key = frozenset((vertex_panel(i), vertex_panel(j)))
        spring_panel_pairs[key].append(gap)

    print(f"Seam springs: {len(pairs)} across {len(spring_panel_pairs)} panel-pairs (gaps in cm):")
    for key, gaps in spring_panel_pairs.items():
        gaps = np.array(gaps) * 100.0
        print(f"  {sorted(key)}: {len(gaps)} springs, gap [{gaps.min():.1f}, {gaps.max():.1f}], median {np.median(gaps):.1f}")

    named_pairs = {frozenset((s[0]["panel"], s[1]["panel"])) for s in stitches}
    missing = named_pairs - set(spring_panel_pairs)
    extra = set(spring_panel_pairs) - named_pairs
    if missing:
        print(f"  WARNING: specification.json names stitches between {missing} with no matching seam springs.")
    if extra:
        print(f"  Note: seam springs also connect {extra}, not explicitly named as a stitch in specification.json.")


def smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def compute_rest_state_and_mass(panel_verts, panel_indices, n_particles, density, mass_floor_frac=0.2):
    """FEM rest state (inv_D, |area|) and per-particle mass (density * area/3
    per corner, floored to `mass_floor_frac` of the *mean* particle mass).

    A handful of sliver triangles from panels_2d.npz's triangulation would
    otherwise leave a few particles with mass orders of magnitude below the
    rest of the mesh. That's harmless for the FEM/mass itself, but those
    outlier-light particles set the worst-case explicit-integration
    stability bound for the sewing springs (see derive_stable_spring_params)
    -- flooring them is a standard, physically defensible step (real fabric
    doesn't have near-zero-mass regions) that avoids a few slivers dictating
    how stiff every seam spring in the whole garment is allowed to be.
    """
    inv_D, areas_signed = panel_triangle_data(panel_verts, panel_indices)
    areas_abs = np.abs(areas_signed)
    mass = np.zeros(n_particles, dtype=np.float64)
    for t, (i, j, k) in enumerate(panel_indices):
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
    frequency), a single point-mass force pair (a sewing spring between two
    particles, or a particle pressed against the -- effectively infinite-mass
    -- static body collider) has no such slack:
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

    A safety net, not expected to fire much here: the flat, spec-placed panels
    for this garment already sit outside the body's bounding box at frame 0
    (verified for 2E2EL4UZUS/01709_straight), unlike a curved geometric
    estimate such as target_shape.obj. Kept for robustness on other garments.
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


def weld_average(unwelded_to_welded, n_welded, positions):
    """Average unwelded (possibly still slightly separated) seam duplicates
    back into GarmentCode's true welded topology."""
    sums = np.zeros((n_welded, 3), dtype=np.float64)
    counts = np.zeros(n_welded, dtype=np.float64)
    np.add.at(sums, unwelded_to_welded, positions)
    np.add.at(counts, unwelded_to_welded, 1.0)
    return sums / counts[:, None]


def compare_to_ground_truth(welded_positions_m, sim_obj_path):
    gt_vertices_cm, gt_faces = load_obj(sim_obj_path)
    gt_m = gt_vertices_cm * 0.01
    if gt_m.shape != welded_positions_m.shape:
        raise RuntimeError(
            f"welded result has {welded_positions_m.shape} vertices, ground truth {sim_obj_path} has "
            f"{gt_m.shape} -- topology mismatch, aborting comparison."
        )
    dist_m = np.linalg.norm(welded_positions_m - gt_m, axis=1)
    metrics = {
        "mean_cm": float(dist_m.mean() * 100.0),
        "median_cm": float(np.median(dist_m) * 100.0),
        "p90_cm": float(np.percentile(dist_m, 90) * 100.0),
        "max_cm": float(dist_m.max() * 100.0),
        "rmse_cm": float(np.sqrt((dist_m**2).mean()) * 100.0),
    }
    return metrics, gt_faces


def build_model(args, mat, q0_m, panel_indices, inv_D, panel_areas_abs, particle_mass, face_panel_id, body_vertices, body_faces, particle_radius, spring_pairs, spring_gap0_m, spring_ke_start, spring_kd):
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
        vertices=q0_m.tolist(),
        indices=panel_indices.flatten().tolist(),
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
    # compute_rest_state_and_mass) -- inv_D's sign reflects each panel's own
    # local-2D-frame winding (some panels, e.g. skirt_back/wb_back, are
    # negative there, and that's fine: q0 and inv_D are derived from this
    # identical signed 2D data via the same face index order, so the
    # deformation gradient F = Ds_world @ inv_D built from them is
    # self-consistently near-identity at rest regardless of sign -- what
    # matters is that panel_indices matches *_sim.obj's own winding exactly,
    # asserted by the caller). particle_mass is already floored there.
    if tri_end - tri_start != len(inv_D):
        raise RuntimeError(
            f"triangle count mismatch: add_cloth_mesh kept {tri_end - tri_start} of {len(inv_D)} triangles "
            "(some were dropped as 3D-degenerate) -- rest-state overrides below assume a 1:1, in-order match."
        )
    if len(particle_mass) != len(builder.particle_mass):
        raise RuntimeError("particle_mass length mismatch against builder's particle count.")

    # Stretch rest state: flat 2D pattern, not the (already-flat-per-panel,
    # but jointly non-planar) 3D q0 add_cloth_mesh would otherwise derive it from.
    builder.tri_poses[tri_start:tri_end] = inv_D.tolist()
    builder.tri_areas[tri_start:tri_end] = panel_areas_abs.tolist()
    builder.particle_mass = particle_mass.tolist()

    # Bending rest state: every interior edge is panel-interior by construction
    # (unwelded topology -- no vertex, hence no edge, is shared across panels).
    adjacency = MeshAdjacency(panel_indices)
    f0, f1 = adjacency.edge_tri_indices[:, 0], adjacency.edge_tri_indices[:, 1]
    interior = f1 >= 0
    cross_panel = face_panel_id[f0[interior]] != face_panel_id[f1[interior]]
    if cross_panel.any():
        raise RuntimeError("found a cross-panel edge in the unwelded mesh -- panel vertex ranges are not disjoint as assumed.")

    edge_rest_angle = np.array(builder.edge_rest_angle, dtype=np.float64)
    if len(edge_rest_angle) != len(f0):
        raise RuntimeError("edge count mismatch between builder and recomputed adjacency -- ordering assumption broke")
    edge_rest_angle[interior] = 0.0
    builder.edge_rest_angle = edge_rest_angle.tolist()

    # Sewing springs: rest length starts at each pair's initial 3D gap (matches
    # what add_spring would auto-compute from builder.particle_q, since that
    # was populated by add_cloth_mesh from these same q0 positions) and is
    # annealed down to 0 over the simulation's ramp loop; see build_model's
    # caller.
    spring_start = len(builder.spring_indices) // 2
    for i, j in spring_pairs:
        builder.add_spring(int(i), int(j), spring_ke_start, spring_kd, control=0.0)
    spring_end = len(builder.spring_indices) // 2
    if spring_end - spring_start != len(spring_pairs):
        raise RuntimeError("spring count mismatch after add_spring calls.")

    model = builder.finalize(device=args.device, requires_grad=args.requires_grad)
    # soft_contact_ke/kd start at 0 -- the sim loop ramps them in over the back half
    # of the seam anneal (see main()); args.contact_ke/kd are their final values.
    model.soft_contact_ke = 0.0
    model.soft_contact_kd = 0.0
    model.soft_contact_mu = args.friction
    model.particle_max_velocity = args.max_velocity
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--panels", default=str(DEFAULT_PANELS))
    parser.add_argument("--specification", default=str(DEFAULT_SPECIFICATION))
    parser.add_argument("--sim-props", default=str(DEFAULT_SIM_PROPS))
    parser.add_argument("--body-measurements", default=str(DEFAULT_BODY_MEASUREMENTS))
    parser.add_argument("--sim-obj", default=str(DEFAULT_SIM_OBJ), help="Ground truth for the final per-vertex comparison.")
    parser.add_argument("--body", default=None, help="Override the source body mesh; default is resolved from --body-measurements' body_sample.")
    parser.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "outputs_semi_implicit_unwelded"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--substeps", type=int, default=32)
    parser.add_argument("--num-frames", type=int, default=300)
    parser.add_argument("--anneal-frac", type=float, default=0.4, help="Fraction of the run (by substep) over which sewing springs anneal to rest_length=0, spring_ke=--spring-ke-end.")
    parser.add_argument("--requires-grad", action="store_true", help="Finalize with requires_grad=True (still no wp.Tape here).")
    parser.add_argument(
        "--skip-push-outside",
        action="store_true",
        help="Skip pre-pushing q0 vertices to >= particle_radius + body_margin outside the body surface.",
    )

    # sim_props.yaml material defaults are read at runtime and used unless overridden here.
    parser.add_argument("--friction", type=float, default=None)
    parser.add_argument("--density", type=float, default=None, help="Areal density [kg/m^2].")
    parser.add_argument("--tri-ke", type=float, default=None)
    parser.add_argument("--tri-ka", type=float, default=None)
    parser.add_argument("--tri-kd", type=float, default=None)
    parser.add_argument("--edge-ke", type=float, default=None)
    parser.add_argument("--edge-kd", type=float, default=None)
    parser.add_argument("--body-margin", type=float, default=None)
    parser.add_argument("--contact-ke", type=float, default=1.0e4)
    parser.add_argument("--contact-kd", type=float, default=1.0e2)
    parser.add_argument("--particle-radius", type=float, default=None)
    parser.add_argument("--max-velocity", type=float, default=None, help="Per-particle speed clamp [m/s].")
    parser.add_argument("--spring-ke-start", type=float, default=None, help="Sewing spring stiffness at anneal start. Default: 1%% of sim_props.yaml's spring_ke.")
    parser.add_argument("--spring-ke-end", type=float, default=None, help="Sewing spring stiffness once annealed. Default: sim_props.yaml's spring_ke.")
    parser.add_argument("--spring-kd", type=float, default=None, help="Sewing spring damping, held constant throughout. Default: sim_props.yaml's spring_kd.")
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
    if args.spring_ke_end is None:
        args.spring_ke_end = mat.get("spring_ke", 50000.0)
    if args.spring_ke_start is None:
        args.spring_ke_start = args.spring_ke_end * 0.01
    if args.spring_kd is None:
        args.spring_kd = mat.get("spring_kd", 10.0)

    panel_verts, panel_indices_raw, face_panel_id, unwelded_to_welded, n_welded, panel_order, panel_ranges = load_panels(args.panels)
    panel_verts_cm = panel_verts  # panels_2d.npz is stored in centimeters
    spec_panels, stitches = load_specification(args.specification)
    if set(panel_order) != set(spec_panels.keys()):
        raise RuntimeError(f"panels_2d.npz panel names {sorted(panel_order)} != specification.json panel names {sorted(spec_panels.keys())}")

    q0_cm = lift_panels_to_3d(panel_verts_cm, panel_ranges, panel_order, spec_panels)
    q0_m = q0_cm * 0.01
    # tri_poses/tri_areas (via compute_rest_state_and_mass's panel_triangle_data call,
    # below) must be in the same units as q0_m (meters) -- panel_verts_cm left in
    # centimeters there would make the rest frame ~100x larger than the world frame,
    # i.e. F = Ds_world @ Dm_rest^-1 comes out ~0.01*I at t=0 (looks like the cloth is
    # compressed to 1% of rest size), which tri_ke then "corrects" with a violent
    # expansion force.
    panel_verts_m = panel_verts_cm * 0.01

    body_name = load_source_body_name(args.body_measurements)
    body_path = Path(args.body) if args.body else (DEFAULT_BODY_DIR / f"{body_name}.obj")
    body_vertices, body_faces = load_obj(body_path)

    gt_vertices_cm, gt_faces = load_obj(args.sim_obj)
    if n_welded != len(gt_vertices_cm):
        raise RuntimeError(f"welded vertex count {n_welded} != ground truth {args.sim_obj} vertex count {len(gt_vertices_cm)}")

    # panels_2d.npz's own face index order already matches *_sim.obj's winding
    # exactly for every panel in this garment (verified against ground truth) --
    # keep it completely as-authored. Some panels (skirt_back, wb_back) come out
    # with negative shoelace area under that order in their own local 2D frame;
    # that's fine for tri_poses (build_model computes the deformation gradient
    # from this same signed 2D data and the matching q0 positions, so it stays
    # self-consistent regardless of sign) and is handled explicitly for mass
    # (build_model uses abs(area) there). Reversing index order to force
    # positive area, as an earlier version of this script did, would have
    # flipped these panels' outward normals relative to *_sim.obj's convention
    # -- exactly what produced backface-culled "holes" in the rendered drape.
    panel_indices = panel_indices_raw
    welded_faces = unwelded_to_welded[panel_indices]

    gt_cycles = {normalize_face_cycle(g) for g in gt_faces}
    n_winding_mismatch = sum(1 for r in welded_faces if normalize_face_cycle(r) not in gt_cycles)
    if n_winding_mismatch:
        raise RuntimeError(
            f"{n_winding_mismatch}/{len(welded_faces)} welded faces from panels_2d.npz do not match *_sim.obj's "
            "winding -- aborting (expected an exact match; see the comment above)."
        )
    print(f"Topology and winding match *_sim.obj exactly ({len(welded_faces)} faces).")

    particle_radius = args.particle_radius
    if particle_radius is None:
        particle_radius = float(np.clip(average_edge_length(q0_m, panel_indices) * 0.5, 0.003, 0.01))

    if not args.skip_push_outside:
        clearance = particle_radius + args.body_margin
        q0_m, n_pushed = push_outside_body(q0_m, body_vertices, body_faces, clearance)
        print(f"Pushed {n_pushed}/{len(q0_m)} q0 vertices out to >= {clearance * 1000:.2f}mm from the body surface.")

    spring_pairs, spring_gap0_m = build_seam_springs(unwelded_to_welded, q0_m)
    summarize_stitches(stitches, spring_pairs, spring_gap0_m, panel_ranges)

    inv_D, panel_areas_abs, particle_mass, n_floored, mass_floor = compute_rest_state_and_mass(
        panel_verts_m, panel_indices, len(q0_m), args.density
    )
    if n_floored:
        print(f"Floored {n_floored}/{len(particle_mass)} particle masses up to {mass_floor * 1000:.4f} g for sewing-spring stability.")

    frame_dt = 1.0 / args.fps
    sim_dt = frame_dt / args.substeps
    min_mass = particle_mass.min()

    # Sewing springs: both endpoints are dynamic particles -> reduced mass mu = min_mass/2.
    ke_end, kd, capped, ke_stable, kd_stable = derive_stable_ke_kd(min_mass / 2.0, sim_dt, args.spring_ke_end, args.spring_kd)
    if capped:
        print(
            f"Capped sewing spring params for explicit-integration stability (given sim_dt={sim_dt:.2e}s and the "
            f"floored minimum particle mass): ke {args.spring_ke_end:.1f} -> {ke_end:.1f} (stable bound {ke_stable:.1f}), "
            f"kd {args.spring_kd:.1f} -> {kd:.1f} (stable bound {kd_stable:.3f}). sim_props.yaml's own spring_ke/spring_kd "
            "are tuned for GarmentCode's native solver/timestep, not this one."
        )
    args.spring_ke_end = ke_end
    args.spring_kd = kd
    args.spring_ke_start = min(args.spring_ke_start, args.spring_ke_end * 0.01)

    # Body contact: the body collider is static (effectively infinite mass), so the
    # reduced mass is just the (floored) particle's own mass mu = min_mass -- the same
    # explicit-integration instability that hit the sewing springs applies here too
    # (contact_ke=1e4's default was ~43x over this bound), and was the actual cause of
    # the wild, far-outside-the-body-bbox vertex excursions seen after fixing springs
    # alone: unstable contact response, not unclosed seams.
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
    print(f"Garment: {len(q0_m)} unwelded verts ({n_welded} welded), {len(panel_indices)} tris, {len(spring_pairs)} sewing springs")
    print(f"particle_radius = {particle_radius:.5f} m, density={args.density}, friction={args.friction}")
    print(f"tri_ke={args.tri_ke} tri_ka={args.tri_ka} tri_kd={args.tri_kd} edge_ke={args.edge_ke} edge_kd={args.edge_kd}")
    print(f"spring_ke {args.spring_ke_start:.2f} -> {args.spring_ke_end:.2f}, spring_kd={args.spring_kd:.3f}, anneal_frac={args.anneal_frac}")
    print(f"contact_ke={args.contact_ke:.2f}, contact_kd={args.contact_kd:.3f}")

    model = build_model(
        args, mat, q0_m, panel_indices, inv_D, panel_areas_abs, particle_mass, face_panel_id, body_vertices, body_faces,
        particle_radius, spring_pairs, spring_gap0_m, args.spring_ke_start, args.spring_kd,
    )
    solver = newton.solvers.SolverSemiImplicit(model=model)

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()

    collision_pipeline = newton.CollisionPipeline(model)
    contacts = collision_pipeline.contacts()

    n_frames = args.timing_only if args.timing_only > 0 else args.num_frames
    total_substeps = n_frames * args.substeps
    anneal_substeps = max(1, int(args.anneal_frac * total_substeps))
    # Body collision starts at 0 and ramps in only over the back half of the
    # seam anneal, finishing exactly when it does. A wrap-around seam (e.g.
    # the skirt's side seam) needs its two sides to travel *through* where
    # the leg will be to meet each other -- with collision active from frame
    # 1, contact holds them apart at the body surface and the seam can never
    # fully close, leaving a visible gap. Letting the garment sew itself
    # closed first (as a free-floating, body-agnostic shape) and only then
    # discovering the body avoids that, while still ramping (not switching)
    # contact stiffness in so there is no sudden-penetration shock once it
    # does start pushing back.
    contact_start_substep = max(1, int(0.5 * anneal_substeps))
    contact_end_substep = anneal_substeps

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
            if global_substep <= anneal_substeps:
                frac = smoothstep(global_substep / anneal_substeps)
                rest_length_now = (spring_gap0_m * (1.0 - frac)).astype(np.float32)
                ke_now = np.full(len(spring_pairs), args.spring_ke_start + (args.spring_ke_end - args.spring_ke_start) * frac, dtype=np.float32)
                model.spring_rest_length.assign(rest_length_now)
                model.spring_stiffness.assign(ke_now)

            if global_substep <= contact_end_substep:
                contact_frac = smoothstep((global_substep - contact_start_substep) / (contact_end_substep - contact_start_substep))
                model.soft_contact_ke = args.contact_ke * contact_frac
                model.soft_contact_kd = args.contact_kd * contact_frac

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

    final_unwelded_m = state_0.particle_q.numpy()
    welded_m = weld_average(unwelded_to_welded, n_welded, final_unwelded_m)

    final_path = out_dir / "final_drape.obj"
    write_obj(final_path, welded_m, welded_faces)
    print(f"Final drape (welded) written to {final_path}")

    metrics, _ = compare_to_ground_truth(welded_m, args.sim_obj)
    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Per-vertex comparison against {args.sim_obj}: {metrics}")
    print(f"Metrics written to {metrics_path}")


if __name__ == "__main__":
    main()
