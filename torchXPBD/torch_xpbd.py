"""Differentiable XPBD garment drape (Dress Anyone Sec. 3.1 / 3.2), in PyTorch.

Sec. 3.2's adjoint (Eq. 5-7) is reverse-mode autodiff through the XPBD state-update
recursion of Eq. 4. So instead of hand-deriving that recursion (where GarmentCode's
Warp fork went wrong -- its built-in XPBD adjoint was found off by 5-20 orders of
magnitude), the forward pass here is written as plain differentiable PyTorch ops and
``torch.autograd`` supplies the backward pass. No Warp / GarmentCode simulator
dependency -- constraint choices below follow the paper's own text, not GarmentCode's
implementation (its XPBD integrator, on inspection, doesn't even solve a triangle-FEM
constraint despite configuring one; its actual effective stretch model is a plain
edge-distance spring network. This module intentionally does NOT replicate that).

Constraint set:
  * in-plane membrane -- the "cloth triangle constraint" of Eq. 7 (Sec. 4.3), now
    DiffXPBD's own Green-strain orthotropic Saint Venant-Kirchhoff formulation
    (own choice per user instruction -- Dress Anyone's Eq. 7 itself doesn't specify
    a strain measure, it names Baraff & Witkin 1998 in its Related Work; an earlier
    version of this constraint used THAT norm-based stretch/shear formulation
    instead, see git history / _solve_triangle_strain's own docstring for the
    area/compliance-scaling reasoning behind switching to DiffXPBD's): per
    triangle, express the current 3D edge vectors in the triangle's MATERIAL
    (u, v) frame -- taken directly from the 2D sewing pattern coordinates (not a
    derived 3D rest length) -- as a deformation gradient F, then the Green strain
    E = 0.5*(F^T F - I), giving constraint vector C = [E_uu, E_vv, 2*E_uv]. Because
    u, v are literally the Sec. 4.3 optimization variable x̄, this constraint's own
    rest reference is differentiable w.r.t. x̄ through the exact same leaf tensors
    the lift (rest_mesh_builder.build_rest_mesh_from_pattern_torch) uses --
    matching Eq. 7's ∂Δx/∂x̄ without any separate 3D-rest-length bookkeeping (no
    more welding-distorted edge lengths, no more simulation.orig_lens_refresh --
    that machinery existed only to patch up 3D edge lengths at welded seams for
    the stretch constraint this module no longer has).
  * bending -- dihedral-angle constraint on adjacent-triangle pairs, ke=1, kd=0
    (GarmentCode uses kd=10; zeroed here along with the other constraints' kd,
    see XPBDConfig.tri_kd -- the paper doesn't specify bending at all).
  * body collision -- avatar only (no self-collision, matching the existing setup);
    one-sided closest-point push-out via libigl, PLUS Coulomb friction
    (``friction_mu``, ported from GarmentCode's ``body_friction`` -- see
    ``_solve_collision``'s docstring). The paper's Sec. 3 doesn't specify a
    friction model, so the friction TERM itself is a from-Warp addition, not a
    paper-following one -- but empirically load-bearing (see that docstring):
    without it, a real drape never conforms to the body's silhouette the way
    GarmentCode's own does.
  * body reference-shape drag (``_solve_body_reference_drag``) -- ported from
    Warp's ``reference_shape_drag_constraint`` kernel's flag==2 branch ("body
    collision resolution"), NOT its flag==1 branch (see below). Warp's own
    trigger for this is computed in ``warp/sim/collide.py``'s
    ``create_soft_contacts``: once a particle's signed distance to the body,
    net of ``body_collision_thickness``, drops below ``-soft_contact_margin``
    (GarmentCode's hardcoded 0.2, ``sim_config.py``), it gets snapped FULLY
    onto the closest surface point that substep (no compliance/damping --
    Warp's own dlambda has no ``ke``/``kd`` term for this branch, just
    ``|err| / invmass``, which cancels to a full correction) -- a harder,
    binary "you've actually penetrated -- get out now" correction layered on
    top of the ordinary soft push-out above (which is a weak, proportional
    spring and can be outrun by a fast free-fall before it catches the
    particle). This is what actually conforms a freely-falling flat-pattern
    rest mesh to the body's curved silhouette -- NOT a continuous "hug the
    body part" force (see the "Deliberately NOT replicated" flag==1 note
    below for that distinction; an earlier version of this docstring
    conflated the two before this was traced through Warp's actual source).
    Reuses the SAME once-per-frame ``closest_points``/``normals`` correspondence
    ``_solve_collision`` already queries (Warp re-queries fresh every substep
    for flagged particles; this module keeps the frame-frozen correspondence
    already established elsewhere in this file, a smaller version of the same
    approximation ``_body_collision_targets`` already documents) and snaps to
    the SAME ``collision_thickness``-offset shell ``_solve_collision`` targets
    (Warp's snap targets the bare surface, zero thickness offset, for this
    specific branch only -- a ~0.25cm difference, judged not worth tracking a
    second offset convention for).
  * stitching -- a zero-rest-length compliant distance constraint between each
    seam's two corresponding unwelded vertices (see ``_solve_stitch``). Neither
    the paper's Sec. 3/4.3 nor DiffXPBD specifies a stitching constraint formula
    at all (Sec. 4.4.1's per-seam loss weight and Eq. 10's pattern-matching term
    are both about the LOSS, not the simulator's constraint set) -- own design
    choice, own approximation, flagged per CLAUDE.md. The panels are lifted to 3D
    and simulated WITHOUT welding (rest_mesh_builder.build_rest_mesh_from_pattern_
    torch's unwelded output is passed straight to simulate_drape); this constraint
    is what holds the seams together during the drape instead. Seam pairs come
    from GarmentCode's own verts_loc_glob correspondence (exact, not a 3D nearest-
    neighbour heuristic) -- see rest_mesh_builder.stitch_pairs_from_weld_map.
    Bending is NOT propagated across a stitched seam (bending quads only form
    within a panel's own unwelded faces) -- an accepted consequence, not
    separately compensated for: real seams behave more like a hinge than a
    welded, bending-continuous sheet, so this is a defensible simplification, not
    just a shortcut.
  * attachment -- GarmentCode's own warm-up pin (``Cloth._add_attachment_labels``
    / Warp's ``attachment_constraint`` kernel, ``add_attachment``), ported to
    this module's own constraint style (see ``_solve_attachment``): a ONE-SIDED
    half-space spring restricting labeled vertices (e.g. ``vertex_labels.yaml``'s
    ``lower_interface`` -- a waistband loop) to ``dot(x - target_point, norm) >=
    0``, active only for the first ``config.attachment_frames`` frames, then
    released. Without it a real garment's own weight can slide it past the body
    before gravity + collision alone establish contact (the failure this
    constraint exists to prevent -- see module-level usage note below). Ported
    from GarmentCode/Warp (not the paper -- neither Dress Anyone Sec. 3 nor
    DiffXPBD describes an attachment mechanism), so this IS a departure from
    ``torch_xpbd`` being a pure paper-following reimplementation, same as the
    stitch constraint above.

Constraint solve (Eq. 1-3): each compliant constraint (triangle strain, bending,
stitch, reference drag, attachment) now keeps its own per-instance Lagrange
multiplier lambda, reset to 0 at the START of each substep and accumulated
across ``config.solver_iters`` inner passes within that substep -- Delta-lambda
_= -(C(x) + alpha_tilde*lambda) / (grad_C^T M^-1 grad_C + alpha_tilde)_, matching
Eq. 2 literally (an earlier version of this module dropped the ``alpha_tilde *
lambda`` term entirely, equivalent to hard-coding ``solver_iters=1`` with no
persistence -- found, on a real (not toy) mesh, to produce a ~1e77-magnitude
simulator adjoint through the full 150-frame drape; see optimize_pattern.py's
``_clip_grad_from_sim``). Each constraint TYPE is solved against the position
already updated by the PREVIOUS constraint type this same inner pass (Gauss-
Seidel across types -- triangle strain, then bending, then collision, then body
reference drag, then reference-shape drag, then stitch, then attachment, in
that fixed order, repeated ``solver_iters`` times); within one constraint
TYPE's own batch (e.g. all triangles' Cu at once) the update is still Jacobi --
this is the standard practical granularity for a vectorized/parallel PBD/XPBD
solver (matching e.g. Warp's own per-kernel-type structure), not literal
per-primitive sequential Gauss-Seidel, which would kill vectorization. That
batched Jacobi update is AVERAGED per vertex, not summed -- see
``_jacobi_divisor`` for why (summing over-relaxes by a vertex's primitive
valence and makes the solver inject energy; this module did sum, and inflated
a real garment without bound, until that was traced). Own
choice, flagged per CLAUDE.md -- the paper's Eq. 2 says "solved iteratively"
but doesn't specify an iteration count OR a Jacobi/Gauss-Seidel ordering.
``_solve_collision``/``_solve_body_reference_drag`` are NOT compliant Eq. 2
constraints (contact push-out and a hard non-compliant snap, respectively, same
as before this change) so they carry no lambda of their own -- they still get
re-evaluated against the latest ``x`` each inner pass, same as everything else,
which is a genuine (if lambda-free) refinement over the old single-pass version.

Deliberately NOT replicated:
  * Warp's ``reference_shape_drag_constraint`` flag==1 branch ("self-collision
    resolution"): triggered by ``find_intersecting_particles`` (``collide.py``)
    when the garment's OWN spring network detects two differently-labeled
    regions self-intersecting (e.g. a sleeve panel tangling through the torso),
    and drags the offending particles back toward their own panel's assigned
    body-part reference submesh. Needs the full ``panel_assignment`` machinery
    (per-panel body-part voting against a body segmentation this pipeline
    doesn't load -- see the earlier feasibility discussion) AND garment self-
    collision detection (disabled in this pipeline's own setup, see
    simulation/warp_drape.py). Narrow in scope (only fires on actual self-
    intersection between mislabeled regions) and not needed for garments
    without such regions (e.g. a plain skirt has no sleeve/torso pair to
    tangle) -- deferred until a garment that actually needs it comes up.

``target_shape``-based reference-shape drag (``_solve_reference_drag`` /
``XPBDConfig.reference_drag_ke``/``reference_drag_kd``) is a SEPARATE, older
mechanism from the ``_solve_body_reference_drag`` constraint above -- it isn't
what fixes the free-hanging-cone problem (that's the flag==2 port). It adds a
weak per-vertex "reference-shape drag" toward ``target_shape`` DURING the
forward drape, not just as a loss target afterward -- an own approximation,
not something either paper specifies (Dress Anyone's own text only uses
target_shape as the Sec. 4.4.1 L_SM loss target, Eq. 8, never as an in-sim
physics term). ``optimization.optimize_pattern`` no longer passes
``target_shape`` into ``simulate_drape`` for this reason (removed for strict
paper fidelity) -- this module still accepts and implements the parameter
(kept for tests/experiments that want it, e.g. pulling a drape toward a
known-good target outside the optimization loop), it's just unused by the
main retargeting pipeline now.

Time stepping follows the paper's own Eq. 4, including its velocity damping
``v_{n+1} = (tau/dt)(x_{n+1} - x_n)`` -- not Warp's damping mechanism. ``tau`` is
applied per SUBSTEP as ``tau_sub = config.velocity_damping ** (1/substeps)``, not
the raw ``config.velocity_damping=0.95`` value itself: the paper's Eq. 4 defines
tau at a single (1/fps) frame timestep and doesn't describe a substep structure
at all, so applying the frame-rate tau once per substep instead compounds
``tau**substeps`` of decay per frame -- confirmed empirically to make terminal
free-fall velocity scale as ~1/substeps (see tests/tests_sim/probe_velocity_
damping_substeps.py). ``tau_sub`` is this module's own resolution, chosen so its
substeps-fold composition reproduces ``config.velocity_damping`` exactly once
per frame regardless of ``config.substeps``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import igl
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from utils.data_class import Mesh

_EPS = 1e-9


class _USDDrapeWriter:
    """Writes one animated USD mesh (the draping garment, one time sample per
    FRAME) plus a static USD mesh (the target body) to a single stage, for
    inspecting a drape's frame-by-frame trajectory in usdview/Blender -- e.g.
    to check whether the garment is still sliding across a single drape call,
    not just between optimization iterations. Independent of Warp (the old
    warp_drape.py wrote USD via warp.sim.render.SimRenderer); uses pxr directly.
    """

    def __init__(self, path, garment_faces: np.ndarray, target_body: Optional[Mesh], fps: float):
        from pxr import Usd, UsdGeom

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.stage = Usd.Stage.CreateNew(str(path))
        UsdGeom.SetStageUpAxis(self.stage, UsdGeom.Tokens.y)
        UsdGeom.SetStageMetersPerUnit(self.stage, 0.01)  # this pipeline is cm-native
        self.stage.SetTimeCodesPerSecond(fps)
        self.stage.SetFramesPerSecond(fps)

        self.garment = UsdGeom.Mesh.Define(self.stage, "/garment")
        face_counts = [3] * len(garment_faces)
        self.garment.CreateFaceVertexCountsAttr(face_counts)
        self.garment.CreateFaceVertexIndicesAttr(garment_faces.reshape(-1).tolist())

        if target_body is not None:
            body = UsdGeom.Mesh.Define(self.stage, "/target_body")
            body.CreatePointsAttr([tuple(v) for v in target_body.vertices])
            body_face_counts = [3] * len(target_body.faces)
            body.CreateFaceVertexCountsAttr(body_face_counts)
            body.CreateFaceVertexIndicesAttr(np.asarray(target_body.faces).reshape(-1).tolist())

    def write_frame(self, frame: int, x: torch.Tensor) -> None:
        points = [tuple(v) for v in x.detach().cpu().numpy()]
        self.garment.GetPointsAttr().Set(points, time=frame)

    def close(self) -> None:
        self.stage.Save()


@dataclass
class XPBDConfig:
    tri_ke: float = 10000.0            # in-plane stretch (Cu, Cv) compliance
    tri_ka: float = 10000.0            # in-plane shear (Cshear) compliance
    tri_kd: float = 0.0                # shared stretch/shear damping -- zeroed (was 1.0):
                                        # paper's Eq. 4 only specifies velocity_damping
                                        # (tau); this per-constraint kd is an own addition
                                        # on top of that, zeroed so it doesn't muddy the
                                        # dt/tau timestep-semantics diagnosis (see
                                        # tests/tests_sim/probe_velocity_damping_substeps.py).
    bend_ke: float = 1.0
    bend_kd: float = 0.0               # was 10.0 (GarmentCode's value) -- zeroed, same
                                        # reasoning as tri_kd above.
    fabric_density: float = 1.0        # mass per unit area, GarmentCode units (cm)
    collision_thickness: float = 0.25  # cm -- default_sim_props.yaml body_collision_thickness
    friction_mu: float = 0.5           # GarmentCode's default_sim_props.yaml body_friction;
                                        # Coulomb friction coefficient for _solve_collision's
                                        # tangential correction. Tune empirically if needed.
    body_reference_drag_margin: float = 0.2  # cm -- GarmentCode's sim_config.py hardcoded
                                        # soft_contact_margin (not in default_sim_props.yaml).
                                        # _solve_body_reference_drag's hard-snap trigger:
                                        # signed_dist - collision_thickness < -this.
    gravity: Tuple[float, float, float] = (0.0, -9.81, 0.0)  # NOT physically-consistent cm/s^2
                                        # (that would be -981) -- GarmentCode's own Warp pipeline
                                        # applies real Earth-gravity magnitude directly to its
                                        # cm-scale garment coordinates (garment.py's own pre-
                                        # existing unit inconsistency, not fixed here), and every
                                        # constraint stiffness this module borrows from GarmentCode
                                        # (tri_ke, attachment_ke, ...) was tuned against THAT value.
                                        # Using the physically-correct -981 here (100x stronger)
                                        # was found to sink/slide garments far past the target body
                                        # -- not a genuine physics gap, just this mismatch.
    fps: float = 60.0
    substeps: int = 10
    solver_iters: int = 4               # inner Gauss-Seidel-across-constraint-types
                                        # passes per substep, each with persistent
                                        # lambda accumulation (Eq. 2) -- paper doesn't
                                        # specify a value (GarmentCode's own Warp XPBD
                                        # uses 1, relying on substeps alone -- see
                                        # module docstring); tune if needed.
    zero_gravity_steps: int = 10
    static_threshold: float = 0.03
    velocity_damping: float = 0.95     # tau, paper Eq. 4
    max_frames: int = 300
    max_velocity: float = 25.0         # cm/s -- default_sim_props.yaml global_max_velocity;
                                        # a numerical safety clamp (Warp's apply_particle_deltas
                                        # applies the same one), not part of the paper's Eq. 4
    reference_drag_ke: float = 200.0   # weak vs. tri_ke=10000 -- nudge, not override
    reference_drag_kd: float = 0.0     # was 5.0 -- zeroed, same reasoning as tri_kd above.
    stitch_ke: float = 1.0e5           # stiffer than tri_ke=10000 -- seams should
                                        # end up near-coincident, not just "pulled".
                                        # Own choice (see module docstring); no
                                        # paper/DiffXPBD value to match.
    stitch_kd: float = 0.0             # was 5.0 -- zeroed, same reasoning as tri_kd above.
    attachment_ke: float = 1000.0      # GarmentCode's default_sim_props.yaml
    attachment_kd: float = 10.0        # attachment_stiffness/damping[0] (lower_interface)
    attachment_frames: int = 400       # GarmentCode's attachment_frames default. NOTE this
                                        # exceeds this config's own max_frames=300 default --
                                        # with both left at their defaults the attachment
                                        # constraint stays active for the WHOLE drape rather
                                        # than being released partway through (GarmentCode's
                                        # own warm-up-then-release behavior needs max_frames
                                        # raised above attachment_frames, or attachment_frames
                                        # lowered, to reproduce exactly).


# ── fixed topology (built once from ``faces``; independent of vertex positions) ──

def _jacobi_divisor(n_vertices: int, *index_arrays, dtype, device) -> torch.Tensor:
    """``(V, 1)``: how many primitives of one constraint type touch each vertex
    (clamped to >= 1), used to AVERAGE that type's batched correction instead of
    summing it.

    Each of this module's constraint types solves its whole batch Jacobi-style
    against one ``x`` snapshot and ``index_add_``s every primitive's full
    correction into a shared ``delta`` (see the module docstring's
    constraint-solve note). Summing them over-relaxes by a vertex's valence: an
    interior cloth vertex is shared by ~6 triangles, so it moves ~6x as far as
    any one triangle's constraint asked for. That is the standard failure mode
    of a vectorized Jacobi PBD/XPBD batch, and the standard fix is the averaged
    (mass-splitting) Jacobi update of Macklin et al. 2014 "Unified Particle
    Physics for Real-Time Applications" Sec. 3 -- divide each vertex's
    accumulated delta by the number of constraints that wrote to it.

    Without this, the solver injects energy rather than dissipating it: on this
    repo's GarmentCode skirt, a drape whose only non-equilibrium term was the
    seam-closing stitch constraint inflated without bound (the garment's bbox
    grew from +-35 cm to +-96 cm over 200 frames, every particle pinned at
    ``config.max_velocity``) instead of sewing shut. With it, the same drape
    closes its 30 cm mean seam gap to <0.01 cm and stays bounded.
    """
    counts = torch.zeros(n_vertices, dtype=dtype, device=device)
    for idx in index_arrays:
        counts.index_add_(0, idx, torch.ones_like(idx, dtype=dtype))
    return counts.clamp_min(1.0)[:, None]


def _build_bending_quads(faces: np.ndarray) -> np.ndarray:
    """``(K, 4)`` ``(i, j, k, l)`` per interior edge: ``(k, l)`` is the shared edge,
    ``i``/``j`` the opposite vertices of its two triangles (Warp's ``bending_constraint``
    indexing). Boundary edges (one adjacent triangle) get no bending constraint."""
    edge_to_faces: Dict[Tuple[int, int], list] = {}
    for f_idx, f in enumerate(faces):
        for a, b in ((0, 1), (1, 2), (2, 0)):
            v1, v2 = int(f[a]), int(f[b])
            key = (min(v1, v2), max(v1, v2))
            edge_to_faces.setdefault(key, []).append(f_idx)

    quads = []
    for (k, l), face_ids in edge_to_faces.items():
        if len(face_ids) != 2:
            continue
        f1, f2 = face_ids
        opp1 = [v for v in faces[f1] if v not in (k, l)][0]
        opp2 = [v for v in faces[f2] if v not in (k, l)][0]
        quads.append((int(opp1), int(opp2), k, l))
    return np.asarray(quads, dtype=np.int64).reshape(-1, 4)


# ── mass / rest quantities (differentiable functions of the rest vertices) ──

def _particle_mass(vertices: torch.Tensor, faces: torch.Tensor, density: float) -> torch.Tensor:
    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    area = 0.5 * torch.linalg.cross(v1 - v0, v2 - v0).norm(dim=1)
    contrib = density * area / 3.0
    mass = torch.zeros(vertices.shape[0], dtype=vertices.dtype, device=vertices.device)
    mass.index_add_(0, faces[:, 0], contrib)
    mass.index_add_(0, faces[:, 1], contrib)
    mass.index_add_(0, faces[:, 2], contrib)
    return mass


def _dihedral_geometry(x: torch.Tensor, quads: torch.Tensor):
    """Shared geometry for the bending constraint -- unit face normals, shared-edge
    unit tangent/length, winding-consistent sign flip, and the dihedral angle."""
    i, j, k, l = quads[:, 0], quads[:, 1], quads[:, 2], quads[:, 3]
    x1, x2, x3, x4 = x[i], x[j], x[k], x[l]

    n1 = torch.linalg.cross(x3 - x1, x4 - x1)
    n2 = torch.linalg.cross(x4 - x2, x3 - x2)
    n1 = n1 / n1.norm(dim=1, keepdim=True).clamp_min(_EPS)
    n2 = n2 / n2.norm(dim=1, keepdim=True).clamp_min(_EPS)

    e = x4 - x3
    e_len = e.norm(dim=1)
    e_hat = e / e_len.clamp_min(_EPS)[:, None]

    cos_theta = (n1 * n2).sum(dim=1).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    flip = -torch.sign((torch.linalg.cross(n1, n2) * e).sum(dim=1))
    angle = torch.acos(cos_theta)

    return dict(x1=x1, x2=x2, x3=x3, x4=x4, n1=n1, n2=n2, e_hat=e_hat, e_len=e_len,
                flip=flip, angle=angle, i=i, j=j, k=k, l=l)


# ── per-substep constraint solves (Jacobi: each scatter-adds into ``delta``) ──

def _solve_triangle_strain(x, v, inv_mass, faces, material_uv, ke, ka, kd, dt,
                            lambda_u, lambda_v, lambda_shear, jacobi_divisor):
    """In-plane Green-strain membrane constraint -- DiffXPBD's own cloth
    formulation (orthotropic Saint Venant-Kirchhoff membrane): per triangle,
    ``F = D_s @ D_m^-1`` (deformation gradient from the material (u, v) frame
    to the current 3D positions) and Green strain ``E = 0.5*(F^T F - I)``, with
    constraint vector ``C = [E_uu, E_vv, 2*E_uv]``. ``material_uv`` is
    ``(F, 3, 2)``: each triangle corner's rest position in the panel's own 2D
    pattern (u, v) frame -- the differentiable optimization variable itself,
    not a derived 3D quantity. At the lifted rest state (x taken straight from
    the lift, before any dynamics) F = I exactly, so C = 0 there, as required.

    Replaces an earlier norm-based stretch/shear formulation (``Cu = |wu|-1``
    etc., Baraff & Witkin 1998, as Dress Anyone's own Eq. 7 names it) with
    DiffXPBD's Green-strain one (own choice per user instruction, not
    Dress Anyone's -- Dress Anyone's Eq. 7 doesn't specify a strain measure,
    just cites Baraff & Witkin by name). ``E_uu = 0.5*(wu.wu - 1)``,
    ``E_vv = 0.5*(wv.wv - 1)`` replace the old ``|wu|-1``/``|wv|-1`` (quadratic
    in the edge vectors instead of requiring a norm+normalize -- also removes
    the old ``n_u``/``n_v`` gradient-direction normalization and its ``_EPS``
    clamp entirely, since ``dE_uu/dwu = wu`` needs no normalization). Shear
    (``2*E_uv = wu.wv``) is algebraically the SAME expression as the old
    ``Cshear`` -- only its area/compliance scaling below changes.

    Area scaling: DiffXPBD's compliance is ``alpha_tilde^-1 = A * K`` against
    the UNSCALED strain ``C = E`` (its own Eq. in Sec. 4.3, area only enters
    the stiffness, not the constraint) -- NOT the old formulation's
    ``C = A * strain`` against a plain scalar ``ke``/``ka`` (equivalent to an
    elastic energy quadratic in area, ``U ~ (A*strain)^2``, vs. DiffXPBD's
    linear-in-area ``U ~ A*strain^2``: the old form made per-triangle stiffness
    implicitly mesh-resolution-dependent -- refining a triangulation at fixed
    ke/ka would stiffen it, not preserve macroscopic behavior). Implemented
    here by folding area into the XPBD compliance ``alpha`` (``1/(A*ke*dt^2)``
    instead of ``1/(ke*dt^2)``) while leaving ``C``/its gradients area-free;
    ``gamma = kd/(ke*dt)`` is deliberately NOT area-scaled (kd/ke are meant as
    the SAME per-unit-area moduli, so area cancels out of their ratio -- moot
    right now since kd=0, see XPBDConfig.tri_kd, but kept correct for when/if
    kd is reintroduced).

    ``lambda_u``/``lambda_v``/``lambda_shear`` are this constraint's per-triangle
    Lagrange multipliers, persisted and accumulated by the caller across
    ``config.solver_iters`` inner passes within one substep (Eq. 2); Cu, Cv,
    Cshear are solved Jacobi-style against the SAME ``x`` snapshot within one
    call (see module docstring's constraint-solve note on this granularity).
    Returns ``(x + delta, new_lambda_u, new_lambda_v, new_lambda_shear)``.
    """
    i, j, k = faces[:, 0], faces[:, 1], faces[:, 2]
    x0, x1, x2 = x[i], x[j], x[k]
    v0, v1, v2 = v[i], v[j], v[k]
    w0, w1, w2 = inv_mass[i], inv_mass[j], inv_mass[k]

    u0, p0 = material_uv[:, 0, 0], material_uv[:, 0, 1]
    u1, p1 = material_uv[:, 1, 0], material_uv[:, 1, 1]
    u2, p2 = material_uv[:, 2, 0], material_uv[:, 2, 1]
    du1, dv1 = u1 - u0, p1 - p0
    du2, dv2 = u2 - u0, p2 - p0
    det = du1 * dv2 - du2 * dv1

    dx1, dx2 = x1 - x0, x2 - x0
    wu = (dv2[:, None] * dx1 - dv1[:, None] * dx2) / det[:, None]
    wv = (du1[:, None] * dx2 - du2[:, None] * dx1) / det[:, None]
    area = (0.5 * det.abs()).clamp_min(_EPS)  # (F,) -- degenerate-triangle safety;
                                               # divides alpha below (unlike the old
                                               # area-multiplies-C formulation, this
                                               # one needs the clamp to avoid a 0/0).

    coeff_u0, coeff_u1, coeff_u2 = -(dv2 - dv1) / det, dv2 / det, -dv1 / det
    coeff_v0, coeff_v1, coeff_v2 = -(du1 - du2) / det, -du2 / det, du1 / det

    delta = torch.zeros_like(x)

    def _apply(C, grad0, grad1, grad2, alpha, gamma, lambda_prev):
        denom = (w0 * grad0.pow(2).sum(1) + w1 * grad1.pow(2).sum(1)
                  + w2 * grad2.pow(2).sum(1)).clamp_min(_EPS)
        grad_dot_v = dt * ((grad0 * v0).sum(1) + (grad1 * v1).sum(1) + (grad2 * v2).sum(1))
        dlambda = -(C + gamma * grad_dot_v + alpha * lambda_prev) / ((1.0 + gamma) * denom + alpha)
        delta.index_add_(0, i, (w0 * dlambda)[:, None] * grad0)
        delta.index_add_(0, j, (w1 * dlambda)[:, None] * grad1)
        delta.index_add_(0, k, (w2 * dlambda)[:, None] * grad2)
        return lambda_prev + dlambda

    alpha_ke = 1.0 / (area * ke * dt * dt)
    gamma_ke = kd / (ke * dt)
    alpha_ka = 1.0 / (area * ka * dt * dt)
    gamma_ka = kd / (ka * dt)

    Cu = 0.5 * ((wu * wu).sum(dim=1) - 1.0)
    lambda_u = _apply(Cu, coeff_u0[:, None] * wu, coeff_u1[:, None] * wu,
                       coeff_u2[:, None] * wu, alpha_ke, gamma_ke, lambda_u)

    Cv = 0.5 * ((wv * wv).sum(dim=1) - 1.0)
    lambda_v = _apply(Cv, coeff_v0[:, None] * wv, coeff_v1[:, None] * wv,
                       coeff_v2[:, None] * wv, alpha_ke, gamma_ke, lambda_v)

    Cshear = (wu * wv).sum(dim=1)
    lambda_shear = _apply(Cshear, coeff_u0[:, None] * wv + coeff_v0[:, None] * wu,
                           coeff_u1[:, None] * wv + coeff_v1[:, None] * wu,
                           coeff_u2[:, None] * wv + coeff_v2[:, None] * wu, alpha_ka, gamma_ka, lambda_shear)

    return x + delta / jacobi_divisor, lambda_u, lambda_v, lambda_shear


def _solve_reference_drag(x, v, inv_mass, target_positions, ke, kd, dt, lambda_):
    """Zero-rest-length compliant spring pulling each vertex toward its
    corresponding ``target_shape`` vertex (fixed, infinite-mass anchor) --
    same XPBD spring math as the triangle constraint's per-axis update, just
    against a constant target instead of a triangle. No scatter-add needed:
    the correspondence is already 1:1 (same topology as the rest mesh).
    ``lambda_`` is this constraint's per-vertex multiplier, persisted by the
    caller across solver_iters (Eq. 2). Returns ``(x + delta, new_lambda)``."""
    diff = x - target_positions
    dist = diff.norm(dim=1).clamp_min(_EPS)
    n = diff / dist[:, None]
    C = dist

    alpha = 1.0 / (ke * dt * dt)
    gamma = kd / (ke * dt)
    grad_c_dot_v = dt * (n * v).sum(dim=1)
    dlambda = -(C + gamma * grad_c_dot_v + alpha * lambda_) / ((1.0 + gamma) * inv_mass + alpha)

    delta = (inv_mass * dlambda)[:, None] * n
    return x + delta, lambda_ + dlambda


def _solve_attachment(x, v, inv_mass, indices, target_point, norm, ke, kd, dt, lambda_):
    """One-sided (inequality) attachment constraint -- GarmentCode's ``add_attachment``
    half-space pin (Warp's ``attachment_constraint`` kernel), ported to this module's
    own XPBD constraint style (see module docstring). Restricts ``indices`` to the
    half-space ``dot(x - target_point, norm) >= 0``: pushes a vertex back toward the
    plane only when it has crossed to the wrong side (``err < 0``); contributes
    nothing otherwise (differentiable one-sided gate via ``torch.where``, same idea as
    ``_solve_collision``'s ``clamp(-c, min=0.0)``). E.g. for GarmentCode's
    ``lower_interface`` label (a garment's waistband loop) with ``norm=(0, 1, 0)``,
    this stops the waistband sliding BELOW ``target_point``'s height, without pinning
    it in place otherwise -- meant to be active only during the drape's warm-up frames
    (``config.attachment_frames``), same as GarmentCode's own usage.

    ``lambda_`` is this constraint's per-labeled-vertex multiplier, persisted by the
    caller across solver_iters (Eq. 2); held at 0 while inactive (``active`` gates
    ``dlambda`` too, not just the delta, so lambda doesn't drift while off).
    Returns ``(x + delta, new_lambda)``.
    """
    xi, vi, wi = x[indices], v[indices], inv_mass[indices]
    err = ((xi - target_point) * norm).sum(dim=1)
    grad_dot_v = dt * (vi * norm).sum(dim=1)

    alpha = 1.0 / (ke * dt * dt)
    gamma = kd / (ke * dt)
    dlambda = -(err + gamma * grad_dot_v + alpha * lambda_) / ((1.0 + gamma) * wi + alpha)

    active = (err < 0)
    dlambda = torch.where(active, dlambda, torch.zeros_like(dlambda))
    delta_i = (wi * dlambda)[:, None] * norm[None, :]

    delta = torch.zeros_like(x)
    delta.index_add_(0, indices, delta_i)
    return x + delta, lambda_ + dlambda


def _solve_stitch(x, v, inv_mass, pair_a, pair_b, ke, kd, dt, lambda_, jacobi_divisor):
    """Zero-rest-length compliant distance constraint between corresponding seam
    vertex pairs (``pair_a``/``pair_b``, unwelded rest-mesh indices -- see
    rest_mesh_builder.stitch_pairs_from_weld_map). Same two-point XPBD distance
    constraint used throughout this module, just between two DYNAMIC vertices
    instead of a fixed target (cf. ``_solve_reference_drag``). This is a standard
    PBD/XPBD sewing technique (a generic Macklin et al. 2016 distance constraint
    applied to seam pairs), not something copied from GarmentCode/Warp -- see the
    module docstring for why this constraint's existence/formula is this module's
    own design choice (paper doesn't specify one).

    ``lambda_`` is this constraint's per-pair multiplier, persisted by the caller
    across solver_iters (Eq. 2). Returns ``(x + delta, new_lambda)``."""
    xa, xb = x[pair_a], x[pair_b]
    va, vb = v[pair_a], v[pair_b]
    wa, wb = inv_mass[pair_a], inv_mass[pair_b]

    diff = xa - xb
    dist = diff.norm(dim=1).clamp_min(_EPS)
    n = diff / dist[:, None]
    C = dist

    alpha = 1.0 / (ke * dt * dt)
    gamma = kd / (ke * dt)
    denom = (wa + wb).clamp_min(_EPS)
    grad_c_dot_v = dt * ((n * va).sum(dim=1) - (n * vb).sum(dim=1))
    dlambda = -(C + gamma * grad_c_dot_v + alpha * lambda_) / ((1.0 + gamma) * denom + alpha)

    delta = torch.zeros_like(x)
    delta.index_add_(0, pair_a, (wa * dlambda)[:, None] * n)
    delta.index_add_(0, pair_b, -(wb * dlambda)[:, None] * n)
    return x + delta / jacobi_divisor, lambda_ + dlambda


def _solve_bending(x, v, inv_mass, quads, rest_angle, ke, kd, dt, lambda_, jacobi_divisor):
    """Dihedral-angle bending constraint. ``lambda_`` is this constraint's
    per-quad multiplier, persisted by the caller across solver_iters (Eq. 2).
    Returns ``(x + delta, new_lambda)``."""
    geo = _dihedral_geometry(x, quads)
    n1, n2, e_hat, e_len, flip = geo["n1"], geo["n2"], geo["e_hat"], geo["e_len"], geo["flip"]
    x1, x2, x3, x4 = geo["x1"], geo["x2"], geo["x3"], geo["x4"]
    i, j, k, l = geo["i"], geo["j"], geo["k"], geo["l"]

    grad1 = n1 * (e_len * flip)[:, None]
    grad2 = n2 * (e_len * flip)[:, None]
    grad3 = (n1 * ((x1 - x4) * e_hat).sum(dim=1, keepdim=True)
             + n2 * ((x2 - x4) * e_hat).sum(dim=1, keepdim=True)) * flip[:, None]
    grad4 = (n1 * ((x3 - x1) * e_hat).sum(dim=1, keepdim=True)
             + n2 * ((x3 - x2) * e_hat).sum(dim=1, keepdim=True)) * flip[:, None]

    w1, w2, w3, w4 = inv_mass[i], inv_mass[j], inv_mass[k], inv_mass[l]
    C = geo["angle"] - rest_angle
    denom = (w1 * grad1.pow(2).sum(1) + w2 * grad2.pow(2).sum(1)
             + w3 * grad3.pow(2).sum(1) + w4 * grad4.pow(2).sum(1)).clamp_min(_EPS)

    alpha = 1.0 / (ke * dt * dt)
    gamma = kd / (ke * dt)
    grad_dot_v = dt * ((grad1 * v[i]).sum(1) + (grad2 * v[j]).sum(1)
                        + (grad3 * v[k]).sum(1) + (grad4 * v[l]).sum(1))
    dlambda = -(C + gamma * grad_dot_v + alpha * lambda_) / ((1.0 + gamma) * denom + alpha)

    delta = torch.zeros_like(x)
    delta.index_add_(0, i, (w1 * dlambda)[:, None] * grad1)
    delta.index_add_(0, j, (w2 * dlambda)[:, None] * grad2)
    delta.index_add_(0, k, (w3 * dlambda)[:, None] * grad3)
    delta.index_add_(0, l, (w4 * dlambda)[:, None] * grad4)
    return x + delta / jacobi_divisor, lambda_ + dlambda


def _body_collision_targets(x: torch.Tensor, body_tree: igl.AABB, body_vertices: np.ndarray,
                             body_faces: np.ndarray,
                             body_face_normals: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    """Closest point + outward normal per particle, refreshed once per FRAME
    (matching Warp's ``wp.sim.collide`` cadence), against a body AABB tree built
    once per drape call and reused every frame (the body mesh is static). The
    correspondence search itself is not differentiated -- a standard,
    frozen-correspondence contact-gradient approximation."""
    x_np = x.detach().cpu().numpy().astype(np.float64)
    _, closest_faces, closest_points = body_tree.squared_distance(body_vertices, body_faces, x_np)
    normals = body_face_normals[closest_faces]
    return (torch.from_numpy(np.ascontiguousarray(closest_points)).to(x.dtype),
            torch.from_numpy(np.ascontiguousarray(normals)).to(x.dtype))


def _solve_collision(x, x_prev, closest_points, normals, thickness, friction_mu) -> torch.Tensor:
    """One-sided closest-point push-out (normal correction) plus Coulomb friction
    (tangential correction), the standard PBD/XPBD contact-friction recipe
    (Macklin et al. 2016 "XPBD", Sec. 3.4 / the earlier "Unified Particle
    Physics" paper's friction step): once a particle is in contact (penetrating,
    ``c < 0``), its tangential slide THIS substep (``x - x_prev``, projected onto
    the contact plane -- ``x`` here is ``x_pred``, the pre-constraint predicted
    position, ``x_prev`` is the substep's starting position, matching every
    other constraint in this file which solves against ``x_pred``) is opposed:
    fully cancelled (static friction) if it's within the Coulomb cone
    (``|slide| <= mu * normal_correction``), otherwise capped at the cone's edge
    (dynamic friction). ``friction_mu`` matches GarmentCode's ``body_friction``
    (default_sim_props.yaml, default 0.5) -- torch_xpbd had NO friction at all
    before this (module docstring used to say so explicitly); this was found to
    be the actual reason a freely-falling flat-pattern rest mesh never conformed
    to the body's silhouette the way Warp's real drape does -- contact alone
    (frictionless push-out) lets touched cloth immediately slide back off again,
    while Warp's cloth "sticks" once it touches (confirmed empirically: Warp's
    final drape has ~25% of vertices sitting almost exactly at the collision
    thickness shell vs. ~9% for a frictionless torch_xpbd drape -- friction, not
    ``_solve_body_reference_drag``, turned out to be what that gap was)."""
    c = ((x - closest_points) * normals).sum(dim=1) - thickness
    normal_mag = torch.clamp(-c, min=0.0)  # push out only when penetrating (c < 0)
    normal_delta = normal_mag[:, None] * normals

    slide = x - x_prev
    slide = slide - (slide * normals).sum(dim=1, keepdim=True) * normals  # tangential component
    slide_mag = slide.norm(dim=1).clamp_min(_EPS)
    max_slide = friction_mu * normal_mag
    scale = torch.clamp(max_slide / slide_mag, max=1.0)
    friction_delta = torch.where((normal_mag > 0)[:, None], -slide * scale[:, None], torch.zeros_like(x))

    return normal_delta + friction_delta


def _solve_body_reference_drag(x, closest_points, normals, thickness, margin) -> torch.Tensor:
    """Warp's ``reference_shape_drag_constraint`` flag==2 branch ("body collision
    resolution -- drag to the closest point on the whole body"), see module
    docstring for the full derivation from Warp's source. Unlike
    ``_solve_collision``'s proportional spring, this is a binary hard snap: once
    a particle has crossed ``margin`` past the collision-thickness shell, it is
    moved FULLY onto that shell this substep (Warp's own dlambda for this branch
    has no ke/kd -- ``|err| / invmass``, which cancels to a full correction, not
    a partial one)."""
    c = ((x - closest_points) * normals).sum(dim=1) - thickness
    deep = (c < -margin)[:, None]
    return torch.where(deep, -c[:, None] * normals, torch.zeros_like(x))


# ── public entry point ──────────────────────────────────────────────────────

def simulate_drape(
    rest_vertices: torch.Tensor,
    faces: np.ndarray,
    target_body: Mesh,
    *,
    material_uv: torch.Tensor,
    stitch_pairs: Optional[np.ndarray] = None,
    target_shape: Optional[Mesh] = None,
    attachment_indices: Optional[np.ndarray] = None,
    attachment_target_point: Optional[Tuple[float, float, float]] = None,
    attachment_norm: Tuple[float, float, float] = (0.0, 1.0, 0.0),
    initial_vertices: Optional[torch.Tensor] = None,
    config: Optional[XPBDConfig] = None,
    max_frames: Optional[int] = None,
    usd_path: Optional[str] = None,
) -> torch.Tensor:
    """Drape ``rest_vertices`` (the rest mesh, a leaf-derived tensor requiring
    grad) on ``target_body`` until equilibrium, differentiably.

    ``material_uv`` is ``(F, 3, 2)`` -- see :func:`_solve_triangle_strain`;
    ``rest_mesh_builder.build_rest_mesh_from_pattern_torch`` returns the
    unwelded 2D vertices this is gathered from (same face order as ``faces``).

    ``stitch_pairs``, if given, is ``(K, 2)`` unwelded vertex-index pairs (see
    ``rest_mesh_builder.stitch_pairs_from_weld_map``): ``rest_vertices``/``faces``
    are expected to be the UNWELDED rest mesh (panels lifted to 3D but not
    merged at the seams -- see module docstring), and this constraint is what
    holds the seams together during the drape, solved every substep alongside
    the triangle strain/bending/collision constraints.

    ``target_shape``, if given, must share the rest mesh's exact vertex
    correspondence (Sec. 4.2's refit output does -- same topology as
    ``reference.draped_garment_mesh``/``box_mesh``, already relied on by
    ``losses.compute_target_shape_matching_loss``'s direct per-index diff). Used
    here as a weak drag target during the drape itself (see module docstring),
    not only as the loss target computed afterward.

    ``attachment_indices``, if given, are unwelded vertex indices held to the
    one-sided half-space ``dot(x - attachment_target_point, attachment_norm) >= 0``
    for the drape's first ``config.attachment_frames`` frames (see
    ``_solve_attachment``) -- GarmentCode's own warm-up pin, e.g. its
    ``vertex_labels.yaml``'s ``lower_interface`` label mapped through
    ``unwelded_to_welded`` onto this rest mesh's indexing. Prevents a garment
    from sliding off the body before collision alone can catch it.

    ``initial_vertices``, if given, is where the drape STARTS -- ``rest_vertices``
    then supplies only the rest quantities every constraint measures against
    (particle mass from its triangle areas, and the bending rest angles), not the
    initial state. They coincide by default. Separating them lets the rest state
    stay the true flat sewing pattern while the drape starts from an
    already-assembled layout (e.g. GarmentCode's ``*_boxmesh.obj``, whose seam
    welding stretches seam-adjacent edges by up to ~65x and would otherwise
    poison both derived quantities). Note the initial state is NOT differentiated
    w.r.t. ``rest_vertices`` when it is supplied separately -- Sec. 4.3's
    ``d(Delta x)/d(x_bar)`` then flows through the constraints only, so pass this
    for forward drapes and fixed-start experiments, not to introduce a second
    optimization variable.

    ``usd_path``, if given, writes a debug USD animation of the garment (one
    time sample per frame) alongside ``target_body`` -- open in usdview/Blender
    to inspect the within-one-drape trajectory (e.g. whether it's still
    settling/sliding at the end), not just before/after snapshots. Adds a
    detach + CPU roundtrip per frame, so leave it off for real optimization runs.

    Returns the final particle positions as a torch tensor still attached to the
    autograd graph rooted at ``rest_vertices`` -- call ``.backward(gradient=...)``
    on it (seeded with dL/d(simulated_vertices)) to get dL/d(rest_vertices).
    """
    config = config or XPBDConfig()
    dtype = rest_vertices.dtype
    device = rest_vertices.device

    faces_np = np.asarray(faces, dtype=np.int64)
    faces_t = torch.as_tensor(faces_np, dtype=torch.int64, device=device)
    quads_np = _build_bending_quads(faces_np)
    quads = torch.as_tensor(quads_np, dtype=torch.int64, device=device) if len(quads_np) else None

    stitch_a = stitch_b = None
    if stitch_pairs is not None and len(stitch_pairs) > 0:
        pairs_t = torch.as_tensor(np.asarray(stitch_pairs, dtype=np.int64), device=device)
        stitch_a, stitch_b = pairs_t[:, 0], pairs_t[:, 1]

    target_shape_verts = None
    if target_shape is not None:
        target_shape_verts = torch.as_tensor(target_shape.vertices, dtype=dtype, device=device)
        if target_shape_verts.shape != rest_vertices.shape:
            raise ValueError(
                f"target_shape has {target_shape_verts.shape[0]} vertices, expected "
                f"{rest_vertices.shape[0]} to match the rest mesh (Sec. 4.2's target_shape "
                "must share the rest mesh's topology)."
            )

    attachment_idx = attachment_point_t = attachment_norm_t = None
    if attachment_indices is not None and len(attachment_indices) > 0:
        if attachment_target_point is None:
            raise ValueError("attachment_target_point is required when attachment_indices is given")
        attachment_idx = torch.as_tensor(np.asarray(attachment_indices, dtype=np.int64), device=device)
        attachment_point_t = torch.tensor(attachment_target_point, dtype=dtype, device=device)
        attachment_norm_t = torch.tensor(attachment_norm, dtype=dtype, device=device)

    if initial_vertices is not None and initial_vertices.shape != rest_vertices.shape:
        raise ValueError(
            f"initial_vertices has shape {tuple(initial_vertices.shape)}, expected "
            f"{tuple(rest_vertices.shape)} to match the rest mesh."
        )

    n_vertices = rest_vertices.shape[0]
    tri_divisor = _jacobi_divisor(n_vertices, faces_t[:, 0], faces_t[:, 1], faces_t[:, 2],
                                   dtype=dtype, device=device)
    bend_divisor = (_jacobi_divisor(n_vertices, quads[:, 0], quads[:, 1], quads[:, 2], quads[:, 3],
                                     dtype=dtype, device=device) if quads is not None else None)
    stitch_divisor = (_jacobi_divisor(n_vertices, stitch_a, stitch_b, dtype=dtype, device=device)
                      if stitch_a is not None else None)

    mass = _particle_mass(rest_vertices, faces_t, config.fabric_density)
    inv_mass = 1.0 / mass.clamp_min(_EPS)

    rest_angle = _dihedral_geometry(rest_vertices, quads)["angle"] if quads is not None else None

    body_vertices = np.ascontiguousarray(target_body.vertices, dtype=np.float64)
    body_faces = np.ascontiguousarray(target_body.faces, dtype=np.int64)
    body_face_normals = igl.per_face_normals(body_vertices, body_faces, np.array([0.0, 0.0, 1.0]))
    body_tree = igl.AABB()
    body_tree.init(body_vertices, body_faces)

    gravity_vec = torch.tensor(config.gravity, dtype=dtype, device=device)
    zero_vec = torch.zeros_like(gravity_vec)
    dt = 1.0 / (config.fps * config.substeps)
    # config.velocity_damping (tau, paper Eq. 4) is defined at a 1/fps FRAME
    # timestep, not at this module's internal substep dt -- applying it once
    # per substep instead compounds tau^substeps of decay per frame, so the
    # SAME tau value damps harder in real time as substeps grows (confirmed
    # empirically: a constraint-free free-fall's terminal velocity scaled as
    # ~1/substeps -- see tests/tests_sim/probe_velocity_damping_substeps.py).
    # tau_sub is the per-substep damping whose `substeps`-fold composition
    # reproduces exactly tau at the frame rate: tau_sub**substeps == tau.
    # Own resolution of an ambiguity the paper's Eq. 4 doesn't address (it
    # doesn't describe a substep structure at all) -- flagged per CLAUDE.md.
    tau_sub = config.velocity_damping ** (1.0 / config.substeps)

    n_faces = faces_t.shape[0]
    n_quads = quads.shape[0] if quads is not None else 0
    n_stitch = stitch_a.shape[0] if stitch_a is not None else 0
    n_attach = attachment_idx.shape[0] if attachment_idx is not None else 0

    def _run_frame(x, v, closest_points, normals, gravity, attachment_active):
        """One frame's substeps (fixed collision correspondence throughout).
        Wrapped in ``torch.utils.checkpoint`` below -- with thousands of
        substeps unrolled over a real drape, keeping every substep's
        intermediates for backward exhausts memory; checkpointing recomputes
        this block during backward instead of storing it, at no cost to the
        computed gradient (it recomputes the exact same forward).

        Per substep: predict ONCE, then run ``config.solver_iters`` inner
        Gauss-Seidel-across-constraint-types passes with persistent lambda
        (Eq. 2) -- see module docstring's constraint-solve note."""
        for _ in range(config.substeps):
            v_pred = v + dt * gravity[None, :]
            x_cur = x + dt * v_pred

            lambda_u = torch.zeros(n_faces, dtype=dtype, device=device)
            lambda_v_ = torch.zeros(n_faces, dtype=dtype, device=device)
            lambda_shear = torch.zeros(n_faces, dtype=dtype, device=device)
            lambda_bend = torch.zeros(n_quads, dtype=dtype, device=device) if quads is not None else None
            lambda_refdrag = torch.zeros(x.shape[0], dtype=dtype, device=device) if target_shape_verts is not None else None
            lambda_stitch = torch.zeros(n_stitch, dtype=dtype, device=device) if stitch_a is not None else None
            lambda_attach = torch.zeros(n_attach, dtype=dtype, device=device) if attachment_idx is not None else None

            for _ in range(config.solver_iters):
                x_cur, lambda_u, lambda_v_, lambda_shear = _solve_triangle_strain(
                    x_cur, v_pred, inv_mass, faces_t, material_uv,
                    config.tri_ke, config.tri_ka, config.tri_kd, dt,
                    lambda_u, lambda_v_, lambda_shear, tri_divisor)
                if quads is not None:
                    x_cur, lambda_bend = _solve_bending(x_cur, v_pred, inv_mass, quads, rest_angle,
                                                          config.bend_ke, config.bend_kd, dt, lambda_bend,
                                                          bend_divisor)
                x_cur = x_cur + _solve_collision(x_cur, x, closest_points, normals,
                                                  config.collision_thickness, config.friction_mu)
                x_cur = x_cur + _solve_body_reference_drag(x_cur, closest_points, normals,
                                                            config.collision_thickness,
                                                            config.body_reference_drag_margin)
                if target_shape_verts is not None:
                    x_cur, lambda_refdrag = _solve_reference_drag(
                        x_cur, v_pred, inv_mass, target_shape_verts,
                        config.reference_drag_ke, config.reference_drag_kd, dt, lambda_refdrag)
                if stitch_a is not None:
                    x_cur, lambda_stitch = _solve_stitch(x_cur, v_pred, inv_mass, stitch_a, stitch_b,
                                                           config.stitch_ke, config.stitch_kd, dt, lambda_stitch,
                                                           stitch_divisor)
                if attachment_idx is not None and attachment_active:
                    x_cur, lambda_attach = _solve_attachment(
                        x_cur, v_pred, inv_mass, attachment_idx,
                        attachment_point_t, attachment_norm_t,
                        config.attachment_ke, config.attachment_kd, dt, lambda_attach)

            x_new = x_cur
            v_new = (tau_sub / dt) * (x_new - x)

            # Numerical safety clamp (matches Warp's apply_particle_deltas v_max):
            # cap the velocity magnitude and pull x_new back in accordance with it,
            # so a large single-step collision correction cannot blow up the state.
            v_mag = v_new.norm(dim=1, keepdim=True).clamp_min(_EPS)
            too_fast = v_mag > config.max_velocity
            v_capped = torch.where(too_fast, v_new * (config.max_velocity / v_mag), v_new)
            x_capped = torch.where(too_fast, x + v_capped * dt, x_new)

            v = v_capped
            x = x_capped
        return x, v

    x = rest_vertices if initial_vertices is None else initial_vertices.to(
        dtype=dtype, device=device)
    v = torch.zeros_like(x)
    n_frames = max_frames if max_frames is not None else config.max_frames
    prev_x_np = None

    usd_writer = _USDDrapeWriter(usd_path, faces_np, target_body, config.fps) if usd_path else None
    if usd_writer is not None:
        usd_writer.write_frame(0, x)

    for frame in range(n_frames):
        gravity = gravity_vec if frame >= config.zero_gravity_steps else zero_vec
        closest_points, normals = _body_collision_targets(
            x, body_tree, body_vertices, body_faces, body_face_normals)

        attachment_active = frame < config.attachment_frames
        x, v = checkpoint(_run_frame, x, v, closest_points, normals, gravity, attachment_active,
                           use_reentrant=False)

        if usd_writer is not None:
            usd_writer.write_frame(frame + 1, x)

        if frame >= config.zero_gravity_steps:
            cur = x.detach().cpu().numpy()
            if not np.isfinite(cur).all():
                raise AssertionError(f"drape diverged (NaN/Inf) at frame {frame}")
            if prev_x_np is not None and np.abs(cur - prev_x_np).sum(axis=1).max() < config.static_threshold:
                break
            prev_x_np = cur

    if usd_writer is not None:
        usd_writer.close()

    return x
