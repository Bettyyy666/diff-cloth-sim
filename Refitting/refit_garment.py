#!/usr/bin/env python
"""Garment refitting example built on the garment-refitting submodule.

Drapes the generated_rand_2E2EL4UZUS garment (originally simulated on body
01709_straight, bundled alongside it in the submodule's data/rand_2E2EL4UZUS/
directory -- the refitting algorithm's affine stencils are computed relative
to that exact source body) onto target body 00614_apart via de Goes, Fong &
O'Malley's relaxation/rebinding pipeline (SIGGRAPH Talks 2020).

Run with the dedicated `garment-refitting` conda env (Python 3.12, torch +
libigl + cholespy):
  <conda envs dir>/garment-refitting/bin/python Refitting/refit_garment.py
"""

import argparse
from pathlib import Path

import igl
import numpy as np
import torch

from refitting.manager import GarmentRefittingManager

REPO_ROOT = Path(__file__).resolve().parent.parent
SUBMODULE_ROOT = REPO_ROOT / "garment-refitting"

DEFAULT_GARMENT = REPO_ROOT / "inputs/generated_rand_2E2EL4UZUS/generated_rand_2E2EL4UZUS_sim.obj"
DEFAULT_SOURCE_BODY = SUBMODULE_ROOT / "data/rand_2E2EL4UZUS/01709_straight.obj"
DEFAULT_TARGET_BODY = REPO_ROOT / "inputs/5000_body_shapes_and_measures/meshes/00614_apart.obj"
DEFAULT_OUT = Path(__file__).resolve().parent / "outputs_refitting" / "generated_rand_2E2EL4UZUS_to_00614.obj"

BODY_SCALE_TO_CM = 100.0  # body objs are stored in meters; garment obj/ply are already centimeters


def load_mesh(path, scale=1.0):
    vertices, faces = igl.read_triangle_mesh(str(path))
    vertices = torch.as_tensor(np.asarray(vertices * scale, dtype=np.float32))
    faces = torch.as_tensor(np.asarray(faces, dtype=np.int32))
    return vertices, faces


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--garment", default=str(DEFAULT_GARMENT))
    parser.add_argument("--source-body", default=str(DEFAULT_SOURCE_BODY))
    parser.add_argument("--target-body", default=str(DEFAULT_TARGET_BODY))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument(
        "--tightness-weight",
        type=float,
        default=0.02,
        help="Relaxation weight pulling each garment vertex toward its rebound candidate position. "
        "Lower values let loose fabric (e.g. a skirt hem) relax more freely instead of being pulled "
        "toward an ambiguous closest-body-point reassignment each rebinding step. The library default "
        "(0.1) visibly distorts this garment's hem -- a sweep over tightness_weight found 0.02 "
        "(combined with distance_weight_alpha below) removes the hem-edge-length outliers that default "
        "produces (edges stretching up to 7x the source's ~1cm hem spacing) while still keeping the "
        "waist snugly bound to the body.",
    )
    parser.add_argument(
        "--distance-weight-alpha",
        type=float,
        default=1.0,
        help="Decays tightness_weight by 1/(1 + alpha * initial_distance_to_body), so vertices that "
        "start further from the body (more prone to ambiguous rebinding, e.g. a loose hem) are relaxed "
        "more freely than vertices already close to the body. 0 disables the decay.",
    )
    args = parser.parse_args()

    garment_vertices, garment_faces = load_mesh(args.garment)
    source_body_vertices, source_body_faces = load_mesh(args.source_body, scale=BODY_SCALE_TO_CM)
    target_body_vertices, target_body_faces = load_mesh(args.target_body, scale=BODY_SCALE_TO_CM)

    print(f"Garment:     {garment_vertices.shape[0]} verts, {garment_faces.shape[0]} tris  ({args.garment})")
    print(f"Source body: {source_body_vertices.shape[0]} verts, {source_body_faces.shape[0]} tris  ({args.source_body})")
    print(f"Target body: {target_body_vertices.shape[0]} verts, {target_body_faces.shape[0]} tris  ({args.target_body})")

    manager = GarmentRefittingManager(
        garment_vertices,
        garment_faces,
        source_body_vertices,
        source_body_faces,
        target_body_vertices,
        target_body_faces,
        tightness_weight=args.tightness_weight,
        distance_weight_alpha=args.distance_weight_alpha,
    )
    refit_vertices = manager.refit()
    for stats in manager.history:
        print(
            f"iter {stats.iteration_index}: max_move={stats.max_movement:.4f}cm "
            f"mean_move={stats.mean_movement:.4f}cm "
            f"closest_dist=[{stats.closest_distance_min:.4f}, {stats.closest_distance_mean:.4f}, "
            f"{stats.closest_distance_max:.4f}]cm converged={stats.converged}"
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    igl.write_triangle_mesh(
        str(out_path),
        refit_vertices.numpy().astype(np.float64),
        garment_faces.numpy().astype(np.int64),
    )
    print(f"Refit garment written to {out_path}")


if __name__ == "__main__":
    main()
