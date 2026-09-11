# diff-cloth-sim

Cloth-drape experiments on [GarmentCode](https://github.com/maria-korosteleva/GarmentCode)
sewing patterns. Each subdirectory is a self-contained simulator reading the same
inputs from `inputs/`:

| Directory | Simulator | Entry point |
|---|---|---|
| `torchXPBD/` | XPBD (Macklin et al. 2016), pure differentiable PyTorch | `run_torch_xpbd.py` |
| `SemiImplicit/` | Newton `SolverSemiImplicit` (explicit integration, structural springs) | `drape_semi_implicit_unwelded.py` |
| `Refitting/` | garment refitting to a new body | `refit_garment.py` |

The two use **different conda envs** — see below.

---

## torchXPBD: forward drape

Reproduces [*XPBD: Position-Based Simulation of Compliant Constrained
Dynamics*](https://dl.acm.org/doi/pdf/10.1145/2994258.2994272) (Macklin, Müller,
Chentanez 2016) with GarmentCode data as input: the sample
`generated_rand_2E2EL4UZUS` (a skirt + waistband, 4 panels) draped on its own
source body `01709_straight`.

Pure forward pass — no retargeting, no optimization, no gradient, and no
`target_shape`. The fitted silhouette comes only from the simulator's own
physics: stitching, in-plane membrane strain, bending, a waistband pin, gravity,
and frictional body collision.

### 1. Set up the conda env

The env is exported in [`environment.yml`](environment.yml) (a full pinned export 
of the `garmentcode` env on this machine).

**On Oscar (Brown CCV):**

```bash
module load miniforge3/25.3.0-3
eval "$(conda shell.bash hook)"

conda env create -f environment.yml      # creates an env named `garmentcode`
conda activate garmentcode
```


### 2. Run the pipeline

```bash
conda activate garmentcode
python torchXPBD/run_torch_xpbd.py
```

### 3. Outputs

Written to `torchXPBD/outputs/`:

| File | What |
|---|---|
| `torch_xpbd_forward.usd` | animated drape (one time sample per frame) + static body — open in usdview or Blender |
| `final_drape.obj` | final garment, welded back into `*_sim.obj`'s vertex numbering |
| `metrics.json` | seam closure, per-vertex distance to GarmentCode's own drape, timings, bbox |

Reference numbers for the default run (300 frames, `--init flat`):

### 4. Inputs

All from `inputs/generated_rand_2E2EL4UZUS/`, except the body:

| File | Used for |
|---|---|
| `*_panels_2d.npz` | per-panel 2D pattern meshes, kept **unwelded** (seam duplicates separate — torch_xpbd's native input, panels held together by the stitch constraint) + `unwelded_to_welded` |
| `*_specification.json` | per-panel 3D placement (translation / rotation) used to lift the flat panels into 3D |
| `*_vertex_labels.yaml` | `lower_interface`, the waistband's top loop — GarmentCode's own attachment ("warm-up pin") label |
| `*_body_measurements.yaml` | `_waist_level` (cm) and the source body name |
| `*_boxmesh.obj` | `--init box` only |
| `*_sim.obj` | ground-truth reference for `metrics.json` only — never fed to the simulator |
| `inputs/5000_body_shapes_and_measures/meshes/01709_straight.obj` | collision body. **In meters**, converted to the cm-native pipeline on load |

To run a different sample, change `SAMPLE` at the top of `run_torch_xpbd.py`.

---

## SemiImplicit: a different env

`SemiImplicit/` imports `newton`, which the `garmentcode` env does not have.
Use the `diffsim` env for those scripts:

```bash
module load miniforge3/25.3.0-3 cuda git-lfs
eval "$(conda shell.bash hook)"
conda activate diffsim
python SemiImplicit/drape_semi_implicit_unwelded.py
```
