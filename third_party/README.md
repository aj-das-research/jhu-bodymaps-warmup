# third_party

External code used by the warm-up, kept **inside this project** (not a sibling repo).

| Path | Source | How it gets here |
|------|--------|------------------|
| `SuPreM/` | https://github.com/MrGiovanni/SuPreM | `bash scripts/setup_env_hpc.sh` (git clone; gitignored) |
| `ShapeKit/` | https://github.com/BodyMaps/ShapeKit | `bash scripts/setup_shapekit_hpc.sh` (git clone; gitignored) |

Do not commit the clones. Checkpoint weights live in `data/` and are symlinked into `SuPreM/direct_inference/pretrained_checkpoints/` by the setup script.

ShapeKit uses a **separate** conda env (`shapekit`) so its deps cannot disturb the pinned `suprem` stack. After clone, setup copies `configs/shapekit_vertebrae.yaml` over ShapeKit's `config.yaml` (vertebrae-only; affine reference = `vertebrae_L1.nii.gz` instead of upstream `liver.nii.gz`).
