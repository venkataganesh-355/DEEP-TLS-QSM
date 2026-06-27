# DEEP-TLS-QSM
Official PyTorch implementation of training-free 3D quantitative susceptibility mapping using Deep Image Prior and Sparse Total Least Squares, with reproducible hyperparameters and SNU, LPCNN Datasets.


Research code for training-free three-dimensional quantitative susceptibility
mapping (QSM) with a deep image prior (DIP) and sparse total least squares
(STLS) refinement. This repository accompanies an IEEE manuscript and is
organized to make the reported reconstruction settings auditable.

## Repository contents

- `recon_snu_noise_tls.py`: command-line reconstruction entry point.
- `models/dip_nets.py`: three-dimensional DIP encoder-decoder.
- `utils/handy.py`: QSM operators and NIfTI/NumPy volume I/O.
- `DATASETS.md`: SNU and LPCNN provenance, versions, and access conditions.

No patient or restricted research data are included.

## Installation

The reference environment used to validate this release is Python 3.12.9,
PyTorch 2.6.0, NumPy 2.4.6, and NiBabel 5.3.3.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install a CUDA-specific PyTorch wheel from the official PyTorch instructions
when GPU execution is required.

## Input

The program accepts one three-dimensional `.npy`, `.nii`, or `.nii.gz` volume.
NIfTI voxel size and affine metadata are reused automatically. For NumPy input,
provide voxel size explicitly with `--vox`; otherwise `1 1 1` is used.

Input values outside the object mask must be zero. Use `--is_field` when the
input is already a local field map. Without it, the input is interpreted as a
susceptibility map and converted to a field map before inversion.

## Run

Minimal CPU example:

```bash
python recon_snu_noise_tls.py \
  --data_path data/example.npy \
  --is_field \
  --vox 1 1 1 \
  --device cpu \
  --save_path results/example
```

Run `python recon_snu_noise_tls.py --help` for every configurable setting.
Each run writes reconstructed `chi_*.nii.gz` volumes, estimated
`delta_*.nii.gz` volumes, `best_model.pth`, and `run_metadata.json`.

## Reproducibility

The manuscript experiments use the following reconstruction settings. The
input and output paths change for each dataset or subject.

| Parameter | Command-line option | Baseline | Noise experiment |
| --- | --- | --- | --- |
| Input volume | `--data_path` | dataset-specific | dataset-specific |
| Output directory | `--save_path` | experiment-specific | experiment-specific |
| Field direction | `--z_prjs` | `0 0 1` | `0 0 1` |
| Voxel size | `--vox` | `1 1 1` | `1 1 1` |
| Input type | `--input_type` | `phi` | `phi` |
| Input is a field map | `--is_field` | enabled | enabled |
| DIP optimization index maximum | `--epoch_num` | `500` | `500` |
| Learning rate | `--lr` | `0.001` | `0.001` |
| Network depth | `--depth` | `1` | `1` |
| Base channels | `--base` | `16` | `16` |
| Decoder blocks | `--decoder_block_num` | `1` | `1` |
| STLS outer iterations | `--stls_iter` | `10` | `10` |
| STLS regularization | `--stls_lambda` | `0.001` | `0.001` |
| Dipole-noise model | `--noisy_dipole` | enabled | enabled |
| Dipole-noise level | `--noise_level` | `0.00` | `0.05` |
| Random seed | `--seed` | `3407` | `3407` |

Other optimizer settings retain the implementation defaults: Adam
`beta1=0.5`, `beta2=0.999`, `epsilon=1e-9`, and weight decay `5e-4`. The
StepLR scheduler uses a step size of `20` and decay factor `0.8`. STLS uses a
convergence tolerance of `1e-5`, `10` inner steps, and an inner learning-rate
scale of `0.1`. The TKD threshold is `0.125`, gradient-loss order is `2`, and
padding mode is `half`.

The implementation follows the original experiment loop and evaluates indices
0 through `epoch_num`, inclusive. Therefore, `--epoch_num 500` performs 501 DIP
updates. Each run writes its complete arguments, input SHA-256 checksum,
resolved voxel size, software versions, and hardware details to
`run_metadata.json`.

### Baseline command

```bash
python recon_snu_noise_tls.py \
  --data_path path/to/input_volume.npy \
  --save_path results/baseline \
  --z_prjs 0 0 1 \
  --vox 1 1 1 \
  --input_type phi \
  --is_field \
  --epoch_num 500 \
  --lr 0.001 \
  --depth 1 \
  --base 16 \
  --decoder_block_num 1 \
  --stls_iter 10 \
  --stls_lambda 0.001 \
  --noisy_dipole \
  --noise_level 0.00 \
  --seed 3407
```

### Noise command

```bash
python recon_snu_noise_tls.py \
  --data_path path/to/input_volume.npy \
  --save_path results/noise_005 \
  --z_prjs 0 0 1 \
  --vox 1 1 1 \
  --input_type phi \
  --is_field \
  --epoch_num 500 \
  --lr 0.001 \
  --depth 1 \
  --base 16 \
  --decoder_block_num 1 \
  --stls_iter 10 \
  --stls_lambda 0.001 \
  --noisy_dipole \
  --noise_level 0.05 \
  --seed 3407
```

Replace `path/to/input_volume.npy` and `--save_path` for each experiment. The
same commands also accept `.nii` and `.nii.gz` input volumes.

## Citation

Please cite the associated IEEE paper when using this code. Add the final paper
title, author list, DOI, and public GitHub URL here once those bibliographic
details are assigned.

## Data and licensing

Dataset access and links for SNU, LPCNN, and the QSM 2016 RC-1 challenge are
documented in `DATASETS.md`. A software license has not been selected.
The copyright holder should add an approved license before inviting reuse
beyond peer-review reproducibility.
