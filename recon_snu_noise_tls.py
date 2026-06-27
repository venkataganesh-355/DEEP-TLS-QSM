import argparse
import hashlib
import json
import os
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.fft as fft
import torch.nn as nn
import torch.nn.functional as F

from models.dip_nets import DIPNet
from utils.handy import (
    calculate_3d_bounding_box,
    dipole_convolution_f,
    forward_field_calc,
    generate_dipole,
    load_volume,
    norm,
    save_tensor_as_nii,
    truncate_qsm,
)


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def add_diagonal_noise(dipole, noise_level=0.01):
    noisy_dipole = dipole.clone()
    noise        = noise_level * torch.randn_like(dipole)
    cone_mask    = torch.abs(dipole) < 0.1
    noisy_dipole[cone_mask]  = dipole[cone_mask]  + noise[cone_mask]  * 10
    noisy_dipole[~cone_mask] = dipole[~cone_mask] + noise[~cone_mask]
    print(f"[INFO] Cone region voxels: {cone_mask.sum().item()} / {dipole.numel()}")
    return noisy_dipole


def tv_3d(x):
    dx = x[:, :, 1:, :, :] - x[:, :, :-1, :, :]
    dy = x[:, :, :, 1:, :] - x[:, :, :, :-1, :]
    dz = x[:, :, :, :, 1:] - x[:, :, :, :, :-1]
    dx = F.pad(dx, [0, 0, 0, 0, 0, 1])
    dy = F.pad(dy, [0, 0, 0, 1, 0, 0])
    dz = F.pad(dz, [0, 1, 0, 0, 0, 0])
    return torch.sum(torch.sqrt(dx**2 + dy**2 + dz**2 + 1e-8))


def update_delta_sherman_morrison(b, chi, dipole_padded, padding, mask):
    px, py, pz = padding
    chi_padded = F.pad(chi, [pz, pz, py, py, px, px], mode='circular')
    Ax = torch.real(fft.ifftn(dipole_padded * fft.fftn(chi_padded)))
    Ax = Ax[:, :, px:-px, py:-py, pz:-pz] * mask
    r         = b - Ax
    x_flat    = chi.reshape(-1)
    xTx       = torch.dot(x_flat, x_flat)
    sm_factor = 1.0 / (1.0 + xTx)
    R             = fft.fftn(r)
    X             = fft.fftn(chi)
    raw_outer     = R * torch.conj(X)
    sm_correction = (xTx * sm_factor) * raw_outer
    delta_new     = (raw_outer - sm_correction).real / x_flat.numel()
    return delta_new


def stls_step2_refine(chi_init, b, dipole_padded, delta, padding, mask,
                      stls_lambda, inner_lr=1e-3, inner_steps=10):
    chi = chi_init.detach().clone()
    chi.requires_grad_(True)
    inner_optim = torch.optim.Adam([chi], lr=inner_lr)
    px, py, pz = padding
    for _ in range(inner_steps):
        inner_optim.zero_grad()
        chi_padded   = F.pad(chi,   [pz, pz, py, py, px, px], mode='circular')
        delta_padded = F.pad(delta, [pz, pz, py, py, px, px], mode='constant', value=0)
        corrected    = dipole_padded + delta_padded
        b_bar = torch.real(fft.ifftn(corrected * fft.fftn(chi_padded)))
        b_bar = b_bar[:, :, px:-px, py:-py, pz:-pz] * mask
        loss  = F.mse_loss(b_bar, b, reduction='sum') + stls_lambda * tv_3d(chi * mask)
        loss.backward()
        inner_optim.step()
    return chi.detach() * mask


def sparse_tls(chi_init, b, dipole_padded, padding, mask,
               stls_iter, stls_lambda, tol=1e-5,
               inner_lr=1e-3, inner_steps=10):
    chi   = chi_init.detach().clone()
    delta = torch.zeros_like(chi)
    print(f"  [STLS] Starting outer loop, max_iter={stls_iter}, lambda={stls_lambda}")
    for j in range(stls_iter):
        chi = stls_step2_refine(
            chi, b, dipole_padded, delta, padding, mask,
            stls_lambda, inner_lr=inner_lr, inner_steps=inner_steps
        )
        delta_new  = update_delta_sherman_morrison(b, chi, dipole_padded, padding, mask)
        delta_diff = torch.norm(delta_new - delta).item()
        delta      = delta_new
        print(f"  [STLS] iter {j+1}/{stls_iter} | delta_diff: {delta_diff:.6f}")
        if delta_diff < tol:
            print(f"  [STLS] Converged at iteration {j+1}")
            break
    return chi, delta


def resolve_device(device_name):
    if device_name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device_name == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but no CUDA device is available.')
    return torch.device(device_name)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_run_metadata(args, save_path, device, voxel_size):
    metadata = {
        'command': [sys.executable, *sys.argv],
        'arguments': vars(args),
        'input_sha256': file_sha256(args.data_path),
        'resolved_device': str(device),
        'resolved_voxel_size': list(voxel_size),
        'software': {
            'python': platform.python_version(),
            'numpy': np.__version__,
            'pytorch': torch.__version__,
        },
        'hardware': {
            'platform': platform.platform(),
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda,
            'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
        },
    }
    with (save_path / 'run_metadata.json').open('w', encoding='utf-8') as file:
        json.dump(metadata, file, indent=2)


def main(args):
    seed_torch(args.seed)
    device = resolve_device(args.device)
    save_path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    data, input_affine, input_voxel_size = load_volume(args.data_path)
    data = data.to(device)
    voxel_size = tuple(args.vox or input_voxel_size or (1.0, 1.0, 1.0))
    write_run_metadata(args, save_path, device, voxel_size)

    if args.scale_data != 1.0:
        data = data * args.scale_data
        print(f"[INFO] Data scaled by {args.scale_data:.6f} | "
              f"new range: {data.min().item():.4f} to {data.max().item():.4f}")

    if args.crop_background:
        shape = data.shape[2:]
        bbox = calculate_3d_bounding_box(data)
        if bbox is None:
            raise ValueError('Cannot crop an all-zero input volume.')
        crop_size = [bbox[4], shape[2]-bbox[5], bbox[2], shape[1]-bbox[3],
                     bbox[0], shape[0]-bbox[1]]
        data      = data[:, :, bbox[0]:bbox[1], bbox[2]:bbox[3], bbox[4]:bbox[5]]

    mask = torch.zeros_like(data)
    mask[data != 0] = 1
    if not torch.any(mask):
        raise ValueError('Input volume contains no non-zero voxels.')

    dipole = generate_dipole(data.shape, args.z_prjs, voxel_size, device=device)

    if not args.is_field:
        data = forward_field_calc(
            data, z_prjs=args.z_prjs, vox=voxel_size, need_padding=True, tpe='kspace'
        ) * mask

    tkd = truncate_qsm(data, dipole, ts=args.tkd_threshold)[0]

    if args.input_type == 'pure':
        input_data = forward_field_calc(
            tkd, z_prjs=[0, 0, 1], vox=voxel_size, tpe='kspace', need_padding=True
        ) * mask
    elif args.input_type == 'noise':
        input_data = torch.rand_like(data)
    elif args.input_type == 'tkd':
        input_data = tkd
    else:
        input_data = data

    model = DIPNet(depth=args.depth, base=args.base, decoder_block_num=args.decoder_block_num,
                   encoder_norm=nn.Identity, norm=nn.InstanceNorm3d, use_skip=False).to(device)

    optim = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optim, step_size=args.step, gamma=args.gamma
    )

    _, _, xx, yy, zz = data.shape
    if args.padding_mode == 'none':
        padding = 2, 2, 2
    elif args.padding_mode == 'half':
        padding = xx//2, yy//2, zz//2
    elif args.padding_mode == 'full':
        padding = xx, yy, zz
    else:
        raise ValueError('padding mode not supported')

    crit = nn.L1Loss(reduction='sum')

    px, py, pz    = padding
    _, _, x, y, z = data.shape

    dipole_padded = generate_dipole(
        ((x+2*px), (y+2*py), (z+2*pz)),
        z_prjs=args.z_prjs, vox=voxel_size, device=device
    ).unsqueeze(0).unsqueeze(0)

    if args.noisy_dipole:
        dipole        = add_diagonal_noise(dipole,        noise_level=args.noise_level)
        dipole_padded = add_diagonal_noise(dipole_padded, noise_level=args.noise_level)
        print(f"[INFO] Added {args.noise_level*100:.1f}% cone noise to dipole kernel")

    model.train()

    start_time = time.time()
    best_loss  = float('inf')

    for epoch in range(args.epoch_num + 1):

        optim.zero_grad()
        pred_chi = model(input_data) * mask

        dc = dipole_convolution_f(F.pad(pred_chi, [pz, pz, py, py, px, px], mode='circular'), dipole_padded)
        dc = dc[:, :, px:-px, py:-py, pz:-pz] * mask

        loss = crit(dc, data) + norm(data, pred_chi, dc, args.grad_loss_order)

        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(model.state_dict(), save_path / 'best_model.pth')

        loss.backward()
        optim.step()
        scheduler.step()

        if epoch % args.interval == 0:

            print({'epoch': epoch, 'lr_rate': optim.param_groups[0]['lr'],
                   'loss': loss.item(), 'time': int(time.time() - start_time)})

            chi_stls, delta_final = sparse_tls(
                chi_init      = pred_chi.detach(),
                b             = data,
                dipole_padded = dipole_padded,
                padding       = padding,
                mask          = mask,
                stls_iter     = args.stls_iter,
                stls_lambda   = args.stls_lambda,
                tol           = args.stls_tol,
                inner_lr      = args.lr * args.stls_inner_lr_scale,
                inner_steps   = args.stls_inner_steps
            )

            with torch.no_grad():
                mean     = chi_stls.sum() / (chi_stls != 0).sum()
                chi_stls = (chi_stls - mean) * mask

                if args.crop_background:
                    chi_stls = F.pad(chi_stls, crop_size)

                save_tensor_as_nii(
                    chi_stls, save_path / f'chi_{epoch}', voxel_size, input_affine
                )

                delta_save = delta_final if delta_final.dim() == 5 \
                             else delta_final.unsqueeze(0).unsqueeze(0)
                if args.crop_background:
                    delta_save = F.pad(delta_save, crop_size)
                save_tensor_as_nii(
                    delta_save, save_path / f'delta_{epoch}', voxel_size, input_affine
                )


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Training-free DIP-STLS reconstruction for 3-D QSM volumes.'
    )

    parser.add_argument('--data_path',         type=str,   required=True,
                        help='Input 3-D volume (.npy, .nii, or .nii.gz).')
    parser.add_argument('--is_field',          action='store_true')
    parser.add_argument('--input_type',        choices=['pure','phi','noise','tkd'], default='phi')
    parser.add_argument('--vox',               type=float, nargs=3, default=None,
                        help='Voxel size; defaults to NIfTI metadata or 1 1 1 for NumPy.')
    parser.add_argument('--z_prjs',            type=float, nargs=3, default=[0,0,1])
    parser.add_argument('--device',            choices=['auto', 'cpu', 'cuda'], default='auto')

    parser.add_argument('--lr',                type=float, default=1e-3)
    parser.add_argument('--gamma',             type=float, default=0.8)
    parser.add_argument('--step',              type=int,   default=20)
    parser.add_argument('--adam_beta1',        type=float, default=0.5)
    parser.add_argument('--adam_beta2',        type=float, default=0.999)
    parser.add_argument('--adam_eps',          type=float, default=1e-9)
    parser.add_argument('--weight_decay',      type=float, default=5e-4)

    parser.add_argument('--epoch_num',         type=int,   default=200)
    parser.add_argument('--seed',              type=int,   default=3407)
    parser.add_argument('--grad_loss_order',   type=int,   default=2, choices=[1,2])
    parser.add_argument('--tkd_threshold',     type=float, default=1/8)

    parser.add_argument('--depth',             type=int,   default=1)
    parser.add_argument('--base',              type=int,   default=16)
    parser.add_argument('--decoder_block_num', type=int,   default=1)

    parser.add_argument('--stls_iter',         type=int,   default=10)
    parser.add_argument('--stls_lambda',       type=float, default=1e-3)
    parser.add_argument('--stls_tol',          type=float, default=1e-5)
    parser.add_argument('--stls_inner_steps',  type=int,   default=10)
    parser.add_argument('--stls_inner_lr_scale', type=float, default=0.1)

    parser.add_argument('--noisy_dipole',      action='store_true')
    parser.add_argument('--noise_level',       type=float, default=0.01)

    parser.add_argument('--scale_data',        type=float, default=1.0,
                        help='Multiplicative input scaling applied before reconstruction.')

    parser.add_argument('--crop_background',   action='store_true')
    parser.add_argument('--interval',          type=int,   default=500)
    parser.add_argument('--padding_mode',      type=str,   default='half',
                        choices=['none','half','full'])
    parser.add_argument('--save_path',         type=str,   default='results/')

    args = parser.parse_args()
    main(args)
