import torch
from PIL import Image
import nibabel as nib
from nibabel import filebasedimages
import numpy as np
from typing import Literal
from piqa import SSIM

import sys
sys.path.append("../../datasets/libs/image-utils/src")
from my_io import load_gray_image, print_image
from spatial_transforms import get_bounding_box
from quality_metrics import compute_epi_torch, compute_cnr_torch, psnr_with_mask

ssim = SSIM(n_channels=1)

def compute_metrics(gt: torch.Tensor, mask: torch.Tensor, img: torch.Tensor):
    bbox = get_bounding_box(mask)
    res = {}
    res["PSNR"] = psnr_with_mask(img, gt, mask).item()
    res["SSIM"] = ssim(
        (img * mask)[None, ...][
            ...,
            bbox[0][0] : bbox[1][0],
            bbox[0][1] : bbox[1][1],
        ],
        (gt * mask)[None, ...][
            ...,
            bbox[0][0] : bbox[1][0],
            bbox[0][1] : bbox[1][1],
        ],
    ).item()
    cnr = compute_cnr_torch(img=img, mask=mask) / compute_cnr_torch(img=gt, mask=mask)
    res["CNRI"] = 1 - abs(1 - min(cnr, 1 / cnr))
    res["EPI"] = compute_epi_torch(denoised=img * mask, reference=gt * mask)
    return res

def get_mask(n: int):
    return load_gray_image(
        f"../../datasets/dataset/im_{n}/mask.png",
        as_mask=True,
    )

for a in ['_anscombe', '']:
    print(f'{a}')
    for std in ['0.10','0.15','0.20','0.25']:
        print(f"--- σ = {std} ---")
        results=[]
        for n in range(10, 14):
            mask = get_mask(n).to(torch.float32)
            gt = torch.load(f"../../datasets/dataset/im_{n}/gt.pt", weights_only=True)
            gt = gt.to(torch.float32)
            denoised = torch.load(f'../results/im_{n}{a}_Std{std}_denoised.pt', weights_only=True)
            denoised = denoised.to(torch.float32).squeeze()[None,...]
            results += [compute_metrics(gt, mask, denoised)]
        # print out global metrics
        metrics = {k: np.array([r[k] for r in results]) for k in results[0]}
        summary = {k: {'mean': np.mean(v), 'std': np.std(v)} for k, v in metrics.items()}
        for metric, stats in summary.items():
            mean = stats['mean']
            std = stats['std']
            print(f"   {metric}: mean = {mean:.4f}, std = {std:.4f}")
        # print out per-image metrics
        for i, r in enumerate(results, 1):
            print(f"   Image i_{i}: ", end='')
            print(", ".join(f"{k} = {v:.4f}" for k, v in r.items()))
