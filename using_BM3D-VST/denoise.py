"""
This script goes through the entire dataset and applies BM3D-VST on it.
The results are then dumped onto a CSV file.
"""

import bm3d
import numpy as np
import imageio.v3 as iio
import torch
from piqa import SSIM


def print_image(img: np.ndarray, file_name: str):
    iio.imwrite(uri=file_name, image=img)


def anscombe_transform(x: np.ndarray, sigma: float):
    return np.sqrt(np.clip(x**2 - sigma**2, a_min=0.0, a_max=None))


def get_bounding_box(mask: torch.Tensor):
    indices = torch.nonzero(mask, as_tuple=True)
    top_left = (
        int(indices[-2].min().item()),
        int(indices[-1].min().item()),
    )  # (min_row, min_col)
    bottom_right = (
        int(indices[-2].max().item()),
        int(indices[-1].max().item()),
    )  # (max_row, max_col)
    return top_left, bottom_right


def psnr_with_mask(
    img_1: np.ndarray,
    img_2: np.ndarray,
    mask: np.ndarray,
    data_range=1.0,
):
    mask_size = (mask > 0).sum().item()
    mse = ((img_1 - img_2) ** 2 * mask).sum() / mask_size
    return 10 * np.log10(data_range**2 / mse)


ssim = SSIM(n_channels=1)
for std in ["0.10", "0.15", "0.20"]:
    psnr_list, ssim_list = [], []
    for idx in [str(id) for id in range(1, 10)]:
        folder = f"../datasets/dataset/im_{idx}"
        gt_path = f"{folder}/gt.pt"
        gt_img = torch.load(gt_path, weights_only=True)

        mask = torch.from_numpy(
            iio.imread(f"{folder}/mask.png", mode="L").astype(np.float32)
        )  # Load as grayscale
        mask = (mask - mask.min()) / (mask.max() - mask.min())
        mask[mask > 0] = mask.max()  # ensure the mask is binary (just in case)
        mask = mask[None, ...]
        bbox = get_bounding_box(mask)

        noisy_path = f"{folder}/Std{std}.pt"
        noisy_img = torch.load(noisy_path, weights_only=True)
        denoised_img = bm3d.bm3d(
            anscombe_transform(noisy_img.numpy().squeeze(), sigma=float(std)),
            sigma_psd=float(std),
        ).clip(min=0.0, max=1.0)[
            None, ...
        ]  # use bm3d as default, then clip to (0, 1)

        # Serialize the image as .npy
        np.save(
            f"denoising_results/denoised_{idx}_std{std}", arr=denoised_img.squeeze()
        )

        # Save the denoised image as .png
        print_image(
            (denoised_img.squeeze() * 255).astype(np.uint8),
            f"denoising_results/denoised_{idx}_std{std}.png",
        )

        # Quality metrics computation
        psnr_mask = psnr_with_mask(torch.from_numpy(denoised_img), gt_img, mask).item()
        ssim_mask = ssim(
            (gt_img * mask)[
                ...,
                bbox[0][0] : bbox[1][0],
                bbox[0][1] : bbox[1][1],
            ][None, ...].to(
                torch.float32
            ),  # add batch dimension for piqa's SSIM
            (torch.from_numpy(denoised_img) * mask)[
                ...,
                bbox[0][0] : bbox[1][0],
                bbox[0][1] : bbox[1][1],
            ][None, ...].to(
                torch.float32
            ),  # add batch dimension for piqa's SSIM
        ).item()
        # Print out per-image performance
        print(f"image {idx}, std={std}: {psnr_mask},{ssim_mask}")
        psnr_list.append(psnr_mask)
        ssim_list.append(ssim_mask)
    # Compute statistics on entire dataset
    psnr_mean = np.mean(psnr_list)
    psnr_std = np.std(psnr_list)
    ssim_mean = np.mean(ssim_list)
    ssim_std = np.std(ssim_list)

    print(f"\n--- Final Statistics for std: {std} ----")
    print(f"PSNR - Mean: {psnr_mean:.4f}, Std: {psnr_std:.4f}")
    print(f"SSIM - Mean: {ssim_mean:.4f}, Std: {ssim_std:.4f}\n")
