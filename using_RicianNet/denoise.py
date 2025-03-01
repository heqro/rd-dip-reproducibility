import numpy as np
from PIL import Image
from piqa import SSIM
import torch
import os
os.environ['GLOG_minloglevel'] = '2' # no logging messages in caffe
import caffe

def print_image(img: np.ndarray, file_name: str):
    from imageio import imwrite

    imwrite(uri=file_name, im=img)


def add_rician_noise(img: np.ndarray, std: float) -> np.ndarray:
    img_real = img + std * np.random.randn(*img.shape)
    img_imag = std * np.random.randn(*img.shape)
    return np.sqrt(img_real**2 + img_imag**2)


def get_bounding_box(mask: np.ndarray):
    indices = np.argwhere(mask)
    min_row, min_col = indices.min(axis=0)
    max_row, max_col = indices.max(axis=0)

    return (min_row, min_col), (max_row, max_col)


def pad_image(img: np.ndarray, new_height: int, new_width: int):
    padding = np.zeros((new_height, new_width))
    img_height, img_width = img.shape
    padding[:img_height, :img_width] = img
    return padding


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

dataset_path = "../datasets/dataset"
prototext_path = "model_architecture.prototxt"
weights_path = "model_weights.caffemodel"

caffe.set_mode_cpu()
net = caffe.Net(prototext_path, weights_path, caffe.TEST)
psnr_list, ssim_list = [], []

for idx in range(1, 10):

    gt = np.load(f"{dataset_path}/im_{idx}/gt.npy")
    mask = np.array(Image.open(f"{dataset_path}/im_{idx}/mask.png").convert("L"))
    mask = mask / mask.max()
    bbox = get_bounding_box(mask)

    f = pad_image(np.load(f"{dataset_path}/im_{idx}/Std0.15.npy").squeeze(), 256, 270)
    # ensure that the padding is also Rician noisy (just in case the network needs it)
    f[:, 256:] = add_rician_noise(f[:, 256:], std=0.15)

    gt = pad_image(gt.squeeze(), 256, 270)
    mask = pad_image(mask.squeeze(), 256, 270)
    residue = f - gt

    result = net.forward(data=f.reshape(1, 1, 256, 270))
    result_conv13 = result["conv13"].squeeze().clip(0, 1)

    # Quality metrics computation
    psnr_mask = psnr_with_mask(result_conv13, gt, mask)
    ssim_mask = ssim(
        torch.from_numpy(
            (result_conv13[:256, :256] * mask[:256, :256])[
                ...,
                bbox[0][0] : bbox[1][0],
                bbox[0][1] : bbox[1][1],
            ]
        )[None, None, ...].to(
            torch.float32
        ),  # add batch and channel dimensions for piqa's SSIM
        torch.from_numpy(
            (gt[:256, :256] * mask[:256, :256])[
                ...,
                bbox[0][0] : bbox[1][0],
                bbox[0][1] : bbox[1][1],
            ]
        )[None, None, ...].to(
            torch.float32
        ),  # add batch and channel dimensions for piqa's SSIM
    ).item()
    psnr_list.append(psnr_mask)
    ssim_list.append(ssim_mask)

    # Serialize the image as .npy
    np.save(f"denoising_results/denoised_{idx}", arr=result_conv13[:256, :256])

    # Save the denoised image as .png
    print_image(
        (result_conv13[:256, :256].clip(0, 1).squeeze() * 255).astype(np.uint8),
        f"denoising_results/riciannet_{idx}.png",
    )

    # Print out per-image performance
    print(f"Im {idx}: PSNR={psnr_mask}, SSIM={ssim_mask}")
# Compute statistics on entire dataset
psnr_mean = np.mean(psnr_list)
psnr_std = np.std(psnr_list)
ssim_mean = np.mean(ssim_list)
ssim_std = np.std(ssim_list)

print(f"\nFinal Statistics:")
print(f"PSNR - Mean: {psnr_mean:.4f}, Std: {psnr_std:.4f}")
print(f"SSIM - Mean: {ssim_mean:.4f}, Std: {ssim_std:.4f}")
