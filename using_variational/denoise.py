import argparse
import sys
import imageio.v2 as imageio
import numpy as np

from scipy.special import i0e, i1e
from skimage.restoration import denoise_tv_chambolle

sys.path.append("../libs/image-utils/src")
from quality_metrics import (
    psnr_with_mask,
    ssim_with_mask,
)
from spatial_transforms import get_bounding_box


def load_image_gray(path, as_mask=False):
    # Read as grayscale; imageio returns a NumPy array
    img = imageio.imread(path, pilmode="L")

    if as_mask:
        # Set any non-zero value to the maximum value in the image
        img = img.copy()
        img[img > 0] = img.max()

    return img


def log_image_ml(image: np.ndarray, name: str):
    mlflow.log_image(image, f"{name}.png")


def add_rician_noise(img: np.ndarray, std: float):
    _real = img + std * np.random.randn(*img.shape)
    _imag = std * np.random.randn(*img.shape)
    return (_real**2 + _imag**2) ** 0.5


def r(f: np.ndarray, u: np.ndarray, sigma: float):
    arg = u * f / sigma**2
    return i1e(arg) / i0e(arg)


def parse_args():
    parser = argparse.ArgumentParser(description="Simulation parameters")
    parser.add_argument("--delta_t", type=float, default=0.001)
    parser.add_argument("--lda", type=float, default=0.1)
    parser.add_argument("--sigma_noise", type=float)
    parser.add_argument("--sigma_estimated", type=float)
    parser.add_argument("--n_its", type=int, default=1000)
    parser.add_argument("--anatomy", type=str)
    parser.add_argument(
        "--log_results", action=argparse.BooleanOptionalAction, default=False
    )

    return parser.parse_args()


args = parse_args()
DELTA_T = args.delta_t
LDA = args.lda
SIGMA_NOISE = args.sigma_noise
SIGMA_ESTIMATED = args.sigma_estimated
N = args.n_its
ANATOMY = args.anatomy

path = f"../datasets/dataset/{ANATOMY}/"
gt = imageio.imread(path + "gt.png").astype(np.float32) / 255
mask = load_image_gray(path + "mask.png", as_mask=True).astype(np.float32) / 255


def alpha():
    return (LDA * DELTA_T + SIGMA_ESTIMATED**2) / (LDA * DELTA_T)


def beta():
    return SIGMA_ESTIMATED**2 / (alpha() * LDA)


def get_f_hat(u: np.ndarray, f: np.ndarray, std: float, lda: float, dt: float):
    return (1 / alpha()) * (std**2 / (lda * dt) * u + r(f, u, std) * f)


def convert_to_loggable_img(img):
    return (img * 255).astype(np.uint8)


f = add_rician_noise(gt, SIGMA_NOISE).astype(np.float32)
bbox = get_bounding_box(mask)
u = f
best_f_hat, best_u = None, None
best_psnr, accompanying_ssim = -np.inf, -np.inf
ssim_log = [0.0] * N
psnr_log = [0.0] * N
for i in range(N):
    f_hat = get_f_hat(u, f, SIGMA_ESTIMATED, LDA, DELTA_T)
    u = denoise_tv_chambolle(f_hat, weight=beta())

    psnr_log[i] = psnr_with_mask(u, gt, mask)
    ssim_log[i] = ssim_with_mask(u, gt, mask, bbox)

    if best_psnr < psnr_log[i]:
        best_f_hat = f_hat
        best_u = u
        best_psnr = psnr_log[i]
        accompanying_ssim = ssim_log[i]

if args.log_results:
    # If you intend to use Mlflow, you will definitely have to change the following line
    # to your server's uri (or just delete this line)
    import pandas as pd
    import mlflow

    mlflow.set_tracking_uri("http://10.100.12.175:5000")

    with mlflow.start_run(run_name="Blind Masked Rician iterated denoising") as run:
        mlflow.log_params(
            {
                "delta_T": DELTA_T,
                "lambda": LDA,
                "sigma_noise": SIGMA_NOISE,
                "sigma_estimated": SIGMA_ESTIMATED,
                "n_iterations": N,
                "anatomy": ANATOMY,
            }
        )

        mlflow.log_image(convert_to_loggable_img(f), "Noisy.png")
        mlflow.log_image(convert_to_loggable_img(gt), "GroundTruth.png")

        df = pd.DataFrame(
            {
                "step": np.arange(N),
                "psnr": psnr_log,
                "ssim": ssim_log,
            }
        )
        mlflow.log_table(data=df, artifact_file=run.info.run_id + ".json")
        mlflow.log_metric("Best PSNR", best_psnr)
        mlflow.log_metric("Accompanying SSIM", accompanying_ssim)
        mlflow.log_metric("Last PSNR", psnr_log[-1])
        mlflow.log_metric("Last SSIM", ssim_log[-1])
        mlflow.log_image(convert_to_loggable_img(f_hat), "f_hat_n.png")
        mlflow.log_image(convert_to_loggable_img(u), "u_n_plus_1.png")
else:
    import matplotlib.pyplot as plt

    print(f"Denoising process complete, PSNR: {psnr_log[-1]:.2f}, {ssim_log[-1]:.2f}")
    plt.imsave("Denoised_image.png", u.squeeze(), cmap="gray")
