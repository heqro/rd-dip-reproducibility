# This code is a modified version from that of
# "MRI Rician Noise Reduction Using Recurrent Convolutional Neural Networks"
# by JAVIER GURROLA-RAMOS, TERESA ALARCON, OSCAR DALMAU AND JOSÉ V. MANJÓN.

import yaml
import torch
import nibabel as nib
import numpy as np
from time import time
from os.path import join
from model import DenoiserNet

from utils import (
    correct_model_dict,
    mod_pad,
    add_rician_noise,
    load_mri_image,
    get_rmse,
    get_psnr,
)

with open("./config.yaml", "r") as stream:  # Load YAML all configuration file.
    config = yaml.safe_load(stream)
model_parameters = config["model"]
test_parameters = config["test"]

noise_levels = [0.10, 0.15, 0.20, 0.25]
dataset_path = test_parameters["ixi dataset path"]
results_path = test_parameters["ixi results path"]
pretrained_models_path = test_parameters["pretrained models path"]
save_images = test_parameters["save images"]
device_name = test_parameters["device"]

sequence = model_parameters["sequence"]
normalization_const = model_parameters["normalization constant"]
model_name = "model_braind-T1.pth"

model_path = join(pretrained_models_path, model_name)
model = DenoiserNet(**model_parameters)
device = torch.device(device_name)
print("Using device: {}".format(device))

state_dict = torch.load(model_path)
state_dict = correct_model_dict(state_dict)
model.load_state_dict(state_dict, strict=True)
model.to(device)

vol_name = "t1_icbm_normal_1mm_pn0_rf20"  # your volumetric image of choice
for param in model.parameters():
    param.requires_grad = False
with torch.no_grad():
    model.eval()

    for noise_level in noise_levels:
        data, affine, header = load_mri_image(
            f"../dataset/volumetric_brains/{vol_name}.nii.gz"
        )
        data = np.transpose(data, (1, 0, 2))
        max_val = data.max()
        sigma = max_val * noise_level
        noisy, _, _ = add_rician_noise(data, sigma)
        noisy, data = noisy / normalization_const, data / normalization_const
        noisy, size = mod_pad(noisy, 2, mode="reflect")
        noisy = np.reshape(
            noisy, (1, 1) + noisy.shape
        )  # Expand for channel and for batch.
        # Save noisy volumetric image (uncomment if not needed)
        # torch.save(
        #     torch.from_numpy(noisy),
        #     f"../dataset/volumetric_noisy_brains/{vol_name}_noisy_{noise_level}.pt",
        # )
        # Denoise volumetric image
        # continue
        noisy = torch.from_numpy(noisy).to(device)
        start = time()
        estimated_image = model(noisy)
        end = time()
        # print(f"Time taken: {end-start}")
        # exit(-1)
        estimated_image = (
            estimated_image.detach()
            .cpu()
            .squeeze()
            .clip(0, estimated_image.max())
            .numpy()
        )
        if size[1] > 0:
            estimated_image = estimated_image[size[0] : -size[1], ...]
        if size[3] > 0:
            estimated_image = estimated_image[:, size[2] : -size[3], :]
        if size[5] > 0:
            estimated_image = estimated_image[..., size[4] : -size[5]]
        estimated_image = np.transpose(estimated_image, (1, 0, 2))
        estimated_image = np.around(normalization_const * estimated_image)
        estimated_image = estimated_image.astype("uint16")
        mri_image = nib.Nifti1Image(estimated_image * normalization_const, affine)
        nib.save(
            mri_image,
            f"../dataset/volumetric_denoised_brains/{vol_name}_{noise_level}_test2.nii.gz",
        )
