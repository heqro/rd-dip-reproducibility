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
from noise import add_rician_noise
from quality_metrics import compute_epi_torch, compute_cnr_torch, psnr_with_mask
gt_path='/mnt/data_drive/hrodrigo/rd-dip-reproducibility/using_DRCnet/dataset/volumetric_brains/abdominal.nii.gz'
gt = torch.from_numpy(nib.load(gt_path).get_fdata()[:,:,34]).to(torch.float32)

# def add_padding(img:torch.Tensor):
#     h, w = img.shape
#     target_h, target_w = 480, 480
#     pad_h = target_h - h
#     pad_w = target_w - w
#     pad_top = pad_h // 2
#     pad_bottom = pad_h - pad_top
#     pad_left = pad_w // 2
#     pad_right = pad_w - pad_left
#     return torch.nn.functional.pad(img, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
# gt=add_padding(gt)
mask=(1+0*gt)[None,...]

def compute_metrics(gt: torch.Tensor, mask: torch.Tensor, img: torch.Tensor, data_range:float):
    bbox = get_bounding_box(mask)
    res = {}
    ssim=SSIM(n_channels=1, value_range=data_range)
    res["PSNR"] = psnr_with_mask(img, gt, mask, data_range=data_range).item()
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
results = []
aux=(gt-gt.min()) / (gt.max()-gt.min())
print_image((255*aux).squeeze().numpy().astype(np.uint8), file_name='gttest.png')
for std in ['0.10', '0.15', '0.20', '0.25']:
    dn_path=f'/mnt/data_drive/hrodrigo/rd-dip-reproducibility/using_DRCnet/dataset/volumetric_denoised_brains/abdominal_{std}_test2.nii.gz'
    dn = torch.from_numpy((nib.load(dn_path).get_fdata()[:,:,34]).copy()).to(torch.float32)/63.75
    # dn=add_padding(dn)
    print_image(img=(255*(((dn-dn.min())/(dn.max()-dn.min())))).numpy().astype(np.uint8), file_name=f'denoised_{std}_abdominal.png')
    results += [compute_metrics(gt, mask, dn.clip(0,gt.max()),data_range=gt.max())]
print(results)
