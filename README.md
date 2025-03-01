# RD-DIP - Rician Denoising with Deep Image Prior (External reproducibility scripts)

## What is this repository for?

This repository stores the dependencies, code and datasets for harnessing the methods we compare **[RD-DIP](https://github.com/heqro/rd-dip) (Rician Denoising with Deep Image Prior)** against. 

## Get started 
Simply clone the repository and initialize the `datasets` module:
```
git clone git@github.com:heqro/rd-dip-reproducibility.git
git submodule init
git submodule update --recursive --remote
```


## Reproducing results with BM3D-VST

For reproducing these results, you need to create a virtual environment (`.venv`), activate it and then install the dependencies:
```
cd using_BM3D-VST
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
The experiments are then launched by executing
```
python denoise.py
```
Its default behavior is to print out the quality metrics (PSNR, SSIM) per denoising experiment and the denoised images both in `.png` and in `.npy` in the `denoising_results/` directory. As explained in our paper, the quality metrics are evaluated exclusively only on the region of interest of each image.

## Reproducing results with RicianNet

`caffe` is a rather old software package requiring Python 3.7. The path of least resistance to use RicianNet is by creating a `conda` environment. Assuming `conda` is installed, the following code creates the environment with the right dependencies:
```
cd using_RicianNet
conda env create -f env_riciannet.yml
```
In order to launch the reproducibility code, activate the environment and launch the script:
```
conda activate env_riciannet
python denoise.py
```
Its default behavior is to print out the quality metrics (PSNR, SSIM) per denoising experiment and the denoised images both in `.png` and in `.npy` in the `denoising_results/` directory. As explained in our paper, the quality metrics are evaluated exclusively only on the region of interest of each image.