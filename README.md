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


## Reproducing results with DRCNet 

This code harnesses the original implementation of DRCNet by Gurrola et al. from [here](https://github.com/JavierGurrola/DRCnet/), solely to reproduce the experiments presented in our paper. For questions about the original codebase, please consult the original repository directly. The exact details are mentioned in the following for reproducibility and fairness of comparison:

- **Original source of the code**: We have used the official repository [https://github.com/JavierGurrola/DRCnet](https://github.com/JavierGurrola/DRCnet), commit `bf2cb42`.
- **Reproducibility and fairness**:
  - We use the same `src/model.py` and `src/utils.py` files. They define the structure of the neural network and several utility functions, respectively. We credit authorship for each file.
  - Model weights are identical to those provided by the authors, and stored in our `pretrained_models/` directory. No retraining has been performed; we load these weights directly.
  - Our file `src/contaminate_and_denoise.py` is a modified version of the original `main_text_ixi.py`; it loads, contaminates, denoises, and saves the denoised volume as a `.nii.gz` volume. We adjusted this script to handle output saving and environment variables to our convenience, e.g., path to our data and model parameters.
  - Evaluation done via our own `src/denoise.ipynb` to ensure fair comparison by slicing consistently across methods.

Assuming `conda` is installed, the following code creates the environment with the right dependencies:
```
cd using_DRCNet
conda env create -f drcnet-env.yml
```
In order to launch the reproducibility code, activate the environment and launch the script:
```
conda activate drcnet-env
python contaminate.py
```
Its default behavior is to print out denoised volumetric images to `../dataset/volumetric_denoised_brains`. The user may also choose to save the noisy brains. 

## Reproducing results with ZS-N2N 

This repository is based on the original ZS-N2N implementation by Mansour et al., which provides a convenient Colab notebook [here](https://colab.research.google.com/drive/1i82nyizTdszyHkaHBuKPbWnTzao8HF9b). For any issues or questions about the core method, please refer to the original authors. We converted the original notebook cells into `.py` modules for automation. Our implementation details are as mentioned in here:

  * Original network architecture and loss functions (see `src/network.py`, authorship credited).
  * Exact hyperparameters as in the original (epochs, learning rate, step-size, etc.), set in `src/denoise.py`.
  * Use of the same variance stabilizing transform (VST) as BM3D-VST to ensure identical input conditions between the two methods using VST.

Assuming `conda` is installed, the following code creates the environment with the right dependencies:
```
cd using_ZS-N2N
conda env create -f ns-n2n-env.yaml
```
In order to launch the reproducibility code, activate the environment and launch the script:
```
conda activate ns-n2n-env
python denoise.py
```
Its default behavior is to save the denoised volumetric images to the `../results/` folder. 



* **Reproducibility and fairness**: Our full pipeline and run scripts are available in our public reproducibility repo ([link](https://github.com/heqro/rd-dip-reproducibility/blob/main/using_ZS-N2N)).
