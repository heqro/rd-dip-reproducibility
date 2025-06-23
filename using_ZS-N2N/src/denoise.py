import torch
import torch.optim as optim

import argparse
from network import Net, train, denoise
import sys

sys.path.append("../../datasets/libs/image-utils/src")
from filters import anscombe_transform

device = "cuda"


parser = argparse.ArgumentParser()
parser.add_argument('-n', type=int, help='Image number')
parser.add_argument('-std', type=str, choices=['0.10', '0.15', '0.20', '0.25'], help='Standard deviation')
parser.add_argument('-a', '--anscombe', action='store_true', help='Apply Anscombe transform to noisy image')
args = parser.parse_args()

im_path = f"../../datasets/dataset/im_{args.n}"
noisy = torch.load(f"{im_path}/Std{args.std}.pt", weights_only=True)[None, ...]

if args.anscombe:
    noisy = anscombe_transform(noisy, float(args.std))
noisy = noisy.to(torch.float32).to(device)
# These parameters have been taken from the original paper
max_epoch = 3000     # training epochs
lr = 0.001           # learning rate
step_size = 1000     # number of epochs at which learning rate decays
gamma = 0.5          # factor by which learning rate decays

model = Net(n_chan=1).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

for epoch in range(max_epoch):
    train(model, optimizer, noisy)
    scheduler.step()
denoised_img = denoise(model, noisy)

torch.save(denoised_img.cpu(), f"../results/im_{args.n}_{'anscombe_' if args.anscombe else ''}Std{args.std}_denoised.pt")
