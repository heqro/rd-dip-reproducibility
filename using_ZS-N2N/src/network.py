# This code has been taken from Zero-Shot Noise2Noise
# by Youssef Mansour, Reinhard Heckel

import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, n_chan: int, chan_embed=48):
        super(Net, self).__init__()

        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv1 = nn.Conv2d(n_chan, chan_embed, 3, padding=1)
        self.conv2 = nn.Conv2d(chan_embed, chan_embed, 3, padding=1)
        self.conv3 = nn.Conv2d(chan_embed, n_chan, 1)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.conv3(x)

        return x


def pair_downsampler(img: torch.Tensor):
    # img has shape B C H W
    c = img.shape[1]

    filter1 = torch.FloatTensor([[[[0, 0.5], [0.5, 0]]]]).to(img.device)
    filter1 = filter1.repeat(c, 1, 1, 1)

    filter2 = torch.FloatTensor([[[[0.5, 0], [0, 0.5]]]]).to(img.device)
    filter2 = filter2.repeat(c, 1, 1, 1)

    output1 = F.conv2d(img, filter1, stride=2, groups=c)
    output2 = F.conv2d(img, filter2, stride=2, groups=c)

    return output1, output2


def loss_func(noisy_img: torch.Tensor, model: nn.Module):
    mse = F.mse_loss
    noisy1, noisy2 = pair_downsampler(noisy_img)

    pred1 = noisy1 - model(noisy1)
    pred2 = noisy2 - model(noisy2)

    loss_res = 1 / 2 * (mse(noisy1, pred2) + mse(noisy2, pred1))

    noisy_denoised = noisy_img - model(noisy_img)
    denoised1, denoised2 = pair_downsampler(noisy_denoised)

    loss_cons = 1 / 2 * (mse(pred1, denoised1) + mse(pred2, denoised2))

    loss = loss_res + loss_cons

    return loss


def train(model: nn.Module, optimizer: torch.optim.Optimizer, noisy_img: torch.Tensor):
    loss = loss_func(noisy_img, model)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def denoise(model: nn.Module, noisy_img: torch.Tensor):

    with torch.no_grad():
        pred = torch.clamp(noisy_img - model(noisy_img), 0, 1)

    return pred
