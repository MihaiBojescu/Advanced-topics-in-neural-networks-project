import torch
import typing as t
from torch import nn


class Generator(nn.Module):
    __device: torch.device

    def __init__(self, *args, device: torch.device = torch.device("cpu")) -> None:
        super().__init__(*args)

        self.__device = device

        self.conv_1 = nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False, device=device)
        self.batch_norm_1 = nn.BatchNorm2d(512, device=device)
        self.conv_2 = nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False, device=device)
        self.batch_norm_2 = nn.BatchNorm2d(256, device=device)
        self.conv_3 = nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False, device=device)
        self.batch_norm_3 = nn.BatchNorm2d(128, device=device)
        self.conv_4 = nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False, device=device)
        self.batch_norm_4 = nn.BatchNorm2d(64, device=device)
        self.conv_5 = nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False, device=device)

        self.activation_function = nn.ReLU(True)
        self.output_activation_function = nn.Tanh()

        self.to(device=self.__device, non_blocking=self.__device == "cuda")

    def forward(self, x):
        x = x.to(device=self.__device, non_blocking=self.__device == "cuda")

        x = self.activation_function(self.batch_norm_1(self.conv_1(x)))
        x = self.activation_function(self.batch_norm_2(self.conv_2(x)))
        x = self.activation_function(self.batch_norm_3(self.conv_3(x)))
        x = self.activation_function(self.batch_norm_4(self.conv_4(x)))
        x = self.output_activation_function(self.conv_5(x))

        return x
