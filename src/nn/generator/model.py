import torch
import typing as t
from torch import nn


class Generator(nn.Module):
    __device: torch.device

    def __init__(self, device: torch.device = torch.device("cpu"), *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.__device = device

        self.encoder_1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        self.encoder_1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder_2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.encoder_2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder_3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False)
        self.encoder_3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.pool_3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder_4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False)
        self.encoder_4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False)
        self.pool_4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder_5_1 = nn.Conv2d(512, 1024, kernel_size=3, padding=1, bias=False)
        self.encoder_5_2 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1, bias=False)

        self.upconv_1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder_1_1 = nn.Conv2d(1024, 512, kernel_size=3, padding=1, bias=False)
        self.decoder_1_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False)

        self.upconv_2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder_2_1 = nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False)
        self.decoder_2_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)

        self.upconv_3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder_3_1 = nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False)
        self.decoder_3_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)

        self.upconv_4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder_4_1 = nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False)
        self.decoder_4_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)

        self.outconv = nn.Conv2d(64, 3, kernel_size=1)
        self.activation_function = nn.Sigmoid()

        self.to(device=self.__device, non_blocking=self.__device == "cuda")

    def forward(self, x):
        x = x.to(device=self.__device, non_blocking=self.__device == "cuda")
        x_encoded_list: t.List[torch.Tensor] = []

        x = self.activation_function(self.encoder_1_1(x))
        x = self.activation_function(self.encoder_1_2(x))
        x_encoded_list.append(x)

        x = self.activation_function(self.encoder_2_1(self.pool_1(x)))
        x = self.activation_function(self.encoder_2_2(x))
        x_encoded_list.append(x)

        x = self.activation_function(self.encoder_3_1(self.pool_2(x)))
        x = self.activation_function(self.encoder_3_2(x))
        x_encoded_list.append(x)

        x = self.activation_function(self.encoder_4_1(self.pool_3(x)))
        x = self.activation_function(self.encoder_4_2(x))
        x_encoded_list.append(x)

        x = self.activation_function(self.encoder_5_1(self.pool_4(x)))
        x = self.activation_function(self.encoder_5_2(x))

        x_encoded = x_encoded_list.pop()
        x = self.upconv_1(x)
        x = torch.cat([x, x_encoded], dim=1)
        x = self.activation_function(self.decoder_1_1(x))
        x = self.activation_function(self.decoder_1_2(x))

        x_encoded = x_encoded_list.pop()
        x = self.upconv_2(x)
        x = torch.cat([x, x_encoded], dim=1)
        x = self.activation_function(self.decoder_2_1(x))
        x = self.activation_function(self.decoder_2_2(x))

        x_encoded = x_encoded_list.pop()
        x = self.upconv_3(x)
        x = torch.cat([x, x_encoded], dim=1)
        x = self.activation_function(self.decoder_3_1(x))
        x = self.activation_function(self.decoder_3_2(x))

        x_encoded = x_encoded_list.pop()
        x = self.upconv_4(x)
        x = torch.cat([x, x_encoded], dim=1)
        x = self.activation_function(self.decoder_4_1(x))
        x = self.activation_function(self.decoder_4_2(x))

        x = self.outconv(x)

        return x
