import torch


class Discriminator(torch.nn.Module):
    __device: torch.device

    def __init__(self, *args, device: torch.device):
        super().__init__(*args)

        self.__device = device

        self.__conv_1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, device=device)
        self.__batch_norm_1 = torch.nn.BatchNorm2d(64, device=device)
        self.__conv_2 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, device=device)
        self.__batch_norm_2 = torch.nn.BatchNorm2d(128, device=device)
        self.__conv_3 = torch.nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, device=device)
        self.__batch_norm_3 = torch.nn.BatchNorm2d(256, device=device)
        self.__conv_4 = torch.nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, device=device)
        self.__batch_norm_4 = torch.nn.BatchNorm2d(512, device=device)
        self.__conv_5 = torch.nn.Conv2d(512, 1, kernel_size=2, stride=1, padding=0, device=device)

        self.__activation_function = torch.nn.ReLU()
        self.__output_activation_function = torch.nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(device=self.__device, non_blocking=self.__device == "cuda")

        x = self.__activation_function(self.__batch_norm_1(self.__conv_1(x)))
        x = self.__activation_function(self.__batch_norm_2(self.__conv_2(x)))
        x = self.__activation_function(self.__batch_norm_3(self.__conv_3(x)))
        x = self.__activation_function(self.__batch_norm_4(self.__conv_4(x)))
        x = self.__output_activation_function(self.__conv_5(x))
        x = x[:, :, 0, 0]

        return x
