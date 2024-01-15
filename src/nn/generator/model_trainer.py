import os
import time
import torch
from torch.nn.modules.loss import _Loss
from nn.discriminator.model import Discriminator
from nn.generator.model import Generator


class GeneratorTrainer:
    __discriminator: Discriminator
    __generator: Generator
    __loss_function: torch.nn.Module
    __optimizer: torch.optim.Optimizer
    __device: torch.device
    __exports_path: str

    def __init__(
        self,
        discriminator: Discriminator,
        generator: Generator,
        loss_function: _Loss,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        learning_rate: float,
        exports_path: str = "/tmp",
    ) -> None:
        self.__discriminator = discriminator
        self.__generator = generator
        self.__loss_function = loss_function()
        self.__optimizer = optimizer(self.__generator.parameters(), lr=learning_rate)
        self.__device = device
        self.__exports_path = exports_path

        self.__device = device

    def run(self, size: torch.Size) -> torch.Tensor:
        self.__discriminator.eval()
        self.__generator.train()

        self.__optimizer.zero_grad()

        static_image_batch = torch.rand(size).to(self.__device)
        generated_image_batch = self.__generator(static_image_batch)

        labels = self.__discriminator(generated_image_batch)
        loss = self.__loss_function(labels, torch.ones(size[0], 1).to(self.__device))

        loss.backward()
        self.__optimizer.step()

        return loss

    def export(self) -> None:
        os.makedirs(self.__exports_path, exist_ok=True)
        torch.save(self.__generator.state_dict(), f"{self.__exports_path}/generator_{time.time_ns()}.pt")
