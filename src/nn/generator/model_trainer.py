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
        exports_path: str = "/tmp",
    ) -> None:
        self.__discriminator = discriminator
        self.__generator = generator
        self.__loss_function = loss_function
        self.__optimizer = optimizer
        self.__device = device
        self.__exports_path = exports_path

        self.__device = device

    def run(self, noise_batch: torch.Tensor) -> torch.Tensor:
        self.__discriminator.train()
        self.__generator.train()
        self.__optimizer.zero_grad()

        fake_image_batch = self.__generator(x=noise_batch)
        fake_image_batch_discriminated = self.__discriminator(x=fake_image_batch)

        generator_loss = self.__loss_function(fake_image_batch_discriminated).to(self.__device)
        generator_loss.backward()
        self.__optimizer.step()

        return generator_loss

    def export(self) -> None:
        os.makedirs(self.__exports_path, exist_ok=True)
        torch.save(self.__generator.state_dict(), f"{self.__exports_path}/generator_{time.time_ns()}.pt")
