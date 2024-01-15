import time
import torch
from nn.discriminator.model import Discriminator

from nn.generator.model import Generator

class GeneratorTrainer:
    __generator: Generator 
    __optimizer: torch.optim.Optimizer 
    __criterion: torch.nn.Module 
    __device: torch.device 
    __exports_path: str

    def __init__(
        self,
        generator: Generator, 
        optimizer: torch.optim.Optimizer, 
        criterion: torch.nn.Module, 
        device: torch.device, 
        learning_rate: float,
        exports_path: str = "/tmp",
    ) -> None:
        
        self.__generator = generator 
        self.__optimizer = optimizer(self.__generator.parameters(), lr=learning_rate)
        self.__criterion = criterion 
        self.__device = device 
        self.__exports_path = exports_path
        
        self.__device = device

    def run(self, discriminator: Discriminator, size: torch.Size) -> torch.Tensor:
        self.__generator.to(self.__device)
        discriminator.to(self.__device)
        self.__optimizer.zero_grad()

        static_image_batch = torch.rand(size).to(self.__device)
        generated_image_batch = self.__generator(static_image_batch)

        labels = discriminator(generated_image_batch)
        loss = self.__criterion(labels, torch.ones(size[0], 1).to(self.__device))

        loss.backward()
        self.__optimizer.step()

    def export(self) -> None:
        torch.save(self.__generator.state_dict(), f"{self.__exports_path}/generator_{time.time_ns()}.pt")
