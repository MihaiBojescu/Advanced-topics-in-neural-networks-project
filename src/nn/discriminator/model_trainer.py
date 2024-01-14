import time
import torch
from torch.nn import Module
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader


class DiscriminatorTrainer:
    __discriminator: Module
    __discriminator: Module
    __loss_function: torch.nn.modules.loss._Loss
    __optimiser: torch.optim.Optimizer
    __device: torch.device
    __exports_path: str

    def __init__(
        self,
        discriminator: Module,
        generator: Module,
        loss_function: _Loss,
        optimiser: Optimizer,
        learning_rate: float,
        device: torch.device = torch.device("cpu"),
        exports_path: str = "/tmp",
    ) -> None:
        self.__discriminator = discriminator
        self.__generator = generator
        self.__loss_function = loss_function()
        self.__optimiser = optimiser(self.__discriminator.parameters(), lr=learning_rate)
        self.__device = device
        self.__exports_path = exports_path

        self.__loss_function = self.__loss_function.to(device=self.__device, non_blocking=self.__device == "cuda")

    def run(
        self,
        batched_dataloader: DataLoader,
    ) -> torch.Tensor:
        self.__discriminator.train()
        self.__generator.eval()
        training_loss = torch.Tensor([0], device=self.__device)

        for entry in batched_dataloader:
            real_image, _ = entry
            fake_image = self.__generator(x=torch.rand(real_image.shape))
            real_image_target = torch.zeros((real_image.shape[0], 1)).fill_(0.85)
            fake_image_target = torch.zeros((real_image.shape[0], 1)).fill_(0.00)

            real_image = real_image.to(device=self.__device, non_blocking=self.__device == "cuda")
            fake_image = fake_image.to(device=self.__device, non_blocking=self.__device == "cuda")
            real_image_target = real_image_target.to(device=self.__device, non_blocking=self.__device == "cuda")
            fake_image_target = fake_image_target.to(device=self.__device, non_blocking=self.__device == "cuda")

            self.__optimiser.zero_grad()
            real_image_result = self.__discriminator(x=real_image)
            fake_image_result = self.__discriminator(x=fake_image)
            loss = self.__loss_function(real_image_result, real_image_target) + self.__loss_function(
                fake_image_result, fake_image_target
            )
            loss.backward()
            self.__optimiser.step()

            training_loss += loss

        return training_loss

    def export(self) -> None:
        torch.save(self.__discriminator.state_dict(), f"{self.__exports_path}/discriminator_{time.time_ns()}.pt")
