import time
import torch
from torch.nn import Module
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer


class DiscriminatorTrainer:
    __discriminator: Module
    __loss_function: torch.nn.modules.loss._Loss
    __optimizer: torch.optim.Optimizer
    __device: torch.device
    __exports_path: str

    def __init__(
        self,
        discriminator: Module,
        generator: Module,
        loss_function: _Loss,
        optimizer: Optimizer,
        learning_rate: float,
        device: torch.device = torch.device("cpu"),
        exports_path: str = "/tmp",
    ) -> None:
        self.__discriminator = discriminator
        self.__generator = generator
        self.__loss_function = loss_function
        self.__optimizer = optimizer(self.__discriminator.parameters(), lr=learning_rate)
        self.__device = device
        self.__exports_path = exports_path

    def run(self, real_image_batch: torch.Tensor) -> torch.Tensor:
        self.__discriminator.train()
        self.__generator.eval()

        fake_image_batch = self.__generator(x=torch.rand(real_image_batch.shape))
        real_image_batch_target = torch.zeros((real_image_batch.shape[0], 1)).fill_(0.85)
        fake_image_batch_target = torch.zeros((real_image_batch.shape[0], 1)).fill_(0.00)

        real_image_batch = real_image_batch.to(device=self.__device, non_blocking=self.__device == "cuda")
        fake_image_batch = fake_image_batch.to(device=self.__device, non_blocking=self.__device == "cuda")
        real_image_batch_target = real_image_batch_target.to(
            device=self.__device, non_blocking=self.__device == "cuda"
        )
        fake_image_batch_target = fake_image_batch_target.to(
            device=self.__device, non_blocking=self.__device == "cuda"
        )

        self.__optimizer.zero_grad()

        real_image_batch_discriminated = self.__discriminator(x=real_image_batch)
        fake_image_batch_discriminated = self.__discriminator(x=fake_image_batch)

        discriminator_loss = self.__loss_function(
            real_image_batch, fake_image_batch, real_image_batch_discriminated, fake_image_batch_discriminated
        )
        discriminator_loss.backward()
        self.__optimizer.step()

        return discriminator_loss

    def export(self) -> None:
        torch.save(self.__discriminator.state_dict(), f"{self.__exports_path}/discriminator_{time.time_ns()}.pt")
