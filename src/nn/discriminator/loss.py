import torch
from torch.nn import Module
from torch.autograd import Variable, grad


class WassersteinWithGradientPenaltyLoss:
    __gradient_penalty_rate: float
    __discriminator: Module
    __device: torch.device

    def __init__(
        self,
        gradient_penalty_rate: float,
        discriminator: Module,
        device: torch.device = torch.device("cpu")
    ) -> None:
        self.__gradient_penalty_rate = gradient_penalty_rate
        self.__discriminator = discriminator
        self.__device = device

    def __call__(
        self,
        real_image: torch.Tensor,
        fake_image: torch.Tensor,
        real_image_discriminated: torch.Tensor,
        fake_image_discriminated: torch.Tensor,
    ) -> torch.Tensor:
        return (
            fake_image_discriminated.mean()
            - real_image_discriminated.mean()
            + self.__gradient_penalty(real_image=real_image, fake_image=fake_image)
        )

    def __gradient_penalty(self, real_image: torch.Tensor, fake_image: torch.Tensor):
        batch_size = real_image.shape[0]
        alpha = torch.rand(batch_size, 1, 1, 1).to(device=self.__device)
        alpha = alpha.expand_as(real_image).to(device=self.__device)
        interpolation = Variable(alpha * real_image + (1 - alpha) * fake_image, requires_grad=True).to(
            device=self.__device
        )
        interpolated_output = self.__discriminator(interpolation)

        gradients = grad(
            inputs=interpolation,
            outputs=interpolated_output,
            grad_outputs=torch.ones(interpolated_output.shape, device=self.__device),
            create_graph=True,
            retain_graph=True,
        )
        gradient = gradients[0].view(batch_size, -1)
        gradient_norm = torch.sqrt(torch.sum(gradient**2, dim=1) + torch.finfo(real_image.dtype).eps)

        return self.__gradient_penalty_rate * ((gradient_norm - 1) ** 2).mean()
