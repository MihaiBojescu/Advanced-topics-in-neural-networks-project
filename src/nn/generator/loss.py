import torch


class WassersteinLoss:
    def __call__(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return -y_hat.mean()
