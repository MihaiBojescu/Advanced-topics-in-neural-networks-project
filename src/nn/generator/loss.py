import torch


class WassersteinLoss:
    def __call__(self, y: torch.Tensor) -> torch.Tensor:
        return -y.mean()
