import os
import time
import typing as t
import torch
from torchvision.transforms.v2 import ToPILImage
from matplotlib import pyplot


class Sampler:
    __good_sample_threshold: float
    __samples_path: str
    __transform: ToPILImage

    def __init__(self, good_sample_threshold: float, samples_path: str) -> None:
        self.__good_sample_threshold = good_sample_threshold
        self.__samples_path = samples_path
        self.__create_samples_path(samples_path)
        self.__transform = ToPILImage()

    def __create_samples_path(self, samples_path: str) -> None:
        if os.path.exists(samples_path):
            return

        os.mkdir(samples_path)

    def sample(
        self,
        batched_generator_sample: torch.Tensor,
        batched_discriminator_sample: torch.Tensor,
        epoch: t.Optional[int] = None,
    ):
        for generator_sample, discriminator_sample in zip(batched_generator_sample, batched_discriminator_sample):
            figure, subplots = pyplot.subplots(1, 2)

            # noise_sample_image = self.__transform(noise_sample)
            generator_sample_image = self.__transform(generator_sample)
            discriminator_sample_image = torch.zeros(generator_sample.shape).permute(1, 2, 0)
            discriminator_sample_image[:, :] = torch.Tensor(
                [0.0, 1.0, 0.0] if discriminator_sample.item() > self.__good_sample_threshold else [1.0, 0.0, 0.0]
            )

            figure.tight_layout(pad=2.0)
            subplots[0].imshow(generator_sample_image)
            subplots[0].set_title("Generator output")
            subplots[0].set_xlabel("Width")
            subplots[0].set_ylabel("Height")
            subplots[1].imshow(discriminator_sample_image)
            subplots[1].set_title("Discriminator output")
            subplots[1].axis("off")

            epoch_string = f"{epoch}_" if epoch is not None else ""
            figure.savefig(f"{self.__samples_path}/sample_{epoch_string}{time.time_ns()}.jpg")
