import typing as t
import torch
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from nn.discriminator.model_trainer import DiscriminatorTrainer
from nn.generator.model_trainer import GeneratorTrainer


class GanTrainer:
    __discriminator_trainer: DiscriminatorTrainer
    __generator_trainer: GeneratorTrainer
    __generator_trainer_run_frequency: int

    def __init__(
        self,
        discriminator_trainer: DiscriminatorTrainer,
        generator_trainer: GeneratorTrainer,
        generator_trainer_run_frequency: int,
    ):
        self.__discriminator_trainer = discriminator_trainer
        self.__generator_trainer = generator_trainer
        self.__generator_trainer_run_frequency = generator_trainer_run_frequency

    def run(self, epochs: int, batched_images_dataloader: DataLoader):
        epoch_progress_bar = tqdm(range(epochs), desc="Training")

        for _ in epoch_progress_bar:
            for batch_index, image_batch in enumerate(batched_images_dataloader):
                self.__discriminator_trainer.run(real_image_batch=image_batch)

                if batch_index % self.__generator_trainer_run_frequency != 0:
                    continue

                self.__generator_trainer.run(size=image_batch.shape)

        self.__discriminator_trainer.export(epoch=epochs)
        self.__generator_trainer.export(epoch=epochs)
