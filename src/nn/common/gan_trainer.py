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
    __best_discriminator_loss: t.Optional[torch.Tensor]
    __best_generator_loss: t.Optional[torch.Tensor]
    __checkpoint_epoch_threshold: t.Optional[int]

    def __init__(
        self,
        discriminator_trainer: DiscriminatorTrainer,
        generator_trainer: GeneratorTrainer,
        generator_trainer_run_frequency: int,
        checkpoint_epoch_threshold: t.Optional[int] = None,
    ):
        self.__discriminator_trainer = discriminator_trainer
        self.__generator_trainer = generator_trainer
        self.__generator_trainer_run_frequency = generator_trainer_run_frequency
        self.__checkpoint_epoch_threshold = checkpoint_epoch_threshold
        self.__best_discriminator_loss = None
        self.__best_generator_loss = None

    def run(self, epochs: int, batched_images_dataloader: DataLoader, log_callback):
        epoch_progress_bar = tqdm(range(epochs), desc="Training")

        for epoch in epoch_progress_bar:

            discriminator_loss_total = 0
            generator_loss_total = 0
            summed_loss_total = 0

            for batch_index, image_batch in enumerate(batched_images_dataloader):
                discriminator_loss = self.__discriminator_trainer.run(real_image_batch=image_batch)
                self.__add_discriminator_checkpoint(epoch=epoch, loss=discriminator_loss)

                if batch_index == 0 and batch_index % self.__generator_trainer_run_frequency != 0:
                    continue

                generator_loss = self.__generator_trainer.run(batch_size=image_batch.shape[0])
                self.__add_generator_checkpoint(epoch=epoch, loss=generator_loss)

                discriminator_loss_total += discriminator_loss
                generator_loss_total += generator_loss
                summed_loss_total = discriminator_loss_total + generator_loss_total
            
            log_callback("discriminator_loss", discriminator_loss_total, epoch+1)
            log_callback("generator_loss", generator_loss_total, epoch+1)
            log_callback("summed_loss", summed_loss_total, epoch+1)

        self.__discriminator_trainer.export()
        self.__generator_trainer.export()

    def __add_discriminator_checkpoint(self, epoch: int, loss: torch.Tensor) -> None:
        if self.__best_discriminator_loss is None:
            self.__best_discriminator_loss = loss

        if torch.greater_equal(loss, self.__best_discriminator_loss) or epoch < self.__checkpoint_epoch_threshold:
            return

        self.__best_discriminator_loss = loss
        self.__discriminator_trainer.export()

    def __add_generator_checkpoint(self, epoch: int, loss: torch.Tensor) -> None:
        if self.__best_generator_loss is None:
            self.__best_generator_loss = loss

        if torch.greater_equal(loss, self.__best_generator_loss) or epoch < self.__checkpoint_epoch_threshold:
            return

        self.__best_generator_loss = loss
        self.__generator_trainer.export()
