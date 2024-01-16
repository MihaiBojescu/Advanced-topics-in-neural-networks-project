import wandb
import torch
from torchvision.transforms.v2 import Compose, Resize, RandomVerticalFlip, RandomHorizontalFlip, ToDtype, Lambda
from torch.utils.data.dataloader import DataLoader
from nn.dataset.cacheable_tensor_dataset import CacheableTensorDataset
from nn.discriminator.loss import WassersteinWithGradientPenaltyLoss
from nn.generator.loss import WassersteinLoss
from nn.util.device import get_default_device
from nn.common.gan_trainer import GanTrainer
from nn.dataset.image_dataset import ImageDataset
from nn.discriminator.model import Discriminator
from nn.discriminator.model_trainer import DiscriminatorTrainer
from nn.generator.model import Generator
from nn.generator.model_trainer import GeneratorTrainer
from nn.util.sampler import Sampler

wandb_project_name = "RN.Monet.GANs.LR_Decay"
wandb_sweep_config = {
    "method": "bayes",
    "metric": {"name": "summed_loss", "goal": "minimize"},
    "parameters": {
        "discriminator_learning_rate": {"min": 0.0001, "max": 0.0002},
        "generator_learning_rate": {"min": 0.0001, "max": 0.0002},
        "generator_trainer_run_frequency": {"min": 4, "max": 6},
        "batch_size": {"values": [16, 32, 64]},
        "gradient_penalty_rate": {"min": 9, "max": 11},
        "discriminator_lr_decay_step_size": {"min": 1, "max": 30},
        "generator_lr_decay_step_size": {"min": 1, "max": 30},
        "discriminator_lr_decay_gamma": {"min": 0.05, "max": 0.2},
        "generator_lr_decay_gamma": {"min": 0.05, "max": 0.2},
    },
}


def main():
    wandb.login()
    sweep_id = wandb.sweep(wandb_sweep_config, project=wandb_project_name)
    wandb.agent(sweep_id, function=wandb_run)

    # train_with_hyperparams(
    #     discriminator_learning_rate=0.0009919817634528208,
    #     generator_learning_rate=0.0013811454108388283,
    #     generator_trainer_run_frequency=1,
    #     batch_size=32,
    #     gradient_penalty_rate=14,
    #     epochs=500,
    #     discriminator_lr_decay_gamma=0.1,
    #     discriminator_lr_decay_step_size=10,
    #     generator_lr_decay_gamma=0.1,
    #     generator_lr_decay_step_size=10,
    # )

def wandb_run():

    with wandb.init():
        # vars
        discriminator_learning_rate = wandb.config.discriminator_learning_rate
        generator_learning_rate = wandb.config.generator_learning_rate
        generator_trainer_run_frequency = wandb.config.generator_trainer_run_frequency
        batch_size = wandb.config.batch_size
        gradient_penalty_rate = wandb.config.gradient_penalty_rate
        epochs = 30
        discriminator_lr_decay_gamma=wandb.config.discriminator_lr_decay_gamma
        discriminator_lr_decay_step_size=wandb.config.discriminator_lr_decay_step_size
        generator_lr_decay_gamma=wandb.config.generator_lr_decay_gamma
        generator_lr_decay_step_size=wandb.config.generator_lr_decay_step_size

        train_with_hyperparams(
            discriminator_learning_rate=discriminator_learning_rate,
            generator_learning_rate=generator_learning_rate,
            generator_trainer_run_frequency=generator_trainer_run_frequency,
            batch_size=batch_size,
            gradient_penalty_rate=gradient_penalty_rate,
            epochs=epochs,
            discriminator_lr_decay_gamma=discriminator_lr_decay_gamma,
            discriminator_lr_decay_step_size=discriminator_lr_decay_step_size,
            generator_lr_decay_gamma=generator_lr_decay_gamma,
            generator_lr_decay_step_size=generator_lr_decay_step_size,
        )


def train_with_hyperparams(
    discriminator_learning_rate: float,
    discriminator_lr_decay_step_size: int,
    discriminator_lr_decay_gamma: float,

    generator_learning_rate: float,
    generator_lr_decay_step_size: int,
    generator_lr_decay_gamma: float,

    generator_trainer_run_frequency: int,
    batch_size: int,
    gradient_penalty_rate: int,
    epochs: int,
):
    device = get_default_device()
    generator = Generator(device=device)
    discriminator = Discriminator(device=device)

    discriminator_loss_function = WassersteinWithGradientPenaltyLoss(
            discriminator=discriminator, gradient_penalty_rate=gradient_penalty_rate, device=device
        )
    discriminator_optimizer = torch.optim.Adam(
        params=discriminator.parameters(), lr=discriminator_learning_rate, betas=(0.5, 0.999)
    )
    generator_loss_function = WassersteinLoss()
    generator_optimizer = torch.optim.Adam(
        params=generator.parameters(), lr=generator_learning_rate, betas=(0.5, 0.999)
    )

    discriminator_trainer = DiscriminatorTrainer(
        device=device,
        discriminator=discriminator,
        generator=generator,
        loss_function=discriminator_loss_function,
        optimizer=discriminator_optimizer,
        exports_path="./wandb_exports",
    )
    discriminator_scheduler = torch.optim.lr_scheduler.StepLR(
        discriminator_optimizer, 
        step_size=discriminator_lr_decay_step_size, 
        gamma=discriminator_lr_decay_gamma
    )

    generator_trainer = GeneratorTrainer(
        device=device,
        discriminator=discriminator,
        generator=generator,
        optimizer=generator_optimizer,
        loss_function=generator_loss_function,
        exports_path="./wandb_exports",
    )

    generator_scheduler = torch.optim.lr_scheduler.StepLR(
        generator_optimizer, 
        step_size=generator_lr_decay_step_size, 
        gamma=generator_lr_decay_gamma
    )

    gan_trainer = GanTrainer(
        discriminator_trainer=discriminator_trainer,
        generator_trainer=generator_trainer,
        generator_trainer_run_frequency=generator_trainer_run_frequency,
        checkpoint_epoch_threshold=101,
        discriminator_lr_scheduler=discriminator_scheduler,
        generator_lr_scheduler=generator_scheduler
    )

    # Resize transform will be used to test the model. Will be removed afterwards.
    transforms = Compose(
        [
            Resize([64, 64]),
            ToDtype(torch.float32, scale=True),
        ]
    )

    dataset = ImageDataset(data_path="./data/datasets/monet_jpg", transforms=transforms)
    cached_dataset = CacheableTensorDataset(dataset=dataset, cache=True)
    batched_image_dataloader = DataLoader(dataset=cached_dataset, batch_size=batch_size, shuffle=True)

    gan_trainer.run(epochs, batched_image_dataloader, lambda key, value, epoch: {})#wandb.log({key: value}, step=epoch))

    sample(discriminator=discriminator, generator=generator)


def sample(discriminator: Discriminator, generator: Generator):
    sampler = Sampler(good_sample_threshold=0.85, samples_path="./data/samples/lr_scheduling")
    noise = torch.rand((4, 100, 1, 1))

    fake_images = generator(noise)
    fake_images_discriminated = discriminator(fake_images)

    sampler.sample(fake_images, fake_images_discriminated)


if __name__ == "__main__":
    main()
