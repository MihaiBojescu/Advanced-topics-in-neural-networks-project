import wandb
import torch
from torchvision.transforms.v2 import Compose, Resize, RandomVerticalFlip, RandomHorizontalFlip, ToDtype
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

wandb_project_name = "RN.Monet.GANs.2"
wandb_sweep_config = {
    'method': 'bayes',
    'metric': {
        'name': 'summed_loss',
        'goal': 'minimize'
    },
    'parameters': {
        'discriminator_learning_rate': {
            'min': 0.00001,
            'max': 0.01
        },
        'generator_learning_rate': {
            'min': 0.00001,
            'max': 0.01
        },
        'generator_trainer_run_frequency': {
            'min': 1,
            'max': 20
        },
        'batch_size': {
            'values': [32, 64, 128]
        },
        'gradient_penalty_rate': {
            'min': 5,
            'max': 15
        }
    }
}

def main():
    wandb.login()
    sweep_id = wandb.sweep(wandb_sweep_config, project=wandb_project_name)
    wandb.agent(sweep_id, function=wandb_run) 

def wandb_run():
    with wandb.init():
        device = get_default_device()
        generator = Generator(device=device)
        discriminator = Discriminator(device=device)

        # vars
        discriminator_learning_rate = wandb.config.discriminator_learning_rate
        generator_learning_rate = wandb.config.generator_learning_rate
        generator_trainer_run_frequency = wandb.config.generator_trainer_run_frequency
        batch_size = wandb.config.batch_size
        gradient_penalty_rate = wandb.config.gradient_penalty_rate

        discriminator_loss_function = lambda: WassersteinWithGradientPenaltyLoss(
            discriminator=discriminator, gradient_penalty_rate=gradient_penalty_rate, device=device
        )
        discriminator_optimizer = torch.optim.Adam
        generator_loss_function = WassersteinLoss
        generator_optimizer = torch.optim.Adam

        discriminator_trainer = DiscriminatorTrainer(
            device=device,
            discriminator=discriminator,
            generator=generator,
            loss_function=discriminator_loss_function,
            optimizer=discriminator_optimizer,
            learning_rate=discriminator_learning_rate,
            exports_path="./wandb_exports"
        )

        generator_trainer = GeneratorTrainer(
            device=device,
            discriminator=discriminator,
            generator=generator,
            optimizer=generator_optimizer,
            loss_function=generator_loss_function,
            learning_rate=generator_learning_rate,
            exports_path="./wandb_exports"
        )

        gan_trainer = GanTrainer(
            discriminator_trainer=discriminator_trainer,
            generator_trainer=generator_trainer,
            generator_trainer_run_frequency=generator_trainer_run_frequency,
            checkpoint_epoch_threshold=101
        )


        # Resize transform will be used to test the model. Will be removed afterwards.
        transforms = Compose([
            Resize([32, 32]),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            ToDtype(torch.float32, scale=True)
        ])

        dataset = ImageDataset(data_path="./data/datasets/monet_jpg", transforms=transforms)
        cached_dataset = CacheableTensorDataset(dataset=dataset, cache=True)
        batched_image_dataloader = DataLoader(dataset=cached_dataset, batch_size=batch_size)

        gan_trainer.run(100, batched_image_dataloader, lambda key, value, epoch: wandb.log({key: value}, step=epoch))


if __name__ == "__main__":
    main()
