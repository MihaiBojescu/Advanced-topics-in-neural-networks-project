import torch
from torchvision.transforms.v2 import Compose, Resize, RandomVerticalFlip, RandomHorizontalFlip
from torch.utils.data.dataloader import DataLoader
from nn.dataset.cacheable_tensor_dataset import CacheableTensorDataset
from nn.util.device import get_default_device
from nn.common.gan_trainer import GanTrainer
from nn.dataset.image_tensor_dataset import ImageTensorDataset
from nn.discriminator.model import Discriminator
from nn.discriminator.model_trainer import DiscriminatorTrainer
from nn.generator.model import Generator
from nn.generator.model_trainer import GeneratorTrainer


def main():
    wandb_sweep()

def wandb_sweep():
    #constants
    device = get_default_device()
    
    # Resize transform will be used to test the model. Will be removed afterwards.
    transforms = Compose([
        Resize([32, 32]),
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
    ])

    dataset = ImageTensorDataset(data_path="./data/datasets/monet_jpg", transforms=transforms)
    cached_dataset = CacheableTensorDataset(dataset=dataset, cache=True)
    batched_image_dataloader = DataLoader(dataset=cached_dataset, batch_size=32)

    #vars
    discriminator_loss_function = torch.nn.CrossEntropyLoss
    discriminator_optimizer = torch.optim.Adam
    discriminator_learning_rate = 0.0001

    generator_loss_function = torch.nn.CrossEntropyLoss
    generator_optimizer = torch.optim.Adam
    generator_learning_rate = 0.0001
    generator_trainer_run_frequency = 1


    generator = Generator(device=device)
    discriminator = Discriminator(device=device)

    discriminator_trainer = DiscriminatorTrainer(
        discriminator=discriminator,
        generator=generator,
        optimizer=discriminator_optimizer,
        loss_function=discriminator_loss_function,
        learning_rate=discriminator_learning_rate,
        exports_path="./data/exports",
        device=device
    )

    generator_trainer = GeneratorTrainer(
        discriminator=discriminator,
        generator=generator,
        optimizer=generator_optimizer,
        loss_function=generator_loss_function,
        learning_rate=generator_learning_rate,
        exports_path="./data/exports",
        device=device,
    )

    gan_trainer = GanTrainer(
        discriminator_trainer=discriminator_trainer,
        generator_trainer=generator_trainer,
        generator_trainer_run_frequency=generator_trainer_run_frequency
    )

    gan_trainer.run(10, batched_image_dataloader)
    

if __name__ == "__main__":
    main()
