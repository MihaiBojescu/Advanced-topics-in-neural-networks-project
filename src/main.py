import torch
from nn.common.gan_trainer import GanTrainer
from nn.dataset.image_tensor_dataset import ImageTensorDataset
from nn.discriminator.model import Discriminator
from nn.discriminator.model_trainer import DiscriminatorTrainer
from nn.generator.model import Generator
from nn.generator.model_trainer import GeneratorTrainer


def main():
    wandb_sweep()
    pass

def wandb_sweep():
    #constants
    device = get_device()
    
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
        loss_function=discriminator_loss_function,
        optimizer=discriminator_optimizer,
        learning_rate=discriminator_learning_rate,
        device=device
    )

    generator_trainer = GeneratorTrainer(
        discriminator=discriminator,
        generator=generator,
        optimizer=generator_optimizer,
        loss_function=generator_loss_function,
        device=device,
        learning_rate=generator_learning_rate
    )

    gan_trainer = GanTrainer(
        discriminator_trainer=discriminator_trainer,
        generator_trainer=generator_trainer,
        generator_trainer_run_frequency=generator_trainer_run_frequency
    )

    gan_trainer.run(10, get_dataloader())

    pass

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mos')
    else:
        return torch.device('cpu')
    
def get_dataloader():
    return ImageTensorDataset(data_path="./data/datasets/monet_jpg")

if __name__ == "__main__":
    main()
