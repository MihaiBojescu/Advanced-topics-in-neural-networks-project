from nn.discriminator.model import Discriminator
from nn.generator.model import Generator
from torch.utils.data import DataLoader
import torch

def train_generator(
        generator: Generator, 
        discriminator: Discriminator, 
        optimizer: torch.optim.Optimizer, 
        criterion: torch.nn.Module, 
        device: torch.device, 
        num_epochs: int,
        batch_size: int
) -> None:
    
    generator.to(device)
    discriminator.to(device)

    for epoch in range(num_epochs):

        # do this X times per epoch?

        # Zero the gradients on each iteration
        optimizer.zero_grad()

        # Generate fake images
        static_images = torch.rand((batch_size, 3, 64, 64)).to(device)
        fake_images = generator(static_images)

        # Calculate the generator loss
        outputs = discriminator(fake_images)
        loss = criterion(outputs, torch.ones(batch_size, 1).to(device))

        # Backprop and optimize the generator
        loss.backward()
        optimizer.step()
        print(f"Epoch [{epoch+1}/{num_epochs}], g_loss: {loss.item()}")
    

generator = Generator(device=torch.device("cuda"))
discriminator = Discriminator(device=torch.device("cuda"))

# Define the optimizer for the generator
optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002)

# Define the loss criterion
criterion = torch.nn.BCELoss()

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the number of epochs
num_epochs = 100

# Define the batch size
batch_size = 64

# Now you can call the function
train_generator(generator, discriminator, optimizer, criterion, device, num_epochs, batch_size)