import torch
from torch.utils.data import DataLoader, TensorDataset
from dataset import SpatiotemporalTensorDataset
from model import Generator, Discriminator
from train import train_gan
import matrix

# Configuration settings
input_dim = 100   # Dimension of noise vector
hidden_dim = 256  # Hidden layer dimension

lat_dim = 25      # Latitude dimension
long_dim = 28     # Longitude dimension
num_samples = 208075 # Number of rows
num_epochs = 100   # Number of training epochs
batch_size = 128   # Batch size

# this is just for demo: creating a synthetic spatiotemporal data for demo
# spatiotemporal_data = torch.randn(num_samples, time_steps, lat_dim, long_dim)  # Shape: (num_samples, time_steps, lat_dim, long_dim)

print("Creating training dataset")
train, test, validation = matrix.split(.8, .1, .1)


time_steps = matrix.time_steps  # Number of time steps
print("Training dataset done")
spatiotemporal_data = torch.tensor(train).clone().detach().requires_grad_(True)
spatiotemporal_data = spatiotemporal_data.double()

# Create Dataset and DataLoader
print("Creating dataset")
dataset = SpatiotemporalTensorDataset(spatiotemporal_data)
print("Creating dataset done")
print("Creating dataloader")
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
print("Creating dataloader done")

# Initialize the GAN models
print("Initializing GAN models")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = Generator(input_dim, hidden_dim, time_steps, lat_dim, long_dim).to(device)
discriminator = Discriminator(1, lat_dim, long_dim, hidden_dim).to(device)
print("Initialized")

# Train the GAN
print("Training GAN model")
train_gan(train_loader, generator, discriminator, num_epochs, device, input_dim)
print("GAN model trained")

