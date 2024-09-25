import torch
from torch.utils.data import DataLoader
from dataset import SpatiotemporalTensorDataset
from model import Generator, Discriminator
from train import train_gan
import matrix

# Configuration settings
input_dim = 100   # Dimension of noise vector
hidden_dim = 256  # Hidden layer dimension
time_steps = 10   # Number of time steps
lat_dim = 32      # Latitude dimension
long_dim = 32     # Longitude dimension
num_samples = 1000
num_epochs = 100   # Number of training epochs
batch_size = 32    # Batch size

# this is just for demo: creating a synthetic spatiotemporal data for demo
# spatiotemporal_data = torch.randn(num_samples, time_steps, lat_dim, long_dim)  # Shape: (num_samples, time_steps, lat_dim, long_dim)
#replace this with our data
print("Creating training dataset")
train, test, validation = matrix.split(.8, .1, .1)
print("Training dataset done")
spatiotemporal_data = train


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
train_gan(train_loader, generator, discriminator, num_epochs, device)
print("GAN model trained")

