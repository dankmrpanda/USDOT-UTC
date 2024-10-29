import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset_w import SpatiotemporalTensorDataset
from model_w import Generator, Discriminator
import matplotlib.pyplot as plt

def generate_and_save_samples(generator, input_dim, epoch, device):
    # Generate noise for fake data
    z = torch.randn(1, input_dim).to(device)
    # Generate fake data using the generator
    with torch.no_grad():
        generated_data = generator(z)
    # Detach and convert to CPU for visualization
    generated_data = generated_data.cpu().numpy()

    # Assuming spatial_dim is 2 (latitude, longitude)
    latitudes = generated_data[0, :, 0]  # Extract latitude
    longitudes = generated_data[0, :, 1]  # Extract longitude

    # Plot the latitudes and longitudes
    plt.scatter(longitudes, latitudes, marker='o')
    plt.title(f'Generated Data at Epoch {epoch}')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True)
    plt.savefig(f'generated_data_epoch_{epoch}.png')
    plt.show()



def train_gan(train_loader, generator, discriminator, num_epochs, device, input_dim):
    criterion = nn.BCELoss()
    # optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    # optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.000005, betas=(0.5, 0.999))
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0003, betas=(0.5, 0.999))


    for epoch in range(num_epochs):
        for i, real_data in enumerate(train_loader):
            # real_data = real_data.to(device) # Move real data to device
            real_data = real_data.float().to(device)
            # Train Discriminator
            optimizer_D.zero_grad() # Reset discriminator gradients
            
            # a) Real Data
            real_labels = torch.ones(real_data.size(0), 1).to(device)
            real_output = discriminator(real_data)
            d_loss_real = criterion(real_output, real_labels)

            # b) Fake Data
            z = torch.randn(real_data.size(0), input_dim).to(device)
            fake_data = generator(z)
            fake_labels = torch.zeros(fake_data.size(0), 1).to(device)
            fake_output = discriminator(fake_data.detach())
            d_loss_fake = criterion(fake_output, fake_labels)

            # c) Combine Discriminator Losses and Update
            if i % 2 == 0:
                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                optimizer_D.step()

            # Train Generator
            for _ in range(4):
                optimizer_G.zero_grad()
                z = torch.randn(real_data.size(0), input_dim).to(device)  # Create new noise vector
                fake_data = generator(z)  # Generate new fake data for each step
                
                fake_output = discriminator(fake_data)  # No detach here for generator training
                g_loss = criterion(fake_output, real_labels)  # Real labels for generator loss
                g_loss.backward()  # Removed retain_graph=True
                optimizer_G.step()
            
        # progress
        # if i % 50 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{i}/{len(train_loader)}], "
                  f"Loss_D: {d_loss.item():.4f}, Loss_G: {g_loss.item():.4f}")
        # Call this function at the end of each epoch in your training loop
        # generate_and_save_samples(generator, input_dim, epoch, device)
