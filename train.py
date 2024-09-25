import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import SpatiotemporalTensorDataset
from model import Generator, Discriminator

def train_gan(train_loader, generator, discriminator, num_epochs, device, input_dim):
    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    for epoch in range(num_epochs):
        for i, real_data in enumerate(train_loader):
            real_data = real_data.to(device) # Move real data to device

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
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_D.step()

            # Train Generator
            optimizer_G.zero_grad()
            fake_output = discriminator(fake_data.detach()) # Pass fake data through discriminator (detached)
            # fake_output = discriminator(fake_data)
            g_loss = criterion(fake_output, real_labels)
            g_loss.backward()
            optimizer_G.step()
            
        # progress
        if i % 50 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{i}/{len(train_loader)}], "
                  f"Loss_D: {d_loss.item():.4f}, Loss_G: {g_loss.item():.4f}")