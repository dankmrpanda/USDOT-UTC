

import torch
import torch.optim as optim
import torch.nn as nn
torch.cuda.empty_cache()
def train_gan(train_loader, generator, discriminator, num_epochs, device, input_dim, lat_dim, lon_dim):
    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    generator.train()
    discriminator.train()

    for epoch in range(num_epochs):
        for i, real_data in enumerate(train_loader):
            real_data = real_data.to(device).long()  # Convert to long for indexing

            batch_size, time_steps, _ = real_data.shape

            # Create an empty grid for real data
            real_data_grid = torch.zeros(batch_size, 1, time_steps, lat_dim, lon_dim, device=device)

            # Extract indices
            lat_indices = real_data[:, :, 0]  # Shape: (batch_size, time_steps)
            lon_indices = real_data[:, :, 1]  # Shape: (batch_size, time_steps)

            # Clamp indices to valid range
            lat_indices = torch.clamp(lat_indices, 0, lat_dim - 1)
            lon_indices = torch.clamp(lon_indices, 0, lon_dim - 1)

            # Prepare batch and time indices
            batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, time_steps)
            time_indices = torch.arange(time_steps, device=device).unsqueeze(0).expand(batch_size, -1)

            # Set the positions in the grid to 1
            real_data_grid[batch_indices, 0, time_indices, lat_indices, lon_indices] = 1.0

            real_data = real_data_grid.float()  # Now shape is [batch_size, 1, time_steps, lat_dim, lon_dim]

            # Train Discriminator
            optimizer_D.zero_grad()

            # Real Data
            real_labels = torch.ones(batch_size, 1).to(device)
            real_output = discriminator(real_data)
            d_loss_real = criterion(real_output, real_labels)

            # Fake Data
            z = torch.randn(batch_size, input_dim).to(device)
            fake_data = generator(z)
            fake_labels = torch.zeros(batch_size, 1).to(device)
            fake_output = discriminator(fake_data.detach())
            d_loss_fake = criterion(fake_output, fake_labels)

            # Combine Discriminator Losses and Update
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_D.step()

            # Train Generator
            optimizer_G.zero_grad()
            fake_output = discriminator(fake_data)
            g_loss = criterion(fake_output, real_labels)
            g_loss.backward()
            optimizer_G.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss_D: {d_loss.item():.4f}, Loss_G: {g_loss.item():.4f}")
        
        
        
# import torch
# import torch.optim as optim
# import torch.nn as nn
# from torch.utils.data import DataLoader
# from dataset import SpatiotemporalTensorDataset
# from model import Generator, Discriminator

# def train_gan(train_loader, generator, discriminator, num_epochs, device, input_dim):
#     criterion = nn.BCELoss()
#     optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
#     optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

#     for epoch in range(num_epochs):
#         for i, real_data in enumerate(train_loader):
#             real_data = real_data.to(device) # Move real data to device
#             # Train Discriminator
#             optimizer_D.zero_grad() # Reset discriminator gradients
            
#             # a) Real Data
#             real_labels = torch.ones(real_data.size(0), 1).to(device)
#             real_output = discriminator(real_data)
#             d_loss_real = criterion(real_output, real_labels)

#             # b) Fake Data
#             z = torch.randn(real_data.size(0), input_dim).to(device)
#             fake_data = generator(z)
#             fake_labels = torch.zeros(fake_data.size(0), 1).to(device)
#             fake_output = discriminator(fake_data.detach())
#             d_loss_fake = criterion(fake_output, fake_labels)

#             # c) Combine Discriminator Losses and Update
#             d_loss = d_loss_real + d_loss_fake
#             d_loss.backward()
#             optimizer_D.step()

#             # Train Generator
#             optimizer_G.zero_grad()
#             # fake_output = discriminator(fake_data.detach()) # Pass fake data through discriminator (detached)
#             fake_output = discriminator(fake_data)
#             g_loss = criterion(fake_output, real_labels)
#             g_loss.backward()
#             optimizer_G.step()
            
#         # progress
#         # if i % 50 == 0:
#         print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{i}/{len(train_loader)}], "
#                   f"Loss_D: {d_loss.item():.4f}, Loss_G: {g_loss.item():.4f}")