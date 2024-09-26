import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, input_dim, hidden_dim, time_steps, lat_dim, lon_dim):
        super(Generator, self).__init__()

        self.time_steps = time_steps
        self.lat_dim = lat_dim
        self.lon_dim = lon_dim

        self.main_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.lat_long_mlp = nn.Sequential(
            # nn.Linear(time_steps * lat_dim * lon_dim, lat_dim*lon_dim),
            nn.Linear(60625, 625),
            nn.Linear(625, 256),
            nn.Linear(256, 32),
            nn.ReLU(),
            # nn.Linear(hidden_dim, lat_dim * lon_dim)
        )

    def forward(self, z):
        latent = self.main_mlp(z)
        print(latent)
        lat_long_factors = self.lat_long_mlp(latent)
        # lat_long_factors = lat_long_factors.view(-1, 1, self.time_steps, self.lat_dim, self.lon_dim)
        return lat_long_factors


class Discriminator(nn.Module):
    def __init__(self, input_channels, time_steps, lat_dim, lon_dim, hidden_dim):
        super(Discriminator, self).__init__()
        self.conv_net = nn.Sequential(
            nn.Conv3d(input_channels, hidden_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv3d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv3d(hidden_dim * 2, 1, kernel_size=3, padding=1),
            nn.Flatten()
        )

        self.fc = nn.Sequential(
            nn.Linear(time_steps * lat_dim * lon_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.float()
        print(f"Input x shape: {x.shape}")
        conv_out = self.conv_net(x)  # x should be (batch_size, channels, time_steps, lat_dim, lon_dim)
        output = self.fc(conv_out)
        return output
    
    
    
    
# import torch
# import torch.nn as nn

# class Generator(nn.Module):
#     def __init__(self, input_dim, hidden_dim, time_steps, lat_dim, long_dim):
#         super(Generator, self).__init__()

#         self.lat_dim = lat_dim
#         self.long_dim = long_dim
#         self.main_mlp = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#         )

#         self.lat_long_mlp = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, lat_dim * long_dim)
#         )

#     def forward(self, z):
#         latent = self.main_mlp(z)
#         lat_long_factors = self.lat_long_mlp(latent)
#         lat_long_factors = lat_long_factors.view(-1, self.lat_dim, self.long_dim)
#         return lat_long_factors


# class Discriminator(nn.Module):
#     def __init__(self, input_channels, lat_dim, long_dim, hidden_dim):
#         super(Discriminator, self).__init__()
#         self.conv_net = nn.Sequential(
#             nn.Conv2d(input_channels, hidden_dim, kernel_size=(3, 3), padding=1),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=(3, 3), padding=1),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(hidden_dim * 2, 1, kernel_size=(3, 3), padding=1),
#             nn.Flatten()
#         )

#         self.fc = nn.Sequential(
#             nn.Linear(lat_dim * long_dim, 1),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         conv_out = self.conv_net(x.unsqueeze(1).float())  # Add channel dimension
#         print(f'conv_out shape: {conv_out.shape}')
#         output = self.fc(conv_out)
#         return output
