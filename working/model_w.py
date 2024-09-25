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

import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, input_dim, hidden_dim, time_steps, spatial_dim):
        super(Generator, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.time_steps = time_steps
        self.spatial_dim = spatial_dim

        # Define a fully connected layer that will convert input noise into the required output shape
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, time_steps * spatial_dim),  # Adjust the output size to (time_steps * spatial_dim)
            nn.Tanh()  # Optional: Use Tanh for output scaling between -1 and 1, you can adjust if necessary
        )

    def forward(self, z):
        # Pass noise through the network
        out = self.fc(z)
        # Reshape output to (batch_size, time_steps, spatial_dim)
        out = out.view(z.size(0), self.time_steps, self.spatial_dim)
        return out



# class Discriminator(nn.Module):
#     def __init__(self, input_channels, lat_dim, long_dim, hidden_dim):
#         super(Discriminator, self).__init__()
#         # self.lat_dim = lat_dim
#         # self.long_dim = long_dim
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

class Discriminator(nn.Module):
    def __init__(self, channels, time_steps, spatial_dim, hidden_dim):
        super(Discriminator, self).__init__()
        self.time_steps = time_steps
        self.spatial_dim = spatial_dim
        self.hidden_dim = hidden_dim

        # Flatten the input (batch_size, time_steps, spatial_dim) to (batch_size, time_steps * spatial_dim)
        self.fc = nn.Sequential(
            nn.Linear(time_steps * spatial_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),  # Output a single probability (real or fake)
            nn.Sigmoid()  # Use Sigmoid to output a probability
        )

    def forward(self, x):
        # Flatten input to (batch_size, time_steps * spatial_dim)
        x = x.view(x.size(0), -1)
        # Pass through the fully connected layers
        out = self.fc(x)
        return out