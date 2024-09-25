import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, input_dim, hidden_dim, time_steps, lat_dim, long_dim):
        super(Generator, self).__init__()

        self.lat_dim = lat_dim
        self.long_dim = long_dim
        self.main_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.lat_long_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, lat_dim * long_dim)
        )

    def forward(self, z):
        latent = self.main_mlp(z)
        lat_long_factors = self.lat_long_mlp(latent)
        lat_long_factors = lat_long_factors.view(-1, self.lat_dim, self.long_dim)
        return lat_long_factors


class Discriminator(nn.Module):
    def __init__(self, input_channels, lat_dim, long_dim, hidden_dim):
        super(Discriminator, self).__init__()
        # self.lat_dim = lat_dim
        # self.long_dim = long_dim
        self.conv_net = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_dim * 2, 1, kernel_size=(3, 3), padding=1),
            nn.Flatten()
        )
        # input_shape = (1, 1, lat_dim, long_dim) 
        # print(input_shape)
        # conv_out_size = self.conv_net(torch.randn(input_shape)).shape
        # print(conv_out_size)
        # fc_input_size = conv_out_size[1]
        # print(fc_input_size)
        
        # self.transform = nn.Linear(128, 194)
        # self.fc = nn.Linear(128, 194)
        # Dynamically calculate the flattened output size from the conv layers
        # example_input = torch.randn(1, input_channels, self.lat_dim, self.long_dim)
        # conv_out_size = self.conv_net(example_input).view(-1).size(0)
        # self.intermediate_fc = nn.Linear(conv_out_size, 896)
        self.fc = nn.Sequential(
            nn.Linear(lat_dim * long_dim, 1),
            # nn.Linear(128, 194),
            nn.Sigmoid()
        )

    def forward(self, x):
        conv_out = self.conv_net(x.unsqueeze(1).float())  # Add channel dimension
        print(f'conv_out shape: {conv_out.shape}')
        # intermediate_output = self.intermediate_fc(conv_out)
        # output = self.fc(intermediate_output)
        # transformed_out = self.transform(conv_out)
        # conv_out = torch.flatten(conv_out, start_dim=1)
        # print(f'Flattened conv_out shape: {conv_out.shape}')
        output = self.fc(conv_out)
        return output

