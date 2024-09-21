import torch
import torch.nn as nn
import torch.optim as optim
import matrix

# Generator
class Generator(nn.Module):
    def __init__(self, noise_dim, output_shape):
        super(Generator, self).__init__()
        self.fc = nn.Linear(noise_dim, 128 * 8 * 8 * 8)
        self.relu = nn.ReLU()
        self.conv_trans1 = nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1)
        self.conv_trans2 = nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1)
        self.conv_trans3 = nn.ConvTranspose3d(32, output_shape[0], kernel_size=4, stride=2, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        x = x.view(x.size(0), 128, 8, 8, 8)
        x = self.conv_trans1(x)
        x = self.relu(x)
        x = self.conv_trans2(x)
        x = self.relu(x)
        x = self.conv_trans3(x)
        x = self.tanh(x)
        return x

# Discriminator
class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv3d(input_shape[0], 32, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1)
        self.fc = nn.Linear(128 * 8 * 8 * 8, 1)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

# Initialize models
noise_dim = 100
output_shape = (1, 28, 32, 825683)  # Example shape: 1 channel, 64x64 spatial, 64 temporal
generator = Generator(noise_dim=noise_dim, output_shape=output_shape)
discriminator = Discriminator(input_shape=output_shape)
batch_size = 32
train_x, test_x, validation_x = matrix.split(.8, .1, .1)
# Loss and optimizers
criterion = nn.BCELoss()
gen_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
disc_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Example Training Loop
epochs = 10000
for epoch in range(epochs):
    # Generate fake data
    noise = torch.randn(batch_size, noise_dim)
    fake_data = generator(noise)

    # Train Discriminator on real and fake data
    disc_optimizer.zero_grad()

    # real_data = torch.randn(batch_size, *output_shape)  # Replace with actual data
    real_data = train_x
    real_labels = torch.ones(batch_size, 1)
    fake_labels = torch.zeros(batch_size, 1)

    real_loss = criterion(discriminator(real_data), real_labels)
    fake_loss = criterion(discriminator(fake_data.detach()), fake_labels)
    disc_loss = real_loss + fake_loss

    disc_loss.backward()
    disc_optimizer.step()

    # Train Generator
    gen_optimizer.zero_grad()

    gen_loss = criterion(discriminator(fake_data), real_labels)  # Trick the discriminator
    gen_loss.backward()
    gen_optimizer.step()

    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Disc Loss: {disc_loss.item()}, Gen Loss: {gen_loss.item()}')

