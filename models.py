import torch

# Neural networks that can be used on the next day wildfire spread dataset
# Make sure the training data is scrubbed of any target fire masks that have missing data

class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        x = x.reshape(-1, 1, 32, 32)
        return x


class BinaryClassifierCNN(torch.nn.Module):
    def __init__(self, in_channels, image_size):
        flattened_conv2_output_dimensions = (image_size//4)**2
        out_channels = image_size**2
        super(BinaryClassifierCNN, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 16, 5, 1, 2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, 5, 1, 2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(32 * flattened_conv2_output_dimensions, out_channels)
        )
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.out(x)
        x = x.reshape(-1, 1, 32, 32)
        return x


# Tutorial for the autoencoder: https://www.youtube.com/watch?v=345wRyqKkQ0
class Reshape(torch.nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args
    
    def forward(self, x):
        return x.view(self.shape)


class Trim(torch.nn.Module):
    def __init__(self, *args):
        super(Trim, self).__init__()
    
    def forward(self, x):
        return x[:, :, :32, :32]


class ConvolutionalAutoencoder(torch.nn.Module):
    def __init__(self):
        super(ConvolutionalAutoencoder, self).__init__()
        
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(12, 16, 3, 1, 0), # 32 x 32 -> 30 x 30
            torch.nn.LeakyReLU(0.01),
            torch.nn.Dropout(0.1),
            torch.nn.Conv2d(16, 32, 3, 2, 0), # 30 x 30 -> 14 x 14
            torch.nn.LeakyReLU(0.01),
            torch.nn.Dropout(0.1),
            torch.nn.Conv2d(32, 32, 3, 2, 0), # 14 x 14 -> 6 x 6
            torch.nn.Flatten(),
            torch.nn.Linear(1152, 2) # 1152 = 32 * 6  * 6
        )
        
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(2, 1152),
            Reshape(-1, 32, 6, 6),
            torch.nn.ConvTranspose2d(32, 32, 3, 1, 0), # 6 x 6 -> 8 x 8
            torch.nn.LeakyReLU(0.01),
            torch.nn.ConvTranspose2d(32, 16, 3, 2, 1), # 8 x 8 -> 15 x 15
            torch.nn.LeakyReLU(0.01),
            torch.nn.ConvTranspose2d(16, 16, 3, 2, 0), # 15 x 15 -> 31 x 31
            torch.nn.LeakyReLU(0.01),
            torch.nn.ConvTranspose2d(16, 1, 3, 1, 0), # 31 x 31 -> 33 x 33
            Trim()
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
