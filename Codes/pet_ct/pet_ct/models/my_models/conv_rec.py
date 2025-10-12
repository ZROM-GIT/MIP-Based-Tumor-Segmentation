import torch
import torch.nn as nn

class Conv_rec(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv_rec, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        )

        self.decoder = nn.Sequential(
            nn.Conv3d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(16, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.upsample = nn.Upsample(size=(1, 400, 400), mode='trilinear')

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.upsample(x)
        return x

# # Assuming x is the unknown dimension. Make sure to replace it with the actual value.
# x = 28  # Replace with the actual value of x
#
# # Create a sample input
# sample_input = torch.randn(1, 1, 16, 400, x)
#
# # Instantiate the model
# model = Conv_rec(in_channels=1, out_channels=2)
#
# # Pass the input through the model
# output = model(sample_input)
#
# # Check the shape of the output
# print(output.shape)
