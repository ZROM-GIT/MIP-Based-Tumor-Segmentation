import torch


class TinyModel(torch.nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 out_channels: int = 2
                 ) -> None:
        super(TinyModel, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.input_layer = self._input()
        self.output_layer = self._output()

    def _input(self):
        seq = torch.nn.Sequential(
            torch.nn.Conv3d(in_channels=1, out_channels=25, kernel_size=(5, 21, 3), padding=(0, 10, 1),
                            padding_mode='zeros'),
            torch.nn.ReLU(),
            torch.nn.Conv3d(in_channels=25, out_channels=100, kernel_size=(5, 21, 3), padding=(0, 10, 1),
                            padding_mode='zeros'),
            torch.nn.ReLU(),
            torch.nn.Conv3d(in_channels=100, out_channels=400, kernel_size=(5, 21, 3), padding=(0, 10, 1),
                            padding_mode='zeros'),
            torch.nn.ReLU(),
            torch.nn.Conv3d(in_channels=400, out_channels=400, kernel_size=(4, 21, 3), padding=(0, 10, 1),
                            padding_mode='zeros'),
            torch.nn.ReLU()
            )

        return seq

    def _output(self):
        seq = torch.nn.Sequential(torch.nn.Flatten(start_dim=2, end_dim=3),
                                  torch.nn.Unflatten(dim=0, unflattened_size=(1, 1)),
                                  torch.nn.Conv3d(in_channels=1, out_channels=2, kernel_size=3, padding=1),
                                  torch.nn.ReLU(),
                                  )

        return seq

    def forward(self, x):
        x = self.input_layer(x)
        x = self.output_layer(x)
        return x


if __name__ == '__main__':
    input = torch.rand(size=(1, 1, 16, 400, 600))

    model = TinyModel()

    output = model(input)
