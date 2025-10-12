
import torch


class rec2d3d(torch.nn.Module):
    def __init__(self,
                 num_of_mips=16,
                 depth=1):
        super().__init__()
        self.num_of_mips = num_of_mips
        self.depth = depth
        self.inplace = True

        self.input_layer = self._input_layer()
        self.depth_layers = self._depth_layers()
        self.output_layer = self._output_layer()

    def _input_layer(self):
        seq = torch.nn.Sequential()

        seq.append(torch.nn.Flatten(start_dim=2, end_dim=-1))
        seq.append(torch.nn.Linear(in_features=self.num_of_mips*400, out_features=400, bias=True))
        seq.append(torch.nn.ReLU(inplace=self.inplace))

        return seq

    def _depth_layers(self):
        seq = torch.nn.Sequential()

        for _ in range(self.depth):
            seq.append(torch.nn.Linear(in_features=400, out_features=400, bias=True))
            seq.append(torch.nn.ReLU(inplace=self.inplace))

        return seq

    def _output_layer(self):
        seq = torch.nn.Sequential(torch.nn.Linear(in_features=400, out_features=400*400, bias=True),
                                  torch.nn.ReLU(inplace=self.inplace)
                                  )

        # Apply Sigmoid final activation
        seq.append(torch.nn.Sigmoid())

        # Reshape output
        seq.append(torch.nn.Unflatten(dim=2, unflattened_size=(400, 400, 1)))

        # # Apply Sigmoid final activation
        # seq.append(torch.nn.Sigmoid())

        return seq

    def forward(self, x):
        self.ORIG_SHAPE = x.shape
        self.LENGTH = self.WIDTH = self.ORIG_SHAPE[3]
        if self.ORIG_SHAPE[2] != self.num_of_mips:
            raise ValueError(f'Number of MIPs in input is {self.ORIG_SHAPE[2]}, expected {self.num_of_mips}')

        x = self.input_layer(x)
        x = self.depth_layers(x)
        x = self.output_layer(x)

        return x


# # Crete instance of model
# mymodel = rec2d3d(num_of_mips=16, depth=10).to('cuda')
#
# # Check number of parameters
# pytorch_total_params = sum(p.numel() for p in mymodel.parameters() if p.requires_grad)
#
# # Input to try
# input = torch.ones(size=(16, 400)).to('cuda')
#
# # Prediction of model
# output = mymodel(input)

