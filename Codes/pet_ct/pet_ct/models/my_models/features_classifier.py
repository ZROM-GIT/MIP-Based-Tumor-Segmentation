import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureClassifier(nn.Module):
    def __init__(self, input_size=768, num_classes=4, hidden_sizes=[512, 256, 128], dropout_rate=0.0):
        """
        Neural network classifier for 1D feature tensors.

        Args:
            input_size (int): Size of input features (default: 768)
            num_classes (int): Number of output classes (default: 4)
            hidden_sizes (list): List of hidden layer sizes (default: [512, 256, 128])
            dropout_rate (float): Dropout rate for regularization (default: 0.3)
        """
        super(FeatureClassifier, self).__init__()

        self.input_size = input_size
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        # Build the network layers
        layers = []
        prev_size = input_size

        # Hidden layers
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size

        # Output layer (no activation, will be handled by loss function)
        layers.append(nn.Linear(prev_size, num_classes))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize network weights using Xavier/Glorot initialization."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 768)

        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes)
        """
        # Flatten if needed (in case input has extra dimensions)
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        # Ensure input has correct size
        if x.size(1) != self.input_size:
            raise ValueError(f"Expected input size {self.input_size}, got {x.size(1)}")

        return self.network(x)

    def predict_class(self, x):
        x = nn.functional.softmax(x, dim=1)
        predicted_class = torch.argmax(x, dim=1)
        return predicted_class

    def predict(self, x):
        """
        Make predictions (returns class indices).

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Predicted class indices
        """
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits, dim=1)

    def predict_proba(self, x):
        """
        Get prediction probabilities.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Class probabilities
        """
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=1)