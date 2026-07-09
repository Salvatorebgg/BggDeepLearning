# -*- coding: utf-8 -*-

"""
BggDeepLearning PyTorch MLP model for tabular clinical data

File:
python/bggdeep/models/deep_learning/mlp.py

Purpose:
1. Define a flexible Multi-Layer Perceptron (MLP) for clinical risk prediction
2. Support configurable hidden layers, dropout, batch normalization
3. Compatible with the project's preprocessed tabular data
4. Output binary classification probability via sigmoid

Clinical task:
poor_outcome prediction from structured clinical variables
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn as nn


@dataclass
class MLPConfig:
    """
    Configuration for tabular MLP model.

    Parameters
    ----------
    input_dim:
        Number of input features (set from data automatically).

    hidden_dims:
        List of hidden layer dimensions. Each element defines one hidden layer.
        Default [128, 64, 32] means 3 hidden layers.

    dropout:
        Dropout rate applied after each hidden layer (except the last).
        0.0 means no dropout.

    use_batch_norm:
        Whether to apply BatchNorm1d after each hidden layer.

    activation:
        Activation function name. Supported: "relu", "gelu", "leaky_relu".

    random_seed:
        Random seed for reproducibility.
    """

    input_dim: int = 26
    hidden_dims: List[int] = field(default_factory=lambda: [128, 64, 32])
    dropout: float = 0.3
    use_batch_norm: bool = True
    activation: str = "relu"
    random_seed: int = 42


class TabularMLP(nn.Module):
    """
    Multi-Layer Perceptron for tabular clinical data.

    Architecture:
    Input -> [Linear -> BatchNorm -> Activation -> Dropout] * N -> Linear -> Sigmoid

    The final layer outputs a single logit, and the forward method
    applies sigmoid to produce a probability in [0, 1].
    """

    def __init__(self, config: MLPConfig) -> None:
        super().__init__()

        self.config = config
        self._build_network()

    def _get_activation(self) -> nn.Module:
        """
        Get activation function by name.
        """
        name = self.config.activation.lower()

        if name == "relu":
            return nn.ReLU(inplace=True)
        elif name == "gelu":
            return nn.GELU()
        elif name == "leaky_relu":
            return nn.LeakyReLU(negative_slope=0.01, inplace=True)
        else:
            raise ValueError(
                f"Unsupported activation: {name}. "
                f"Use relu, gelu, or leaky_relu."
            )

    def _build_network(self) -> None:
        """
        Build MLP layers dynamically from config.
        """
        layers: List[nn.Module] = []
        in_features = self.config.input_dim

        for i, hidden_dim in enumerate(self.config.hidden_dims):
            # Linear layer
            layers.append(nn.Linear(in_features, hidden_dim))

            # Batch normalization
            if self.config.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            # Activation
            layers.append(self._get_activation())

            # Dropout
            if self.config.dropout > 0:
                layers.append(nn.Dropout(self.config.dropout))

            in_features = hidden_dim

        # Final classification head
        layers.append(nn.Linear(in_features, 1))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """
        Initialize Linear layer weights with Kaiming uniform,
        and biases with zeros.
        """
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x:
            Input tensor of shape (batch_size, input_dim).

        Returns
        -------
        torch.Tensor
            Predicted probability of shape (batch_size, 1), values in [0, 1].
        """
        logits = self.network(x)
        probabilities = torch.sigmoid(logits)
        return probabilities

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict probability (convenience method, same as forward).
        """
        self.eval()
        with torch.no_grad():
            return self.forward(x)

    def get_model_size(self) -> int:
        """
        Return total number of trainable parameters.
        """
        return sum(
            param.numel()
            for param in self.parameters()
            if param.requires_grad
        )

    def summary(self) -> str:
        """
        Return a human-readable model summary string.
        """
        lines = []
        lines.append("=" * 60)
        lines.append("TabularMLP Model Summary")
        lines.append("-" * 60)
        lines.append(f"Input dimension: {self.config.input_dim}")
        lines.append(f"Hidden layers: {self.config.hidden_dims}")
        lines.append(f"Dropout: {self.config.dropout}")
        lines.append(f"Batch normalization: {self.config.use_batch_norm}")
        lines.append(f"Activation: {self.config.activation}")
        lines.append(f"Trainable parameters: {self.get_model_size():,}")
        lines.append("-" * 60)

        for name, module in self.network.named_children():
            lines.append(f"  {name}: {module}")

        lines.append("=" * 60)

        return "\n".join(lines)


def build_mlp_from_data(
    input_dim: int,
    hidden_dims: List[int] | None = None,
    dropout: float = 0.3,
    use_batch_norm: bool = True,
    activation: str = "relu",
    random_seed: int = 42,
) -> TabularMLP:
    """
    Factory function to build an MLP model.

    This is a convenience function that infers hidden_dims
    based on input_dim if not explicitly provided.

    Rule of thumb:
    - If input_dim <= 20: use [64, 32]
    - If input_dim <= 50: use [128, 64, 32]
    - If input_dim > 50: use [256, 128, 64, 32]
    """
    if hidden_dims is None:
        if input_dim <= 20:
            hidden_dims = [64, 32]
        elif input_dim <= 50:
            hidden_dims = [128, 64, 32]
        else:
            hidden_dims = [256, 128, 64, 32]

    config = MLPConfig(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        use_batch_norm=use_batch_norm,
        activation=activation,
        random_seed=random_seed,
    )

    return TabularMLP(config)


# Quick self-test
if __name__ == "__main__":
    # Create a small model and run a forward pass
    config = MLPConfig(input_dim=26, hidden_dims=[64, 32], dropout=0.3)
    model = TabularMLP(config)

    print(model.summary())

    # Test forward pass
    batch_size = 16
    x_dummy = torch.randn(batch_size, 26)
    model.eval()
    with torch.no_grad():
        probs = model(x_dummy)
    print(f"\nInput shape: {x_dummy.shape}")
    print(f"Output shape: {probs.shape}")
    print(f"Output range: [{probs.min().item():.4f}, {probs.max().item():.4f}]")
    print(f"Output sample: {probs[:5].squeeze().tolist()}")
    print("\nMLP module self-test: PASSED")
