"""
models_baseline.py - Baseline Model Implementations for TQF Comparison

This module implements three baseline neural network architectures for
"apples-to-apples" comparison with the TQF-ANN model:

1. FC-MLP: Fully-connected multi-layer perceptron
2. CNN-L5: 5-layer convolutional neural network
3. ResNet-18-Scaled: Scaled-down ResNet-18 for MNIST

All models are parameter-matched to approximately 650,000 parameters
to ensure fair comparison with TQF-ANN. Parameter matching is handled
by the param_matcher module.

KEY FEATURES:
=============
- Parameter-matched architectures (~650K params each)
- Consistent training interface across all models
- Proper handling of MNIST input (28x28 grayscale)
- Batch normalization and dropout for regularization
- Type-hinted code for clarity
- Factory pattern via MODEL_REGISTRY

USAGE:
======
from models_baseline import get_model

# Get parameter-matched models
mlp = get_model('FC-MLP')
cnn = get_model('CNN-L5')
resnet = get_model('ResNet-18-Scaled')
tqf = get_model('TQF-ANN', R=20, hidden_dim=80)

# All models have same interface
logits = model(x)  # x: (batch, 784) or (batch, 1, 28, 28)

Author: Nathan O. Schmidt
Organization: Cold Hammer Research & Development LLC
License: MIT License
Date: February 2026
"""

# MIT License
#
# Copyright (c) 2026 Nathan O. Schmidt, Cold Hammer Research & Development LLC
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

from config import DROPOUT_DEFAULT
# LAZY IMPORT: Import TQFANN only when needed to avoid slow startup
# from models_tqf import TQFANN  # Moved to get_model() function
from param_matcher import (
    tune_mlp_for_params,
    tune_cnn_for_params,
    tune_resnet_for_params,
    TARGET_PARAMS
)


class FCMLP(nn.Module):
    """
    Fully-connected multi-layer perceptron baseline for MNIST.

    A simple feedforward network with ReLU activations and dropout
    regularization. Serves as the simplest baseline to demonstrate
    the benefits of more sophisticated architectures.

    Architecture:
        Input (784) -> Linear -> ReLU -> Dropout -> ... -> Linear -> Output (10)

    The number and size of hidden layers are automatically tuned via
    param_matcher.py to match the target parameter count (~650K).

    Why this baseline?
    - Simplest possible architecture
    - No spatial structure exploitation
    - No inductive biases
    - Pure capacity comparison

    Example:
        >>> model = FCMLP(hidden_sizes=[460, 460, 460])
        >>> x = torch.randn(32, 784)
        >>> logits = model(x)
        >>> logits.shape
        torch.Size([32, 10])
    """

    def __init__(
        self,
        in_features: int = 784,
        hidden_sizes: Optional[List[int]] = None,
        num_classes: int = 10,
        dropout: float = DROPOUT_DEFAULT
    ):
        """
        Initialize FC-MLP.

        Args:
            in_features: Input dimension (default: 784 for MNIST)
            hidden_sizes: List of hidden layer sizes (auto-tuned if None)
            num_classes: Number of output classes (default: 10)
            dropout: Dropout probability (default: 0.2)
        """
        super().__init__()

        # Validate inputs
        assert in_features > 0, "in_features must be positive"
        assert num_classes > 0, "num_classes must be positive"
        assert 0.0 <= dropout < 1.0, "dropout must be in [0, 1)"

        if hidden_sizes is None:
            # Auto-tune for target parameter count
            hidden_sizes = tune_mlp_for_params(num_layers=3)

        self.in_features: int = in_features
        self.hidden_sizes: List[int] = hidden_sizes
        self.num_classes: int = num_classes

        # Build sequential network
        layers: List[nn.Module] = []
        prev_dim: int = in_features

        for hidden_dim in hidden_sizes:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))

        self.network: nn.Sequential = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through network.

        Args:
            x: Input tensor (shape: B x 784 or B x 1 x 28 x 28)

        Returns:
            Logits (shape: B x num_classes)
        """
        # Flatten to (batch, 784) if needed
        x_flat: torch.Tensor = x.view(x.size(0), -1)
        return self.network(x_flat)

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class CNNL5(nn.Module):
    """
    5-layer convolutional neural network baseline for MNIST.

    A standard CNN with alternating convolution-pooling blocks followed
    by fully-connected layers. Exploits spatial structure but uses
    conventional grid-based convolutions (not graph convolutions).

    Architecture:
        Conv -> ReLU -> Pool -> Conv -> ReLU -> Pool -> Conv -> ReLU -> Pool -> FC -> FC

    Features:
    - Batch normalization for training stability
    - Max pooling for spatial downsampling
    - Dropout for regularization
    - Parameter-matched to ~650K

    Why this baseline?
    - Exploits spatial locality (unlike MLP)
    - Standard CNN architecture
    - Grid-based convolutions (compare to TQF graph convolutions)

    Example:
        >>> model = CNNL5(conv_channels=[64, 128, 256])
        >>> x = torch.randn(32, 1, 28, 28)
        >>> logits = model(x)
        >>> logits.shape
        torch.Size([32, 10])
    """

    def __init__(
        self,
        in_channels: int = 1,
        conv_channels: Optional[List[int]] = None,
        num_classes: int = 10,
        dropout: float = DROPOUT_DEFAULT
    ):
        """
        Initialize CNN-L5.

        Args:
            in_channels: Input image channels (default: 1 for MNIST)
            conv_channels: List of conv layer output channels (auto-tuned if None)
            num_classes: Number of output classes (default: 10)
            dropout: Dropout probability (default: 0.2)
        """
        super().__init__()

        if conv_channels is None:
            # Auto-tune for target parameter count
            conv_channels, fc_size = tune_cnn_for_params()
            self.fc_size: int = fc_size
        else:
            self.fc_size: int = 256  # Default if channels provided manually

        self.in_channels: int = in_channels
        self.conv_channels: List[int] = conv_channels
        self.num_classes: int = num_classes

        # Convolutional layers with batch normalization
        self.conv1 = nn.Conv2d(in_channels, conv_channels[0], kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(conv_channels[0])
        self.pool1 = nn.MaxPool2d(2, 2)  # 28x28 -> 14x14

        self.conv2 = nn.Conv2d(conv_channels[0], conv_channels[1], kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(conv_channels[1])
        self.pool2 = nn.MaxPool2d(2, 2)  # 14x14 -> 7x7

        self.conv3 = nn.Conv2d(conv_channels[1], conv_channels[2], kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(conv_channels[2])
        self.pool3 = nn.MaxPool2d(2, 2)  # 7x7 -> 3x3

        # Calculate flattened feature size
        # After 3 pooling layers: 28 -> 14 -> 7 -> 3
        feature_size: int = conv_channels[2] * 3 * 3

        # Fully-connected layers
        self.fc1 = nn.Linear(feature_size, self.fc_size)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(self.fc_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through network.

        Args:
            x: Input tensor (shape: B x 1 x 28 x 28 or B x 784)

        Returns:
            Logits (shape: B x num_classes)
        """
        # Ensure 4D input (batch, channels, height, width)
        if x.dim() == 2:
            x = x.view(x.size(0), 1, 28, 28)
        elif x.dim() == 3:
            x = x.unsqueeze(1)

        # Convolutional blocks
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))

        # Flatten and fully-connected layers
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ResNet18Scaled(nn.Module):
    """
    Scaled-down ResNet-18 for MNIST (28x28 input).

    Implements residual learning with skip connections, scaled down to
    match the target parameter count (~650K). Residual connections help
    with gradient flow and enable training of deeper networks.

    Architecture:
        - Initial convolution (7x7, stride 2)
        - N residual blocks with channel doubling
        - Global average pooling
        - Fully-connected classifier

    Features:
    - Residual connections (identity shortcuts)
    - Batch normalization for stability
    - Progressive channel doubling
    - Parameter-matched to ~650K

    Why this baseline?
    - State-of-art architecture (residual learning)
    - Gradient flow via skip connections
    - Deeper network possible
    - Compare residual vs graph-based propagation

    Example:
        >>> model = ResNet18Scaled()
        >>> x = torch.randn(32, 1, 28, 28)
        >>> logits = model(x)
        >>> logits.shape
        torch.Size([32, 10])
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        base_channels: Optional[int] = None,
        num_blocks: Optional[int] = None
    ):
        """
        Initialize scaled ResNet-18.

        Args:
            in_channels: Input image channels (default: 1 for MNIST)
            num_classes: Number of output classes (default: 10)
            base_channels: Base channel count (auto-tuned if None)
            num_blocks: Number of residual blocks (auto-tuned if None)
        """
        super().__init__()

        if base_channels is None or num_blocks is None:
            # Auto-tune for target parameter count
            num_blocks, base_channels = tune_resnet_for_params()

        self.in_channels: int = in_channels
        self.num_classes: int = num_classes
        self.base_channels: int = base_channels
        self.num_blocks: int = num_blocks

        # Initial convolution
        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)

        # Build residual blocks with channel doubling
        # Channels: [base, base, 2*base, 2*base, 4*base, 4*base, ...]
        blocks: List[nn.Module] = []
        in_ch: int = base_channels

        for block_idx in range(num_blocks):
            # Double channels every 2 blocks
            out_ch: int = base_channels * (2 ** (block_idx // 2))
            # Stride 2 when channels double (every 2 blocks after first)
            stride: int = 2 if (block_idx > 0 and block_idx % 2 == 0) else 1

            # Create residual block
            blocks.append(self._make_simple_block(in_ch, out_ch, stride))
            in_ch = out_ch

        self.residual_blocks = nn.Sequential(*blocks)

        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_ch, num_classes)

    def _make_simple_block(self, in_channels: int, out_channels: int, stride: int) -> nn.Module:
        """
        Create a simple residual block.

        Each block consists of:
        - Two 3x3 convolutions with batch normalization
        - ReLU activations
        - Skip connection (identity or projection)

        Args:
            in_channels: Input channels
            out_channels: Output channels
            stride: Stride for first convolution

        Returns:
            Residual block module
        """
        class SimpleBlock(nn.Module):
            """Simple residual block with two convolutions."""

            def __init__(self, in_ch: int, out_ch: int, stride: int):
                super().__init__()

                # First convolution (may downsample)
                self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride,
                                      padding=1, bias=False)
                self.bn1 = nn.BatchNorm2d(out_ch)

                # Second convolution
                self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1,
                                      padding=1, bias=False)
                self.bn2 = nn.BatchNorm2d(out_ch)

                # Skip connection (identity or projection)
                if in_ch != out_ch or stride != 1:
                    self.skip = nn.Sequential(
                        nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                        nn.BatchNorm2d(out_ch)
                    )
                else:
                    self.skip = nn.Identity()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                """Forward through residual block."""
                identity = self.skip(x)
                out = F.relu(self.bn1(self.conv1(x)))
                out = self.bn2(self.conv2(out))
                out = out + identity
                out = F.relu(out)
                return out

        return SimpleBlock(in_channels, out_channels, stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through network.

        Args:
            x: Input tensor (shape: B x 784 or B x 1 x 28 x 28 or B x 28 x 28)

        Returns:
            Logits (shape: B x num_classes)
        """
        # Ensure 4D input (batch, channels, height, width)
        if x.dim() == 2:
            x = x.view(x.size(0), 1, 28, 28)
        elif x.dim() == 3:
            x = x.unsqueeze(1)

        # Forward through network
        out: torch.Tensor = F.relu(self.bn1(self.conv1(x)))
        out = self.residual_blocks(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Model registry for factory pattern
MODEL_REGISTRY: dict = {
    'FC-MLP': FCMLP,
    'CNN-L5': CNNL5,
    'ResNet-18-Scaled': ResNet18Scaled,
    # TQF-ANN loaded lazily in get_model() to avoid slow import
}


def get_model(name: str, **kwargs) -> nn.Module:
    """
    Factory function to instantiate models from registry.

    Provides a unified interface for creating any baseline or TQF model
    with consistent parameter handling.

    Args:
        name: Model name from MODEL_REGISTRY
        **kwargs: Model-specific arguments

    Returns:
        Instantiated model

    Raises:
        ValueError: If model name not in registry

    Example:
        >>> model = get_model('FC-MLP')
        >>> model = get_model('CNN-L5', conv_channels=[32, 64, 128])
        >>> model = get_model('TQF-ANN', R=20, hidden_dim=80, fibonacci_mode='none')
    """
    # Handle TQF-ANN with lazy import (avoid slow startup)
    if name == 'TQF-ANN':
        from models_tqf import TQFANN
        return TQFANN(**kwargs)

    if name not in MODEL_REGISTRY:
        available_models: List[str] = list(MODEL_REGISTRY.keys()) + ['TQF-ANN']
        raise ValueError(
            f"Unknown model: {name}. Available: {available_models}"
        )

    model_class = MODEL_REGISTRY[name]
    return model_class(**kwargs)
