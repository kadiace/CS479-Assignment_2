"""
Pytorch implementation of MLP used in NeRF (ECCV 2020).
"""

from typing import Tuple
from typeguard import typechecked

from jaxtyping import Float, jaxtyped
import torch
import torch.nn as nn


class NeRF(nn.Module):
    """
    A multi-layer perceptron (MLP) used for learning neural radiance fields.

    For architecture details, please refer to 'NeRF: Representing Scenes as
    Neural Radiance Fields for View Synthesis (ECCV 2020, Best paper honorable mention)'.

    Attributes:
        pos_dim (int): Dimensionality of coordinate vectors of sample points.
        view_dir_dim (int): Dimensionality of view direction vectors.
        feat_dim (int): Dimensionality of feature vector within forward propagation.
    """

    def __init__(
        self,
        pos_dim: int,
        view_dir_dim: int,
        feat_dim: int = 256,
    ) -> None:
        """
        Constructor of class 'NeRF'.
        """
        super().__init__()

        # TODO
        self.mlp1 = nn.Sequential(
            nn.Linear(pos_dim, feat_dim),  # Input layer
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim),  # Hidden layers
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim),  # Hidden layers
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim),  # Hidden layers
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim),  # Hidden layers
            nn.ReLU(),
        )

        self.mlp2 = nn.Sequential(
            nn.Linear(pos_dim + feat_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim),  # Hidden layers
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim),  # Hidden layers
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim),  # Hidden layers
        )

        self.density = nn.Sequential(
            nn.Linear(feat_dim, 1),
            nn.ReLU()
        )

        self.mlp3 = nn.Sequential(
            nn.Linear(feat_dim + view_dir_dim, feat_dim//2),
            nn.ReLU(),
            nn.Linear(feat_dim//2, 3),
            nn.Sigmoid()
        )

    @jaxtyped
    @typechecked
    def forward(
        self,
        pos: Float[torch.Tensor, "num_sample pos_dim"],
        view_dir: Float[torch.Tensor, "num_sample view_dir_dim"],
    ) -> Tuple[Float[torch.Tensor, "num_sample 1"], Float[torch.Tensor, "num_sample 3"]]:
        """
        Predicts color and density.

        Given sample point coordinates and view directions,
        predict the corresponding radiance (RGB) and density (sigma).

        Args:
            pos: The positional encodings of sample points coordinates on rays.
            view_dir: The positional encodings of ray directions.

        Returns:
            sigma: The density predictions evaluated at the given sample points.
            radiance: The radiance predictions evaluated at the given sample points.
        """

        # TODO
        # Concatenate pos and view_dir
        x = self.mlp1(pos)
        x = torch.concat([x, pos], dim = -1)
        x = self.mlp2(x)

        density = self.density(x)

        x = torch.concat([view_dir, x], dim = -1)
        color = self.mlp3(x)

        return (density, color)
