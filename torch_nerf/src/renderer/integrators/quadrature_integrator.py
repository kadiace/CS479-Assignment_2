"""
Integrator implementing quadrature rule.
"""

from typing import Tuple
from typeguard import typechecked

from jaxtyping import Float, jaxtyped
import torch
from torch_nerf.src.renderer.integrators.integrator_base import IntegratorBase


class QuadratureIntegrator(IntegratorBase):
    """
    Numerical integrator which approximates integral using quadrature.
    """

    @jaxtyped
    @typechecked
    def integrate_along_rays(
        self,
        sigma: Float[torch.Tensor, "num_ray num_sample"],
        radiance: Float[torch.Tensor, "num_ray num_sample 3"],
        delta: Float[torch.Tensor, "num_ray num_sample"],
    ) -> Tuple[Float[torch.Tensor, "num_ray 3"], Float[torch.Tensor, "num_ray num_sample"]]:
        """
        Computes quadrature rule to approximate integral involving in volume rendering.
        Pixel colors are computed as weighted sums of radiance values collected along rays.

        For details on the quadrature rule, refer to 'Optical models for
        direct volume rendering (IEEE Transactions on Visualization and Computer Graphics 1995)'.

        Args:
            sigma: Density values sampled along rays.
            radiance: Radiance values sampled along rays.
            delta: Distance between adjacent samples along rays.

        Returns:
            rgbs: Pixel colors computed by evaluating the volume rendering equation.
            weights: Weights used to determine the contribution of each sample to the final pixel color.
                A weight at a sample point is defined as a product of transmittance and opacity,
                where opacity (alpha) is defined as 1 - exp(-sigma * delta).
        """
        # TODO
        # HINT: Look up the documentation of 'torch.cumsum'.
        # Calculate opacity (alpha) for each sample
        delta_ = delta.to(device = sigma.device) # [num_ray, num_sample]
        alpha = 1 - torch.exp(-sigma * delta_) + 1e-10 # [num_ray, num_sample]

        # Compute transmittance
        transmittance = torch.exp(-torch.cumsum(sigma * delta_, dim=0)) + 1e-10 # [num_ray, num_sample]

        # Compute weights as the product of transmittance and alpha
        weights = transmittance * alpha # [num_ray, num_sample]

        # Calculate the final pixel color (rgbs) by summing up weighted radiance values
        rgbs = torch.sum(radiance * weights.unsqueeze(2), dim=1) / torch.sum(weights, dim=1).unsqueeze(1)
        rgbs = torch.clamp(rgbs, 0.0, 1.0)

        return rgbs, weights
