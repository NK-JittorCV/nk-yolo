"""Activation modules."""

import jittor as jt
from jittor import nn

class AGLU(nn.Module):
    """Unified activation function module from https://github.com/kostas1515/AGLU."""

    def __init__(self, device=None, dtype=None) -> None:
        """Initialize the Unified activation function."""
        super().__init__()
        self.act = nn.Softplus(beta=-1.0)
        self.lambd = nn.Parameter(nn.init.uniform_(jt.empty(1, device=device, dtype=dtype)))  # lambda parameter
        self.kappa = nn.Parameter(nn.init.uniform_(jt.empty(1, device=device, dtype=dtype)))  # kappa parameter

    def execute(self, x: jt.Var) -> jt.Var:
        """Compute the forward pass of the Unified activation function."""
        lam = jt.clamp(self.lambd, min=0.0001)
        return jt.exp((1 / lam) * self.act((self.kappa * x) - jt.log(lam)))
