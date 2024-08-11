import torch
from torch import nn

class FullyConnectedNetwork(nn.Module):
    def __init__(self, obs_shape: int | tuple[int], action_shape: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(torch.prod(torch.tensor(obs_shape)).item(), 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, action_shape)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x.flatten(start_dim=1))
