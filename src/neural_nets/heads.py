import torch
import torch.nn.functional as F
from torch import nn

from .types import ActivationFactory
from .utils import orthogonal_initialization_

class BasicActorHead(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_main_actions: int,
        activation: ActivationFactory,
    ) -> None:
        super().__init__()
        self.main_actor_linear = nn.Sequential(
            nn.Linear(in_features=d_model+1, out_features=d_model),
            activation(),
            nn.Linear(in_features=d_model, out_features=n_main_actions)
        )
        
