from enum import Enum
from collections.abc import Callable

from torch import nn


ActivationFactory = Callable[[], nn.Module]

class RewardSpace(Enum):
    FINAL_WINNER = ...
    DIFF = ...
    