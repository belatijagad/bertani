"""Policy-neutral action tensor containers for batched environments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


IntegerArray = NDArray[np.integer[Any]]


@dataclass(frozen=True, slots=True)
class ActionBatch:
    """Reusable action buffers compatible with :meth:`bertani.VecEnv.step`."""

    unit_actions: IntegerArray
    market_actions: IntegerArray
    market_lengths: IntegerArray


__all__ = ["ActionBatch", "IntegerArray"]
