from typing import Any
import numpy as np


class Feature:

    def __init__(self, r: np.random.Generator, distribution: str) -> None:
        self.r = r
        self.distribution = distribution

    def __call__(self, *args: Any, **kwds: Any) -> float:
        return getattr(self.r, self.distribution)()
