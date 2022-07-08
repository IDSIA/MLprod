from abc import ABC, abstractmethod
from readline import get_endidx
import numpy as np

class Generator(ABC):

    @abstractmethod
    def sample(self):
        raise NotImplementedError("Method not implemented")


class BetaGenerator(Generator):

    def __init__(self, a: float, b: float) -> None:
        super().__init__()

        assert(a > 0)
        assert(b > 0)

        self.a: float = a
        self.b: float = b


    def sample(self) -> float:
        return np.random.beta(self.a, self.b)

    
class NormalGenerator(Generator):

    def __init__(self, loc: float, scale: float) -> None:
        super().__init__()

        self.loc: float = loc
        self.scale: float = scale

    def sample(self) -> float:
        return np.random.normal(self.loc, self.scale)