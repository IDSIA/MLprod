from typing import Any
import numpy as np


def sample_int(r: np.random.Generator, min_val: int, max_val: int, a: float, b: float, size: int=1) -> np.ndarray[Any, np.dtype[np.int64]]:
    """Samples one or more integer from a beta distribution given two bounds.

    :param r:
      Random number generator.
    :param min_val:
      Minimum value (inclusive).
    :param max_val:
      Maximum value (exclusive).
    :param a:
      Parameter alpha for beta distirbution.
    :param b:
      Parameter beta for beta distirbution.
    :param size:
      Number of samples to generate (if 1, an int is returned).
    """
    n = max_val - min_val
    p = r.beta(a, b, n)
    p /= p.sum()
    c = r.choice(n, p=p, size=size) + min_val
    return c


def sample_float(r: np.random.Generator, min_val: float, max_val: float, a: float, b: float) -> float:
    """Samples one float from a beta distribution given two inclusive bounds.

    :param r:
      Random number generator.
    :param min_val:
      Minimum value.
    :param max_val:
      Maximum value.
    :param a:
      Parameter alpha for beta distirbution.
    :param b:
      Parameter beta for beta distirbution.
    """
    return r.beta(a, b) * (max_val - min_val) + min_val


def sample_bool(r: np.random.Generator, threshold: float, a: float, b: float) -> bool:
    """Samples one boolean from a beta distribution given a threshold. The 
    threshold is the probability to have this boolean.

    :param r:
      Random number generator.
    :param threshold:
      Threshold used for sample, return a True when the generated internal 
      value is below this threshold.
    :param a:
      Parameter alpha for beta distirbution.
    :param b:
      Parameter beta for beta distirbution.
    """
    return r.beta(a, b) < threshold


def sample_list(r: np.random.Generator, objects: list, a: float, b: float, size: int=1) -> Any:
    """Samples one or more objects from the given list using a beta distribution.

    :param r:
      Random number generator.
    :param objects:
      Objects to choice from.
    :param a:
      Parameter alpha for beta distirbution.
    :param b:
      Parameter beta for beta distirbution.
    :param size:
      Number of samples to generate.
    """
    n = len(objects)
    p = r.beta(a, b, n)
    p /= p.sum()
    c = r.choice(n, p=p, size=size)
    o = [objects[i] for i in c]
    if size == 1:
      return  o[0]
    return o
