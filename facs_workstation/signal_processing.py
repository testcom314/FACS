import math

"""Signal smoothing primitives."""

from .config import DEFAULT_SMOOTHING
from .utils import safe_float

class SignalSmoother:
    def __init__(self, alpha=DEFAULT_SMOOTHING):
        self.alpha = float(alpha)
        self.values = {}

    def reset(self):
        self.values.clear()

    def update(self, key, value):
        value = safe_float(value)
        if not math.isfinite(value):
            return value
        if key not in self.values or not math.isfinite(self.values[key]):
            self.values[key] = value
        else:
            self.values[key] = self.alpha * value + (1.0 - self.alpha) * self.values[key]
        return self.values[key]
