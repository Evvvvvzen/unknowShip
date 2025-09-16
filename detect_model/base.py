from abc import ABC, abstractmethod
from typing import List
import numpy as np
from tools.types import Detection

class Detector(ABC):
    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Detection]:
        ...
