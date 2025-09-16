from abc import ABC, abstractmethod
from typing import List
import numpy as np
from tools.types import Detection, TrackInfo

class Tracker(ABC):
    @abstractmethod
    def update(self, dets: List[Detection], frame: np.ndarray) -> List[TrackInfo]:
        ...
