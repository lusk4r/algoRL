from abc import ABC, abstractmethod
import numpy as np


class RLAgent(ABC):    
    @abstractmethod
    def execute(self, obs: np.array, reward: float) -> np.array:
        ...
    
    @abstractmethod
    def save_model(self) -> None:
        ...

    @abstractmethod
    def load_model(self) -> None:
        ...

    @abstractmethod
    def set_test_setup(self) -> None:
        ...
