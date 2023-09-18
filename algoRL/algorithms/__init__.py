import os
from abc import ABC, abstractmethod
import numpy as np


class RLAgent(ABC):    
    def __init__(self) -> None:    
        self.model_dir = os.path.join(\
            os.path.abspath(os.path.dirname(__file__)), '..', 'models')
        if not os.path.isdir(self.model_dir):
            os.makedirs(self.model_dir)        

    @abstractmethod
    def execute(self, obs: np.array, reward: float, terminal: bool) -> np.array:
        ...
    
    @abstractmethod
    def save_model(self, model_name: str) -> None:
        ...

    @abstractmethod
    def load_model(self, model_name: str) -> None:
        ...

    @abstractmethod
    def set_test_setup(self) -> None:
        ...
