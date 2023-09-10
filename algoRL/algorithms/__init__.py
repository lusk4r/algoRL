from abc import ABC, abstractmethod
import numpy as np


class RLAlgorithm(ABC):
    def __init__(self, actions: np.array, states: np.array) -> None:
        self.actions = actions
        self.states = states     

    @abstractmethod
    def execute(self, obs: np.array, reward: np.array) -> np.array:
        ...
