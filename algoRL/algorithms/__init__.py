from typing import Tuple, Dict, Any
from abc import ABC, abstractmethod
import numpy as np


class RLAlgorithm(ABC):

    @abstractmethod
    def execute(self, obs: np.array, reward: np.array) -> np.array:
        ...


class ApproximateRL(RLAlgorithm):
    def __init__(self, actions: np.array,
                 states_info: Dict[str, Any]) -> None:        
        self.actions = actions
        self.action_index = None
        self.states_info = states_info
        self.curr_state_index = None           
        self.prev_state_index = None   
    
    def get_nearest_state_index(self, obs: np.array) -> Tuple[int]:
        ids = (obs - self.states_info['low'])/self.states_info['delta']
        return tuple([round(id) for id in ids])
        
    @abstractmethod
    def episode_start_setup(self, obs: np.array, action: np.array):
        ...

    @abstractmethod
    def episode_exit_setup(self):
        ...

    @abstractmethod
    def next_action_strategy(self) -> Tuple[float, int]:
        ...

    def execute(self, obs: np.array, reward: np.array) -> np.array:
        ...