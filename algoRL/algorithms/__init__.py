from typing import Union
from abc import ABC, abstractmethod
import numpy as np


class RLAlgorithm(ABC):
    def __init__(self, actions: np.array, states: np.array) -> None:
        self.actions = actions
        self.states = states     

    @abstractmethod
    def execute(self, obs: np.array, reward: np.array) -> np.array:
        ...


class ApproximateRL(RLAlgorithm):
    def __init__(self, actions: np.array, states: np.array) -> None:
        self.actions = actions
        self.states = states  
        self.action_index = None
        self.curr_state_index = None           
        self.prev_state_index = None   
    
    def get_nearest_state_index(self, obs: np.array) -> int:
        min_dist = np.inf            
        min_id = -1                      
        for s_index, s in enumerate(self.states):
            if s_index != self.prev_state_index and \
                min_dist > np.linalg.norm(s - obs):        
                min_id = s_index      
                min_dist = np.linalg.norm(s - obs)
        return min_id                                      

    @abstractmethod
    def episode_start_setup(self, obs: np.array, action: np.array):
        ...

    @abstractmethod
    def episode_exit_setup(self):
        ...

    @abstractmethod
    def next_action_strategy(self) -> Union[float, int]:
        ...
