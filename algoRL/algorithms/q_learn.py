import random 
from typing import Any
from algorithms import RLAlgorithm
import numpy as np


class QLearning(RLAlgorithm):
    def __init__(self,
                  actions: np.array, states: np.array, 
                  lr: float = .1, 
                  discount_rate: float = .95) -> None:
       super().__init__(actions, states)
       self.q_star = np.zeros(shape=(states.shape[0], actions.shape[0]))
       self.action_index = None
       self.curr_state_index = None           
       self.prev_state_index = None
       self.lr = lr
       self.discount_rate = discount_rate
       self.epsilon = 0.85
       self.epsilon_decay = 0.05

    def decay_epsilon(self):
        if self.epsilon >= 0.05:    
            self.epsilon -= self.epsilon_decay      
    
    def get_nearest_state_index(self, obs: np.array) -> int:
        min_dist = np.inf            
        min_id = -1                      
        for s_index, s in enumerate(self.states):
            if s_index != self.prev_state_index and \
                min_dist > np.linalg.norm(s - obs):        
                min_id = s_index      
                min_dist = np.linalg.norm(s - obs)
        return min_id                                      
                
    def execute(self, obs: np.array, reward: np.array) -> Any:                
        if self.prev_state_index is None:
            raise ValueError("Set previous state index")
        
        self.curr_state_index = self.get_nearest_state_index(obs=obs)    
                            
        next_action_index = np.argmax(self.q_star[self.curr_state_index,:])

        self.q_star[self.prev_state_index, self.action_index] += \
            self.lr*(reward+self.discount_rate*
                    self.q_star[self.curr_state_index, next_action_index] - 
                    self.q_star[self.prev_state_index, self.action_index])
        
        if random.random() < self.epsilon:            
            next_action_index = random.randint(0, self.actions.shape[0]-1)
        
        self.action_index = next_action_index        
        self.prev_state_index = self.curr_state_index                

        return self.actions[self.action_index]