import random 
from typing import Any, Union
from algorithms import ApproximateRL
import numpy as np


class QLearning(ApproximateRL):
    def __init__(self,
                  actions: np.array, states: np.array, 
                  lr: float = .1, 
                  discount_rate: float = .95) -> None:
       super().__init__(actions, states)
       self.q_star = np.zeros(shape=(states.shape[0], actions.shape[0]))       
       self.lr = lr
       self.discount_rate = discount_rate
               
    def episode_start_setup(self, obs: np.array, action: np.array):
        self.prev_state_index = self.get_nearest_state_index(obs=obs)
        self.action_index = action

    def episode_exit_setup(self):
        ...

    def next_action_strategy(self) -> Union[float, int]:
        """
            get the maximum of the Q value with respect action.

            returns: expected optimal value and the index related to the chosen action.
        """
        next_action_index = np.argmax(self.q_star[self.curr_state_index,:])
        return self.q_star[self.curr_state_index, next_action_index], next_action_index

    def execute(self, obs: np.array, reward: np.array) -> Any:                
        if self.prev_state_index is None:
            raise ValueError("Set previous state index")
        
        self.curr_state_index = self.get_nearest_state_index(obs=obs)                                    
        optimal_val, next_action_index = self.next_action_strategy()

        self.q_star[self.prev_state_index, self.action_index] += \
            self.lr*(reward+self.discount_rate*optimal_val - 
                    self.q_star[self.prev_state_index, self.action_index])
             
        self.action_index = next_action_index        
        self.prev_state_index = self.curr_state_index                

        return self.actions[self.action_index]
    

class EpsilonGreedyQLearning(QLearning):
    def __init__(self, actions: np.array, states: np.array, lr: float = 0.1, discount_rate: float = 0.95) -> None:
        super().__init__(actions, states, lr, discount_rate)
        self.epsilon = 0.85
        self.epsilon_decay = 0.000001

    def episode_exit_setup(self):
        if self.epsilon >= 0.05:    
            self.epsilon -= self.epsilon_decay          

    def next_action_strategy(self) -> Union[float, int]:
        """
            epsilon greedy strategy: 
                with probability epsilon 
                    choose an action at random
                with probability 1-epsilon 
                    get the maximum of the Q value with respect action 
        
            returns: expected optimal value and the index related to the chosen action.
        """ 
        next_action_index = np.argmax(self.q_star[self.curr_state_index,:])
        optimal_exp_val = self.q_star[self.curr_state_index, next_action_index]
        if random.random() < self.epsilon:            
            next_action_index = random.randint(0, self.actions.shape[0]-1)

        return optimal_exp_val, next_action_index


class ExplorationFuncQLearning(QLearning):
    def __init__(self, actions: np.array, states: np.array, lr: float = 0.1, discount_rate: float = 0.95) -> None:
        super().__init__(actions, states, lr, discount_rate)      
        self.curiosity = 100
        self.curiosity_decay = 1.5
        # n is the number of times action `a` was chosen in state `s` 
        self.n = self.q_star.copy() 
        
    def episode_exit_setup(self):
        if self.curiosity >= 0.1:    
            self.curiosity /= self.curiosity_decay

    def next_action_strategy(self) -> Union[float, int]:
        """
            get the maximum of the exploration function f with respect action.
                f(q, n) = q + K/(1+n)
                    where K is the curiosity hyperparameter

            returns: expected optimal value and the index related to the chosen action.
        """ 
        expl_func = self.q_star[self.curr_state_index, :] + self.curiosity/(1+self.n[self.curr_state_index, :])
        next_action_index = np.argmax(expl_func)
        optimal_exp_val = expl_func[next_action_index]
        self.n[self.curr_state_index, next_action_index] += 1
        
        return optimal_exp_val, next_action_index
    

class Sarsa(EpsilonGreedyQLearning):
    def next_action_strategy(self) -> Union[float, int]:
        """
            epsilon greedy strategy: 
                with probability epsilon 
                    choose an action at random
                with probability 1-epsilon 
                    get the maximum of the Q value with respect action 
        
            returns: expected optimal value and the index related to the chosen action.
        """ 
        next_action_index = np.argmax(self.q_star[self.curr_state_index,:])        
        if random.random() < self.epsilon:            
            next_action_index = random.randint(0, self.actions.shape[0]-1)

        return self.q_star[self.curr_state_index, next_action_index], next_action_index        
