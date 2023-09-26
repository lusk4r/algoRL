import os 
import random 
from abc import abstractmethod
from typing import Any, Tuple, List
from algorithms import RLAgent
import numpy as np
from algorithms.utils import get_states_delta_from_n_intervals
from collections import namedtuple


class TabularValueRLAgent(RLAgent):
    def __init__(self, actions: np.array,
                 obs_ranges: Tuple[Any], 
                 n_intervals: List[int]) -> None:        
        super().__init__()
        states_delta = get_states_delta_from_n_intervals(obs_ranges=obs_ranges, n_intervals=n_intervals)    
        
        states_grid = namedtuple("states_grid", ["delta_dim", "low_val", "n_intervals"])
        self.states_info: Tuple[Any] = states_grid(delta_dim=states_delta,
                                              low_val=obs_ranges[0, :],
                                              n_intervals=[d+1 for d in n_intervals])
        self.actions = actions
        self.action_index = None        
        self.curr_state_index = None           
        self.prev_state_index = None   
    
    def get_nearest_state_index(self, obs: np.array) -> Tuple[int]:
        ids = (obs - self.states_info.low_val)/self.states_info.delta_dim
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
    

class QLearningAgent(TabularValueRLAgent):
    def __init__(self, actions: np.array,
                 obs_ranges: Tuple[Any],                 
                 n_intervals: List[int],
                 lr: float = .1,                 
                 discount_rate: float = .95) -> None:          
        super().__init__(actions=actions, obs_ranges=obs_ranges,
                         n_intervals=n_intervals)                     
        self.q_star = np.zeros(shape=tuple(self.states_info.n_intervals) + (len(actions), ))       
        self.lr = lr
        self.discount_rate = discount_rate        
               
    def episode_start_setup(self, obs: np.array, action_index: int):
        self.prev_state_index = self.get_nearest_state_index(obs=obs)
        self.action_index = action_index

    def set_test_setup(self) -> None:        
        self.load_model(model_name='q_step')

    def episode_exit_setup(self):
        self.save_model(model_name='q_step')    

    def next_action_strategy(self) -> Tuple[float, int]:
        """
            get the maximum of the Q value with respect action.

            returns: expected optimal value and the index related to the chosen action.
        """                
        next_action_index = np.argmax(self.q_star[self.curr_state_index])        
        return self.q_star[self.curr_state_index + (next_action_index,)], next_action_index

    def execute(self, obs: np.array, reward: float, terminal: bool) -> Any:                
        if self.prev_state_index is None:
            raise ValueError("Set previous state index")
        
        self.curr_state_index = self.get_nearest_state_index(obs=obs)                                    
        optimal_val, next_action_index = self.next_action_strategy()
                
        self.q_star[self.prev_state_index + (self.action_index, )] += \
            self.lr*(reward+self.discount_rate*optimal_val - 
                    self.q_star[self.prev_state_index + (self.action_index, )])                    

        self.action_index = next_action_index        
        self.prev_state_index = self.curr_state_index                

        return self.actions[self.action_index]
        
    def save_model(self, model_name: str) -> None:
        model_path = os.path.join(self.model_dir, f'{model_name}.npy')
        np.save(model_path, self.q_star)
    
    def load_model(self, model_name: str) -> None:    
        model_path = os.path.join(self.model_dir, f'{model_name}.npy')        
        if not os.path.isfile(model_path):
            raise ValueError(f"Model {model_path} not available.",
                              "Start a new training routine or insert a previously trained model into `models` folder.")
        if os.path.isfile(model_path):
            self.q_star = np.load(model_path)
    

class EpsilonGreedyAgent(QLearningAgent):
    def __init__(self, actions: np.array,
                  obs_ranges: Tuple[Any],
                  n_intervals: List[int], 
                  lr: float = .1, 
                  discount_rate: float = .95) -> None:        
        super().__init__(actions=actions,
                         n_intervals=n_intervals, 
                         obs_ranges=obs_ranges,
                         lr=lr, discount_rate=discount_rate)             
        self.epsilon = 0.5
        self.epsilon_decay = 0.0001

    def episode_exit_setup(self):
        if self.epsilon >= 0.05:    
            self.epsilon -= self.epsilon_decay          
        super().episode_exit_setup()

    def set_test_setup(self) -> None:
        self.epsilon = 0.05
        super().set_test_setup()

    def next_action_strategy(self) -> Tuple[float, int]:
        """
            epsilon greedy strategy: 
                with probability epsilon 
                    choose an action at random
                with probability 1-epsilon 
                    get the maximum of the Q value with respect action 
        
            returns: expected optimal value and the index related to the chosen action.
        """ 
        next_action_index = np.argmax(self.q_star[self.curr_state_index])        
        optimal_exp_val = self.q_star[self.curr_state_index + (next_action_index, )]
        if random.uniform(0.0, 1.0) < self.epsilon:                                
            next_action_index = random.randint(0, self.q_star.shape[-1]-1)

        return optimal_exp_val, next_action_index


class ExplorationFuncAgent(QLearningAgent):
    def __init__(self, actions: np.array,
                  obs_ranges: Tuple[Any],
                  n_intervals: List[int],
                  lr: float = .1, 
                  discount_rate: float = .95) -> None:        
        super().__init__(actions=actions,
                         obs_ranges=obs_ranges,
                         n_intervals=n_intervals, 
                         lr=lr, discount_rate=discount_rate)                 
        self.curiosity = 1
        self.curiosity_decay = 0.001
        # n is the number of times action `a` was chosen in state `s` 
        self.n = self.q_star.copy() 
        
    def episode_exit_setup(self):
        if self.curiosity >= 0.0:    
            self.curiosity -= self.curiosity_decay
        super().episode_exit_setup()

    def set_test_setup(self) -> None:
        self.curiosity = 0.0
        super().set_test_setup()

    def next_action_strategy(self) -> Tuple[float, int]:
        """
            get the maximum of the exploration function f with respect action.
                f(q, n) = q + K/(1+n)
                    where K is the curiosity hyperparameter

            returns: expected optimal value and the index related to the chosen action.
        """ 
        expl_func = self.q_star[self.curr_state_index] + self.curiosity/(1+self.n[self.curr_state_index])
        next_action_index = np.argmax(expl_func)
        optimal_exp_val = expl_func[next_action_index]
        self.n[self.curr_state_index + (next_action_index, )] += 1
        
        return optimal_exp_val, next_action_index
    

class SarsaAgent(EpsilonGreedyAgent):
    def __init__(self, actions: np.array,
                  obs_ranges: Tuple[Any],
                  n_intervals: List[int],
                  lr: float = .1, 
                  discount_rate: float = .95) -> None:        
        super().__init__(actions=actions,
                         n_intervals=n_intervals,
                         obs_ranges=obs_ranges,
                         lr=lr, discount_rate=discount_rate)                 

    def next_action_strategy(self) -> Tuple[float, int]:
        """
            epsilon greedy strategy: 
                with probability epsilon 
                    choose an action at random
                with probability 1-epsilon 
                    get the maximum of the Q value with respect action 
        
            returns: expected optimal value and the index related to the chosen action.
        """ 
        next_action_index = np.argmax(self.q_star[self.curr_state_index])        
        if random.uniform(0.0, 1.0) < self.epsilon:            
            next_action_index = random.randint(0, self.q_star.shape[-1]-1)

        return self.q_star[self.curr_state_index + (next_action_index, )], next_action_index        
