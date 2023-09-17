import os
import numpy as np
from abc import abstractmethod
from copy import deepcopy
from random import sample, uniform, randint
from typing import Deque, Optional
from dataclasses import dataclass 
from collections import deque, namedtuple
from algorithms import RLAgent
from torch import save, load, from_numpy, max, cat
from torch.nn import Module, Linear, Sequential, MSELoss
from torch.nn.functional import relu
from torch.cuda import is_available
from torch.optim import AdamW


class DeepValueRLAgent(RLAgent):
    @abstractmethod
    def episode_start_setup(self, obs: np.array, action: np.array):
        ...

    @abstractmethod
    def episode_exit_setup(self):
        ...
    

class FCNetwork(Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.l_1 = Linear(in_features, 128)
        self.l_2 = Linear(128, 128)
        self.l_3 = Linear(128, out_features)

    def forward(self, x):
        x = relu(self.l_1(x))
        x = relu(self.l_2(x))
        return self.l_3(x)    
    

class SimpleDQNAgent(DeepValueRLAgent):    
    def __init__(self, obs_ranges: np.array, actions: np.array,                   
                 batch_size: int, lr: float,
                 epsilon: float = 0.5, epsilon_decay: float = 0.001, 
                 device: Optional[str] = None) -> None: 
        self.device = device 
        if self.device is None:
            self.device = 'cuda' if is_available() else 'cpu'                    

        self.memory = ReplayMemory(examples=deque([], maxlen=10**6))        
        self.policy_nn = FCNetwork(in_features=obs_ranges.shape[1], 
                                   out_features=actions.reshape(-1, 1).shape[0]).to(self.device)        
        self.target_nn = deepcopy(self.policy_nn).to(self.device)
        self.optimizer = AdamW(self.policy_nn.parameters(), lr=lr, amsgrad=True)
        self.loss_func = MSELoss()
        self.losses = []

        self.batch_size = batch_size 
        self.lr = lr
        self.epsilon = epsilon 
        self.epsilon_decay = epsilon_decay
        self.discount = 0.95        

        self.action_space = actions        

        self.steps_before_update = 10 
        self.update_target_count = 0

        
    def episode_start_setup(self, obs: np.array, action_index: int):            
        self.prev_state = obs
        self.action_index = action_index

    def set_test_setup(self) -> None:        
        self.epsilon = 0.05
        self.load_model(model_name='dqn_episode')
    
    def episode_exit_setup(self):
        if self.epsilon >= 0.05:    
            self.epsilon -= self.epsilon_decay    

        self.save_model(model_name='dqn_episode')  

        mean_loss = np.array(self.losses)/len(self.losses)
        print(f"avg loss: {mean_loss}")

    def execute(self, obs: np.array, reward: np.array) -> float:                                
        # put data in replay memory 
        self.memory.insert_example(Example(s_t=self.prev_state, 
                                           a_t=self.action_index, 
                                           s_next_t=obs, 
                                           reward_t=reward))
        
        # optimization loop 
        # sample data
        try:            
            samples = self.memory.get_examples(sample_dim=self.batch_size)
        except ReplayMemoryNotReady:     
            self.action_index = randint(0, len(self.action_space)-1)       
            self.prev_state = obs
            return self.action_space[self.action_index]
        
        # prepare training set 
        x, y = None, None
        for (s_t, a_t, s_next_t, reward_t) in samples:                    
            current_q = self.policy_nn(from_numpy(s_t).to(self.device))            
            next_q = self.target_nn(from_numpy(s_next_t).to(self.device))

            current_q[a_t] += reward_t + self.discount * max(next_q)
                                    
            if x is None:
                x = from_numpy(s_t).unsqueeze(0).to(self.device)
                y = current_q.unsqueeze(0).to(self.device)
            else:
                x = cat((x, from_numpy(s_t).unsqueeze(0).to(self.device)), dim=0).to(self.device)
                y = cat((y, current_q.unsqueeze(0).to(self.device)), dim=0).to(self.device)

        # train policy nn 
        loss = self.loss_func(self.policy_nn(x), y)        
        self.losses.append(loss)        
        self.optimizer.zero_grad() # intialize 
        loss.backward()
        self.optimizer.step()

        # target nn weights update logic
        if self.update_target_count >= self.steps_before_update:
            self.target_nn.load_state_dict(self.policy_nn.state_dict())
            self.update_target_count = 0        
        else:
            self.update_target_count += 1      

        # update state and action 
        self.prev_state = obs        

        if uniform(0.0, 1.0) < self.epsilon:
            self.action_index = randint(0, len(self.action_space)-1)
        else:
            self.action_index = self.policy_nn.forward(from_numpy(obs).to(self.device)).argmax()
        
        return self.action_space[self.action_index]

    def load_model(self, model_name: str) -> None:
        model_path = os.path.join(self.model_dir, f'{model_name}')
        if not os.path.isfile(model_path):
            raise ValueError(f"Model {model_path} not available.",
                              "Start a new training routine or insert a previously trained model into `models` folder.")
        self.policy_nn.load_state_dict(
            load(model_path)
            )
    
    def save_model(self, model_name: str) -> None:
        save(self.policy_nn.state_dict(), 
             os.path.join(self.model_dir, f'{model_name}'))        


Example = namedtuple("example", ['s_t', 'a_t', 's_next_t', 'reward_t'])


class ReplayMemoryNotReady(Exception):
    ...


@dataclass
class ReplayMemory:
    examples: Deque[Example]

    def get_examples(self, sample_dim: int):
        if sample_dim > len(self.examples):
            raise ReplayMemoryNotReady
        # randomly select a subset of examples from the replay memory
        return sample(self.examples, k=sample_dim)

    def insert_example(self, example: Example):
        self.examples.append(example)

