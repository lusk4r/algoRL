import os
import numpy as np
from abc import abstractmethod
from copy import deepcopy
from random import sample, uniform, randint
from typing import Any, Deque, Optional
from dataclasses import dataclass 
from collections import deque, namedtuple
from algorithms import RLAgent

import torch
from torch import nn, optim, cuda

try:
    import tinygrad.tensor as tiny_tensor
    from tinygrad import nn as tiny_nn
    from tinygrad.nn.optim import AdamW
    from tinygrad.nn.state import safe_save, safe_load, get_parameters, get_state_dict, load_state_dict
except ImportError: 
    ...


class DeepValueRLAgent(RLAgent):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def episode_start_setup(self, obs: np.array, action: np.array, terminal: bool):
        ...

    @abstractmethod
    def episode_exit_setup(self):
        ...
    

class FCNetwork(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.relu = nn.ReLU()
        self.l_1 = nn.Linear(in_features, 128)
        self.l_2 = nn.Linear(128, 128)
        self.l_3 = nn.Linear(128, out_features)

    def forward(self, x):
        x = self.relu(self.l_1(x))
        x = self.relu(self.l_2(x))
        return self.l_3(x)    
    

class TinyFCNetwork:
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()        
        self.l_1 = tiny_nn.Linear(in_features, 128)
        self.l_2 = tiny_nn.Linear(128, 128)
        self.l_3 = tiny_nn.Linear(128, out_features)
    
    def __call__(self, x: tiny_tensor.Tensor) -> tiny_tensor.Tensor:       
        x = self.l_1(x).relu()        
        x = self.l_2(x).relu()
        return self.l_3(x)
    

class SimpleDQNAgent(DeepValueRLAgent):    
    def __init__(self, obs_ranges: np.array, actions: np.array,                   
                 batch_size: int, lr: float,
                 epsilon: float = 0.5, epsilon_decay: float = 0.001, 
                 device: Optional[str] = None) -> None: 
        super().__init__()          
        self.device = device 
        if self.device is None:
            self.device = 'cuda' if cuda.is_available() else 'cpu'                    

        self.memory = ReplayMemory(examples=deque([], maxlen=50000))        
        self.q_value_nn = FCNetwork(in_features=obs_ranges.shape[1], 
                                    out_features=actions.reshape(-1, 1).shape[0]).to(self.device)        
    
        self.target_nn = deepcopy(self.q_value_nn).to(self.device)
        self.optimizer = optim.AdamW(self.q_value_nn.parameters(), lr=lr, amsgrad=True)    
        self.loss_func = nn.MSELoss()        
        self.losses = []

        self.batch_size = batch_size 
        self.lr = lr
        self.epsilon = epsilon 
        self.epsilon_decay = epsilon_decay
        self.discount = 0.95        

        self.action_space = actions        

        self.steps_before_update = 5 
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
    
        mean_loss = np.array(self.losses).sum()/len(self.losses)
        print(f"avg loss: {mean_loss}")        

    def execute(self, obs: np.array, reward: np.array, terminal: bool) -> float:                                
        # put data in replay memory 
        self.memory.insert_example(Example(s_t=torch.from_numpy(self.prev_state), 
                                           a_t=self.action_index, 
                                           s_next_t=torch.from_numpy(obs), 
                                           reward_t=reward, 
                                           terminal=terminal))
        
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
        with torch.no_grad():
            for (s_t, a_t, s_next_t, reward_t, terminal) in samples:                    
                current_q = self.q_value_nn(s_t.to(self.device))            
                next_q = self.target_nn(s_next_t.to(self.device))

                if terminal:
                    #current_q[a_t] = reward_t
                    current_q[a_t] = 0
                else:
                    current_q[a_t] = reward_t + self.discount * torch.max(next_q)
                                        
                if x is None:
                    x = s_t.unsqueeze(0)
                    y = current_q.unsqueeze(0)
                else:
                    x = torch.cat((x, s_t.unsqueeze(0)), dim=0)
                    y = torch.cat((y, current_q.unsqueeze(0)), dim=0)
        x = x.to(self.device)
        y = y.to(self.device)

        # train q_value_nn
        loss = self.loss_func(self.q_value_nn(x), y)        
        self.losses.append(loss.item())        
        
        self.optimizer.zero_grad() 
        loss.backward()
        self.optimizer.step()

        # update target_nn weights 
        if self.update_target_count >= self.steps_before_update:
            self.target_nn.load_state_dict(self.q_value_nn.state_dict())
            self.update_target_count = 0        
        else:
            self.update_target_count += 1      



        # update state and action 
        self.prev_state = obs        

        if uniform(0.0, 1.0) < self.epsilon:
            self.action_index = randint(0, len(self.action_space)-1)
        else:            
            self.action_index = torch.argmax(self.q_value_nn(torch.from_numpy(obs).to(self.device)), dim=0).item()            
        
        return self.action_space[self.action_index]

    def load_model(self, model_name: str) -> None:
        model_path = os.path.join(self.model_dir, f'{model_name}')
        if not os.path.isfile(model_path):
            raise ValueError(f"Model {model_path} not available.",
                              "Start a new training routine or insert a previously trained model into `models` folder.")
        self.q_value_nn.load_state_dict(
            torch.load(model_path)
            )
    
    def save_model(self, model_name: str) -> None:
        torch.save(self.q_value_nn.state_dict(), 
             os.path.join(self.model_dir, f'{model_name}'))        


class TinyDQNAgent(DeepValueRLAgent):    
    def __init__(self, obs_ranges: np.array, actions: np.array,                   
                 batch_size: int, lr: float,
                 epsilon: float = 0.5, epsilon_decay: float = 0.001, 
                 device: Optional[str] = None) -> None: 
        super().__init__()          
        self.device = device         
        if self.device is None:            
            self.device = 'GPU' if cuda.is_available() else 'CPU'                    

        self.memory = ReplayMemory(examples=deque([], maxlen=50000))       
        self.q_value_nn = TinyFCNetwork(in_features=obs_ranges.shape[1], 
                                        out_features=actions.reshape(-1, 1).shape[0])
        self.target_nn = deepcopy(self.q_value_nn)
        self.optimizer = AdamW(get_parameters(self.q_value_nn), lr=lr)        
        self.loss_func = None
        self.losses = []

        self.batch_size = batch_size 
        self.lr = lr
        self.epsilon = epsilon 
        self.epsilon_decay = epsilon_decay
        self.discount = 0.95        

        self.action_space = actions        

        self.steps_before_update = 5 
        self.update_target_count = 0
        
    def episode_start_setup(self, obs: np.array, action_index: int):            
        self.prev_state = obs
        self.action_index = action_index

    def set_test_setup(self) -> None:        
        self.epsilon = 0.05
        self.load_model(model_name='dqn_episode.safetensors')
    
    def episode_exit_setup(self):            
        if self.epsilon >= 0.05:    
            self.epsilon -= self.epsilon_decay    

        self.save_model(model_name='dqn_episode.safetensors')  
    
        mean_loss = np.array(self.losses).sum()/len(self.losses)
        print(f"avg loss: {mean_loss}")        

    def execute(self, obs: np.array, reward: np.array, terminal: bool) -> float:                                
        # put data in replay memory 
        self.memory.insert_example(Example(s_t=tiny_tensor.Tensor(self.prev_state), 
                                           a_t=self.action_index, 
                                           s_next_t=tiny_tensor.Tensor(obs), 
                                           reward_t=reward, 
                                           terminal=terminal))
        
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
        for (s_t, a_t, s_next_t, reward_t, terminal) in samples:                    
            current_q = self.q_value_nn(s_t.to(self.device))            
            next_q = self.target_nn(s_next_t.to(self.device))

            if terminal:
                current_q.numpy()[int(a_t)] = 0
            else:                
                current_q.numpy()[int(a_t)] = reward_t + self.discount * next_q.max().numpy()
                                    
            if x is None:
                x = s_t.unsqueeze(0)
                y = current_q.unsqueeze(0)
            else:                
                x = x.cat(s_t.unsqueeze(0), dim=0)
                y = y.cat(current_q.unsqueeze(0), dim=0)                
        x = x.to(self.device)
        y = y.to(self.device)

        # train q_value_nn
        loss = ((self.q_value_nn(x) - y)**2/y.shape[0]).sum()          
        self.losses.append(loss.numpy())        
        
        self.optimizer.zero_grad() 
        loss.backward()
        self.optimizer.step()

        # update target_nn weights 
        if self.update_target_count >= self.steps_before_update:            
            load_state_dict(self.target_nn, get_state_dict(self.q_value_nn)) 
            self.update_target_count = 0        
        else:
            self.update_target_count += 1      

        # update state and action 
        self.prev_state = obs        

        if uniform(0.0, 1.0) < self.epsilon:
            self.action_index = randint(0, len(self.action_space)-1)
        else:            
            self.action_index = self.q_value_nn(tiny_tensor.Tensor(obs).to(self.device)).argmax(axis=0).numpy()
        
        return self.action_space[int(self.action_index)]

    def load_model(self, model_name: str) -> None:
        model_path = os.path.join(self.model_dir, f'{model_name}')
        if not os.path.isfile(model_path):
            raise ValueError(f"Model {model_path} not available.",
                              "Start a new training routine or insert a previously trained model into `models` folder.")
        # and load it back in
        state_dict = safe_load(model_path)
        load_state_dict(self.q_value_nn, state_dict)        
    
    def save_model(self, model_name: str) -> None:
        state_dict = get_state_dict(self.q_value_nn)
        safe_save(state_dict, os.path.join(self.model_dir, f'{model_name}'))
        

Example = namedtuple("example", ['s_t', 'a_t', 's_next_t', 'reward_t', 'terminal'])


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

