"""
Solve the MountainCar environment with Q-learning algorithm. 
"""

import keyboard
import random
from typing import Tuple
import gymnasium as gym
import numpy as np
from algorithms.q_learn import QLearning


def reset_env(env: gym.Env) -> Tuple[np.array]:
    observation, _ = env.reset(seed=42)
    action =  random.randint(0, 2)
    return observation, action


def main():
    env = gym.make("MountainCar-v0", render_mode='human')        
    states = np.meshgrid(np.linspace(-1.2, 0.6, 20), np.linspace(-.07, .07, 20))    
    q_learn = QLearning(states=np.array(states).T.reshape(-1, 2),
                        actions=np.array([0, 1, 2]))
            
    for _ in range(100):
        observation, action = reset_env(env=env)
        q_learn.prev_state_index = q_learn.get_nearest_state_index(obs=observation)
        q_learn.action_index = np.where(q_learn.actions == action)
              
        terminated = False
        while not terminated:                
            # actions 
            # + discrete: action {
            #                       0: accelerate left
            #                       1: don't accelerate
            #                       2: accelerate right
            #                    }                    

            observation, reward, terminated, _, _ = env.step(action)                   
            action = q_learn.execute(obs=observation, reward=reward)        
            
            # keyboard commands
            if keyboard.is_pressed('q') or keyboard.is_pressed('esc'):    
                break 
            elif keyboard.is_pressed('f'):
                env.metadata['render_fps'] = 1000
            elif keyboard.is_pressed('s'):
                env.metadata['render_fps'] = 30

        q_learn.decay_epsilon()
            
    env.close()
        

if __name__ == "__main__":
    main()