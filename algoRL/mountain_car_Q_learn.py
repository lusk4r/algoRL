"""
Solve the MountainCar environment with Q-learning algorithm. 
"""

import keyboard
import random
from typing import Tuple
import gymnasium as gym
import numpy as np
from algorithms.q_learning import EpsilonGreedyQLearning, ExplorationFuncQLearning, Sarsa


def reset_env(env: gym.Env) -> Tuple[np.array]:
    observation, _ = env.reset(seed=42)
    action =  random.randint(0, 2)
    return observation, action


def main():
    env = gym.make("MountainCar-v0", render_mode='human')        
    states = np.meshgrid(np.linspace(-1.2, 0.6, 20), np.linspace(-.07, .07, 20))    

    #q_learn = QLearning(states=np.array(states).T.reshape(-1, 2),
    #                    actions=np.array([0, 1, 2]))
    #q_learn = EpsilonGreedyQLearning(states=np.array(states).T.reshape(-1, 2),
    #                                 actions=np.array([0, 1, 2]))
    #q_learn = ExplorationFuncQLearning(states=np.array(states).T.reshape(-1, 2),
    #                                   actions=np.array([0, 1, 2]))        
    q_learn = Sarsa(states=np.array(states).T.reshape(-1, 2),
                    actions=np.array([0, 1, 2]))            

    for _ in range(100):
        observation, action = reset_env(env=env)
        q_learn.episode_start_setup(obs=observation, action=action)

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
            if keyboard.is_pressed('q'):    
                # exit from episode
                break 
            if keyboard.is_pressed('esc'):
                # close program
                env.close()
                exit(-1)
            elif keyboard.is_pressed('f'):
                # fast execution of the simulation
                env.metadata['render_fps'] = 10000
            elif keyboard.is_pressed('s'):
                # slow execution of the simulation
                env.metadata['render_fps'] = 30            
                
        q_learn.episode_exit_setup()            
    env.close()
        

if __name__ == "__main__":
    main()