"""
Solve the MountainCar environment with Q-learning algorithm. 
"""
import keyboard
import random
from typing import Tuple, Dict, Any
import gymnasium as gym
import numpy as np
from algorithms.q_learning import QLearning, EpsilonGreedyQLearning, ExplorationFuncQLearning, Sarsa
from algorithms.utils import get_states_delta_from_n_intevals


def reset_env(env: gym.Env) -> Tuple[np.array]:
    observation, _ = env.reset(seed=42)
    action =  random.randint(0, 2)
    return observation, action


def main(test: bool):
    env = gym.make("MountainCar-v0", render_mode='human' if test else None)  
    n_intervals = [20, 20]      
    states_delta = get_states_delta_from_n_intevals(env=env, n_intervals=n_intervals)
    states_info: Dict[str, Any] = {
        "delta": states_delta,
        "low": env.observation_space.low,
        "dims": [d+1 for d in n_intervals]
        }

    #q_learn = QLearning(states_info=states_info,    
    #                    actions=np.array([0, 1, 2]))

    #q_learn = EpsilonGreedyQLearning(states_info=states_info,
    #                                 actions=np.array([0, 1, 2]))
    
    #q_learn = ExplorationFuncQLearning(states_info=states_info,
    #                                   actions=np.array([0, 1, 2]))        
    
    q_learn = Sarsa(states_info=states_info,
                    actions=np.array([0, 1, 2]))            
    
    if test:
        q_learn.load_model()
        q_learn.set_test_setup()
        
    n_steps = 0    
    steps_list = []
    for episode in range(20000):
        if episode%50 == 0:
            print(f"episode: {episode} n_steps: {np.array(steps_list).mean()}", flush=True)
            q_learn.save_model()  
            steps_list = []

        steps_list.append(n_steps)
        observation, action = reset_env(env=env)
        q_learn.episode_start_setup(obs=observation, action=action)
        
        terminated = False                
        n_steps = 0
        while not terminated:
            # actions 
            # + discrete: action {
            #                       0: accelerate left
            #                       1: don't accelerate
            #                       2: accelerate right
            #                    }                    

            observation, reward, terminated, _, _ = env.step(action)                                           

            action = q_learn.execute(obs=observation, reward=reward)        
            n_steps += 1            
            
            if True:
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
                    env.metadata['render_fps'] = 1000
                elif keyboard.is_pressed('s'):
                    # slow execution of the simulation
                    env.metadata['render_fps'] = 30                                              
                
        q_learn.episode_exit_setup()                  
    env.close()
        

if __name__ == "__main__":
    main(test=False)