"""
Solve the MountainCar environment with Q-learning algorithm. 
"""
import keyboard
import random
from typing import Tuple
import gymnasium as gym
import numpy as np
from algorithms import RLAgent
from algorithms.tabularValue import TabularValueRLAgent, QLearningAgent, \
                                    EpsilonGreedyAgent, ExplorationFuncAgent, SarsaAgent


def reset_env(env: gym.Env) -> Tuple[np.array]:
    observation, _ = env.reset(seed=42)
    action =  random.randint(0, 2)
    return observation, action


def run(test: bool, n_episodes: int, env: gym.Env,  agent: RLAgent):    

    if test:
        agent.load_model()
        agent.set_test_setup()
        
    n_steps = 0    
    steps_list = [0]
    for episode in range(n_episodes):
        if episode%50 == 0:
            print(f"episode: {episode} n_steps: {np.array(steps_list).mean()}", flush=True)
            agent.save_model()  
            steps_list = []

        steps_list.append(n_steps)
        observation, action = reset_env(env=env)
        agent.episode_start_setup(obs=observation, action=action)
        
        terminated = False                
        n_steps = 0
        while not terminated:         
            observation, reward, terminated, _, _ = env.step(action)                                           

            action = agent.execute(obs=observation, reward=reward)        
            n_steps += 1            
            
            if test:
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
                
        agent.episode_exit_setup()                  
    env.close()
        

def main():
    env = gym.make("MountainCar-v0", render_mode=None) 
   
   # actions 
            # + discrete: action {
            #                       0: accelerate left
            #                       1: don't accelerate
            #                       2: accelerate right
            #                    }                    
    obs_ranges = np.array([env.observation_space.low, env.observation_space.high])
    actions = np.array([0, 1, 2])

    #agent = QLearningAgent(obs_ranges=obs_ranges, actions=actions)
    #agent = EpsilonGreedyAgent(obs_ranges=obs_ranges, actions=actions)
    #agent = ExplorationFuncAgent(obs_ranges=obs_ranges, actions=actions)
    agent = SarsaAgent(obs_ranges=obs_ranges, actions=actions)

    n_episodes=2000

    # training phase 
    print("\n\nTraining:\n")     
    run(test=False, n_episodes=n_episodes, env=env, agent=agent)

    # testing phase
    print("\n\nTesting:\n")
    env = gym.make("MountainCar-v0", render_mode='human')
    run(test=True, n_episodes=n_episodes, env=env, agent=agent)


if __name__ == "__main__":        
    main()