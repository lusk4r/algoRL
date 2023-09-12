from typing import List
from gymnasium import Env


def get_states_delta_from_n_intevals(*, env: Env, n_intervals: List[int]) -> List[float]:
    if len(n_intervals) < len(env.observation_space.high):
        raise ValueError('`n_intervals` and `observation_space` dimensions are different')
    return env.observation_space.high - env.observation_space.low / n_intervals


def get_n_intervals_from_states_delta(*, env: Env, states_delta: List[float]) -> List[int]:
    if len(states_delta) < len(env.observation_space.high):
        raise ValueError('`states_delta` and `observation_space` dimensions are different')
    return env.observation_space.high - env.observation_space.low / states_delta
