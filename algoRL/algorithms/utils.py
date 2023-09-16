import numpy as np
from typing import List
from gymnasium import Env


def get_states_delta_from_n_intervals(*, obs_ranges: np.array, n_intervals: List[int]) -> List[float]:
    if len(n_intervals) < len(obs_ranges[1, :]):
        raise ValueError('`n_intervals` and `observation_space` dimensions are different')
    return (obs_ranges[1, :] - obs_ranges[0, :]) / n_intervals


def get_n_intervals_from_states_delta(*, obs_ranges: np.array, states_delta: List[float]) -> List[int]:
    if len(states_delta) < len(obs_ranges[1, :]):
        raise ValueError('`states_delta` and `observation_space` dimensions are different')
    return (obs_ranges[1, :] - obs_ranges[0, :]) / states_delta
