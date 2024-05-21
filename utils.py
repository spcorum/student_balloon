
import numpy as np
import yaml
import gym
from stationseeker import StationSeeker
from ppo import PPO


def get_env(seed):
    return gym.make('BalloonLearningEnvironment-v0')


def get_agent(env, agent_name, config, ckpt, seed):
    if config is not None:
        with open(config) as f:
            config = yaml.load(f, yaml.CLoader)
    else:
        config = {}

    if agent_name == 'stationseeker':
        if ckpt is not None:
            print('Warning: The "stationseeker" agent does not support checkpoints')
        return StationSeeker(env, **config)
    elif agent_name == 'ppo':
        a = PPO(env, **config)
        if ckpt is not None:
            a.load(ckpt)
        return a


def get_station_distance(state: np.array):
    # Index found in balloon_learning_environment/env/features.py at line 203
    dist = state[...,7]
    # dist is normalized in [0,1] using f(x) = x/(x+250)
    # xy+250y=x ==> x=-250y/(y-1)
    # Clip with a small epsilon for numerical safety
    return 250.0 * dist / np.maximum(1.0 - dist, 1e-5)


def get_twr50(states: np.array):
    return (get_station_distance(states) <= 50.0).mean()

