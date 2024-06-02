
import numpy as np
import yaml
import gym
from stationseeker import StationSeeker
from ppo import PPO
from random_walk import RandomWalk
from perciatelli import Perciatelli
from ble_qrdqn import QRDQN
from ble_vdqn import VDQN as BLE_VDQN
from vdqn import VDQN
from pt_dqn_eval_wrapper import PtDQNEvalWrapper as PT_DQN
from knobbified_ppo import KnobbifiedPPO


def get_env(seed, env_name='BalloonLearningEnvironment-v0'):
    env = gym.make(env_name)
    env.seed(seed)
    return env

def get_agent(env, agent_name, config, ckpt, seed):
    if config is not None:
        with open(config) as f:
            config = yaml.load(f, yaml.CLoader)
    else:
        config = {}

    if agent_name == 'station-seeker':
        if ckpt is not None:
            print('Warning: The "station-seeker" agent does not support checkpoints')
        return StationSeeker(env, **config)
    elif agent_name == 'ppo':
        a = PPO(env, **config)
        if ckpt is not None:
            a.load(ckpt)
        return a
    elif agent_name == 'random-walk':
        if ckpt is not None:
            print('Warning: The "random-walk" agent does not support checkpoints')
        return RandomWalk(env, **config)
    elif agent_name == 'perciatelli':
        if ckpt is not None:
            print('Warning: The "perciatelli" agent does not support checkpoints')
        return Perciatelli(env, **config)
    elif agent_name == 'qrdqn':
        a = QRDQN(env, **config)
        if ckpt is not None:
            a.load(ckpt)
        return a
    elif agent_name == 'vdqn':
        a = VDQN(env, **config)
        if ckpt is not None:
            a.load(ckpt)
        return a
    elif agent_name == 'ble_vdqn':
        a = BLE_VDQN(env, **config)
        if ckpt is not None:
            a.load(ckpt)
        return a
    elif agent_name == 'pt_dqn':
        a = PT_DQN(env, **config)
        if ckpt is not None:
            a.load(ckpt)
        return a
    elif agent_name == 'knob_ppo':
        a = KnobbifiedPPO(env, **config)
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

