
import numpy as np
import torch
import gym
import os
import argparse
from pathlib import Path
from tqdm import tqdm
from general import get_logger, Progbar, export_plot
from utils import get_agent, get_env, get_twr50


    
def train(env, agent, logger, num_iters, eps_per_iter, max_ep_length, pbar=True):
    """
    Performs training.
    We assume no early termination, so all episodes are max_ep_length.
    Returns a list of total episode rewards
    """

    assert isinstance(env.action_space, gym.spaces.Discrete)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    ep_rewards = []

    if pbar: T = tqdm(total=num_iters*eps_per_iter*max_ep_length, desc='Running training')

    for i in range(num_iters):
        logger.info(f'[ITERATION {i}]')

        agent.begin_iteration()

        for j in range(eps_per_iter):
            total_ep_reward = 0.0

            state = env.reset()
            action = agent.begin_episode(state)

            for k in range(max_ep_length):
                state, reward, done, info = env.step(action)
                total_ep_reward += reward

                if done:
                    break

                if k < max_ep_length - 1:
                    action = agent.get_action(reward, state)

                if pbar: T.update(1)

            agent.end_episode(reward, done)
            
            logger.info(f'[ITER {i} EPISODE {j}]: Total reward: {total_ep_reward:04.2f}')
            ep_rewards.append(total_ep_reward)

        agent.end_iteration()

        if pbar: T.n = (i+1)*eps_per_iter*max_ep_length; T.refresh()

    if pbar: T.close()

    return ep_rewards



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', type=str, default='stationseeker',
                        choices=['stationseeker', 'ppo'])
    parser.add_argument('--config', type=Path, default=None)
    #parser.add_argument('--ckpt', type=Path, default=None)
    parser.add_argument('--iters', type=int, default=200)
    parser.add_argument('--eps-per-iter', dest='eps_per_iter', type=int, default=5)
    parser.add_argument('--max-ep-length', dest='max_ep_length', type=int, default=960)
    parser.add_argument('--out-dir', dest='out_dir', type=Path, default=None)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    if args.out_dir is None:
        args.out_dir = Path(f'./results/{args.agent}_seed={args.seed}')
    if args.out_dir.exists():
        print('Warning:', args.out_dir, 'already exists')
    else:
        args.out_dir.mkdir(parents=True)


    env = get_env(args.seed)
    agent = get_agent(env, args.agent, args.config, None, args.seed)
    logger = get_logger(args.out_dir / 'train_log.txt')

    results = train(env, agent, logger, args.iters, args.eps_per_iter, args.max_ep_length)


    print()
    print('Training finished')
    print('Average episode reward:', np.mean(results), '+/-', np.std(results))
    
    results_path = args.out_dir / 'train_rewards.npy'
    np.save(results_path, np.array(results))
    print('Rewards saved to', results_path)

    ckpt_path = args.out_dir / f'ckpt-{args.iters}.pt'
    agent.save(ckpt_path)
    print('Checkpoint saved to', ckpt_path)

    results_path = args.out_dir / 'train_config.npy'
    results = {}
    results['config'] = agent.config
    results['command'] = args.__dict__
    np.save(results_path, results)
    print('Config saved to', results_path)

