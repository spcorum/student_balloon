
import numpy as np
import torch
import gym
import os
import argparse
from pathlib import Path
from tqdm import tqdm
from general import get_logger, Progbar, export_plot
from utils import get_agent, get_env, get_twr50


    
def eval(env, agent, logger, num_iters, max_ep_length, pbar=True):
    """
    Performs evaluation.
    We assume no early termination, so all episodes are max_ep_length.
    Returns a dict of results with entries:
        - "state" (num_iters, max_ep_length+1, state_dim)
        - "action" (num_iters, max_ep_length, action_dim)
        - "reward" (num_iters, max_ep_length)
        - "twr50" (scalar)
    """

    assert isinstance(env.action_space, gym.spaces.Discrete)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    balloon_state_dim = 4

    # We'll need to store them eventually anyway, so just make the arrays now
    all_states = np.zeros((num_iters, max_ep_length+1, state_dim))
    all_actions = np.zeros((num_iters, max_ep_length, action_dim), dtype=int)
    all_rewards = np.zeros((num_iters, max_ep_length))
    all_balloon_states = np.zeros((num_iters, max_ep_length, balloon_state_dim))
    def get_balloon_state():
        bs = env.arena._balloon.state
        x, y = bs.x.km, bs.y.km
        power_load = bs.power_load.watts
        battery_charge = bs.battery_charge.watt_hours
        return [x, y, power_load, battery_charge]


    if pbar: T = tqdm(total=num_iters*max_ep_length, desc='Running eval')

    for i in range(num_iters):
        logger.info(f'[ITERATION {i}]')

        state = env.reset()
        agent.begin_iteration()
        action = agent.begin_episode(state)
        
        for j in range(max_ep_length):
            all_states[i,j,:] = state
            all_actions[i,j,:] = action
            all_balloon_states[i,j,:] = get_balloon_state()

            state, reward, done, info = env.step(action)
            all_rewards[i,j] = reward

            if done:
                logger.info('Warning: A terminal state was reached, but the eval',
                      'metrics assume there is no early termination')
                break

            if j < max_ep_length - 1:
                action = agent.get_action(reward, state)
            
            if pbar: T.update(1)

        all_states[i,-1,:] = state
        all_balloon_states[i,-1,:] = get_balloon_state()
        agent.end_episode(reward, done)
        agent.end_iteration()

        logger.info(f'[EPISODE {i}]: Total reward: {all_rewards[i].sum():04.2f}')
        if pbar: T.n = (i+1)*max_ep_length; T.refresh()


    if pbar: T.close()


    return {
        'state': all_states,
        'action': all_actions,
        'reward': all_rewards,
        'twr50': get_twr50(all_states.reshape(-1, state_dim)),
        'balloon_state': all_balloon_states,
        'balloon_state_comment': '[x (km), y (km), power_load (watts), battery_charge (watt-hours)]',
    }




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', type=str, default='station-seeker',
                        choices=['station-seeker', 'ppo', 'random-walk', 'perciatelli'])
    parser.add_argument('--config', type=Path, default=None)
    parser.add_argument('--ckpt', type=Path, default=None)
    parser.add_argument('--iters', type=int, default=200)
    parser.add_argument('--max-ep-length', dest='max_ep_length', type=int, default=960)
    parser.add_argument('--out-dir', dest='out_dir', type=Path, default=None)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    if args.out_dir is None:
        if args.ckpt is None:
            args.out_dir = Path(f'./results/{args.agent}_seed={args.seed}')
        else:
            args.out_dir = args.ckpt.parent
    if args.out_dir.exists():
        print('Warning:', args.out_dir, 'already exists')
    else:
        args.out_dir.mkdir(parents=True)


    env = get_env(args.seed)
    agent = get_agent(env, args.agent, args.config, args.ckpt, args.seed)
    logger = get_logger(args.out_dir / 'eval_log.txt')

    agent.eval()
    results = eval(env, agent, logger, args.iters, args.max_ep_length)


    print()
    print('Evaluation finished')
    ep_rewards = results['reward'].sum(1)
    print('Average episode reward:', ep_rewards.mean(), '+/-', ep_rewards.std())
    print('TWR50:', results['twr50'])
    
    results_path = args.out_dir / 'eval_results.npy'
    results['config'] = dict(**agent.config)
    results['command'] = args.__dict__
    np.save(results_path, results)
    print('Results saved to', results_path)

