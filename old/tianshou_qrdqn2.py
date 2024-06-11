from tianshou.data import Collector, VectorReplayBuffer
#from tianshou.highlevel.logger import LoggerFactoryDefault
from tianshou.policy import QRDQNPolicy
from tianshou.policy.base import BasePolicy
from tianshou.trainer import OffpolicyTrainer
from tianshou.utils.net.common import Net
from tianshou.env.venvs import BaseVectorEnv
from tianshou.env.worker.dummy import DummyEnvWorker

import argparse
# gymnasium-0.29.1
import gym
import numpy as np
import torch
import os
from utils import get_agent, get_env, get_twr50



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--gamma', type=float)
    parser.add_argument('--num_quantiles', type=int)
    parser.add_argument('--n_step', type=int)
    parser.add_argument('--target_update_freq', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--steps_per_epoch', type=int)
    parser.add_argument('--steps_per_collect', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--eps_test', type=int)
    parser.add_argument('--test_num', type=int)
    parser.add_argument('--update_per_step', type=int)
    parser.add_argument('--buffer_size', type=int)
    parser.add_argument('--frames_stack', type=int)
    parser.add_argument('--log_path')
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    def make_env():
        return get_env(args.seed)
    env = BaseVectorEnv([make_env], DummyEnvWorker)

    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.n
    policy_net = Net(
        state_shape=state_shape,
        action_shape=action_shape,
        hidden_sizes=[128, 128, 128],
    )
    optim = torch.optim.Adam(policy_net.parameters(), lr=args.lr)
    policy: QRDQNPolicy = QRDQNPolicy(
        model=policy_net,
        optim=optim,
        action_space=env.action_space,
        discount_factor=args.gamma,
        num_quantiles=args.num_quantiles,
        estimation_step=args.n_step,
        target_update_freq=args.target_update_freq,
    )

    buffer = VectorReplayBuffer(
        args.buffer_size,
        buffer_num=1,
        ignore_obs_next=True,
        save_only_last_obs=True,
        stack_num=args.frames_stack,
    )

    train_collector = Collector(policy, env, buffer, exploration_noise=True)
    test_collector = Collector(policy, env, exploration_noise=True)


    def train_fn(epoch: int, env_step: int) -> None:
        # nature DQN setting, linear decay in the first 1M steps
        if env_step <= 1e6:
            eps = args.eps_train - env_step / 1e6 * (args.eps_train - args.eps_train_final)
        else:
            eps = args.eps_train_final
        policy.set_eps(eps)
        #if env_step % 1000 == 0:
        #    logger.write("train/env_step", env_step, {"train/eps": eps})

    def test_fn(epoch: int, env_step) -> None:
        policy.set_eps(args.eps_test)


    #def stop_fn(mean_rewards: float) -> bool:
    #    if env.spec.reward_threshold:
    #        return mean_rewards >= env.spec.reward_threshold
    #    return False

    def save_best_fn(policy: BasePolicy) -> None:
        torch.save(policy.state_dict(), os.path.join(args.log_path, "policy.pth"))


    result = OffpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=args.epochs,
        step_per_epoch=args.steps_per_epoch,
        step_per_collect=args.steps_per_collect,
        episode_per_test=args.test_num,
        batch_size=args.batch_size,
        train_fn=train_fn,
        test_fn=test_fn,
        stop_fn=None, #stop_fn,
        save_best_fn=save_best_fn,
        #logger=logger,
        update_per_step=args.update_per_step,
        test_in_train=False,
    ).run()

    print(result)
