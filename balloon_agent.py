import numpy as np
import torch
import torch.nn as nn
import gym
import os
from general import get_logger, Progbar, export_plot


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self




class BalloonAgent(nn.Module):
    '''
    Base class for wrappers around balloon_learning_environment agents
    '''

    def __init__(self, env, **kwargs):
        super().__init__()
        self.env = env
        self.config = AttrDict(**kwargs)

        # discrete vs continuous action space
        self.discrete = isinstance(env.action_space, gym.spaces.Discrete)
        self.observation_dim = self.env.observation_space.shape[0]
        self.action_dim = (
            self.env.action_space.n if self.discrete else self.env.action_space.shape[0]
        )

        self.init_policy()


    def save(self, ckpt_path):
        torch.save(self.state_dict(), ckpt_path)

    def load(self, ckpt_path):
        self.load_state_dict(torch.load(ckpt_path))



    def init_policy(self):
        raise NotImplementedError()

    def get_action(self, reward, observation):
        raise NotImplementedError()
    
    def begin_episode(self, observation):
        raise NotImplementedError()

    def end_episode(self, reward, terminal):
        raise NotImplementedError()
    
    def begin_iteration(self):
        pass
    
    def end_iteration(self):
        pass







    # DEPRECATED
    #  |
    #  V


    
    def _train(self):
        """
        Performs training
        """
        averaged_total_rewards = []

        for i in range(self.config.num_iters):
            self.logger.info(f'[ITERATION {i}]')

            for t in range(self.config.num_episodes):
                total_rewards = []


                state = self.env.reset()
                action = self.begin_episode(state)
                
                for j in range(self.config.max_episode_length):

                    state, reward, done, info = self.env.step(action)
                    total_rewards.append(reward)

                    # TODO: optionally record transition in replay buffer

                    if done:
                        break

                    if j < self.config.max_episode_length - 1:
                        action = self.get_action(reward, state)

                self.end_episode(reward, done)

                # compute reward statistics for this batch and log
                #avg_reward = np.mean(total_rewards)
                #sigma_reward = np.sqrt(np.var(total_rewards) / len(total_rewards))
                #msg = "[EPISODE {}]: Average reward: {:04.2f} +/- {:04.2f}".format(
                #    t, avg_reward, sigma_reward
                #)
                avg_reward = np.sum(total_rewards)
                msg = "[EPISODE {}]: Average reward: {:04.2f}".format(
                    t, avg_reward
                )
                averaged_total_rewards.append(avg_reward)
                self.logger.info(msg)

                if self.config.record and (last_record > self.config.record_freq):
                    self.logger.info("Recording...")
                    last_record = 0
                    self.record()

        self.logger.info("- Training done.")
        np.save(self.config.scores_output, averaged_total_rewards)
        export_plot(
            averaged_total_rewards,
            "Score",
            self.config.env_name,
            self.config.plot_output,
        )


    def evaluate(self, env=None, num_episodes=1):
        """
        Evaluates the return for num_episodes episodes.
        Not used right now, all evaluation statistics are computed during training
        episodes.
        """
        if env == None:
            env = self.env
        paths, rewards = self.sample_path(env, num_episodes)
        avg_reward = np.mean(rewards)
        sigma_reward = np.sqrt(np.var(rewards) / len(rewards))
        msg = "Average reward: {:04.2f} +/- {:04.2f}".format(avg_reward, sigma_reward)
        self.logger.info(msg)
        return avg_reward

    def record(self):
        """
        Recreate an env and record a video for one episode
        """
        env = gym.make(self.config.env_name)
        env.seed(self.seed)
        env = gym.wrappers.Monitor(
            env, self.config.record_path, video_callable=lambda x: True, resume=True
        )
        self.evaluate(env, 1)

    def run(self):
        """
        Apply procedures of training for a PG.
        """
        # record one game at the beginning
        if self.config.record:
            self.record()
        # model
        self.train()
        # record one game at the end
        if self.config.record:
            self.record()

