
#
# PPO, adapted from CS234 HW2 to work with our balloon environment code.
#
# Arguments: --agent "ppo" --config configs/ppo_lr=1e-5.yml
#

import numpy as np
import torch
import torch.nn.functional as F
import gym
import itertools
import copy
import os
from general import get_logger, Progbar, export_plot
from baseline_network import BaselineNetwork
from network_utils import build_mlp, device, np2torch
from policy import CategoricalPolicy, GaussianPolicy
from balloon_agent import BalloonAgent


def extend_array(x, y):
    if x is None:
        return y
    else:
        return np.concatenate([x,y], axis=0)




class PPO(BalloonAgent):


    def init_policy(self):
        self.lr = self.config.learning_rate
        self.baseline_network = BaselineNetwork(self.env, self.config)
        network = build_mlp(self.observation_dim, self.action_dim, self.config.n_layers, self.config.layer_size)
        network = network.to(device)
        if self.discrete:
            self.policy = CategoricalPolicy(network)
        else:
            self.policy = GaussianPolicy(network, self.action_dim)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)


    def get_action(self, reward, observation):
        a, logprob = self.policy.act(observation.reshape(1, *observation.shape), return_log_prob=True)
        if self.training:
            self.ep_states.append(observation)
            self.ep_actions.append(a[0])
            self.ep_rewards.append(reward)
            self.ep_logprobs.append(logprob[0])
        return a

    def begin_episode(self, observation):
        a, logprob = self.policy.act(observation.reshape(1, *observation.shape), return_log_prob=True)
        if self.training:
            self.ep_states = [observation]
            self.ep_actions = [a[0]]
            self.ep_rewards = []
            self.ep_logprobs = [logprob[0]]
        return a

    def end_episode(self, state, reward, terminal):
        if self.training:
            self.ep_rewards.append(reward)
            ep_returns = self.get_returns(self.ep_rewards)
            ep_adv = self.get_advantages(ep_returns, self.ep_states)
            self.iter_states = extend_array(self.iter_states, np.array(self.ep_states))
            self.iter_actions = extend_array(self.iter_actions, np.array(self.ep_actions))
            #self.iter_rewards = extend_array(self.iter_rewards, np.array(self.ep_rewards))
            self.iter_logprobs = extend_array(self.iter_logprobs, np.array(self.ep_logprobs))
            self.iter_returns = extend_array(self.iter_returns, np.array(ep_returns))
            self.iter_adv = extend_array(self.iter_adv, ep_adv)
    
    def begin_iteration(self):
        if self.training:
            self.iter_states = None
            self.iter_actions = None
            #self.iter_rewards = None
            self.iter_logprobs = None
            self.iter_returns = None
            self.iter_adv = None
    
    def end_iteration(self):
        if self.training:

            for _ in range(self.config.update_steps):
                self.baseline_network.update_baseline(self.iter_returns, self.iter_states)
                self.update_policy(self.iter_states, self.iter_actions, self.iter_adv, self.iter_logprobs)

    def get_returns(self, rewards):
        returns = [rewards[-1]]
        for i in reversed(range(0, len(rewards)-1)):
            returns.append(rewards[i] + self.config.discount * returns[-1])
        returns = returns[::-1]
        return returns

    def get_advantages(self, returns, states):
        advantages = self.baseline_network.calculate_advantage(np.array(returns), np.array(states))
        if self.config.normalize_advantage:
            mean = advantages.mean()
            std = advantages.std()
            if std > 0.0:
                advantages = (advantages - mean) / std
            else:
                advantages = advantages - mean
        return advantages





    def update_policy(self, observations, actions, advantages, old_logprobs):
        """
        Args:
            observations: np.array of shape [batch size, dim(observation space)]
            actions: np.array of shape
                [batch size, dim(action space)] if continuous
                [batch size] (and integer type) if discrete
            advantages: np.array of shape [batch size, 1]
            old_logprobs: np.array of shape [batch size]

        Perform one update on the policy using the provided data using the PPO clipped
        objective function.

        To compute the loss value, you will need the log probabilities of the actions
        given the observations as before. Note that the policy's action_distribution
        method returns an instance of a subclass of torch.distributions.Distribution,
        and that object can be used to compute log probabilities.

        Note:
            - PyTorch optimizers will try to minimize the loss you compute, but you
            want to maximize the policy's performance.
        """
        observations = np2torch(observations)
        actions = np2torch(actions)
        advantages = np2torch(advantages)
        old_logprobs = np2torch(old_logprobs)

        self.optimizer.zero_grad()
        action_distro = self.policy.action_distribution(observations)
        log_probs = action_distro.log_prob(actions)
        z = torch.exp(log_probs - old_logprobs)
        loss = -torch.minimum(z * advantages, torch.clip(z, 1-self.config.eps_clip, 1+self.config.eps_clip) * advantages).mean()
        loss.backward()
        self.optimizer.step()










    def _train(self):
        """
        Performs training

        You do not have to change or use anything here, but take a look
        to see how all the code you've written fits together!
        """
        last_record = 0

        self.init_averages()
        all_total_rewards = (
            []
        )  # the returns of all episodes samples for training purposes
        averaged_total_rewards = []  # the returns for each iteration

        for t in range(self.config.num_batches):

            # collect a minibatch of samples
            paths, total_rewards = self.sample_path(self.env)
            all_total_rewards.extend(total_rewards)
            observations = np.concatenate([path["observation"] for path in paths])
            actions = np.concatenate([path["action"] for path in paths])
            rewards = np.concatenate([path["reward"] for path in paths])
            old_logprobs = np.concatenate([path["old_logprobs"] for path in paths])

            # compute Q-val estimates (discounted future returns) for each time step
            returns = self.get_returns(paths)
            advantages = self.calculate_advantage(returns, observations)

            # run training operations
            for k in range(self.config.update_freq):
                self.baseline_network.update_baseline(returns, observations)
                self.update_policy(observations, actions, advantages, 
                                   old_logprobs)

            # logging
            if t % self.config.summary_freq == 0:
                self.update_averages(total_rewards, all_total_rewards)
                self.record_summary(t)

            # compute reward statistics for this batch and log
            avg_reward = np.mean(total_rewards)
            sigma_reward = np.sqrt(np.var(total_rewards) / len(total_rewards))
            msg = "[ITERATION {}]: Average reward: {:04.2f} +/- {:04.2f}".format(
                    t, avg_reward, sigma_reward
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

    def sample_path(self, env, num_episodes=None):
        """
        Sample paths (trajectories) from the environment.

        Args:
            num_episodes: the number of episodes to be sampled
                if none, sample one batch (size indicated by config file)
            env: open AI Gym envinronment

        Returns:
            paths: a list of paths. Each path in paths is a dictionary with
                path["observation"] a numpy array of ordered observations in the path
                path["actions"] a numpy array of the corresponding actions in the path
                path["reward"] a numpy array of the corresponding rewards in the path
            total_rewards: the sum of all rewards encountered during this "path"

        You do not have to implement anything in this function, but you will need to
        understand what it returns, and it is worthwhile to look over the code
        just so you understand how we are taking actions in the environment
        and generating batches to train on.
        """
        episode = 0
        episode_rewards = [] 
        paths = []
        t = 0

        while num_episodes or t < self.config.batch_size:
            state = env.reset()
            states, actions, old_logprobs, rewards = [], [], [], []
            episode_reward = 0

            for step in range(self.config.max_ep_len):
                states.append(state)
                # Note the difference between this line and the corresponding line
                # in PolicyGradient.
                action, old_logprob = self.policy.act(states[-1][None], return_log_prob = True)
                assert old_logprob.shape == (1,)
                action, old_logprob = action[0], old_logprob[0]
                state, reward, done, info = env.step(action)
                actions.append(action)
                old_logprobs.append(old_logprob)
                rewards.append(reward)
                episode_reward += reward
                t += 1
                if done or step == self.config.max_ep_len - 1:
                    episode_rewards.append(episode_reward)
                    break
                if (not num_episodes) and t == self.config.batch_size:
                    break

            path = {
                "observation": np.array(states),
                "reward": np.array(rewards),
                "action": np.array(actions),
                "old_logprobs": np.array(old_logprobs)
            }
            paths.append(path)
            episode += 1
            if num_episodes and episode >= num_episodes:
                break

        return paths, episode_rewards
