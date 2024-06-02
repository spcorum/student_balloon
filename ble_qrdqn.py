
from balloon_learning_environment.agents import agent
from balloon_learning_environment.agents.quantile_agent import QuantileAgent
from balloon_learning_environment.agents.marco_polo_exploration import MarcoPoloExploration
from balloon_learning_environment.agents import dopamine_utils
from dopamine.jax import networks
from dopamine.jax.agents.quantile.quantile_agent import JaxQuantileAgent
from dopamine.jax.agents.dqn import dqn_agent
import torch
import numpy as np

from balloon_agent import BalloonAgent
from network_utils import build_mlp, device, np2torch

from pathlib import Path
import functools



class QRDQN(BalloonAgent):

    def init_policy(self):
        #self.network = build_mlp(self.observation_dim, self.action_dim, self.config.n_layers, self.config.layer_size)
        #self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.config.learning_rate)
        self.policy = JaxQuantileAgent(
            self.action_dim,
            observation_shape=(1, self.observation_dim,),
            observation_dtype=np.float32,
            stack_size=1,
            network=networks.QuantileNetwork,
            kappa=1.0,
            num_atoms=self.config.num_quantiles,
            gamma=self.config.discount,
            update_horizon=5,
            min_replay_history=500,
            update_period=4,
            target_update_period=100,
            epsilon_fn=dqn_agent.identity_epsilon,
            epsilon_train=0.0,
            epsilon_eval=0.0,
            #epsilon_decay_period=self.config.epsilon_decay_period,
            #replay_scheme='prioritized',
            optimizer='adam',
            summary_writer=None,
            #summary_writing_frequency=500,
            seed=self.config.seed,
            allow_partial_reload=False,
        )
        if self.config.explore:
            self._exploration_wrapper = MarcoPoloExploration(
                self.action_dim,
                observation_shape=(1, self.observation_dim,),
                )
        else:
            self._exploration_wrapper = None

    def get_action(self, reward, observation):
        action = self.policy.step(reward, observation)
        if self.policy.eval_mode:
            return action
        if self._exploration_wrapper is not None:
            action = self._exploration_wrapper.step(
                reward, observation, action)
        return action

    def begin_episode(self, observation):
        action = self.policy.begin_episode(observation)
        if self.policy.eval_mode:
            return action
        if self._exploration_wrapper is not None:
            action = self._exploration_wrapper.begin_episode(observation, action)
        return action

    def end_episode(self, state, reward, terminal):
        self.policy.end_episode(reward, terminal)

    def begin_iteration(self):
        pass

    def end_iteration(self):
        #self.policy._train_step()
        pass

    def save(self, ckpt_path):
        ckpt_path = Path(ckpt_path)
        checkpoint_dir = str(ckpt_path.parent)
        iteration_number = int(ckpt_path.stem.split('-')[-1].split('.')[-1])
        dopamine_utils.save_checkpoint(
            checkpoint_dir, iteration_number,
            functools.partial(JaxQuantileAgent.bundle_and_checkpoint, self.policy))

    def load(self, ckpt_path):
        ckpt_path = Path(ckpt_path)
        checkpoint_dir = str(ckpt_path.parent)
        iteration_number = int(ckpt_path.stem.split('-')[-1].split('.')[-1])
        dopamine_utils.load_checkpoint(
            checkpoint_dir, iteration_number,
            functools.partial(JaxQuantileAgent.unbundle, self.policy))
        
    def train(self):
        super().train()
        self.policy.eval_mode = False

    def eval(self):
        super().eval()
        self.policy.eval_mode = True
