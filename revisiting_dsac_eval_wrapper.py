
#
# A wrapper for loading model checkpoints trained with the Revisiting
# Discrete SAC code for eval. See Revisiting_Discrete_SAC/src/balloon_sac.py
#
# Arguments: --agent dsac --config configs/revisiting_dsac.yml
#


import sys
sys.path.append('./Revisiting_Discrete_SAC/src/')

from Revisiting_Discrete_SAC.src.libs.discrete_sac import DiscreteSACDevPolicy
from Revisiting_Discrete_SAC.src.tianshou.utils.net.discrete import Actor, Critic
from Revisiting_Discrete_SAC.src.tianshou.data.batch import Batch
from balloon_agent import BalloonAgent
from network_utils import build_mlp, device
import torch
import torch.nn as nn
import torch.nn.functional as F



class Net(nn.Module):
    def __init__(self, config, state_dim, action_dim):
        super().__init__()
        self.net = build_mlp(
            state_dim,
            action_dim,
            config.n_layers,
            config.hidden_size
        )
        self.config = config
    def forward(self, x, state):    # needs to be called "state" (used as kwarg)
        x = torch.as_tensor(x, device=self.config.device, dtype=torch.float32)
        return self.net(x), None


class RevisitingDSacEvalWrapper(BalloonAgent):

    
    def init_policy(self):
        self.config.device = device
        actor = Actor(Net(self.config, self.observation_dim, self.action_dim),
                      self.action_dim, device=self.config.device, softmax_output=False,
                      preprocess_net_output_dim=self.action_dim)
        critic1 = Critic(Net(self.config, self.observation_dim, self.action_dim),
                         last_size=self.action_dim, device=self.config.device,
                         preprocess_net_output_dim=self.action_dim)
        critic2 = Critic(Net(self.config, self.observation_dim, self.action_dim),
                         last_size=self.action_dim, device=self.config.device,
                         preprocess_net_output_dim=self.action_dim)
        self.policy = DiscreteSACDevPolicy(
            actor,
            None,
            critic1,
            None,
            critic2,
            None,
            self.config.tau,
            self.config.gamma,
            self.config.alpha,
            estimation_step=self.config.n_step,
            reward_normalization=self.config.rew_norm,

            use_avg_q=self.config.avg_q,
            use_clip_q=self.config.clip_q,
            clip_q_epsilon=self.config.clip_q_epsilon,
            use_entropy_penalty=self.config.entropy_penalty,
            entropy_penalty_beta=self.config.entropy_penalty_beta,
        ).to(device)
        self.policy.eval()
        self.policy._deterministic_eval = True


    def get_action(self, reward, observation):
        observation = torch.as_tensor(observation, device=device).unsqueeze(0)
        act = self.policy(Batch({'obs': observation, 'info': None}))
        return act['act'].cpu().item()
    
    
    def begin_episode(self, observation):
        return self.get_action(None, observation)

    def end_episode(self, observation, reward, terminal):
        pass
    
    def load(self, ckpt_path):
        data = torch.load(ckpt_path)
        self.policy.load_state_dict(data)
    