
#
# A wrapper for loading model checkpoints trained with pt_dqn.py for eval.
#
# Arguments: --agent pt_dqn --config configs/pt_dqn_ble.yml
#

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from balloon_agent import BalloonAgent
from network_utils import device



class DQN(nn.Module):

    def __init__(self, config, n_observations, n_actions):
        self.config = config
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, self.config.layer_size)
        self.hlayers = torch.nn.ModuleList([
            nn.Linear(self.config.layer_size, self.config.layer_size) for _ in range(self.config.n_layers-2)
        ])
        self.layer3 = nn.Linear(self.config.layer_size, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        for l in self.hlayers:
            x = F.relu(l(x))
        return self.layer3(x)


class PtDQNEvalWrapper(BalloonAgent):

    def init_policy(self):
        self.policy_net = DQN(self.config, self.observation_dim, self.action_dim).to(device)
        self.target_net = DQN(self.config, self.observation_dim, self.action_dim).to(device)



    def get_action(self, reward, observation):
        with torch.no_grad():
            observation = torch.from_numpy(observation).unsqueeze(0).to(device)
            return self.policy_net(observation).max(1).indices.view(1, 1).cpu().item()


    def begin_episode(self, observation):
        return self.get_action(None, observation)


    def end_episode(self, observation, reward, terminal):
        pass


    def begin_iteration(self):
        pass

