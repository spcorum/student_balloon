
# Largely copied from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

from balloon_learning_environment.agents.random_walk_agent import RandomWalkAgent
from balloon_agent import BalloonAgent

from replaybuffer import ReplayBuffer
from eps_decay import EpsDecay

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



class VDQN(BalloonAgent):

    def __copy_network(self, src, dst):
        dst.load_state_dict(src.state_dict())
    
    def init_policy(self):
        self.device = torch.device(self.config.device)
        layers = [
            nn.Linear(self.observation_dim, self.config.layer_size),
            nn.ReLU(),
        ]
        for _ in range(self.config.n_layers-2):
            layers.extend([
                nn.Linear(self.config.layer_size, self.config.layer_size),
                nn.ReLU(),
            ])
        layers.append(nn.Linear(self.config.layer_size, self.action_dim))
        self.actor_network = nn.Sequential(*layers).to(self.device)     # Actor network that learns
        #self.target_network = nn.Sequential(*layers).to(self.device)    # Frozen target network for bootstrapping
        #self.__copy_network(self.actor_network, self.target_network)
        self.optimizer = torch.optim.Adam(self.actor_network.parameters(), lr=self.config.learning_rate)

        self.replaybuffer = ReplayBuffer(self.config.replay_size)
        self.eps = EpsDecay(self.config.eps_init, self.config.eps_final, self.config.eps_decay_steps)
        if self.config.use_random_walk:
            self.explorer = RandomWalkAgent(num_actions=self.action_dim, observation_shape=self.observation_dim)
        else:
            self.explorer = None


    def get_action(self, reward, observation):
        self.replaybuffer.push(torch.from_numpy(self.last_state_action[0]),
                               torch.tensor([self.last_state_action[1]]),
                               torch.from_numpy(observation),
                               torch.tensor([reward]))

        eps = self.eps.step()
        if np.random.random() < eps:
            if self.explorer:
                action = self.explorer.step(reward, observation)
            else:
                action = self.env.action_space.sample()
        else:
            with torch.no_grad():
                Q = self.actor_network(torch.from_numpy(observation).to(self.device))
                action = torch.argmax(Q).cpu().item()

        self.last_state_action = [observation, action]
        self.__optimize()       # optimize every step
        return action
    
    
    def begin_episode(self, observation):
        if self.explorer:
            explorer_action = self.explorer.begin_episode(observation)
        eps = self.eps.step()
        if np.random.random() < eps:
            if self.explorer:
                action = explorer_action
            else:
                action = self.env.action_space.sample()
        else:
            with torch.no_grad():
                Q = self.actor_network(torch.from_numpy(observation).to(self.device))
                action = torch.argmax(Q).cpu().item()
                
        self.last_state_action = [observation, action]
        self.__optimize()       # optimize every step
        return action


    def end_episode(self, reward, terminal):
        self.last_state_action = []
        if self.explorer:
            self.explorer.end_episode(reward, terminal)
    
    def begin_iteration(self):
        pass

    def end_iteration(self):
        self.__optimize()
        #self.__copy_network(self.actor_network, self.target_network)

    def __optimize(self):
        if len(self.replaybuffer) < self.config.batch_size:
            return
        batch = self.replaybuffer.sample(self.config.batch_size)
        state_batch = torch.stack(batch.state).to(self.device)
        action_batch = torch.stack(batch.action).to(self.device)
        next_state_batch = torch.stack(batch.next_state).to(self.device)
        reward_batch = torch.stack(batch.reward).to(self.device)
        state_action_values = self.actor_network(state_batch).gather(1, action_batch)
        with torch.no_grad():
            target_state_action_values = torch.max(self.actor_network(next_state_batch), dim=1).values
            target_state_action_values = target_state_action_values * self.config.discount + reward_batch
        loss = ((state_action_values - target_state_action_values) ** 2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(self.actor_network.parameters(), 100)
        self.optimizer.step()

    

    def save(self, ckpt_path):
        ckpt_path = Path(ckpt_path)
        checkpoint_dir = Path(ckpt_path.parent)
        iteration_number = int(ckpt_path.stem.split('-')[-1].split('.')[-1])
        self.replaybuffer.save(checkpoint_dir / f'replay_{iteration_number}.pkl')
        super().save(ckpt_path)

    def load(self, ckpt_path):
        ckpt_path = Path(ckpt_path)
        checkpoint_dir = Path(ckpt_path.parent)
        iteration_number = int(ckpt_path.stem.split('-')[-1].split('.')[-1])
        self.replaybuffer.load(checkpoint_dir / f'replay_{iteration_number}.pkl')
        super().load(ckpt_path)
