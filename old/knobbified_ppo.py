
#
# This is a second PPO implementation we attempted to adapt, offering more features
# and hyperparameters for us to tune in accordance with our project mentor's advice.
# Adapted from:
# - https://github.com/XinJingHao/PPO-Discrete-Pytorch
#
# This model did not work well, so we stuck with our original PPO model (see ppo.py)
#
# Arguments: --agent knob_ppo --config configs/old/knobbified_ppo_lambd=1.0.yml
#





import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
from balloon_agent import BalloonAgent
from network_utils import device, build_mlp
import math
import copy



class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, n_layers, layer_size):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, net_width)
        self.l2 = nn.Linear(net_width, net_width)
        self.l3 = nn.Linear(net_width, action_dim)

    def forward(self, state):
        n = torch.tanh(self.l1(state))
        n = torch.tanh(self.l2(n))
        return n

    def pi(self, state, softmax_dim = 0):
        n = self.forward(state)
        prob = F.softmax(self.l3(n), dim=softmax_dim)
        return prob

class Critic(nn.Module):
    def __init__(self, state_dim,net_width):
        super(Critic, self).__init__()

        self.C1 = nn.Linear(state_dim, net_width)
        self.C2 = nn.Linear(net_width, net_width)
        self.C3 = nn.Linear(net_width, 1)

    def forward(self, state):
        v = torch.relu(self.C1(state))
        v = torch.relu(self.C2(v))
        v = self.C3(v)
        return v
    


class KnobbifiedPPO(BalloonAgent):

    def init_policy(self):
        self.gamma = self.config.gamma
        self.lambd = self.config.lambd
        self.entropy_coef = self.config.entropy_coef
        self.entropy_coef_decay = self.config.entropy_coef_decay
        self.batch_size = self.config.batch_size
        self.T_horizon = self.config.steps_until_update
        self.n_actor_layers = self.config.n_actor_layers
        self.n_critic_layers = self.config.n_critic_layers
        self.actor_layer_size = self.config.actor_layer_size
        self.critic_layer_size = self.config.critic_layer_size
        self.update_steps = self.config.update_steps
        self.eps_clip = self.config.eps_clip
        self.l2_reg = self.config.l2_reg
        self.normalize_advantage = self.config.normalize_advantage
        self.lr = self.config.learning_rate



        '''Build Actor and Critic'''
        self.actor = build_mlp(self.observation_dim, self.action_dim, self.n_actor_layers, self.actor_layer_size).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic = build_mlp(self.observation_dim, 1, self.n_critic_layers, self.critic_layer_size).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

        '''Build Trajectory holder'''
        self.s_hoder = np.zeros((self.T_horizon, self.observation_dim), dtype=np.float32)
        self.a_hoder = np.zeros((self.T_horizon, 1), dtype=np.int64)
        self.r_hoder = np.zeros((self.T_horizon, 1), dtype=np.float32)
        self.s_next_hoder = np.zeros((self.T_horizon, self.observation_dim), dtype=np.float32)
        self.logprob_a_hoder = np.zeros((self.T_horizon, 1), dtype=np.float32)
        self.done_hoder = np.zeros((self.T_horizon, 1), dtype=np.bool_)
        self.dw_hoder = np.zeros((self.T_horizon, 1), dtype=np.bool_)



    def get_action(self, reward, observation):
        if self.last_state_action_logprob is not None:
            s, a, l = self.last_state_action_logprob
            self._put_data(s, a, reward, observation, l, False, False, idx=self.step_idx)
            self.step_idx += 1

        s = torch.from_numpy(observation).float().to(device)
        with torch.no_grad():
            pi = F.softmax(self.actor(s), dim=0)
            if not self.training:
                action = torch.argmax(pi).item()
                return action
            else:
                m = torch.distributions.Categorical(pi)
                action = m.sample().item()
                logprob = pi[action].item()

        self.last_state_action_logprob = [observation, action, logprob]
        return action
    
    def begin_episode(self, observation):
        self.step_idx = 0
        self.last_state_action_logprob = None
        return self.get_action(None, observation)

    def end_episode(self, observation, reward, terminal):
        if self.last_state_action_logprob is not None:
            s, a, l = self.last_state_action_logprob
            self._put_data(s, a, reward, observation, l, True, terminal, idx=self.step_idx)
    
    def begin_iteration(self):
        pass
    
    def end_iteration(self):
        self.optimize()

    def _put_data(self, s, a, r, s_next, logprob_a, done, dw, idx):
        self.s_hoder[idx] = s
        self.a_hoder[idx] = a
        self.r_hoder[idx] = r
        self.s_next_hoder[idx] = s_next
        self.logprob_a_hoder[idx] = logprob_a
        self.done_hoder[idx] = done
        self.dw_hoder[idx] = dw


    def optimize(self):
        self.entropy_coef *= self.entropy_coef_decay #exploring decay
        '''Prepare PyTorch data from Numpy data'''
        s = torch.from_numpy(self.s_hoder).to(device)
        a = torch.from_numpy(self.a_hoder).to(device)
        r = torch.from_numpy(self.r_hoder).to(device)
        s_next = torch.from_numpy(self.s_next_hoder).to(device)
        old_prob_a = torch.from_numpy(self.logprob_a_hoder).to(device)
        done = torch.from_numpy(self.done_hoder).to(device)
        dw = torch.from_numpy(self.dw_hoder).to(device)

        ''' Use TD+GAE+LongTrajectory to compute Advantage and TD target'''
        with torch.no_grad():
            vs = self.critic(s)
            vs_ = self.critic(s_next)

            '''dw(dead and win) for TD_target and Adv'''
            deltas = r + self.gamma * vs_ * (~dw) - vs
            deltas = deltas.cpu().flatten().numpy()
            adv = [0]

            '''done for GAE'''
            for dlt, done in zip(deltas[::-1], done.cpu().flatten().numpy()[::-1]):
                advantage = dlt + self.gamma * self.lambd * adv[-1] * (~done)
                adv.append(advantage)
            adv.reverse()
            adv = copy.deepcopy(adv[0:-1])
            adv = torch.tensor(adv).unsqueeze(1).float().to(device)
            td_target = adv + vs
            if self.normalize_advantage:
                adv = (adv - adv.mean()) / ((adv.std() + 1e-4))  #sometimes helps

        """PPO update"""
        #Slice long trajectopy into short trajectory and perform mini-batch PPO update
        optim_iter_num = int(math.ceil(s.shape[0] / self.batch_size))

        for _ in range(self.update_steps):
            #Shuffle the trajectory, Good for training
            perm = np.arange(s.shape[0])
            np.random.shuffle(perm)
            perm = torch.LongTensor(perm).to(device)
            s, a, td_target, adv, old_prob_a = \
                s[perm].clone(), a[perm].clone(), td_target[perm].clone(), adv[perm].clone(), old_prob_a[perm].clone()

            '''mini-batch PPO update'''
            for i in range(optim_iter_num):
                index = slice(i * self.batch_size, min((i + 1) * self.batch_size, s.shape[0]))

                '''actor update'''
                prob = F.softmax(self.actor(s[index]), dim=1)
                entropy = torch.distributions.Categorical(prob).entropy().sum(0, keepdim=True)
                prob_a = prob.gather(1, a[index])
                ratio = torch.exp(torch.log(prob_a) - torch.log(old_prob_a[index]))  # a/b == exp(log(a)-log(b))

                surr1 = ratio * adv[index]
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * adv[index]
                a_loss = -torch.min(surr1, surr2) - self.entropy_coef * entropy

                self.actor_optimizer.zero_grad()
                a_loss.mean().backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 40)
                self.actor_optimizer.step()

                '''critic update'''
                c_loss = (self.critic(s[index]) - td_target[index]).pow(2).mean()
                for name, param in self.critic.named_parameters():
                    if 'weight' in name:
                        c_loss += param.pow(2).sum() * self.l2_reg

                self.critic_optimizer.zero_grad()
                c_loss.backward()
                self.critic_optimizer.step()