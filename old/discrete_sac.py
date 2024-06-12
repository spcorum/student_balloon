
#
# This was our first (incomplete) attempt at implementing Discrete SAC ourselves, following
# the design of the Revisiting Discrete SAC paper:
# - https://github.com/coldsummerday/Revisiting-Discrete-SAC
#
# We found this implementation too demanding (it didn't fit into our existing training code)
# so we opted to directly use the original repo's code. See Revisiting_Discrete_SAC/
#
# **This code is not runnable, but is included for completeness**
#


#from Revisiting_Discrete_SAC.src.libs.discrete_sac import DiscreteSACDevPolicy

import torch
import torch.nn as nn
import torch.nn.functional as F
from network_utils import build_mlp
from replaybuffer import PrioritizedReplayBuffer
from decay import LinearDecay


class DiscreteSAC(nn.Module):

    
    def init_policy(self):
        self.actor = build_mlp(self.observation_dim, self.action_dim,
                               self.config.actor_n_layers, self.config.actor_layer_size)
        self.critic1 = build_mlp(self.observation_dim, self.action_dim,
                               self.config.critic_n_layers, self.config.critic_layer_size)
        self.critic2 = build_mlp(self.observation_dim, self.action_dim,
                               self.config.critic_n_layers, self.config.critic_layer_size)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), self.config.actor_lr)
        self.critic1_optim = torch.optim.Adam(self.critic1.parameters(), self.config.critic_lr)
        self.critic2_optim = torch.optim.Adam(self.critic2.parameters(), self.config.critic_lr)
        self.replaybuffer = PrioritizedReplayBuffer(self.config.replay_size, self.config.replay_alpha)
        self.replay_beta = LinearDecay(self.config.replay_beta_init, 1.0, self.config.replay_beta_steps)

    
    # Copied from Revisiting_Discrete_SAC/src/tianshou_frozen/policy/base.py
    def soft_update(self, tgt: nn.Module, src: nn.Module, tau: float) -> None:
        """Softly update the parameters of target module towards the parameters \
        of source module."""
        for tgt_param, src_param in zip(tgt.parameters(), src.parameters()):
            tgt_param.data.copy_(tau * src_param.data + (1 - tau) * tgt_param.data)


    def get_action(self, reward, observation):
        if self.training:
            self.replaybuffer.add((self.last_state_action[0],
                                   self.last_state_action[1],
                                   observation,
                                   reward, False), )

        logits = self.actor(...) # TODO
        dist = torch.distributions.Categorical(logits=logits)
        if self._deterministic_eval and not self.training:
            action = logits.argmax(axis=-1)
        else:
            action = dist.sample()

        self.last_state_action = [observation, action]
        return action
    
    def begin_episode(self, observation):
        action = self.get_action(None, observation) # check if None works
        self.last_state_action = [observation, action]

    def end_episode(self, observation, reward, terminal):
        if self.training:
            self.replaybuffer.push(torch.from_numpy(self.last_state_action[0]),
                                   torch.tensor([self.last_state_action[1]]),
                                   None if terminal else torch.from_numpy(observation),
                                   torch.tensor([reward]))
        self.last_state_action = []
    
    def begin_iteration(self):
        pass
    
    def end_iteration(self):
        pass

    def __optimize(self):
        beta = self.replay_beta.step()
        batch = self.replaybuffer.sample(self.config.batch_size, beta)

        target_q = batch.returns.flatten()

        # ?
        old_entropy = batch.pop("old_entropy", None)
        weight = batch.pop("weight", 1.0)
        act = to_torch(
            batch.act[:, np.newaxis], device=target_q.device, dtype=torch.long
        )

        # critic 1
        current_q1 = self.critic1(batch.obs).gather(1, act).flatten()
        td1 = current_q1 - target_q
        clipq_ratio = 0.0

        if self.use_clip_q:
            with torch.no_grad():
                q1_old = self.critic1_old(batch.obs).gather(1, act).flatten()
            clipped_q1 = q1_old + torch.clamp(current_q1 - q1_old, -self.clip_q_epsilon,
                                              self.clip_q_epsilon)
            q1_loss_1 = F.mse_loss(current_q1, target_q) * weight
            q1_loss_2 = F.mse_loss(clipped_q1, target_q) * weight
            critic1_loss = torch.maximum(q1_loss_1, q1_loss_2)
            clipq_ratio = torch.mean((q1_loss_2 >= q1_loss_1).float()).item()
            #state_dict['state/clipped_q1'] = clipped_q1.mean().item()
        else:
            critic1_loss = (td1.pow(2) * weight).mean()

        self.critic1_optim.zero_grad()
        critic1_loss.backward()
        self.critic1_optim.step()


        # critic 2
        current_q2 = self.critic2(batch.obs).gather(1, act).flatten()
        td2 = current_q2 - target_q
        if self.use_clip_q:
            with torch.no_grad():
                q2_old = self.critic2_old(batch.obs).gather(1, act).flatten()
            clipped_q2 = q2_old + torch.clamp(current_q2 - q2_old, -self.clip_q_epsilon,
                                              self.clip_q_epsilon)

            q2_loss_1 = F.mse_loss(current_q2, target_q) * weight
            q2_loss_2 = F.mse_loss(clipped_q2, target_q) * weight
            critic2_loss = torch.maximum(q2_loss_1, q2_loss_2)
            clipq_ratio = (clipq_ratio + torch.mean((q2_loss_2 >= q2_loss_1).float()).item()) / 2.0
            #state_dict['state/clipped_q2'] = clipped_q2.mean().item()
        else:
            critic2_loss = (td2.pow(2) * weight).mean()

        self.critic2_optim.zero_grad()
        critic2_loss.backward()
        self.critic2_optim.step()
        batch.weight = (td1 + td2) / 2.0  # prio-buffer

        # actor
        dist = self(batch).dist # ?
        entropy = dist.entropy()
        with torch.no_grad():
            current_q1a = self.critic1(batch.obs)
            current_q2a = self.critic2(batch.obs)

            if self.use_avg_q:
                q = torch.mean(torch.stack([current_q1a, current_q2a], dim=-1), dim=-1)

                min_q = torch.min(current_q1a, current_q2a).detach()
                #state_dict['state/avgq/min_q'] = min_q.mean().item()
            else:
                q = torch.min(current_q1a, current_q2a)

        actor_loss = -(self._alpha * entropy + (dist.probs * q).sum(dim=-1)).mean()

        if self.use_entropy_penalty:
            old_entropy = to_torch(
                old_entropy, device=entropy.device, dtype=torch.float
            )
            entropy_loss = F.mse_loss(old_entropy, entropy)
            #state_dict['loss/entropy_loss'] = entropy_loss.item()
            #state_dict['state/old_entropy'] = old_entropy.mean().item()
            actor_loss = actor_loss + self.entropy_penalty_beta * entropy_loss

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        if self._is_auto_alpha:
            log_prob = -entropy.detach() + self._target_entropy
            alpha_loss = -(self._log_alpha * log_prob).mean()
            self._alpha_optim.zero_grad()
            alpha_loss.backward()
            self._alpha_optim.step()
            self._alpha = self._log_alpha.detach().exp()

        # self.sync_weight():
        self.soft_update(self.actor_old, self.actor, self.tau)
        self.soft_update(self.critic_old, self.critic, self.tau)
