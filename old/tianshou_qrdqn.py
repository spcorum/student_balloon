

import torch
import tianshou
from tianshou.data import ReplayBuffer, Collector, VectorReplayBuffer
from tianshou.data.batch import Batch
from tianshou.utils.net.common import Net





class QRDQN(nn.Module):

    
    def init_policy(self):
        self.policy_net = Net(
            state_shape=(self.observation_dim,),
            action_shape=(self.action_dim,),
            hidden_sizes=[128, 128, 128]
        )
        self.optim = torch.optim.Adam(self.policy_net.parameters(), lr=self.config.learning_rate)
        self.policy = tianshou.policy.QRDQNPolicy(
            self.policy_net,
            self.optim,
            discount_factor=self.config.discount,
            num_quantiles=self.config.num_quantiles,
            estimation_step=self.config.estimation_step,
            target_update_freq=self.config.target_update_freq,
            reward_normalization=self.config.reward_normalization,
        )
        self.buffer = ReplayBuffer(
            args.buffer_size,
            ignore_obs_next=True,
            save_only_last_obs=True,
            stack_num=args.frames_stack,
        )

    def get_action(self, reward, observation):
        batch = Batch()
        action_batch = self.policy.forward()
        # action_batch has keys: "act", "logits", "state" (the hidden state)
    
    def begin_episode(self, observation):
        raise NotImplementedError()

    def end_episode(self, reward, terminal):
        raise NotImplementedError()
    
    def begin_iteration(self):
        pass
    
    def end_iteration(self):
        pass

