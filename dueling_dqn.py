



class DuelingDQNBase(dqn_agent):
    pass





class DuelingDQN(nn.Module):

    
    def init_policy(self):

        self.model = DuelingDQNBase()


    def get_action(self, reward, observation):
        return self.model.action(observation)
    
    def begin_episode(self, observation):
        raise NotImplementedError()

    def end_episode(self, reward, terminal):
        raise NotImplementedError()
    
    def begin_iteration(self):
        pass
    
    def end_iteration(self):
        pass
