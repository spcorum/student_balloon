
from balloon_learning_environment.agents.random_walk_agent import RandomWalkAgent
from balloon_agent import BalloonAgent



class RandomWalk(BalloonAgent):

    
    def init_policy(self):
        self.p = RandomWalkAgent(self.action_dim, self.observation_dim)


    def get_action(self, reward, observation):
        return self.p.step(reward, observation)
    
    
    def begin_episode(self, observation):
        return self.p.begin_episode(observation)


    def end_episode(self, observation, reward, terminal):
        return self.p.end_episode(reward, terminal)
    
    