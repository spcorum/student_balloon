
#
# A wrapper around the Perciatelli model from the Bellemare paper,
# reported as state of the art.
# - https://github.com/google/balloon-learning-environment
#
# Arguments: --agent "perciatelli"
# (no config required)
#

from balloon_learning_environment.agents.perciatelli44 import Perciatelli44
from balloon_agent import BalloonAgent



class Perciatelli(BalloonAgent):

    
    def init_policy(self):
        self.p = Perciatelli44(self.action_dim, [self.observation_dim])


    def get_action(self, reward, observation):
        return self.p.step(reward, observation)
    
    
    def begin_episode(self, observation):
        return self.p.begin_episode(observation)


    def end_episode(self, observation, reward, terminal):
        return self.p.end_episode(reward, terminal)
    
    