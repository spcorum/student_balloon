
#
# A wrapper around the Station Seeker model from the Bellemare paper.
# - https://github.com/google/balloon-learning-environment
#
# Arguments: --agent "station-seeker"
# (no config required)
#

from balloon_learning_environment.agents.station_seeker_agent import StationSeekerAgent
from balloon_agent import BalloonAgent



class StationSeeker(BalloonAgent):

    
    def init_policy(self):
        self.p = StationSeekerAgent(self.action_dim, self.observation_dim)


    def get_action(self, reward, observation):
        return self.p.step(reward, observation)
    
    
    def begin_episode(self, observation):
        return self.p.begin_episode(observation)


    def end_episode(self, observation, reward, terminal):
        return self.p.end_episode(reward, terminal)
    
    