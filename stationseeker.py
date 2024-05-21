
from balloon_learning_environment.agents.station_seeker_agent import StationSeekerAgent
from balloon_agent import BalloonAgent



class StationSeeker(BalloonAgent):

    
    def init_policy(self):
        self.ss = StationSeekerAgent(self.action_dim, self.observation_dim)


    def get_action(self, reward, observation):
        return self.ss.step(reward, observation)
    
    
    def begin_episode(self, observation):
        return self.ss.begin_episode(observation)


    def end_episode(self, reward, terminal):
        return self.ss.end_episode(reward, terminal)
    
    