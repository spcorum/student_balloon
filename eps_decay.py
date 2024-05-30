
import math


class EpsDecay():

    def __init__(self,
        init_eps: float,
        final_eps: float,
        total_steps: int
    ):
        self.init_eps = init_eps
        self.final_eps = final_eps
        self.total_steps = total_steps
        self.steps = 0

    def get(self):
        if self.steps >= self.total_steps:
            return self.final_eps
        return self.final_eps + (self.init_eps - self.final_eps) / math.exp(self.steps / self.total_steps)

    def step(self):
        self.steps += 1
        return self.get()