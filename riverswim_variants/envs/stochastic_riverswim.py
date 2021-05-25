import gym
import numpy as np
from numpy.random import normal

from gym import error, spaces, utils
from gym.utils import seeding
from gym.envs.toy_text import discrete


LEFT = 0
RIGHT = 1

def categorical_sample(prob_n, np_random):
    """
    Sample from categorical distribution
    Each row specifies class probabilities
    """
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    return (csprob_n > np_random.rand()).argmax()


class StochasticRiverSwimEnv(discrete.DiscreteEnv):
    """
    Riverswim but rewards are now sampled according to a gaussian
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, nS=6):
        # Defining the number of actions
        nA = 2
        
        # Defining the reward system and dynamics of RiverSwim environment
        P, isd = self.__init_dynamics(nS, nA)
        
        super(StochasticRiverSwimEnv, self).__init__(nS, nA, P, isd)

    def __init_dynamics(self, nS, nA):
        
        # P[s][a] == [(probability, nextstate, reward, done), ...]
        P = {}
        for s in range(nS):
            P[s] = {a: [] for a in range(nA)}

        # Rewarded Transitions
        # NOTE: The rewards are functions, i.e. the generating distribution
        P[0][LEFT] = [(1., 0, lambda:normal(5/1000, 1), 0)]
        P[nS-1][RIGHT] = [(0.9, nS-1, lambda:normal(1, 1), 0), (0.1, nS-2, lambda:normal(1, 1), 0)]

        # Left Transitions
        for s in range(1, nS):
            P[s][LEFT] = [(1., max(0, s-1), lambda:normal(0, 0), 0)]

        # RIGHT Transitions
        for s in range(1, nS - 1):
            P[s][RIGHT] = [(0.3, min(nS - 1, s + 1), lambda:normal(0, 0), 0), (0.6, s, lambda:normal(0, 0), 0), (0.1, max(0, s-1), lambda:normal(0, 0), 0)]

        P[0][RIGHT] = [(0.3, 0, lambda:normal(0, 0), 0), (0.7, 1, lambda:normal(0, 0), 0)]

        # Starting State Distribution
        isd = np.zeros(nS)
        isd[0] = 1.

        return P, isd


    def step(self, a):
        transitions = self.P[self.s][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, d = transitions[i]
        r = r()
        self.s = s
        self.lastaction = a
        return (int(s), r, d, {"prob": p})


    def render(self, mode='human'):
        pass

    def close(self):
        pass
