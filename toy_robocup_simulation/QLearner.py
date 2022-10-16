"""
by: Razvan Ciuca
date: oct 10 2022

This file contains the class that implements the logic behind Q-Learning with a general function approximator
for Q(s,a). We use epsilon-random exploration, a replay buffer that we treat using importance sampling.

See Chapters 4 to 5 of Sutton & Barto: Reinforcement Learning
"""

import torch as t
from NNmodels import *

class QLearner:
    """
    NOT USED RIGHT NOW, MAIN Q-LEARNING LOGIC IS IN train.py
    """
    def __init__(self, explore_eps, ):
        pass

    def train(self, ):
        pass

    def sample(self):
        pass