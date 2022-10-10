"""
by: Razvan Ciuca
date: oct 10 2022

This file contains the class that implements the logic behind Q-Learning with a general function approximator
for Q(s,a). We use epsilon-random exploration, a replay buffer that we treat using importance sampling.

See Chapters 4 to 5 of Sutton & Barto: Reinforcement Learning
"""

class QLearner:

    def __init__(self, epsilon):