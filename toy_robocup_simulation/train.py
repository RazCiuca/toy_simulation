"""
by: Razvan Ciuca
date: oct 10 2022

This file contains the main logic loop that trains our neural network via Q-learning to learn to play our
simulated robocup environment

See https://www.deeplearningbook.org/ chapter 8 for optimisation algorithms we use for deep learning
and https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html for a good tutorial and
boilerplate code for a basic training loop

"""
import torch as t
import torch.optim as optim
import numpy as np

import matplotlib.pyplot as plt

from NNmodels import *
from QLearner import *
from RobocupEnv import *


if __name__ == "__main__":

    episode_length = 2000
    max_training_iter = 100
    explor_eps = 0.1

    n_games = 100
    team_size = 6
    input_size = 16 * (1+team_size)
    output_size = 10  # for 9 possible actions for acceleration and 1 for kicking
    print(f"input size: {input_size}")

    # ==============================================================================================================
    # Initialising Agents and Environment
    # ==============================================================================================================

    env = RobocupEnv(n_games, team_size)
    model = MLP([input_size, 128, output_size])
    optimizer = t.optim.SGD(model.parameters(), lr=0.01, momentum=0.8)

    # ==============================================================================================================
    # Main Training Loop
    # ==============================================================================================================

    for i in range(0, max_training_iter):

        optimizer.zero_grad()

        for t in range(0, episode_length):

            # getting state for both teams from environment and passing it through network
            state_team_A, state_team_B = env.get_state()
            Q_values_A = model(state_team_A)
            Q_values_B = model(state_team_B)

            assert state_team_A.shape == t.Size([n_games, team_size, input_size])
            assert Q_values_A.shape == t.Size([n_games, team_size, output_size])

            # the greedy actions
            greedy_actions_A = t.argmax(Q_values_A, dim=2)
            greedy_actions_B = t.argmax(Q_values_B, dim=2)

            # todo: write code that samples from actions
            # sample actions from model output
            # uniformly random with probability epsilon
            actions = None

            # Do time step for all games and get rewards and new states
            rewards_A, rewards_B = env.time_step(actions)
            new_state_team_A, new_state_team_B = env.get_state()

            # find Q values for next state
            next_Q_values_A = model(new_state_team_A)
            next_Q_values_B = model(new_state_team_B)

            objective = t.mean((Q_values_A - (rewards_A + t.max(next_Q_values_A, dim=2)))**2)/episode_length \
                      + t.mean((Q_values_B - (rewards_B + t.max(next_Q_values_B, dim=2)))**2)/episode_length

            # store gradients in network, while we don't call optimizer.zero_grad(), gradients are added to model.w.grad
            objective.backward()

        optimizer.step()



    # ==============================================================================================================
    # Saving model and State trajectories
    # ==============================================================================================================

    # ==============================================================================================================
    # Plotting
    # ==============================================================================================================


