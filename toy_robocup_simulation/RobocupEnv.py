"""
by: Razvan Ciuca
date: oct 10 2022

file containing the RobocupEnv class, which contains all the logic that defines our game

"""

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import numpy as np
import torch as t
import torch.nn.functional as F

class RobocupEnv:
    """
    For simplicity we will discretise the possible action space of our robot. At each time step it will have the
    following actions:
    - 9 possible choices of acceleration: either 0, or a constant in one of 8 directions
    - 1 shooting action, with a cooldown
    """
    def __init__(self, n_games, team_size):

        self.n_games = n_games         # number of simultaneous games played
        self.team_size = team_size     # number of players in each team
        self.state_size = 16 * (1+self.team_size)  # the dimension of our state

        # game constants
        self.L_x = 10.4                # Length of field in meters
        self.L_y = 7.4                 # Width of field in meters
        self.ball_diam = 0.043         # diameter of playing ball, in meters, NOT USED RIGHT NOW
        self.ball_mass = 0.046         # mass of playing ball, in meters, NOT USED RIGHT NOW

        # robot constants
        self.robot_radius = 0.178      # radius of robot in meters, NOT USED RIGHT NOW
        self.robot_mass = 2.62         # robot mass in kg, NOT USED RIGHT NOW
        self.accel = 6.0               # linear acceleration of robot, in m/s^2
        self.max_speed = 4.0           # max speed of robot, in m/s

        # shooting constants
        self.k_repul_baseline = 0.1    # baseline repulsive constant
        self.k_kick_multiple = 10.0    # temporary multiplicative increase in repulsive contant after kick
        self.kick_halflife = 0.5       # kick half-life in seconds
        self.halflives_before_kick = 8  # number of halflives to wait before being able to kick again

        # physical constants
        self.g = 9.81                  # gravitational acceleration, in m/s^2
        self.friction_coeff_wheels = 0.1  # friction coefficient between wheels and terrain, needs to be tested
        self.friction_coeff_ball = 2e-3  # friction coefficient between rolling ball and terrain

        # simulation constants
        self.time = 0                  # time since start of episode, in seconds
        self.sim_delta_t = 0.01        # simulation time-step, in seconds
        self.control_delta_t = 0.1     # the delta_t between agent decisions
        self.sim_steps_between_actions = int(self.control_delta_t/self.sim_delta_t)

        # ==============================================================================================================
        # Pytorch Arrays
        # ==============================================================================================================

        # initializing positions of both teams randomly within the field
        # self.pos_agents is an array of size [n_games, 2* team_size, 2], hence for example
        # self.pos_agents[i,j,1] is the y coordinate of the j-th player of the i-th game
        self.pos_agents = t.cat([self.L_x * t.rand(n_games, 2 * team_size, 1),
                                 self.L_y * t.rand(n_games, 2 * team_size, 1)], dim=2)
        self.pos_ball = t.cat([self.L_x * t.rand(n_games, 1, 1), self.L_y * t.rand(n_games, 1, 1)], dim=2)

        # keeping track of current "kick multiple", after a kick action, this becomes self.k_kick_multiple, then
        # exponentially decreases with time
        self.kick_tracking = t.zeros(n_games, 2 * team_size, 1)

        # initializing velocities and accelerations of both teams to zero
        self.vel_agents = t.zeros(n_games, 2 * team_size, 2)
        self.vel_ball = t.zeros(n_games, 1, 2)

        # Initializing the accelerations to Zero
        # These are control arrays, the agent will decide what goes into them
        self.accel_agents = t.zeros(n_games, 2 * team_size, 2)
        self.accel_ball = t.zeros(n_games, 1, 2)

        # initializing arrays containing the distance between ball and goal
        # these have shape [n_games]
        self.ball_goal_dist_team_A = (
                    (self.pos_ball[:, :, 0] ** 2 + (self.pos_ball[:, :, 1] - self.L_y / 2) ** 2) ** 0.5).squeeze()
        self.ball_goal_dist_team_B = (((self.pos_ball[:, :, 0] - self.L_x) ** 2 + (
                    self.pos_ball[:, :, 1] - self.L_y / 2) ** 2) ** 0.5).squeeze()

        # storing states
        self.prev_state_team_A = None
        self.prev_state_team_B = None

    def get_state(self):
        """
        :return: should return the state suitable for passing to our model for action prediction

        this function should return the state that we will pass to the neural network to make decisions
        for all players.

        It should include
        1. the position and velocity of the ball at t_n and t_(n-1)
        2. the position and velocity of the controlled agent at t_n and t_(n-1)
        3. the position and velocity of teammates (including itself) at t_n and t_(n-1)
        4. the position and velocity of opponents at t_n and t_(n-1)

        our total state size is 16*(1+self.team_size)

        """
        # these are here as convenient reminders of the sizes of useful tensors
        assert self.pos_ball.shape == t.Size([self.n_games, 1, 2])
        assert self.vel_ball.shape == t.Size([self.n_games, 1, 2])
        assert self.pos_agents.shape == t.Size([self.n_games, 2*self.team_size, 2])
        assert self.vel_agents.shape == t.Size([self.n_games, 2*self.team_size, 2])

        # extract the positions and velocities of each team
        pos_agents_A = self.pos_agents[:, :self.team_size]
        pos_agents_B = self.pos_agents[:, self.team_size:]
        vel_agents_A = self.vel_agents[:, :self.team_size]
        vel_agents_B = self.vel_agents[:, self.team_size:]

        # combine the velocity and position in a single tensor
        pos_vel_A = t.cat([pos_agents_A, vel_agents_A], dim=2)
        pos_vel_B = t.cat([pos_agents_B, vel_agents_B], dim=2)

        assert pos_vel_A.shape == t.Size([self.n_games, self.team_size, 4])
        assert pos_vel_B.shape == t.Size([self.n_games, self.team_size, 4])

        # for every player in the team we need a whole copy of the positions and velocities of everyone
        # this is massaging the tensors so that state_pos_team_A[i,j] has size [self.team_size, 2]
        # and hence contains the position of the whole team, whatever the j is.
        state_pos_team_A = pos_agents_A.reshape(self.n_games, -1).unsqueeze(1).repeat(1, self.team_size, 1)
        state_vel_team_A = vel_agents_A.reshape(self.n_games, -1).unsqueeze(1).repeat(1, self.team_size, 1)
        state_pos_team_B = pos_agents_B.reshape(self.n_games, -1).unsqueeze(1).repeat(1, self.team_size, 1)
        state_vel_team_B = vel_agents_B.reshape(self.n_games, -1).unsqueeze(1).repeat(1, self.team_size, 1)

        assert state_pos_team_A.shape == t.Size([self.n_games, self.team_size, 2*self.team_size])
        assert state_vel_team_A.shape == t.Size([self.n_games, self.team_size, 2*self.team_size])

        # repeat the state of the ball for all agents
        pos_vel_ball = t.cat([self.pos_ball, self.vel_ball], dim=2)
        state_ball = pos_vel_ball.repeat(1, self.team_size, 1)

        assert state_ball.shape == t.Size([self.n_games, self.team_size, 4])

        # now we finally combine everything into a state for each team
        # we have two different states because the two teams need to be treated differently, an agent in team A
        # needs to be told (via the position of arguments) which team is his
        state_team_A = t.cat([pos_vel_A, state_ball, state_pos_team_A, state_vel_team_A, state_pos_team_B, state_vel_team_B], dim=2)
        state_team_B = t.cat([pos_vel_B, state_ball, state_pos_team_B, state_vel_team_B, state_pos_team_A, state_vel_team_A], dim=2)

        assert state_team_A.shape == t.Size([self.n_games, self.team_size, 2*(2*(2+2*self.team_size))])

        if self.prev_state_team_A is None:
            full_state_team_A = t.cat([state_team_A, state_team_A], dim=2)
        else:
            full_state_team_A = t.cat([state_team_A, self.prev_state_team_A], dim=2)
        if self.prev_state_team_B is None:
            full_state_team_B = t.cat([state_team_B, state_team_B], dim=2)
        else:
            full_state_team_B = t.cat([state_team_B, self.prev_state_team_B], dim=2)

        assert full_state_team_A.shape == t.Size([self.n_games, self.team_size, 16 * (1+self.team_size)])

        self.prev_state_team_A = state_team_A.clone()
        self.prev_state_team_B = state_team_B.clone()

        return full_state_team_A, full_state_team_B

    def get_episode_history(self):
        pass

    def save_episode_to_file(self):
        pass

    def compute_accel_from_actions(self, actions):

        assert self.accel_agents.shape == t.Size([self.n_games, 2 * self.team_size, 2])
        assert self.accel_ball.shape == t.Size([self.n_games, 1, 2])

        # we assume that the first index corresponds to not accelerating at all and the 1:9 indices correspond to
        # directions of acceleration
        no_accel_actions = actions[:, :, 0:1]
        accel_actions = actions[:, :, 1:9]

        # compute the implied acceleration in x for all agents
        self.accel_agents[:, :, 0:1] += self.accel * (1.0 - no_accel_actions) * (
            t.cos(2 * np.pi * t.argmax(accel_actions, dim=2, keepdim=True) / 8))

        # compute the implied acceleration in y for all agents
        self.accel_agents[:, :, 1:2] += self.accel * (1.0 - no_accel_actions) * (
            t.sin(2 * np.pi * t.argmax(accel_actions, dim=2, keepdim=True) / 8))

        # cap velocity for agents
        vel_agents_norm = t.norm(self.vel_agents, dim=2, keepdim=True) + 1e-7
        speed_violations = vel_agents_norm > self.max_speed

        self.vel_agents = self.vel_agents * t.logical_not(speed_violations).float() + (
                self.max_speed * self.vel_agents / vel_agents_norm * speed_violations.float())

    def kick_force_tracking(self, actions):

        # our halflife is self.kick_halflife
        # we want it to decrease like 0.5^(t/halflife) = e^(-ln(2) * t/halflife)
        # this way we automatically adjust the exponential decrease if we change self.sim_delta_t
        alpha = np.exp(-np.log(2.0) * self.sim_delta_t / self.kick_halflife)

        self.kick_tracking *= alpha

        # extract the kick action from the overall actions tensor
        kick_actions = actions[:, :, -1:]  # shape [n_games, 2*team_size, 1]
        assert kick_actions.shape == t.Size([self.n_games, 2 * self.team_size, 1])

        # on those players where they've waited enough time for the kick, add the kick multiple to tracking
        self.kick_tracking += (self.k_kick_multiple * self.k_repul_baseline) * kick_actions * (
                self.kick_tracking < 0.5 ** self.halflives_before_kick)

        assert self.kick_tracking.shape == t.Size([self.n_games, 2 * self.team_size, 1])

    def robot_robot_repulsion(self):
        # now this is a tensor of size [n_games, 2*team_size, 2*team_size, 2]
        # which is simply pos_agents copied 2*n_player times along dim=2
        pos_expanded = self.pos_agents.unsqueeze(2).repeat(1, 1, 2 * self.team_size, 1)
        assert pos_expanded.shape == t.Size([self.n_games, 2 * self.team_size, 2 * self.team_size, 2])

        # agent_pos_diff[k, i, j] constains a tensor of size 2 with the relative position of
        # the i-th to the j-th player in the k-th game
        agent_pos_diff = pos_expanded - pos_expanded.transpose(1, 2)  # size = [n_games, 2*team_size, 2*team_size, 2]
        assert agent_pos_diff.shape == t.Size([self.n_games, 2 * self.team_size, 2 * self.team_size, 2])

        # tensor containing the distance between the i-th and j-th player
        # here we add 1e-7 to avoid zero when we compute the distances with themselves
        # size = [n_games, 2*team_size, 2*team_size, 1]
        agent_distances = t.norm(agent_pos_diff, dim=3, keepdim=True) + 1e-7
        assert agent_distances.shape == t.Size([self.n_games, 2 * self.team_size, 2 * self.team_size, 1])

        # Compute repulsive force from position difference
        # these equations make use of pytorch broadcasting to combine tensors of different shapes
        agent_force_norms = self.k_repul_baseline / (agent_distances ** 2)
        net_agent_forces = t.sum((agent_pos_diff * agent_force_norms) / agent_distances, dim=2, keepdim=False)
        assert agent_force_norms.shape == t.Size([self.n_games, 2 * self.team_size, 2 * self.team_size, 1])
        assert net_agent_forces.shape == t.Size([self.n_games, 2 * self.team_size, 2])

        # add our net forces to the acceleration tensor
        self.accel_agents += net_agent_forces

    def robot_ball_repulsion(self):
        ball_pos_diff = self.pos_ball - self.pos_agents  # size = [n_games, 2*team_size, 2]
        assert ball_pos_diff.shape == t.Size([self.n_games, 2 * self.team_size, 2])

        # tensor containing the distance between the ball and i-th player
        ball_distances = t.norm(ball_pos_diff, dim=2, keepdim=True) + 1e-7  # size = [n_games, 2*team_size, 1]
        assert ball_distances.shape == t.Size([self.n_games, 2 * self.team_size, 1])

        ball_force_norms = self.k_repul_baseline * (1 + self.kick_tracking) / ball_distances ** 2
        net_ball_forces = t.sum(ball_pos_diff * ball_force_norms / ball_distances, dim=1, keepdim=True)

        assert net_ball_forces.shape == t.Size([self.n_games, 1, 2])

        # add our net forces to the acceleration tensor
        self.accel_ball += net_ball_forces

    def robot_ball_friction(self):
        # decrease acceleration in line with the velocity
        self.accel_agents -= self.friction_coeff_wheels * self.g * \
                             self.vel_agents / (t.norm(self.vel_agents, dim=2, keepdim=True) + 1e-7)

        self.accel_ball -= self.friction_coeff_ball * self.g * \
                           self.vel_ball / (t.norm(self.vel_ball, dim=1, keepdim=True) + 1e-7)

    def wall_collisions(self):
        # on the first loop this does agent robot-wall collisions, and on the second ball-wall collisions

        for pos, vel in zip([self.pos_agents, self.pos_ball], [self.vel_agents, self.vel_ball]):
            x_outbound_left = (pos[:, :, 0:1] < 0)
            x_outbound_right = (pos[:, :, 0:1] > self.L_x)
            y_outbound_bottom = (pos[:, :, 1:2] < 0)
            y_outbound_up = (pos[:, :, 1:2] > self.L_y)

            x_outbounds_bools = x_outbound_left + x_outbound_right
            y_outbounds_bools = y_outbound_bottom + y_outbound_up

            # shape [n_games, 2*team_size, 2], consists of either 1 or 0, giving the parameters out of bounds
            outbounds_bools = t.cat([x_outbounds_bools, y_outbounds_bools], dim=2)

            # these are necessary manipulations to prepare our outbound booleans to index the pos_agent arrays
            zeros_temp = t.zeros(pos.shape[0], pos.shape[1], 1).bool()
            x_left_indices = t.cat([x_outbound_left, zeros_temp], dim=2)
            x_right_indices = t.cat([x_outbound_right, zeros_temp], dim=2)
            y_up_indices = t.cat([zeros_temp, y_outbound_up], dim=2)
            y_down_indices = t.cat([zeros_temp, y_outbound_bottom], dim=2)

            vel *= (-1) ** outbounds_bools

            pos[x_left_indices] = 0.0
            pos[x_right_indices] = self.L_x
            pos[y_down_indices] = 0.0
            pos[y_up_indices] = self.L_y

    def pos_vel_update(self):

        # increment position from velocity
        self.pos_agents += self.sim_delta_t * self.vel_agents
        self.pos_ball += self.sim_delta_t * self.vel_ball

        # increment velocity from acceleration
        self.vel_agents += self.sim_delta_t * self.accel_agents
        self.vel_ball += self.sim_delta_t * self.accel_ball

    def get_reward(self):

        new_ball_goal_dist_team_A = ((self.pos_ball[:, :, 0] ** 2 +
                                      (self.pos_ball[:, :, 1] - self.L_y / 2) ** 2) ** 0.5).squeeze()
        new_ball_goal_dist_team_B = (((self.pos_ball[:, :, 0] - self.L_x) ** 2 +
                                      (self.pos_ball[:, :, 1] - self.L_y / 2) ** 2) ** 0.5).squeeze()
        assert new_ball_goal_dist_team_A.shape == t.Size([self.n_games])
        assert new_ball_goal_dist_team_B.shape == t.Size([self.n_games])

        goal_team_A = 1.0 / new_ball_goal_dist_team_A - 1.0 / self.ball_goal_dist_team_A
        goal_team_B = 1.0 / new_ball_goal_dist_team_B - 1.0 / self.ball_goal_dist_team_B

        self.ball_goal_dist_team_A = new_ball_goal_dist_team_A
        self.ball_goal_dist_team_B = new_ball_goal_dist_team_B

        reward_team_A = goal_team_A - goal_team_B
        reward_team_B = -reward_team_A

        return reward_team_A, reward_team_B

    def time_step(self, actions, verbose=False):
        """
        :param actions:
            we assume that actions is a tensor of size [n_games, 2*team_size, 10]
            the first 9 numbers in dim=2 are a one-hot encoding of the acceleration direction, the last
            number is assumed to either be 1 or 0, corresponding to the kick decision
        :param new_action_bool: boolean that states if we are doing a simulation step in which new actions
            are given by the model, or not
        :return reward_team_A: the rewards for team A, the team B rewards are the negatives of that.
        """

        # big simulation loop between asking model to send actions
        for i in range(0, self.sim_steps_between_actions):

            self.time += self.sim_delta_t
            self.accel_agents *= 0.0
            self.accel_ball *= 0.0

            # ==============================================================================================================
            # Computing acceleration from actions tensor and capping velocity
            # ==============================================================================================================

            self.compute_accel_from_actions(actions)
            if verbose: print(f"end of action to accel conversion: {self.pos_agents[0, 0]}")

            # ==============================================================================================================
            # Kick force exponential decrease and tracking
            # ==============================================================================================================

            self.kick_force_tracking(actions)
            if verbose: print(f"end of kick force exponential decrease: {self.pos_agents[0, 0]}")

            # ==============================================================================================================
            # Robot-Robot Repulsion
            # ==============================================================================================================

            self.robot_robot_repulsion()
            if verbose: print(f"end of robot-robot repulsion: {self.pos_agents[0, 0]}")

            # ==============================================================================================================
            # Robot-Ball Repulsion
            # ==============================================================================================================

            self.robot_ball_repulsion()
            if verbose: print(f"end of robot-ball repulsion: {self.pos_agents[0, 0]}")

            # ==============================================================================================================
            # Ball & Robot friction dynamics
            # ==============================================================================================================

            self.robot_ball_friction()
            if verbose: print(f"end of friction compute: {self.pos_agents[0, 0]}")

            # ==============================================================================================================
            # Robot-Wall and Ball-Wall Collisions
            # ==============================================================================================================

            self.wall_collisions()
            if verbose: print(f"end of wall collisions: {self.pos_agents[0, 0]}")

            # ==============================================================================================================
            # Position and Velocity Updates
            # ==============================================================================================================

            self.pos_vel_update()
            if verbose: print(f"end of acceleration compute: {self.pos_agents[0, 0]}")

        # ==============================================================================================================
        # Ball-Goal Distance
        # ==============================================================================================================
        # we assume the goals are at coordinates (0, L_y/2) and (L_x , L_y/2)

        return self.get_reward()


if __name__ == "__main___":

    # ==============================================================================================================
    # Plotting Coordinates of a single Agent
    # ==============================================================================================================

    t.manual_seed(1334315)

    n_games = 10
    team_size = 6

    env = RobocupEnv(n_games, team_size)

    actions = t.cat([F.one_hot(t.floor(9 * t.rand(n_games, 2 * team_size)).long()),
                     t.floor(2 * t.rand(n_games, 2 * team_size, 1)).long()], dim=2)

    assert actions.shape == t.Size([n_games, 2 * team_size, 10])

    time = []
    x = []
    y = []

    for i in range(0, 1000):
        print(env.pos_agents[0, 0])
        if i%10 == 0:
            time.append(i)
            x.append(env.pos_agents[0, 0, 0].item())
            y.append(env.pos_agents[0, 0, 1].item())
        env.time_step(actions, verbose=False)

    # Bouncing Parabolas!! it (probably) works
    plt.subplot(211)
    plt.title("x position over time")
    plt.plot(np.array(time), np.array(x))
    plt.subplot(212)
    plt.title("y position over time")
    plt.plot(np.array(time), np.array(y))
    plt.show()


if __name__ == "__main__":

    # ==============================================================================================================
    # Making Plot of agent trajectories under constant actions
    # ==============================================================================================================

    t.manual_seed(1334315)

    n_games = 10
    team_size = 6

    env = RobocupEnv(n_games, team_size)

    actions = t.cat([F.one_hot(t.floor(9 * t.rand(n_games, 2 * team_size)).long()),
                     t.floor(2 * t.rand(n_games, 2 * team_size, 1)).long()], dim=2)

    assert actions.shape == t.Size([n_games, 2 * team_size, 10])

    time = []
    x_team_A = []
    y_team_A = []
    x_team_B = []
    y_team_B = []

    for i in range(0, 100):

        # fun with changing the action directly every few iterations
        if i % 20 == 0:
            actions = t.cat([F.one_hot(t.floor(9 * t.rand(n_games, 2 * team_size)).long()),
                             t.floor(2 * t.rand(n_games, 2 * team_size, 1)).long()], dim=2)

        if i % 1 == 0:
            time.append(i*env.control_delta_t)

            print(f"time: {time[-1]} + sample agent pos: {env.pos_agents[0, 0]}")

            x_team_A.append(env.pos_agents[0, 0:team_size, 0].numpy().copy())
            y_team_A.append(env.pos_agents[0, 0:team_size, 1].numpy().copy())
            x_team_B.append(env.pos_agents[0, team_size:, 0].numpy().copy())
            y_team_B.append(env.pos_agents[0, team_size:, 1].numpy().copy())

            state_A, state_B = env.get_state()

        reward, new_state = env.time_step(actions, verbose=False)

    x_team_A = np.array(x_team_A)
    y_team_A = np.array(y_team_A)
    x_team_B = np.array(x_team_B)
    y_team_B = np.array(y_team_B)

    for i in range(0, team_size):
        plt.plot(x_team_A[:, i], y_team_A[:, i], color="r")
        plt.plot(x_team_B[:, i], y_team_B[:, i], color="b")

    plt.margins(0.0)
    plt.show()
