"""
by: Razvan Ciuca
date: oct 10 2022

file containing the RobocupEnv class, which contains all the logic that defines our game

todo for more accurate simulation:
- actual motor torque curves to define our acceleration capabilities
- friction between robots when turning against each other.

"""

import torch as t

class RobocupEnv:
    """
    For simplicity we will discretise the possible action space of our robot. At each time step it will have the
    following actions:
    - 9 possible choices of acceleration: either 0, or a constant in one of 8 directions
    - 3 choices of rotational acceleration: either 0, counterclockwise or clockwise
    - 1 shooting action, with a cooldown

    For the simulation, here are the things we take into account:

    1. Pure Acceleration from the motors
    2. collisions between robots
    3. collisions between ball and robots
    4. ball shooting
    5. ball dribbling interactions
    6. field edge effects

    Fundamental problem is needing to not use python for loops, given that they're much too slow.
    So we need to accomplish all these tasks with pytorch array manipulations as much as possible.

    Another problem is the time-slice size. The times at which we can the agent to make decisions and the
    timeslice we use in the simulation are different.

    Another possibility is just to check for robots intersecting and then to simply set their positions
    along the line joining their centers, with their speeds reversed along this direction.

    """
    def __init__(self, n_games, n_players):

        self.n_games = n_games         # number of simultaneous games played
        self.n_players = n_players     # number of players in each team

        # game constants
        self.L_x = 10.4                # Length of field in meters
        self.L_y = 7.4                 # Width of field in meters
        self.ball_diam = 0.043         # diameter of playing ball, in meters
        self.ball_mass = 0.046         # mass of playing ball, in meters

        # robot constants
        self.robot_radius = 0.178      # radius of robot in meters
        self.robot_mass = 2.62         # robot mass in kg
        self.accel = 1.0               # linear acceleration of robot, in m/s^2
        self.max_speed = 2.0           # max speed of robot, in m/s
        self.rot_accel = 1.0           # max rotational acceleration, in rad/s^2
        self.max_rot_speed = 10.0      # max rotational speed, in rad/s
        self.elastic_col_coeff = 0.3   # fraction of energy conserved when robots collide

        # shooting constants, useful for computing ball arcs
        self.shoot_width = 0.07        # width of "foot" that shoots the ball, in meters
        self.shoot_height = 0.05       # height of shooting foot, in meters
        self.shoot_energy = 10.0       # energy of ball kick, in joules

        # physical constants
        self.g = 9.81                  # gravitational acceleration, in m/s^2
        self.friction_coeff_wheels = 0.7  # friction coefficient between wheels and terrain, needs to be tested
        self.friction_coeff_ball = 2e-3  # friction coefficient between rolling ball and terrain

        # simulation constants
        self.sim_delta_t = 0.01        # simulation time-step, in seconds
        self.control_delta_t = 0.1     # the delta_t between agent decisions

        # =============================================
        # Pytorch Arrays
        # =============================================

        # initializing positions of both teams randomly within the field
        # self.pos_agents is an array of size [n_games, 2* n_players, 2], hence for example
        # self.pos_agents[i,j,1] is the y coordinate of the j-th player of the i-th game
        self.pos_agents = t.cat([self.L_x * t.rand(n_games, 2*n_players, 1),
                                self.L_y * t.rand(n_games, 2*n_players, 1)], dim=2)
        self.pos_ball = t.cat([self.L_x * t.rand(n_games, 1, 1), self.L_y * t.rand(n_games, 1, 1)], dim=2)

        # todo: add "kick" tensor corresponding to current strength of kick

        # initializing velocities and accelerations of both teams to zero
        self.vel_agents = t.zeros(n_games, 2*n_players, 2)
        self.vel_ball = t.zeros(n_games, 1, 2)

        # Initializing the accelerations to Zero
        # These are control arrays, the agent will decide what goes into them
        self.accel_agents = t.zeros(n_games, 2*n_players, 2)
        self.accel_ball = t.zeros(n_games, 1, 2)

    def time_step(self, actions):

        # =============================================
        # Computing acceleration from actions tensor
        # =============================================

        # =============================================
        # Position and Velocity calculations
        # =============================================

        # increment position from velocity
        self.pos_agents += self.sim_delta_t * self.vel_agents
        self.pos_ball += self.sim_delta_t * self.vel_ball

        # increment velocity from acceleration
        self.vel_agents += self.sim_delta_t * self.accel_agents
        self.vel_ball += self.sim_delta_t * self.accel_ball

        # =============================================
        # Kick force exponential decrease
        # =============================================

        # =============================================
        # Ball & Robot friction dynamics
        # =============================================

        # =============================================
        # Robot-Robot Repulsion
        # =============================================

        # now this is a tensor of size [n_games, 2*n_players, 2*n_players, 2]
        # which is simply pos_agents copied 2*n_player times along dim=2
        pos_expanded = self.pos_agents.unsqueeze(2).repeat(1, 1, 2*self.n_players, 2)

        # X_dist[k, i, j] constains a tensor of size 2 with the relative position of the i-th to the j-th player
        # in the k-th game
        pos_diff = pos_expanded - pos_expanded.transpose(1, 2)

        # todo: Compute repulsive force from position difference

        # todo: add it to the acceleration tensor

        # =============================================
        # Robot-Ball Repulsion & "kick"
        # =============================================

        # todo: same thing for the the ball-ball

        # =============================================
        # Robot-Wall Collisions
        # =============================================

        # =============================================
        # Ball-Wall Collisions
        # =============================================

        # =============================================
        # Ball-Goal Distance
        # =============================================

        pass

    def get_episode_history(self):
        pass
