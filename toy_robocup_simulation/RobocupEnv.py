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
        self.friction_coeff = 0.7      # friction coefficient between wheels and terrain, needs to be tested

        # simulation constants
        self.sim_delta_t = 0.01        # simulation time-step, in seconds
        self.control_delta_t = 0.1     # the delta_t between agent decisions

        # =============================================
        # Pytorch Arrays
        # =============================================

        # initializing positions of first team randomly within the field
        # self.pos_team_A is an array of size [n_games, n_players, 2], hence
        # self.pos_team_A[i,j,1] is the y coordinate of the j-th player of the i-th game
        self.pos_team_A = t.cat([self.L_x * t.rand(n_games, n_players, 1),
                                 self.L_y * t.rand(n_games, n_players, 1)], dim=2)
        self.pos_team_B = t.cat([self.L_x * t.rand(n_games, n_players, 1),
                                 self.L_y * t.rand(n_games, n_players, 1)], dim=2)

        # initializing velocities and acclerations of both teams to zero
        self.vel_team_A = t.zeros(n_games, n_players, 2)
        self.vel_team_B = t.zeros(n_games, n_players, 2)

        # Initializing the accelerations to Zero
        # These are control arrays, the agent will decide what goes into them
        self.accel_team_A = t.zeros(n_games, n_players, 2)
        self.accel_team_B = t.zeros(n_games, n_players, 2)

        # initializing the rotational position to a random angle
        self.rot_team_A = 2.0 * t.pi * t.rand(n_games, n_players)
        self.rot_team_B = 2.0 * t.pi * t.rand(n_games, n_players)

        # initializing the rotational velocity to zero
        self.rot_vel_team_A = t.zeros(n_games, n_players)
        self.rot_vel_team_B = t.zeros(n_games, n_players)

        # initializing the rotational acceleration to zero
        # These are control arrays, the agent will decide what goes into them
        self.rot_accel_team_A = t.zeros(n_games, n_players)
        self.rot_accel_team_B = t.zeros(n_games, n_players)

    def compute_forces(self):

        pass

    def time_step(self, actions):

        # =============================================
        # Position and Velocity calculations
        # =============================================

        # =============================================
        # Robot-Robot Collisions
        # =============================================

        # =============================================
        # Robot-Ball non-shot non-dribble Collisions
        # =============================================

        # =============================================
        # Robot-Ball shot Collisions
        # =============================================

        # =============================================
        # Robot-Ball dribble Collisions
        # =============================================

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
