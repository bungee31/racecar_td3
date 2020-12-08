# Importing simulation
from gym_env.simulation import python_env

# Importing gym environment
import gym
from gym import spaces

# Other imports
import sys
import random
import numpy as np
import time
from math import pi, sqrt, atan2, cos, sin


# Defining helper functions wrt. reward shaping

def wallhit(scan):
    thresh = 0.30
    return min(scan) < thresh


def goalhit(dist):
    thresh = 0.1
    return dist < thresh


def closer(dist1, dist2):
    return dist2 < dist1


def diff(a1, a2):
    """
    Calculates the difference between two angles as well as the sign from a2 to
    a1.
    """
    phi = abs(a1 - a2) % (2 * pi)
    test = ((a1 - a2 >= 0) & (a1 - a2 <= pi)) or ((a1 - a2 <= -pi) & (a1 - a2 >= -2 * pi))

    if test:
        sign = 1
    else:
        sign = -1

    if phi > pi:
        return 2 * pi - phi, sign
    else:
        return phi, sign


class RacecarEnv(gym.Env):
    """
    Custom Environment that follows gym interface and defines the MIT-racecar
    simulation environment.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, turns=5):
        super(RacecarEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        # self.action_space = spaces.Discrete(1)
        # Example for using image as input:
        # self.observation_space = spaces.Box(low=0, high=255, shape=(1,), dtype=np.uint8)
        self._max_episode_steps = 250
        self.observation_space = spaces.Box(low=0.0, high=255.0, shape=(33, ), dtype=np.float32)
        self.action_space = spaces.Box(low=np.array([0.0, -3.0]), high=np.array([1.0, 3.0]), dtype=np.float32)

        # Defining the maximum distance to goal (used to normalize)
        self.maxgoaldist = 5
        self.alpha = -0.1
        self.turns = turns

        # Spawning the simulation
        self.sim = python_env(turns, 0)

    def observe(self):
        dist = sqrt((self.goal[0] - self.carx) ** 2 + (self.goal[1] - self.cary) ** 2)
        angle_to_goal = atan2((self.goal[1] - self.cary), (self.goal[0] - self.carx))
        phi, sign = diff(self.cartheta, angle_to_goal)
        return dist, phi, sign

    def step(self, action):
        '''
        Defining the step function - returning the observation space, reward and
        whether or not the action caused a reset (hitting goal or wall)
        '''
        # print(f"Action: {action}")
        # Calculating previous distances and angles
        old_dist, old_phi, old_sign = self.observe()
        self.oldx, self.oldy = self.carx, self.cary


        self.sim.action(action)

        # Fetching the resulting lidar as well as postion data
        sensor = self.sim.lidar()
        self.carx, self.cary, self.cartheta = self.sim.car

        # Calculating current distances and angles
        dist, phi, sign = self.observe()

        # Defining the observation vector
        self.obs = list(1 - np.array(sensor) / self.maxgoaldist)
        # act = action[0]
        self.obs.extend([dist / self.maxgoaldist, phi / pi, sign])
        # self.obs.extend(act)

        # Incrementing step counter
        self.current_step += 1

        if wallhit(sensor):
            print('I HIT A WALL!')
            print(f"Action: {action}, reward: -1500")
            return np.array(self.obs).reshape(1, -1), -1500, True, {}

        if goalhit(dist):
            print('I HIT THE GOAL!')
            print(f"Action: {action}, reward: 2000")
            return np.array(self.obs).reshape(1, -1), 2000, True, {}

        # Defining the reward shaping
        reward = self.alpha * dist
        # print(f"Reward: {reward}")
        print(f"Action: {action}, reward: {reward}")

        return np.array(self.obs).reshape(1, -1), reward, False, {}

    def reset(self, seed=64):
        """
        Resetting the environment for a new run
        """
        random.seed(seed)
        np.random.seed(seed)

        self.sim = python_env(self.turns, seed)

        # Initializing the goal
        self.goal = self.sim.goal

        self.sim.spawn(0, 0, 0)

        # Initializing step counter
        self.current_step = 0

        # Recieving sensor information
        sensor = self.sim.lidar()
        self.carx, self.cary, self.cartheta = self.sim.car

        # Calculating distances and angles for the observation vector
        dist, phi, sign = self.observe()

        # Defining the observation vector
        self.obs = list(1 - np.array(sensor) / self.maxgoaldist)
        # act = [0, 0]
        self.obs.extend([dist / self.maxgoaldist, phi / pi, sign])
        # self.obs.extend(act)

        return np.array(self.obs).reshape(1, -1)
