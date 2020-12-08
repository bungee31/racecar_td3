from gym.envs.registration import register
from gym_env.environment import RacecarEnv

register(
    id='racecar-v0',
    entry_point='gym_env:RacecarEnv',
    reward_threshold =-100
)
