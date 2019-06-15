from gym.envs.registration import register

register(
    id='ropod-v0',
    entry_point='gym_ropod.envs:RopodEnv',
)

register(
    id='ropod-nav-discrete-v0',
    entry_point='gym_ropod.envs:RopodNavDiscreteEnv',
)
