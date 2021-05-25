from gym.envs.registration import register

register(
    id='scaled-riverswim-v0',
    entry_point='riverswim_variants.envs:ScaledRiverSwimEnv',
    max_episode_steps=20,
)

register(
    id='stochastic-riverswim-v0',
    entry_point='riverswim_variants.envs:StochasticRiverSwimEnv',
    max_episode_steps=20,
)