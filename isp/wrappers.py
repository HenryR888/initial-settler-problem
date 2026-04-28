""" Wrapper for ISP for use with JAXMarl baselines. 
We call the ISP environment an keep track of episode-level stats while episode runs and then finally store totals when episode ends."""
import jax
import jax.numpy as jnp
import chex
from flax import struct
from functools import partial 

# from gymnax.environments import environment, spaces
from typing import Union, Any

class JaxMARLWrapper(object):
    """Base class for all jaxmarl wrappers."""

    def __init__(self, env):
        self._env = env

    def __getattr__(self, name: str):
        return getattr(self._env, name)
    
    def _batchify_floats(self, x: dict):
        return jnp.stack([x[a] for a in self._env.agents])
    

@struct.dataclass
class LogEnvState:
    env_state: Any
    # running live total for the current episodes:
    episode_returns: float # cumulative return for each agent so far (num_agents,)
    episode_lengths: int # num steps each agent has taken in current episode (num_agent,)
    episode_harvest: float # total amount of harvesting done by each agent during episode (num_agents,)
    episode_invest: float # total number of investing done by each agent during each episode
    episode_punish: float # total number of punishment done by each agent during each episode (num_agents,)
    # snapshot of the just-completed episode (for logging): 
    returned_episode_returns: float
    returned_episode_lengths: int
    returned_episode_harvest: float
    returned_episode_invest: float
    returned_episode_punish: float
    returned_river_level: float # scalar value
    returned_mean_energy: float # scalar value (mean across all agents)
    returned_collapse_rate: float # scalar value
    returned_mean_reputation: float 


class LogWrapper(JaxMARLWrapper):
    """
    Tracks per-episode returns, lengths, and ISP-specific action counts.
    Also, calls step_env directly (ISP handles its own resets internally)
    """

    def __init__(self,env,replace_info: bool = False):
        super().__init__(env)
        self.replace_info = replace_info

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey):
        obs, env_state = self._env.reset(key)
        n = self._env.num_agents
        state = LogEnvState(
            env_state=env_state,
            episode_returns=jnp.zeros((n,)),
            episode_lengths=jnp.zeros((n,), dtype=jnp.int32),
            episode_harvest=jnp.zeros((n,)),
            episode_invest=jnp.zeros((n,)),
            episode_punish=jnp.zeros((n,)),
            returned_episode_returns=jnp.zeros((n,)),
            returned_episode_lengths=jnp.zeros((n,), dtype=jnp.int32),
            returned_episode_harvest=jnp.zeros((n,)),
            returned_episode_invest=jnp.zeros((n,)),
            returned_episode_punish=jnp.zeros((n,)),
            returned_river_level=jnp.zeros(()),
            returned_mean_energy=jnp.zeros(()),
            returned_collapse_rate=jnp.zeros(()),
            returned_mean_reputation=jnp.zeros(()),
        )
        return obs, state
    
    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: LogEnvState,
        action: Union[int, float],
    ):
        obs, env_state, reward, done, info = self._env.step_env(
            key, state.env_state, action
        )
        ep_done = done["__all__"] # check to see that episode is complete

        rewards_vec = self._batchify_floats(reward)
        new_episode_return = state.episode_returns + rewards_vec
        new_episode_length = state.episode_lengths + 1
        new_episode_harvest = state.episode_harvest + info["harvest"]
        new_episode_invest = state.episode_invest + info["invest"]
        new_episode_punish = state.episode_punish + info["punish"]

        state = LogEnvState(
            env_state = env_state,
            # reset running totals when episode ends:
            episode_returns=new_episode_return * (1-ep_done),
            episode_lengths=new_episode_length*(1-ep_done),
            episode_harvest=new_episode_harvest*(1-ep_done),
            episode_invest=new_episode_invest*(1-ep_done),
            episode_punish=new_episode_punish*(1-ep_done),
            # once episode is completed, we snapshot the completed episode's values: 
            returned_episode_returns=state.returned_episode_returns*(1-ep_done)+new_episode_return*ep_done,
            returned_episode_lengths=state.returned_episode_lengths*(1-ep_done) + new_episode_length*ep_done,
            returned_episode_harvest=state.returned_episode_harvest*(1-ep_done)+ new_episode_harvest*ep_done,
            returned_episode_invest=state.returned_episode_invest*(1-ep_done) + new_episode_invest*ep_done,
            returned_episode_punish=state.returned_episode_punish * (1 - ep_done) + new_episode_punish * ep_done,
            returned_river_level=state.returned_river_level * (1-ep_done) + info["river_level"] * ep_done,
            returned_mean_energy=state.returned_mean_energy * (1-ep_done) + info["energy"].mean() * ep_done,
            returned_collapse_rate=state.returned_collapse_rate * (1-ep_done) + info["collapse"].astype(jnp.float32) * ep_done,
            returned_mean_reputation=state.returned_mean_reputation*(1-ep_done) + info["reputations"].mean() * ep_done
        )

        if self.replace_info:
            info = {}
        
        info["episode_returns"] = state.returned_episode_returns
        info["episode_lengths"] = state.returned_episode_lengths
        info["episode_harvest"] = state.returned_episode_harvest
        info["episode_invest"] = state.returned_episode_invest
        info["episode_punish"] = state.returned_episode_punish
        info["returned_river_level"] = state.returned_river_level
        info["returned_mean_energy"] = state.returned_mean_energy
        info["returned_collapse_rate"] = state.returned_collapse_rate
        info["returned_episode"] = jnp.full((self._env.num_agents,), ep_done)
        info["returned_mean_reputation"] = state.returned_mean_reputation

        return obs, state, reward, done, info
