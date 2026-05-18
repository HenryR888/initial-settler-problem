import chex
import jax
import jax.numpy as jnp
from flax.struct import dataclass

NUM_AGENTS = 3
NUM_PATCHES = 3
HARVEST_AMOUNT = 0.2
REGEN_AMOUNT = 0.1
TIMEOUT_DURATION = 5
MAX_STEPS = 500

# Actions:
# 0...NUM_PATCHES-1 = Harvest patch i
# NUM_PATCHES...NUM_PATCHES+NUM_AGENTS-1 = Punish agent j (self-punish is NOOP)
# NUM_PATCHES + NUM_AGENTS = NOOP

class State:
    patch_level: chex.Array #(num_patches,) float32 in [0,1]
    timeout: chex.Array # (num_agents,) int32, steps remaining
    time: int

class CPR:
    def __init__(
        self,
        num_agents: int = NUM_AGENTS,
        num_patches: int = NUM_PATCHES,
        harvest_amount: float = HARVEST_AMOUNT,
        regen_amount: float = REGEN_AMOUNT,
        timeout_duration: int = TIMEOUT_DURATION,
        max_steps: int = MAX_STEPS,
        p_regen: tuple = (0.0, 0.01, 0.05, 0.10), # same probs as from Leibo et al. 2017
    ):
        self.num_agents = num_agents
        self.num_patches = num_patches
        self.harvest_amount = harvest_amount
        self.regent_amount = regen_amount
        self.timeout_duration = timeout_duration
        self.max_steps = max_steps
        self._p_regen = jnp.array(p_regen)
        self.num_actions = num_patches+num_agents+1
        self.obs_size = num_patches + num_agents + num_agents # the obs per agent is the patch levels + normalised timeout per agent, and a one-hot encoded agent id. 

    def reset(self, key: chex.PRNGKey):
        state = State(
            patch_levels = jnp.ones(self.num_patches, dtype = jnp.float32),
            timeout=jnp.zeros(self.num_agents, dtype=jnp.int32),
            time=0,
        )
        return self._get_obs(state), state
    
    def _get_obs(self, state: State) -> chex.Array:
        shared = jnp.concatenate([
            state.patch_levels,
            state.timeout.astype(jnp.float32)/self.timeout_duration
        ]) # shape is: (num_patches + num_agents,)
        agents_ids = jnp.eye(self.num_agents, dtype=jnp.float32) # (num_agents, num_agents)
        return jnp.concatenate([jnp.tile(shared, (self.num_agents, 1)), agents_ids], axis=1)
    
    def step(self, key: chex.PRNGKey, state:State, actions: chex.Array):
        key, regen_key = jax.random.split(key)

        noop = self.num_patches+self.num_agents
        effective_actions = jnp.where(state.timeout>0, noop, actions)

        patch_ids = jnp.arange(self.num_patches)
        punish_action_ids = jnp.arange(self.num_patches, self.num_patches + self.num_agents)

        # create (num_agents, num_patches) array for agent i harvesting patch j: 
        harvest_mask = (effective_actions[:, None] == patch_ids[None, :])

        # create (num_agents, num_patches) agent i punishes agent j, with no self-punishment: 
        punish_mask = (effective_actions[:, None] == punish_action_ids[None, :])
        punish_mask = punish_mask & ~jnp.eye(self.num_agents, dtype=bool)

        # rewawrd is just amount per alive patch harvested: 
        patch_alive = state.patch_levels>0
        rewards = jnp.sum(harvest_mask*patch_alive[None,:], axis=1).astype(jnp.float32)*self.harvest_amount

        # depletion: 
        harvests_per_patch = jnp.sum(harvest_mask, axis=0).astype(jnp.float32)
        new_patch_levels = jnp.maximum(0.0, state.patch_levels - harvests_per_patch*self.harvest_amount)

        # for the punishment, we set the timeout to timeout duration for any punished agent: 
        