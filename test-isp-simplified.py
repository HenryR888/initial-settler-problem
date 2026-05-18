import jax
import jax.numpy as jnp
from isp_simplified import CPR

env = CPR()
key = jax.random.PRNGKey(0)

obs, state = env.reset(key)
print("obs shape:", obs.shape) # should be (3,9)
print("patch_levels:", state.patch_levels)
print("timeout:", state.timeout)

# Step with all agents harvesting patch 0: 
key, subkey = jax.random.split(key)
actions = jnp.array([0, 0, 0])
obs, state, rewards, done, info = env.step(subkey, state, actions)
print("rewards:", rewards)
print("patch_levels after:", state.patch_levels)


key = jax.random.PRNGKey(0)
obs, state = env.reset(key)

total_rewards = jnp.zeros(env.num_agents)
for t in range(env.max_steps):
    key, action_key, step_key = jax.random.split(key, 3)
    # random policy for now
    actions = jax.random.randint(action_key, shape=(env.num_agents,), minval=0, maxval=env.num_actions)
    obs, state, rewards, done, info = env.step(step_key, state, actions)
    total_rewards += rewards
    if done: 
        break

print("total_rewards:", total_rewards)
print("final patch_levels:", state.patch_levels)

