import jax
from isp import ISP

# we instantiate with jit = False so errors are readable:
env = ISP(num_agents=3, jit=False)
key = jax.random.PRNGKey(0)

# test reset:

obs, state = env.reset(key)
print("obs shape:", obs.shape)
print("river_level", state.river_level)
print("energy:", state.energy)
print("reputations:", state.reputations)

# now we test a single step with dummy actions: 

key, subkey = jax.random.split(key)
actions = jax.numpy.array([6 ,6 ,6]) # choosing stay for all 3 agents
obs, state, rewards, done, info = env.step_env(subkey, state, actions)
print("\nAfter one step:")
print("obs shape:", obs.shape)
print("river_level:", state.river_level)
print("energy:", state.energy)
print("rewards:", rewards)
print("done:", done)
print("info:", info)