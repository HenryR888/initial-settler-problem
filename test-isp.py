import jax
import jax.numpy as jnp
from isp import ISP

# we instantiate with jit = False so errors are readable:
env = ISP(num_agents=3, jit=False)
key = jax.random.PRNGKey(0)

# test reset:

obs, state = env.reset(key)
print("obs shape:", obs.shape) # should expect (3,11,11,num_class+2+3+12)
print("river_level", state.river_level)
print("energy:", state.energy)
print("reputations:", state.reputations)
print("last_comms:", state.last_comms) # expect (3,4) zeros


# action set array is as follows: 
# INDEX 0: env_action: possible values are 0-11: 0=turn_left, 1=turn_right, 2=left, 3=right,4=up, 5=down, 6=stay, 7=harvest, 8=invest, 9=punish(0) (i.e. punish agent whose index is 0...and then if the agent's own index is 0, say, then this will be handled in isp.py, since agent cannot punish itself), 10 = punish(1), 11=punish(2)
# INDEX 1: claim_level: possible values are 0-4: 0 = very low [0,0.2); 1 = low [0.2,0.4); 2 = medium [0.4,0.6); 3 = high [0.6,0.8); 4 = very high [0.8,1]
# INDEX 2: accuse_target: 0-3: 0=nobody, 1=agent_0, 2=agent_1, 3=agent_2
# INDEX 3: charge: 0-3: 0=no charge; 1=greedy; 2=freeloading; 3=lying
# INDEX 4: recommend: 0-3: 0=no recommendation' 1= harvest less; 2 = harvest more; 3= invest more 

# now we test a single step, with all agents staying and claiming river level low: 
key, subkey = jax.random.split(key)
actions = jax.numpy.array([
    [6,0,0,0,0],
    [6,0,0,0,0],
    [6,0,0,0,0],
])
obs, state, rewards, done, info = env.step_env(subkey, state, actions)
print("\nAfter one step:")
print("obs shape:", obs.shape)
print("river_level:", state.river_level)
print("energy:", state.energy)
print("reputations:", state.reputations)
print("last_comms:", state.last_comms)
print("rewards", rewards)

# step with harvest action, claim very high river level and no accusation, no charge and no recommendation: 
key, subkey = jax.random.split(key)
actions = jnp.array([
    [7,4,0,0,0], 
    [7,4,0,0,0],
    [7,4,0,0,0],
])
obs, state, rewards, done, info = env.step_env(subkey, state, actions)
print("\nAfter harvest step:")
print("energy:", state.energy)
print("cumulative_harvest:", state.cumulative_harvest)
print("reputations:", state.reputations)
print("last_comms:", state.last_comms)
print("rewards", rewards)

# step with one of the agents accusing agent 1 of greed and recommending to harvest less: 
key, subkey = jax.random.split(key)
actions = jnp.array([
    [6, 2, 2, 1, 1], 
    [6, 2, 0, 0, 0],
    [6, 2, 0, 0, 0],
])
obs, state, rewards, done, info = env.step_env(subkey, state, actions)
print("\nAfter accusation step (agent 0 accuses agent 1 of greed):")
print("reputations:", state.reputations) # agent 1 rep should change
print("last_comms:", state.last_comms)
print("rewards", rewards)

# self accusation guard test: agent 0 accuses itself, which should be ignored: 

key, subkey = jax.random.split(key)
actions = jnp.array([
    [6, 2, 1, 1, 0],
    [6, 2, 0, 0, 0],
    [6, 2, 0, 0, 0],
])
rep_before = state.reputations
obs, state, rewards, done, info = env.step_env(subkey, state, actions)
print("\nAfter self-accusation step (should be ignored):")
print("rep before:", rep_before)
print("rep after: ", state.reputations) # agent 0 reputation should be unchanged
print("rewards", rewards)

# force a greedy state for agent 1 and test the accusation is justified: 
# we expect that agent 1 reputation should drop, and agent 0 reputation should remain unchanged
state_greedy = state.replace(
    cumulative_harvest = jnp.array([0.,10.,0.]),
    cumulative_invest = jnp.array([0., 0., 0.]),
)
actions = jnp.array([
    [6,2,2,1,0],
    [6,2,0,0,0],
    [6,2,0,0,0],
])
_, state_greedy, _,_,_ = env.step_env(subkey, state_greedy, actions)
print("reputations after justified accusation:", state_greedy.reputations)
print("rewards", rewards)


# check that the energy value cna go negative: 
state_low = state.replace(energy=jnp.array([0.02, 0.02, 0.02]))
obs, state_low, rewards, done, info = env.step_env(subkey, state_low, jnp.array([[6,0,0,0,0]]*3))
print("energy after NOOP from near-zero:", state_low.energy)  # should be -0.03, not clipped to 0


# verify that tile_richness is initialised correctly: 

print("tile_richness:", state.tile_richness)
print("num river tiles:", len(env.RIVER))

# check to see that richness is within the observation channel: 
print("obs shape:", obs.shape) # should be 23 not 22, after adding the extra channel

# check that the reward scales with the richness of the river: 

river_loc = jnp.array([env.RIVER[0, 0], env.RIVER[0, 1], 0], dtype=jnp.int16)                                                                                                            
state_on_river = state.replace(                                                                                                                                                          
    agent_locs=state.agent_locs.at[0].set(river_loc)                                                                                                                                     
)
actions = jnp.array([[7,0,0,0,0], [6,0,0,0,0], [6,0,0,0,0]])                                                                                                                             
obs, state_on_river, rewards, done, info = env.step_env(subkey, state_on_river, actions)
print("rewards:", rewards)                                                                                            
print("tile_richness[0]:", state.tile_richness[0])  



# punish respawn test: 
# we place agent 0 on the river, and agent 1 punishes agent...expect agent 0 to spawn on spawn tile

river_loc = jnp.array([env.RIVER[0,0], env.RIVER[0,1], 0], dtype=jnp.int16)
state_test = state.replace(agent_locs=state.agent_locs.at[0].set(river_loc))
actions = jnp.array([[6,0,0,0,0], [9,0,0,0,0], [6,0,0,0,0]])
obs, state_test, rewards, done, info = env.step_env(subkey, state_test, actions)
print("agent 0 loc after being punished:", state_test.agent_locs[0])


# starvation respawn test: 
# we set the agent's energy near to 0, and have him do a NoOp, then drain the energy of agent to below 0 , and should have him respawn with fresh 0.5 energy

state_test = state.replace(energy=jnp.array([0.03, 0.5, 0.5]))
actions = jnp.array([[6,0,0,0,0]]*3)
obs, state_test, rewards, done, info = env.step_env(subkey, state_test, actions)
print("agent 0 energy after starving:", state_test.energy[0])
print("agent 0 loc after starving:", state_test.agent_locs[0])


# guilty respawn test: 
# we force agent 1 into a greedy state and accuse, then should expect him to respawn. 

state_test = state.replace(
    cumulative_harvest=jnp.array([0., 20., 0.]),
    cumulative_invest=jnp.array([0., 0., 0.]),
)
actions = jnp.array([[6,0,2,1,0], [6,0,0,0,0], [6,0,0,0,0]])
obs, state_test, rewards, done, info = env.step_env(subkey, state_test, actions)
print("agent 1 loc after guilty verdict:", state_test.agent_locs[1])
