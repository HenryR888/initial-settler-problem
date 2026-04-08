""" 
Based on PureJaxRL & jaxmarl Implementation of PPO
"""
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any
from flax.training.train_state import TrainState
import distrax
import wandb
import pickle
import os
from PIL import Image
from pathlib import Path
import argparse

from isp.wrappers import LogWrapper
from isp.isp import ISP

CONFIG = {
    "SEED": 42,
    "NUM_SEEDS": 1,
    "LR": 0.0003,
    "NUM_ENVS": 64, # NOTE: change this to 128 or 256 on TPU...this is the number of parallel environments
    "NUM_STEPS": 500, # rollout horizon - i.e. num of env steps before performing policy update
    "TOTAL_TIMESTEPS": 1e8, # total number of time steps before training ends
    "UPDATE_EPOCHS": 2, # number of times we iterate over the rollot data before collecting new data 
    "NUM_MINIBATCHES": 500, # splitting the data into NUM_MINIBATCHES 
    "GAMMA": 0.99,
    "GAE_LAMBDA": 0.95, # used within the Generalised Advantage Estimator for A_t
    "CLIP_EPS": 0.2, # epsilon in the clip objective function...interpretation is that policy is not allowed to change by more than 20% per update step...remember that if we have one bad gradient step then the policy collapses.
    "ENT_COEF": 0.01, # entropy coef for total loss to encourage agent exploration
    "VF_COEF": 0.5, # value loss coefficient for total loss equation
    "MAX_GRAD_NORM": 0.5, # prevents exploding gradients
    "ACTIVATION": "relu",
    "ENV_NAME": "isp",
    "ENV_KWARGS": { #KWARGS stands for keywords as arguments, which allow us to unpack the dictionary as parameters within our clean_up function later on
        "num_agents" : 3,
        "num_inner_steps" : 500, # num of steps before environment resets
    },
    "ANNEAL_LR": False, # NOTE: switch to True if you want learning rate to decay linearly over training
    "GIF_NUM_FRAMES": 250, # length of evaluation for GIF evaluation...this is only for visualising post-training
    # WandB Params
    "ENTITY": "",
    "PROJECT": "isp",
    "WANDB_MODE" : "online", # NOTE: for dev, set to offline
    "WANDB_TAGS": ["3-agents", "ippo-cnn"],
}

class CNN(nn.Module):
    activation: str = "relu"
    dtype: Any = jnp.bfloat16 # reduced precision but much faster

    @nn.compact
    def __call__(self, x):
        x = x.astype(self.dtype) # cast input to bfloat16 for speed
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        x = nn.Conv(
            features=32,
            kernel_size=(5, 5), # apply 32 different 5x5 filters to grid...i.e. learn the spatial patterns in a 5x5 neighbourhood
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            dtype=self.dtype,
            param_dtype=jnp.float32,
        )(x)
        x = activation(x) # add the non-linearity so the network can represent more complicated function
        x = nn.Conv(
            features=32,
            kernel_size=(3, 3), # now apply another set of 32 different 3x3 filters to grid
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            dtype=self.dtype,
            param_dtype=jnp.float32,
        )(x)
        x = activation(x) # apply activation after each convolution
        x = nn.Conv(
            features=32,
            kernel_size=(3, 3), # apply one more time, 32 different 3x3 filters to  grid
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            dtype=self.dtype,
            param_dtype=jnp.float32,
        )(x)
        x = activation(x)
        x = x.reshape((x.shape[0], -1))  # Flatten to get single feature vector per observation
        # The reason we choose to flatten here is because we can compress all the information we have about what exists and where into one compact state embedding,
        # which is often much simpler than keeping the spatial tensor and needing many more convolution layers within the net

        # we get a 64-dimensional vector of learned features from the grid that comes from initial input from our clean_up file. This 64 dimensional embedding then gets sent to policy/value head
        x = nn.Dense(
            features=64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0),
            dtype=self.dtype,
            param_dtype=jnp.float32,
        )(x)
        x = activation(x)

        return x


class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "relu"
    dtype: Any = jnp.bfloat16

    @nn.compact
    def __call__(self, x): # define the forward pass 
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        # this is the embedding from the output of our convolutional neural network, which will be receive by actor (policy) critic (value) head
        embedding = CNN(self.activation, dtype=self.dtype)(x)
        # because we have discrete actions here, the actor will output logits...the name mean, comes from a Gaussian policy which outputs continuous action spaces
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0), # we've got 64 neurons in this layer (since we have a 64-dim embedding)...note that each neuron will receive this entire embedding vector
            dtype=self.dtype, # that kernel_init actually comes from how we initialise the weights. the weights here satisfy W'W =I...this stabilises our training and reduces vanishing/exploding gradients. The reason being that orthogonal matrices allow for preservation of vector length.
            param_dtype=jnp.float32, #...cont from above, this means that ||W_x|| \approx ||x||
        )(embedding) # np.sqrt(2) allows for gain of the output...since half the values (h) become zero, the variance of activations...so, the signal drops quickly, thus we want to boost it to maintain some signal
        actor_mean = activation(actor_mean) # apply non-linearity
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0), # the small gain (0.01) is to keep the initial logit near 0 to prevent huge deviations in the policy training, initially
            dtype=self.dtype,
            param_dtype=jnp.float32,
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean.astype(jnp.float32)) # then use distrax to build out distribution from logits

        # value head...
        critic = nn.Dense( # same logic as actor hidden layer
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0), # notice that we initialise the bias at zero, because we want stable initialisation to just be Wx
            dtype=self.dtype,
            param_dtype=jnp.float32,
        )(embedding)
        critic = activation(critic)
        critic = nn.Dense( # output scalar value prediction per observation
            1, kernel_init=orthogonal(1.0), bias_init=constant(0.0),
            dtype=self.dtype,
            param_dtype=jnp.float32,
        )(critic)

        return pi, jnp.squeeze(critic.astype(jnp.float32), axis=-1)

# NamedTuple allows for python to call objects like t.action or t.state, instead of indexing like t[0], t[1]...
# This transition allows us to store all the information for a specific time step
class Transition(NamedTuple):
    done: jnp.ndarray # did the episode end at this step?
    action: jnp.ndarray # action the agent took
    value: jnp.ndarray # V(s_t)
    reward: jnp.ndarray # r_t
    log_prob: jnp.ndarray # taking log of policy (log_theta(pi(a_t |s_t)))
    obs: jnp.ndarray # observation taken by agent
    info: jnp.ndarray # some extra info that we can log later on


def get_rollout(params, config):
    '''
    Just getting rollout data, which is done after training for visualisation.
    '''
    env = ISP(**config["ENV_KWARGS"])

    network = ActorCritic(env.action_space().n, activation=config["ACTIVATION"])

    key = jax.random.PRNGKey(0)
    key, key_r, key_a = jax.random.split(key, 3)

    done = False

    obs, state = env.reset(key_r)
    state_seq = [state]
    for o in range(config["GIF_NUM_FRAMES"]):
        print(o)
        key, key_a0, key_a1, key_s = jax.random.split(key, 4)

        obs_batch = jnp.stack([obs[a] for a in env.agents]).reshape(-1, *env.observation_space()[0].shape)
        
        pi, value = network.apply(params, obs_batch)
        action = pi.sample(seed=key_a0)
        env_act = unbatchify(
            action, env.agents, 1, env.num_agents
        )           

        env_act = {k: v.squeeze() for k, v in env_act.items()}

        # STEP ENV
        obs, state, reward, done, info = env.step(key_s, state, env_act)
        done = done["__all__"]

        state_seq.append(state)

    return state_seq


def batchify(x: dict, agent_list, num_actors): # convert to format (batched tensor) that neural network expects
    x = jnp.stack([x[:, a] for a in agent_list])
    return x.reshape((num_actors, -1))

def batchify_dict(x: dict, agent_list, num_actors): # same thing as above, but this is done if the dictionary keys are strings
    x = jnp.stack([x[str(a)] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors): # Now we reverse the above by converting the batched array into the dictionary format, which is going to be what our step function in the environment expects
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}

def batchify_int_dict(x: dict, agent_list, num_actors):
    """
    Here we add a new helper function for dictionaries with integer keys, since ISP reward is a dictionary with int keys
    """
    x = jnp.stack([x[a] for a in agent_list]) # stack the dictionary values into a JAX array, and infer the dimension automatically
    return x.reshape((num_actors, -1))


def gini_coefficient(values):
    """
    Calculate the Gini coefficient for a given array of values.
    
    Args:
        values: JAX array of values for which to calculate the Gini coefficient
        
    Returns:
        Gini coefficient (0 = perfect equality, 1 = maximum inequality)
    """
    # Sort values in ascending order
    sorted_values = jnp.sort(values)
    n = len(sorted_values)
    
    # total reward from agents
    total = jnp.sum(sorted_values)
    
    # If all values are zero, return 0 (perfect equality)
    gini = jax.lax.cond(
        total == 0,
        lambda: 0.0, # handle the edge case
        lambda: (2 * jnp.sum(jnp.arange(1, n + 1) * sorted_values) / (n * total) - (n + 1) / n)
    )
    
    return gini


def make_train(config):
    '''
    This function amalgamates all functions needed to do the training of our net using PPO...first we instantiate everything according to our ENV KWARGS
    '''
    env = ISP(**config["ENV_KWARGS"])

    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"] # this is the total number of agent-trajectories...i.e. (number of agents per environment) x (number of environments being run in parallel)

    # total number of outer updates PPO does
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    # splitting up the number of steps throughout all environments in minibatches, and taking the size of those minibatches
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    env = LogWrapper(env, replace_info=False) # enable logging for statistics and tracking

   
    def linear_schedule(count):
        '''
        anneal the learning rate, linearly
        '''
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):
        '''
        Here we perform the entire training run.
        '''
        # INIT NETWORK
        network = ActorCritic(env.action_space().n, activation=config["ACTIVATION"])

        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros((1, *(env.observation_space()[0]).shape)) # here we create a dummy tensor in order to initialise our network in the next step.

        network_params = network.init(_rng, init_x) # initialising neural network with dummy tensor (batch, width, height, channels)...these are the tensors that we shall pass into our Neural Net which come from our CNN

        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )

        # this guy stores everything we need to train the model in one object: the params, how to run the net (apply_fn), the optimiser from above (tx)
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"]) # each parallel environment gets its own reset randomness by splitting the keys
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng) # reset all at once

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            '''
            This function corresponds to one outer PPO update. 
            1.) collect trajectories
            2.) compute the advantages
            3.) update the network with PPO
            '''
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, update_step, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                
                # SOME Qs: When exactly are we obtaining this observation? 
                obs_batch = jnp.transpose(last_obs,(1,0,2,3,4)).reshape(-1, *(env.observation_space()[0]).shape) # flattens environments and agent axes together, to obtain one batch of all actor observations to feed into the network
                
                # ! IMPORTANT: Here is where we apply the network...we give the network the current params of our CNN, and the observation batch of all agents from all environments that we flattened above, and feed forward through the CNN, then this returns the policy distribution (output) from our actor head
                # and then the value estimate for each actor observation (from the critic head)
                pi, value = network.apply(train_state.params, obs_batch)
                action = pi.sample(seed=_rng) # for each observation of each agent, we sample an action from the policy
                log_prob = pi.log_prob(action) # for the PPO alg, we need the log of that prob from the sampled action
                env_act = unbatchify(
                    action, env.agents, config["NUM_ENVS"], env.num_agents # unflattening the action tensor to fit what environment expects 
                )

                env_act = [v for v in env_act.values()] # convert the dictionary values in a list for env.step
                
                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])

                # this steps all the environments in parallel at the same time
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(rng_step, env_state, env_act)
                
                info = jax.tree_map(lambda x: x.reshape((config["NUM_ACTORS"])), info)
                transition = Transition( # as in the transition class defined earlier, this stores the transition object for this particular time step, which is what PPO needs later (log prob and value specifically)
                    batchify_int_dict(done, env.agents, config["NUM_ACTORS"]).squeeze(),
                    action,
                    value,
                    batchify_int_dict(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
                    log_prob,
                    obs_batch,
                    info,
                    )

                runner_state = (train_state, env_state, obsv, update_step, rng) # this updates the observation, and it is where the env_state gets updated
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"] # this repeats the _env_step function for "NUM_STEPS" times...so we have a full trajectory batch before updating 
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, update_step, rng = runner_state

            last_obs_batch = jnp.transpose(last_obs,(1,0,2,3,4)).reshape(-1, *(env.observation_space()[0]).shape)
            _, last_val = network.apply(train_state.params, last_obs_batch) # we need the final value of the next state to compute our advantage using GAE

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )

                    # Temporal Difference residual is computed: delta_t = r_t + gamma.V(s_{t+1})(1-d_t) - V(s_t)
                    # remember gae is calculating the advantage which will be used to update our policy
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = ( # delta_t + (gamma.lambda)(delta_t+1)
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae
                
                _, advantages = jax.lax.scan( # we obtain the advantages every time step and actor here
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value # this second parameter that is returned is the target (Q(s,a) = A_t + V(s_t))...which is what is used to train the critic network

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK for one epoch...remember we'll do multiple passes over the same rollout data
            def _update_epoch(update_state, unused, i):
                def _update_minbatch(train_state, batch_info, network_used):
                    '''
                    gradient descent for one minibatch
                    '''
                    traj_batch, advantages, targets = batch_info # unpack the mini batch data

                    def _loss_fn(params, traj_batch, gae, targets, network_used):
                        # RERUN NETWORK...we do this because above, within the transition object, we stored the old log prob, action and value...now with PPO, we want to compare our current action/value/log_prob to our old ones to updates params accordingly
                        pi, value = network_used.apply(params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)
                        # CALCULATE VALUE LOSS

                        # V_param^{clip}(s_t) = V_old(s_t) + clip(V_param(s_t)-V_old, -eps, eps)...where V_param is the value calculated with the current params 
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        # Now we compute two losses, Loss1 = (V_param - target)^2
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets) # the second loss computed is Loss_2 = (V_clipped - target)^2
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        ) # because we trained over a whole batch and we now have two vectors of losses, we compute the mean of the maximum values, to smooth out the variance and then the 0.5 actually simplifies the grad calc
                        # also note here that we take the maximum of the two losses in order to penalise the largest error

                        # CALCULATE ACTOR LOSS
                        # this is just pi_param/pi_old ratio
                        # Remember that we use log probs instead of prob for numerical stability...in our nets, sometimes the actual probs can be 0.000004 say and 0.00002 and dividing by these numbers, we could have floating point underflow, which results in us dividing by zero and exploding gradients happen...log probs fix this
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8) # normalise the advantage, which stabilises increase/decrease of params in update step
                        loss_actor1 = ratio * gae #pi_param/pi_old * A_t
                        loss_actor2 = ( # clipped part of the final loss function
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2) # take conservative loss which we want to  minimise, that is why we use minimum. Moreover, the neg sign is there because PPO actually seeks to maximise J(theta)...but here in the gradient-based method, we seek to minimise a loss function, which is why we take the negative of it
                        loss_actor = loss_actor.mean() # take the mean because we will have a vector of per sample losses.  
                        entropy = pi.entropy().mean() # we add this entropy bonus term here so that the agent continues to explore the environment, and does not settle on some local minimum too soon

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)


                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    # evaluate the current loss and grads wrt current params
                    total_loss, grads = grad_fn(
                            train_state.params, traj_batch, advantages, targets, network_used
                        )
                    train_state = train_state.apply_gradients(grads=grads) # apply optimiser so that params get updated
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = update_state 
                rng, _rng = jax.random.split(rng) 
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ACTORS"]
                ), "batch size must be equal to number of steps * number of actors"
                permutation = jax.random.permutation(_rng, batch_size) 
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(
                        lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                    )

                shuffled_batch = jax.tree_util.tree_map( # randomise the rollout data, as the rollout data that was collected is sequential, and we want to avoid correlation between transitions
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree_util.tree_map( # split the shuffled rollout data into minibatches
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )

                train_state, total_loss = jax.lax.scan( # run gradient update step through each of the minibatches, and then update network params
                    lambda state, batch_info: _update_minbatch(state, batch_info, network), train_state, minibatches
                )

                update_state = (train_state, traj_batch, advantages, targets, rng)

                return update_state, total_loss
            
            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                lambda state, unused: _update_epoch(state, unused, 0), update_state, None, config["UPDATE_EPOCHS"] # repeat the same update_epoch step for number of epochs
            )
            train_state = update_state[0]
            metric = traj_batch.info
            rng = update_state[-1]

                
            def callback(metric):
                wandb.log(metric)

            update_step = update_step + 1

            # Per-agent harvest and invest breakdown: 
            episode_harvest = metric["episode_harvest"] # shape is (NUM_STEPS, NUM_ENVS, num_agents)
            episode_invest = metric["episode_invest"]

            for agent_id in range(config["ENV_KWARGS"]["num_agents"]):
                metric[f"agent_{agent_id}_harvest"] = episode_harvest[..., agent_id].mean() # over all time steps, and environment, give me the per-agent average of the harvest...i.e. avg agent harvest per step per environment
                metric[f"agent_{agent_id}_invest"] = episode_invest[..., agent_id].mean() # similarly, here we have avg agent invest per step per environment

            metric["total_episode_harvest"] = episode_harvest.sum(axis=-1).mean() # avg harvest per step
            metric["total_episode_invest"] = episode_invest.sum(axis=-1).mean() # avg invest per step

            agent_harvest_means = jnp.array([episode_harvest[..., i].mean() for i in range(config["ENV_KWARGS"]["num_agents"])])
            metric["harvest_gini"] = gini_coefficient(agent_harvest_means) # check gini coeff to determine how equal is distribution of harvesting among agents

            metric = jax.tree_map(lambda x: x.mean(), metric) # then we have one clean dictionary of metrics to gather info about rollout data

            jax.debug.callback(callback, metric)

            runner_state = (train_state, env_state, last_obs, update_step, rng)

            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, 0, _rng)
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )

        return {"runner_state": runner_state, "metrics": metric}

    return train

def single_run(config):
    config = CONFIG
    # wandb stores metrics for every run
    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=config["WANDB_TAGS"],
        config=config,
        mode=config["WANDB_MODE"],
        name=f'ippo_cnn_isp',
    )

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    train_jit = jax.jit(make_train(config))

    # with jax.profiler.trace("profiling/jax-trace"):
    out = jax.vmap(train_jit)(rngs)

    print("** Saving Results **")
    filename = f'{config["ENV_NAME"]}_seed{config["SEED"]}'
    train_state = jax.tree_map(lambda x: x[0], out["runner_state"][0])

    # Save the training params
    save_path = f"./checkpoints/indvidual/{filename}.pkl"
    save_params(train_state, save_path)
    params = load_params(save_path)

    #evaluate(params, Clean_up(**config["ENV_KWARGS"]), save_path, config) # TODO: uncomment once render is implemented in ISP

    #print("** Evaluation Complete **")

    return True

def save_params(train_state, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    params = jax.tree_util.tree_map(lambda x: np.array(x), train_state.params)

    with open(save_path, 'wb') as f:
        pickle.dump(params, f)

def load_params(load_path):
    with open(load_path, 'rb') as f:
        params = pickle.load(f)
    return jax.tree_util.tree_map(lambda x: jnp.array(x), params)

def evaluate(params, env, save_path, config):
    rng = jax.random.PRNGKey(0)
    
    rng, _rng = jax.random.split(rng)
    obs, state = env.reset(_rng)
    done = False
    
    pics = []
    img = env.render(state)
    pics.append(img)
    root_dir = f"evaluation/cleanup"
    path = Path(root_dir + "/state_pics")
    path.mkdir(parents=True, exist_ok=True)

    for o_t in range(config["GIF_NUM_FRAMES"]):

        obs_batch = jnp.stack([obs[a] for a in env.agents]).reshape(-1, *env.observation_space()[0].shape)
        network = ActorCritic(action_dim=env.action_space().n, activation="relu")  # apply the trained params to the current obs
        pi, _ = network.apply(params, obs_batch)
        rng, _rng = jax.random.split(rng)
        actions = pi.sample(seed=_rng)
        env_act = {k: v.squeeze() for k, v in unbatchify(
            actions, env.agents, 1, env.num_agents
        ).items()}

        rng, _rng = jax.random.split(rng)
        obs, state, reward, done, info = env.step(_rng, state, [v.item() for v in env_act.values()])
        done = done["__all__"]
        
        img = env.render(state)
        pics.append(img)
        
        # print('###################')
        # print(f'Actions: {env_act}')
        # print(f'Reward: {reward}')
        # print(f'State: {state.agent_locs}')
        # print(f'State: {state.claimed_indicator_time_matrix}')
        # print("###################")
    
    # GIF
    print(f"Saving Episode GIF")
    pics = [Image.fromarray(np.array(img)) for img in pics]
    n_agents = len(env.agents)
    gif_path = f"{root_dir}/{n_agents}-agents_seed-{config['SEED']}_frames-{o_t + 1}.gif"
    pics[0].save(
        gif_path,
        format="GIF",
        save_all=True,
        optimize=False,
        append_images=pics[1:],
        duration=200,
        loop=0,
    )

    # Log the GIF to WandB
    print("Logging GIF to WandB")
    wandb.log({"Episode GIF": wandb.Video(gif_path, caption="Evaluation Episode", format="gif")})

def main(config):
    single_run(config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, help='Random seed to use (overrides CONFIG)')
    args = parser.parse_args()
    config = CONFIG.copy()
    if args.seed is not None:
        config["SEED"] = args.seed

    main(config)