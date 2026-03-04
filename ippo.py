""" 
Based on PureJaxRL & jaxmarl Implementation of PPO
"""
import sys
sys.path.append('clean_up')

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

from clean_up.wrappers import LogWrapper
from clean_up.clean_up import Clean_up

CONFIG = {
    "SEED": 42,
    "NUM_SEEDS": 1,
    "LR": 0.0003,
    "NUM_ENVS": 64, # NOTE: change this to 128 or 256 on TPU...this is the number of parallel environments
    "NUM_STEPS": 1000, # rollout horizon - i.e. num of env steps before performing policy update
    "TOTAL_TIMESTEPS": 1e8, # total number of time steps before training ends
    "UPDATE_EPOCHS": 2, # number of times we iterate over the rollot data before collecting new data 
    "NUM_MINIBATCHES": 500, # splitting the data into NUM_MINIBATCHES 
    "GAMMA": 0.99,
    "GAE_LAMBDA": 0.95, # used within the Generalised Advantage Estimator for A_t
    "CLIP_EPS": 0.2, # epsilon in the clip objective function...interpretation is that policy is not allowed to change by more than 20% per update step...remember that if we have one bad gradient step then the policy collapses.
    "ENT_COEF": 0.01, # entropy coef
    "VF_COEF": 0.5,
    "MAX_GRAD_NORM": 0.5, # prevents exploding gradients
    "ACTIVATION": "relu",
    "ENV_NAME": "clean_up",
    "ENV_KWARGS": { #KWARGS stands for keywords as arguments, which allow us to unpack the dictionary as parameters within our clean_up function later on
        "num_agents" : 3,
        "num_inner_steps" : 1000, # num of steps before environment resets
        "reward_type" : "fractional",  # NOTE: "shared", "individual", or "saturating"
        "cnn" : True,
        "jit" : True,
        "agent_ids" : True,  # NOTE: switch to True to enable agent ID channels in observations...(each agent can distringuish itself from other)
    },
    "ANNEAL_LR": False, # NOTE: switch to True if you want learning rate to decay linearly over training
    "GIF_NUM_FRAMES": 250, # length of evaluation for GIF evaluation...this is only for visualising post-training
    # WandB Params
    "ENTITY": "",
    "PROJECT": "socialjax",
    "WANDB_MODE" : "online", # NOTE: for dev, set to offline
    "WANDB_TAGS": ["3-agents", "individual_reward", "small-map"],
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


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def get_rollout(params, config):
    env = Clean_up(**config["ENV_KWARGS"])

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


def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[:, a] for a in agent_list])
    return x.reshape((num_actors, -1))

def batchify_dict(x: dict, agent_list, num_actors):
    x = jnp.stack([x[str(a)] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


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
    
    # Handle edge cases
    total = jnp.sum(sorted_values)
    
    # If all values are zero, return 0 (perfect equality)
    gini = jax.lax.cond(
        total == 0,
        lambda: 0.0,
        lambda: (2 * jnp.sum(jnp.arange(1, n + 1) * sorted_values) / (n * total) - (n + 1) / n)
    )
    
    return gini


def make_train(config):
    env = Clean_up(**config["ENV_KWARGS"])

    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]

    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )

    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    env = LogWrapper(env, replace_info=False)


    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):

        # INIT NETWORK
        network = ActorCritic(env.action_space().n, activation=config["ACTIVATION"])

        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros((1, *(env.observation_space()[0]).shape))

        network_params = network.init(_rng, init_x)

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

        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, update_step, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                
                obs_batch = jnp.transpose(last_obs,(1,0,2,3,4)).reshape(-1, *(env.observation_space()[0]).shape)
                
                pi, value = network.apply(train_state.params, obs_batch)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                env_act = unbatchify(
                    action, env.agents, config["NUM_ENVS"], env.num_agents
                )

                env_act = [v for v in env_act.values()]
                
                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])

                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(rng_step, env_state, env_act)
                
                info = jax.tree_map(lambda x: x.reshape((config["NUM_ACTORS"])), info)
                transition = Transition(
                    batchify_dict(done, env.agents, config["NUM_ACTORS"]).squeeze(),
                    action,
                    value,
                    batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
                    log_prob,
                    obs_batch,
                    info,
                    )

                runner_state = (train_state, env_state, obsv, update_step, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, update_step, rng = runner_state

            last_obs_batch = jnp.transpose(last_obs,(1,0,2,3,4)).reshape(-1, *(env.observation_space()[0]).shape)
            _, last_val = network.apply(train_state.params, last_obs_batch)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )

                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae
                
                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused, i):
                def _update_minbatch(train_state, batch_info, network_used):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets, network_used):
                        # RERUN NETWORK
                        pi, value = network_used.apply(params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)
                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)


                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                            train_state.params, traj_batch, advantages, targets, network_used
                        )
                    train_state = train_state.apply_gradients(grads=grads)
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

                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )

                train_state, total_loss = jax.lax.scan(
                    lambda state, batch_info: _update_minbatch(state, batch_info, network), train_state, minibatches
                )

                update_state = (train_state, traj_batch, advantages, targets, rng)

                return update_state, total_loss
            
            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                lambda state, unused: _update_epoch(state, unused, 0), update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            metric = traj_batch.info
            rng = update_state[-1]

                
            def callback(metric):
                wandb.log(metric)

            update_step = update_step + 1

            # Extract per-agent apple data before averaging
            if "episode_apples" in metric:
                # Reshape apples to have agent dim at the end
                episode_apples = metric["episode_apples"].reshape((config["NUM_STEPS"], config["NUM_ENVS"], config["ENV_KWARGS"]["num_agents"]))
                
                # Add individual agent apple metrics
                for agent_id in range(config["ENV_KWARGS"]["num_agents"]):
                    metric[f"agent_{agent_id}_apples"] = episode_apples[...,agent_id].mean()
                
                # Calculate total episode apples across all agents
                metric["total_episode_apples"] = episode_apples.sum(axis=-1).mean()
                
                # Calculate Gini coefficient for apple distribution among agents
                # Take the mean across environments for each agent, then compute Gini across agents
                agent_apple_means = jnp.array([episode_apples[...,agent_id].mean() for agent_id in range(config["ENV_KWARGS"]["num_agents"])])
                metric["apple_gini_coefficient"] = gini_coefficient(agent_apple_means)
                
                # Remove the original episode_apples from metrics
                del metric["episode_apples"]
            
            # Remove cumulative_apples_collected if it exists
            if "cumulative_apples_collected" in metric:
                del metric["cumulative_apples_collected"]
            
            metric = jax.tree_map(lambda x: x.mean(), metric)

            metric["update_step"] = update_step
            metric["env_step"] = update_step * config["NUM_STEPS"] * config["NUM_ENVS"]

            metric["update_step"] = update_step
            metric["env_step"] = update_step * config["NUM_STEPS"] * config["NUM_ENVS"]
            metric["clean_action_info"] = metric["clean_action_info"] * config["ENV_KWARGS"]["num_inner_steps"]

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

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=config["WANDB_TAGS"],
        config=config,
        mode=config["WANDB_MODE"],
        name=f'ippo_cnn_cleanup',
    )

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    train_jit = jax.jit(make_train(config))

    # with jax.profiler.trace("profiling/jax-trace"):
    out = jax.vmap(train_jit)(rngs)

    print("** Saving Results **")
    filename = f'{config["ENV_NAME"]}_seed{config["SEED"]}'
    train_state = jax.tree_map(lambda x: x[0], out["runner_state"][0])

    save_path = f"./checkpoints/indvidual/{filename}.pkl"
    save_params(train_state, save_path)
    params = load_params(save_path)

    evaluate(params, Clean_up(**config["ENV_KWARGS"]), save_path, config)

    print("** Evaluation Complete **")

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
        network = ActorCritic(action_dim=env.action_space().n, activation="relu")  # 使用与训练时相同的参数
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