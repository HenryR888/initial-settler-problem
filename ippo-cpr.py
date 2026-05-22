"""IPPO training for the simplified CPR environment.

Feedforward MLP policy — observation is fully observable and Markovian,
so recurrent memory is not needed.
Each update step = one full episode (episode_length == rollout_length).
"""
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import NamedTuple, Any
from flax.training.train_state import TrainState
import distrax
import wandb
import pickle
import os
import argparse

from isp_simplified.isp_simplified import CPR

CONFIG = {
    "SEED": 42, # use seed 42, 
    "NUM_SEEDS": 1,
    "LR": 3e-4,
    "NUM_ENVS": 64, # NOTE: change this to 128 or 256 on TPU...this is the number of parallel environments
    "NUM_STEPS": 500, # rollout horizon - i.e. num of env steps before performing PPO update
    "TOTAL_TIMESTEPS": 1e8, # total number of time steps before training ends (!NOTE DEMO RUN IS 5e6 timesteps...default is 1e8)
    "UPDATE_EPOCHS": 4, # number of times we iterate over the rollot data before collecting new data 
    "NUM_MINIBATCHES": 8, # NUM_ACTORS (192) must be divisible by this
    "GAMMA": 0.99,
    "GAE_LAMBDA": 0.95, # used within the Generalised Advantage Estimator for A_t
    "CLIP_EPS": 0.2,  # epsilon in the clip objective function...interpretation is that policy is not allowed to change by more than 20% per update step...remember that if we have one bad gradient step then the policy collapses.
    "ENT_COEF": 0.01, # entropy coef for total loss to encourage agent exploration
    "VF_COEF": 0.5, # value loss coefficient for total loss equation
    "MAX_GRAD_NORM": 0.5,
    "ANNEAL_LR": False,
    "ENV_KWARGS": {
        "num_agents": 3,
        "num_patches": 3,
        "timeout_duration":0,# we set this to 0 to disable punishment
    },
    "ENTITY": "henryrochester8-university-of-cape-town",
    "PROJECT": "isp-simplified",
    "WANDB_MODE": "online",
    "WANDB_TAGS": ["3-agents", "ippo-mlp", "cpr", "no-punishment"],
}


class ActorCriticMLP(nn.Module):
    num_actions: int

    @nn.compact
    def __call__(self, obs):
        # obs: (batch, obs_size)
        x = nn.relu(nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(obs))
        x = nn.relu(nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x))

        actor_hidden = nn.relu(nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x))
        pi = distrax.Categorical(
            logits=nn.Dense(self.num_actions, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_hidden)
        )

        critic_hidden = nn.relu(nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x))
        value = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic_hidden)

        return pi, jnp.squeeze(value, axis=-1)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: Any


def make_train(config):
    env = CPR(**config["ENV_KWARGS"])
    num_agents = env.num_agents
    num_patches = env.num_patches
    noop_action = num_patches + num_agents
    obs_size = env.obs_size

    config["NUM_ACTORS"] = num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = int(config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"])
    config["MINIBATCH_SIZE"] = config["NUM_ACTORS"] // config["NUM_MINIBATCHES"]

    assert config["NUM_ACTORS"] % config["NUM_MINIBATCHES"] == 0, (
        f"NUM_ACTORS ({config['NUM_ACTORS']}) must be divisible by NUM_MINIBATCHES ({config['NUM_MINIBATCHES']})"
    )

    def linear_schedule(count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
        return config["LR"] * frac

    def train(rng):
        network = ActorCriticMLP(num_actions=env.num_actions)

        rng, _rng = jax.random.split(rng)
        network_params = network.init(_rng, jnp.zeros((1, obs_size)))

        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(linear_schedule if config["ANNEAL_LR"] else config["LR"], eps=1e-5),
        )
        train_state = TrainState.create(apply_fn=network.apply, params=network_params, tx=tx)

        # Dummy env state for initial runner_state shape (overwritten on first update step): 
        rng, _rng = jax.random.split(rng)
        obsv, env_state = jax.vmap(env.reset)(jax.random.split(_rng, config["NUM_ENVS"]))

        def _update_step(runner_state, unused):
            train_state, _, _, update_step, rng = runner_state

            # Reset envs at the start of each episode:
            rng, _rng = jax.random.split(rng)
            last_obs, env_state = jax.vmap(env.reset)(jax.random.split(_rng, config["NUM_ENVS"]))
            # last_obs: (NUM_ENVS, num_agents, obs_size)

            def _env_step(carry, unused):
                train_state, env_state, last_obs, rng = carry
                rng, _rng = jax.random.split(rng)

                obs_batch = last_obs.reshape(config["NUM_ACTORS"], obs_size)
                pi, value = network.apply(train_state.params, obs_batch)
                action = pi.sample(seed=_rng)      # (NUM_ACTORS,)
                log_prob = pi.log_prob(action)     # (NUM_ACTORS,)

                env_act = action.reshape(config["NUM_ENVS"], num_agents)
                rng, _rng = jax.random.split(rng)
                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0))(
                    jax.random.split(_rng, config["NUM_ENVS"]), env_state, env_act
                )
                # reward: (NUM_ENVS, num_agents)  done: (NUM_ENVS,)

                done_flat = jnp.tile(done[:, None], (1, num_agents)).reshape(-1)  # (NUM_ACTORS,)

                transition = Transition(
                    done=done_flat,
                    action=action,
                    value=value,
                    reward=reward.reshape(-1),
                    log_prob=log_prob,
                    obs=obs_batch,
                    info=info,
                )
                return (train_state, env_state, obsv, rng), transition

            (train_state, env_state, last_obs, rng), traj_batch = jax.lax.scan(
                _env_step, (train_state, env_state, last_obs, rng), None, config["NUM_STEPS"]
            )
            # traj_batch fields have leading dim NUM_STEPS
            # traj_batch.info["patch_levels"]: (NUM_STEPS, NUM_ENVS, num_patches)
            # traj_batch.info["timeout"]: (NUM_STEPS, NUM_ENVS, num_agents)

            # GAE: 
            _, last_val = network.apply(train_state.params, last_obs.reshape(config["NUM_ACTORS"], obs_size))

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(carry, transition):
                    gae, next_value = carry
                    delta = transition.reward + config["GAMMA"] * next_value * (1 - transition.done) - transition.value
                    gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - transition.done) * gae
                    return (gae, transition.value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages, (jnp.zeros_like(last_val), last_val), traj_batch, reverse=True, unroll=16
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # PPO update: 
            def _update_epoch(update_state, unused):
                def _update_minibatch(train_state, batch_info):
                    traj_mb, adv_mb, tgt_mb = batch_info
                    # traj_mb.obs: (NUM_STEPS, mb_size, obs_size)
                    # traj_mb.action: (NUM_STEPS, mb_size)
                    # adv_mb, tgt_mb: (NUM_STEPS, mb_size)

                    def _loss_fn(params, traj_mb, adv_mb, tgt_mb):
                        T, B = config["NUM_STEPS"], config["MINIBATCH_SIZE"]
                        pi, value_seq = network.apply(params, traj_mb.obs.reshape(T * B, obs_size))
                        value_seq = value_seq.reshape(T, B)
                        log_prob = pi.log_prob(traj_mb.action.reshape(T * B)).reshape(T, B)
                        entropy = pi.entropy().reshape(T, B).mean()

                        value_pred_clipped = traj_mb.value + (value_seq - traj_mb.value).clip(
                            -config["CLIP_EPS"], config["CLIP_EPS"]
                        )
                        value_loss = 0.5 * jnp.maximum(
                            jnp.square(value_seq - tgt_mb),
                            jnp.square(value_pred_clipped - tgt_mb),
                        ).mean()

                        ratio = jnp.exp(log_prob - traj_mb.log_prob)
                        gae = (adv_mb - adv_mb.mean()) / (adv_mb.std() + 1e-8)
                        loss_actor = -jnp.minimum(
                            ratio * gae,
                            jnp.clip(ratio, 1 - config["CLIP_EPS"], 1 + config["CLIP_EPS"]) * gae,
                        ).mean()

                        total_loss = loss_actor + config["VF_COEF"] * value_loss - config["ENT_COEF"] * entropy
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    (total_loss, _), grads = grad_fn(train_state.params, traj_mb, adv_mb, tgt_mb)
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)

                permutation = jax.random.permutation(_rng, config["NUM_ACTORS"])
                fields = (traj_batch.done, traj_batch.action, traj_batch.value,
                          traj_batch.reward, traj_batch.log_prob, traj_batch.obs)
                shuffled = jax.tree_util.tree_map(lambda x: jnp.take(x, permutation, axis=1), fields)
                shuffled_adv = jnp.take(advantages, permutation, axis=1)
                shuffled_tgt = jnp.take(targets, permutation, axis=1)

                mb_size = config["MINIBATCH_SIZE"]

                def split_seqs(x):
                    # (NUM_STEPS, NUM_ACTORS, ...) -> (NUM_MINIBATCHES, NUM_STEPS, mb_size, ...)
                    return x.reshape(x.shape[0], config["NUM_MINIBATCHES"], mb_size, *x.shape[2:]).swapaxes(0, 1)

                (done_mb, action_mb, value_mb, reward_mb, log_prob_mb, obs_mb) = (
                    jax.tree_util.tree_map(split_seqs, shuffled)
                )
                adv_mb = shuffled_adv.reshape(config["NUM_STEPS"], config["NUM_MINIBATCHES"], mb_size).swapaxes(0, 1)
                tgt_mb = shuffled_tgt.reshape(config["NUM_STEPS"], config["NUM_MINIBATCHES"], mb_size).swapaxes(0, 1)

                traj_mbs = Transition(
                    done=done_mb, action=action_mb, value=value_mb, reward=reward_mb,
                    log_prob=log_prob_mb, obs=obs_mb,
                    info=jnp.zeros(config["NUM_MINIBATCHES"]),  # unused in loss
                )

                train_state, total_loss = jax.lax.scan(
                    lambda state, batch: _update_minibatch(state, batch),
                    train_state, (traj_mbs, adv_mb, tgt_mb),
                )

                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                lambda state, _: _update_epoch(state, _),
                update_state, None, config["UPDATE_EPOCHS"],
            )
            train_state = update_state[0]
            rng = update_state[-1]

            # Metrics: 
            patch_levels = traj_batch.info["patch_levels"] #(NUM_STEPS, NUM_ENVS, num_patches)
            timeout_arr = traj_batch.info["timeout"] #(NUM_STEPS, NUM_ENVS, num_agents)
            action = traj_batch.action  #(NUM_STEPS, NUM_ACTORS)
            reward = traj_batch.reward #(NUM_STEPS, NUM_ACTORS)

            metric = {
                "mean_patch_level": patch_levels.mean(),
                "patch_survival_rate": (patch_levels > 0).astype(jnp.float32).mean(),
                "min_patch_level_mean": patch_levels.min(axis=-1).mean(), # mean of per-step min patch
                "mean_reward": reward.mean(),
                "episode_total_reward": reward.sum(axis=0).mean(),
                "timeout_rate": (timeout_arr > 0).astype(jnp.float32).mean(),
                "harvest_rate": (action < num_patches).astype(jnp.float32).mean(),
                "punish_rate": ((action >= num_patches) & (action < noop_action)).astype(jnp.float32).mean(),
                "noop_rate": (action == noop_action).astype(jnp.float32).mean(),
                "mean_value": traj_batch.value.mean(),
                "update_step": update_step,
            }

            jax.debug.callback(lambda m: wandb.log(m), metric)

            update_step = update_step + 1
            runner_state = (train_state, env_state, last_obs, update_step, rng)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, 0, _rng)
        runner_state, metric = jax.lax.scan(_update_step, runner_state, None, config["NUM_UPDATES"])
        return {"runner_state": runner_state, "metrics": metric}

    return train


def save_params(train_state, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    params = jax.tree_util.tree_map(lambda x: np.array(x), train_state.params)
    with open(save_path, "wb") as f:
        pickle.dump(params, f)


def single_run(config):
    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=config["WANDB_TAGS"],
        config=config,
        mode=config["WANDB_MODE"],
        name=f"ippo_cpr_seed{config['SEED']}",
    )

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    train_jit = jax.jit(make_train(config))
    out = jax.vmap(train_jit)(rngs)

    train_state = jax.tree_map(lambda x: x[0], out["runner_state"][0])
    save_params(train_state, f"./checkpoints/cpr/seed{config['SEED']}.pkl")

    # Save metrics for offline plotting (aggregated across seeds via vmap)
    metrics_np = jax.tree_util.tree_map(lambda x: np.array(x[0]), out["metrics"])
    tag = "punishment" if config["ENV_KWARGS"]["timeout_duration"] > 0 else "no-punishment"
    results_dir = "./results/cpr"
    os.makedirs(results_dir, exist_ok=True)
    results_path = f"{results_dir}/seed{config['SEED']}_{tag}.pkl"
    with open(results_path, "wb") as f:
        pickle.dump({
            "metrics": metrics_np,
            "config": {k: v for k, v in config.items() if k != "ENV_KWARGS"},
        }, f)
    print(f"Metrics saved to {results_path}")

    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--wandb-mode", type=str, default=None, choices=["online", "offline", "disabled"])
    parser.add_argument("--punishment", action="store_true", help="Enable punishment (timeout_duration=5)")
    args = parser.parse_args()

    config = {**CONFIG, "ENV_KWARGS": {**CONFIG["ENV_KWARGS"]}}
    if args.seed is not None:
        config["SEED"] = args.seed
    if args.wandb_mode is not None:
        config["WANDB_MODE"] = args.wandb_mode
    if args.punishment:
        config["ENV_KWARGS"]["timeout_duration"] = 25
        config["WANDB_TAGS"] = [t for t in config["WANDB_TAGS"] if t != "no-punishment"] + ["punishment"]

    single_run(config)
