import jax
from jax import numpy as jnp
import optax
from Policy_v2 import ActorCritic

def masked_fill(mask, a, fill):
  return jax.lax.select(mask, a, jax.lax.broadcast(fill, a.shape))

def setup_experiment(
  env,
  seed,
  learning_rate,
  n_update_iterations,
  decay_rate
):
  n_devices = jax.device_count()
  rng, rng_reset, rng_params = jax.random.split(jax.random.PRNGKey(seed), 3)

  _, timestep = env.reset(rng_reset)
  grid = jnp.expand_dims(timestep.observation.grid, 0)
  n_actions = env.action_spec.num_values

  network = ActorCritic(int(n_actions))
  params = network.init(rng_params, grid)

  # scheduler = optax.exponential_decay(learning_rate, n_update_iterations, decay_rate)
  # optim = optax.chain(
  #   optax.adam(scheduler),
  #   optax.clip_by_global_norm(1.0)
  # )
  optim = optax.adam(learning_rate)
  opt_state = optim.init(params)

  return n_devices, network.apply, params, optim, opt_state, rng

def get_rng_keys(n_devices, n_envs, rng):
  rng, *rngs_pv = jax.random.split(rng, n_devices * n_envs + 1)
  stacked_rngs = jnp.stack(rngs_pv)
  return rng, stacked_rngs.reshape((n_devices, n_envs) + (stacked_rngs.shape[1],))


def broadcast_to_pv_shape(n_devices, n_envs, params, opt_state, rng):
  broadcast = lambda x: jnp.broadcast_to(x, (n_devices, n_envs) + x.shape)
  params = jax.tree.map(broadcast, params)
  opt_state = jax.tree.map(broadcast, opt_state)

  rng, rngs_pv = get_rng_keys(n_devices, n_envs, rng)
  return params, opt_state, rngs_pv, rng
