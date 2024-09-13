import jax
import jax.numpy as jnp
import flax
import optax
import flashbax as fbx
from dqn.dqn_v2 import DQNetworkDueling
from dqn.anakin import Transition, Params

def init_dqn_model(n_actions, example_obs, rng):
  network = DQNetworkDueling(n_actions)
  params = network.init(rng, example_obs)
  return network, params

def masked_fill(mask, a, fill):
  return jax.lax.select(mask, a, jax.lax.broadcast(fill, a.shape))


def setup_experiment(
  env,
  batch_size,
  learning_rate,
  seed,
  buffer_size,
  start_epsilon,
  end_epsilon,
  steps_epsilon
):


  """Sets up necessary objects: network, optimizer, buffer, epsilon scheduling"""
  n_devices = jax.device_count()
  optim = optax.adam(learning_rate)

  rng, rng_env, rng_params = jax.random.split(jax.random.PRNGKey(seed), 3)
  _, timestep = env.reset(rng_env)
  obs = timestep.observation.board

  n_actions = env.action_spec.num_values
  network = DQNetworkDueling(n_actions)
  params = network.init(rng_params, jnp.expand_dims(obs, (0, 3)))
  opt_state = optim.init(params)

  buffer = fbx.make_flat_buffer(
    max_length=buffer_size,
    min_length=batch_size,
    sample_batch_size=batch_size,
    add_sequences=True,
    add_batch_size=None
  )

  transition = Transition(
    observation=obs,
    action=jnp.zeros((), dtype=jnp.int32),
    discount=jnp.zeros(()),
    reward=jnp.zeros(())
  )

  buffer_state = buffer.init(transition)

  epsilon_schedule_fn = optax.linear_schedule(start_epsilon, end_epsilon, steps_epsilon)

  return n_devices, network, params, optim, opt_state, buffer, buffer_state, epsilon_schedule_fn, rng

def get_rng_keys(n_devices, n_envs, rng):
  rng, *rngs_pv = jax.random.split(rng, n_devices * n_envs + 1)
  stacked_rngs = jnp.stack(rngs_pv)
  return rng, stacked_rngs.reshape((n_devices, n_envs) + (stacked_rngs.shape[1],))

def broadcast_to_pv_shape(n_devices, n_envs, params, opt_state, buffer_state, rng):
  broadcast = lambda x: jnp.broadcast_to(x, (n_devices, n_envs) + x.shape)
  params = jax.tree.map(broadcast, params)
  opt_state = jax.tree.map(broadcast, opt_state)
  buffer_state = jax.tree.map(broadcast, buffer_state)

  params_state = Params(
    online=params,
    target=params,
    update_count=jnp.zeros((n_devices, n_envs))
  )

  rng, rngs_pv = get_rng_keys(n_devices, n_envs, rng)

  return params_state, opt_state, buffer_state, rngs_pv, rng






    
