import chex
import jax
from jax import numpy as jnp
import optax
import rlax
import dqn.utils as utils

@chex.dataclass(frozen=True)
class Transition:
  observation: chex.Array
  action: chex.Array
  discount: chex.Array
  reward: chex.Array

@chex.dataclass(frozen=True)
class Params:
  online: dict
  target: dict
  update_count: int


def get_learner_fn(
env,
rollout_len,
gamma,
buffer,
update_period,
n_iterations,
optim_update_len,
forward_fn,
opt_update_fn,
epsilon_schedule_fn):
  """Define minimal unit of computations in Anakin."""

  def rollout_fn(state, timestep, params_state, rng, rollout_len):
    """Collects data from trajectory in environment"""
    
    def step_fn(data, rng):

      def epsilon_greedy_sample(epsilon, q_values, action_mask, rng):
        uniform_var = jax.random.uniform(rng)
        true_oper = lambda x, y: jnp.argmax(utils.masked_fill(y, x, -jnp.inf))
        false_oper = lambda x, y: jax.random.choice(key=rng, a=len(x), shape=(), p=y / jnp.where(y, 1, 0).sum())
        return jax.lax.cond(epsilon < uniform_var, true_oper, false_oper, q_values, action_mask)

      state, timestep, params_state = data
      obs = timestep.observation.board
      action_mask = timestep.observation.action_mask
      q_values = forward_fn(params_state.online, jnp.expand_dims(obs, (0, 3))).squeeze(0)
      
      epsilon = epsilon_schedule_fn(params_state.update_count)
      action = epsilon_greedy_sample(epsilon, q_values, action_mask, rng)

      next_state, next_timestep = env.step(state, action)
      return (next_state, next_timestep, params_state), Transition(
        observation=obs, 
        action=action,
        discount=timestep.discount,
        reward=next_timestep.reward
        # reward=jax.lax.cond(action_mask[action], lambda x: x, lambda x: x -1e20, next_timestep.reward)
      )
      #rollout will be called with different params in differents learning moments, so it is necessary to return params to use jax.lax.scan
    
    rng_steps = jax.random.split(rng, rollout_len)
    (new_state, new_timestep, params_state), trajectory = jax.lax.scan(
      step_fn,
      (state, timestep, params_state),
      rng_steps
    )
    return new_state, new_timestep, trajectory

  def loss_fn(online_params, target_params, batch):
    """Defines q_learning loss loss on batch"""
    obs_tm1 = batch.first.observation # obs (B, 4, 4)
    a_tm1 = batch.first.action # a (B, )
    reward_t = batch.first.reward # r (B, )
    discount_t = batch.second.discount * gamma # (B, )
    obs_t = batch.second.observation #obs' (B, 4, 4)
    
    q_values_tm1 = forward_fn(online_params, jnp.expand_dims(obs_tm1, 3)) # (B, 4 = n_actions)
    q_values_t = forward_fn(target_params, jnp.expand_dims(obs_t, 3))
    q_selector_t = forward_fn(online_params, jnp.expand_dims(obs_t, 3))

    td_error = jax.vmap(rlax.double_q_learning)(
      q_tm1=q_values_tm1,
      a_tm1=a_tm1,
      r_t=reward_t,
      discount_t=discount_t,
      q_t_value=q_values_t,
      q_t_selector=q_selector_t) #vmap used for vectorizing across batch

    return jnp.mean(jnp.square(td_error))

  def step_and_update_fn(params_state, opt_state, buffer_state, state, timestep, rng):

    """Makes rollout, collects data and updates agent's params"""

    def optim_step(data, rng):

      """Defines a single parameters update"""

      params_state, opt_state, buffer_state = data

      batch = buffer.sample(buffer_state, rng).experience

      online_params, target_params = params_state.online, params_state.target

      grads = jax.grad(loss_fn)(online_params, target_params, batch) 
      # default value for argnums (which specifies the positional args to differentiate with respect to) is argnums=0, so no changes needed
      grads = jax.lax.pmean(grads, axis_name='i')  # reduce mean across cores.
      grads = jax.lax.pmean(grads, axis_name='j')  # reduce mean across batch.

      updates, opt_state = opt_update_fn(grads, opt_state)
      online_params = optax.apply_updates(online_params, updates)

      target_params = optax.periodic_update(online_params, target_params, params_state.update_count + 1, update_period)

      params_state = Params(
        online=online_params,
        target=target_params,
        update_count=params_state.update_count + 1
      )

      return (params_state, opt_state, buffer_state), None
    
    rng, rng_rollout, rng_update = jax.random.split(rng, 3)
    state, timestep, trajectory = rollout_fn(state, timestep, params_state, rng_rollout, rollout_len)

    buffer_state = buffer.add(buffer_state, trajectory)
    
    true_update_fn = lambda p_state, opt_state, b_state, rng: jax.lax.scan(
      optim_step, (p_state, opt_state, b_state), jax.random.split(rng, optim_update_len)
      )[0]
    
    false_update_fn = lambda p_state, opt_state, b_state, _ : (p_state, opt_state, b_state)
    
    params_state, opt_state, buffer_state = jax.lax.cond(
      buffer.can_sample(buffer_state),
      true_update_fn,
      false_update_fn,
      params_state,
      opt_state,
      buffer_state,
      rng_update
    )

    return params_state, opt_state, buffer_state, state, timestep, rng
  
  def learner_fn(params_state, opt_state, buffer_state, states, timesteps, rngs):
    batched_step_and_update_fn = jax.vmap(
      step_and_update_fn,
      axis_name='j'
    ) # vectorize across batch.

    def iterate_fn(_, val):
      params_state, opt_state, buffer_state, states, timesteps, rngs = val
      return batched_step_and_update_fn(params_state, opt_state, buffer_state, states, timesteps, rngs)
    
    return jax.lax.fori_loop(
      0,
      n_iterations,
      iterate_fn,
      (params_state, opt_state, buffer_state, states, timesteps, rngs)
    )
  return learner_fn







#Итого: в цикле по количествку итераций (for i in range(n_iterations)) идет вызов  step_and_update_fn;
#Внутри step_and_update вызывается rollout на rollout_len шагов и записывает собранные данные, затем вызывает optim_update_len обновлений
#параметров модели, оптимизатора и буффера

#то есть 