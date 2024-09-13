import jax
from jax import numpy as jnp
import optax
import chex
import rlax


@chex.dataclass(frozen=True)
class Transition:
  obs: jax.Array
  action: int
  reward: int
  discount: float


def get_learner_fn(
  env,
  gamma,
  n_iterarions,
  coefs,
  forward_fn,
  opt_update_fn,
  rollout_len,
  lambda_gae
):
  def rollout_fn(params, state, timestep, rng):

    def step__fn(carry, rng):
      params, state, timestep = carry

      logits, _ = forward_fn(params, jnp.expand_dims(timestep.observation.grid, 0))

      rng, rng_sample = jax.random.split(rng, 2)
      action = jax.random.categorical(key=rng_sample, logits=logits.squeeze(0))
      next_state, next_timestep = env.step(state, action)

      transition = Transition(
        obs=timestep.observation.grid,
        action=action,
        reward=next_timestep.reward,
        discount=timestep.discount
      )

      return (params, next_state, next_timestep), transition
    
    rng, rng_rollout = jax.random.split(rng, 2)
    rngs_steps = jax.random.split(rng_rollout, rollout_len)
    (params, state, timestep), trajectory = jax.lax.scan(
      step__fn,
      (params, state, timestep),
      rngs_steps
    )
    
    return params, state, timestep, trajectory
  
  def loss_fn(params, trajectory):
    obs = trajectory.obs
    action = trajectory.action.reshape(-1, 1)[:-1]
    reward = trajectory.reward[:-1]
    discount = trajectory.discount

    logits, values = forward_fn(params, obs)

    values = values.squeeze(1)
    advantages = rlax.truncated_generalized_advantage_estimation(
      r_t=reward,
      discount_t=discount[1:] * gamma,
      lambda_=lambda_gae,
      values=values,
    )

    value_loss = jnp.square(advantages)

    logits = logits[:-1]
    log_probs = jax.nn.log_softmax(logits, axis=1)
    chosen_log_probs = jnp.take_along_axis(log_probs, action, axis=1)
    policy_gradient_loss = -chosen_log_probs * jax.lax.stop_gradient(advantages) * discount[:-1]

    probs = jax.nn.softmax(logits, axis=1)
    entropy_loss = -jnp.sum(probs * log_probs, axis=1)

    alpha_1, alpha_2, alpha_3 = coefs
    loss = alpha_1 * policy_gradient_loss + alpha_2 * value_loss - alpha_3 * entropy_loss
    return jnp.mean(loss)
  
  def update_fn(params, opt_state, state, timestep, rng):

    rng, rng_rollout = jax.random.split(rng, 2)

    params, state, timestep, trajectory = rollout_fn(params, state, timestep, rng_rollout)

    grads = jax.grad(loss_fn)(params, trajectory)

    grads = jax.lax.pmean(grads, axis_name="devices")
    grads = jax.lax.pmean(grads, axis_name="envs")
    updates, opt_state = opt_update_fn(grads, opt_state)
    params = optax.apply_updates(params, updates)

    return params, opt_state, state, timestep, rng
  
  def learner_fn(params, opt_state, state, timestep, rng):
    batched_update_fn = jax.vmap(update_fn, axis_name="envs")

    def iterate_fn(_, val):
      params, opt_state, state, timestep, rng = val
      return batched_update_fn(params, opt_state, state, timestep, rng)
    
    return jax.lax.fori_loop(0, n_iterarions, iterate_fn, (params, opt_state, state, timestep, rng))

  return learner_fn