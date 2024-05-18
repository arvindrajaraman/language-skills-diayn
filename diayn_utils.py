from flax import linen as nn
from functools import partial
import jax
import jax.numpy as jnp
from jax import random, jit, vmap, pmap
import optax

from models import QNetCraftax
from utils import grad_norm

# def dqn_init_policy(config, policy_key, dummy_state, dummy_embedding, dummy_skill):
#     qlocal_key, qtarget_key = jax.random.split(policy_key)

#     qlocal = QNetCraftax(
#         action_size=config.action_size,
#         hidden1_size=config.policy_units,
#         hidden2_size=config.policy_units
#     )
#     qlocal_params = qlocal.init(qlocal_key, dummy_state, dummy_skill)
#     qlocal_opt = optax.adam(learning_rate=config.policy_lr)
#     qlocal_opt_state = qlocal_opt.init(qlocal_params)

#     qtarget = QNetCraftax(
#         action_size=config.action_size,
#         hidden1_size=config.policy_units,
#         hidden2_size=config.policy_units
#     )
#     qtarget_params = qtarget.init(qtarget_key, dummy_state, dummy_skill)

#     return (qlocal, qtarget), (qlocal_params, qtarget_params), qlocal_opt, qlocal_opt_state

# @partial(jit, static_argnames=('qlocal', 'action_size', 'vectorization'))
# def dqn_act(key, qlocal, qlocal_params, obs, skill, action_size, vectorization, eps=0.0):
#     key, explore_key, action_key = random.split(key, num=3)
#     action = jnp.where(
#         random.uniform(explore_key, shape=(vectorization,)) > eps,
#         jnp.argmax(qlocal.apply(qlocal_params, obs, skill), axis=1),
#         random.choice(action_key, jnp.arange(action_size), shape=(vectorization,))
#     )
#     return key, action

# @partial(jit, static_argnames=('qlocal', 'qlocal_opt', 'qtarget', 'discrim', 'discrim_opt', 'batch_size', 'skill_size', 'gamma', 'tau', 'reward_gt_coeff'))
# def dqn_update_model(qlocal, qlocal_params, qtarget, qtarget_params, qlocal_opt, qlocal_opt_state, discrim, discrim_params,
#                      discrim_opt, discrim_opt_state, batch_size, skill_size, gamma, tau,
#                      reward_gt_coeff, batch):    
#     def q_loss_fn(qlocal_params, batch):
#         obs, actions, skills, reward_gts, next_obs, next_obs_embeddings, dones = batch['obs'], batch['action'], batch['skill'], batch['reward_gt'], batch['next_obs'], batch['next_obs_embedding'], batch['done']

#         skill_idxs = jnp.argmax(skills, axis=1)
#         skill_preds = jax.nn.log_softmax(discrim.apply(discrim_params, next_obs_embeddings), axis=1)
        
#         selected_logits = skill_preds[jnp.arange(batch_size), skill_idxs]
#         reward_prs = (selected_logits + jnp.log(skill_size))
#         rewards = reward_prs + (reward_gt_coeff * reward_gts)

#         Q_targets_next = qtarget.apply(qtarget_params, next_obs, skills).max(axis=1)
#         Q_targets = rewards + (gamma * Q_targets_next * ~dones)
#         Q_expected = qlocal.apply(qlocal_params, obs, skills)[jnp.arange(batch_size), actions]

#         loss = optax.squared_error(Q_targets, Q_expected).mean()

#         q_value_mean = Q_expected.mean()
#         target_value_mean = Q_targets.mean()
#         reward_mean = rewards.mean()
#         return loss, (q_value_mean, target_value_mean, reward_mean)

#     (q_loss, (q_value_mean, target_value_mean, reward_mean)), q_grads = jax.value_and_grad(q_loss_fn, has_aux=True)(qlocal_params, batch)
#     qlocal_updates, qlocal_opt_state = qlocal_opt.update(q_grads, qlocal_opt_state)
#     qlocal_params = optax.apply_updates(qlocal_params, qlocal_updates)
#     qtarget_params = optax.incremental_update(qlocal_params, qtarget_params, tau)

#     def discrim_loss_fn(discrim_params, batch):
#         next_obs_embeddings, skills = batch['next_obs_embedding'], batch['skill']

#         skill_idxs = jnp.argmax(skills, axis=1)
#         skill_preds = discrim.apply(discrim_params, next_obs_embeddings)
#         skill_pred_idxs = jnp.argmax(skill_preds, axis=1)

#         loss = optax.softmax_cross_entropy(skill_preds, skills).mean()
#         acc = (skill_pred_idxs == skill_idxs).mean()
#         pplx = 2. ** jax.scipy.special.entr(nn.activation.softmax(skill_preds)).sum(axis=1).mean()

#         return loss, (acc, pplx)
    
#     (discrim_loss, (discrim_acc, discrim_pplx)), discrim_grads = jax.value_and_grad(discrim_loss_fn, has_aux=True)(discrim_params, batch)
#     discrim_updates, discrim_opt_state = discrim_opt.update(discrim_grads, discrim_opt_state)
#     discrim_params = optax.apply_updates(discrim_params, discrim_updates)

#     model_metrics = {
#         'critic_loss': q_loss,
#         'q_values': q_value_mean,
#         'target_values': target_value_mean,
#         'qlocal_grad_norm': grad_norm(q_grads),
#         'reward': reward_mean,

#         'discrim_loss': discrim_loss,
#         'discrim_acc': discrim_acc,
#         'discrim_pplx': discrim_pplx,
#         'discrim_grad_norm': grad_norm(discrim_grads)
#     }

#     return qlocal_params, qlocal_opt_state, qtarget_params, discrim_params, discrim_opt_state, model_metrics

# def diayn_discriminator_update(discrim, discrim_params, discrim_opt, discrim_opt_state, batch):
#     def discrim_loss_fn(discrim_params, batch):
#         next_obs_embeddings, skills = batch['next_obs_embedding'], batch['skill']

#         skill_idxs = jnp.argmax(skills, axis=1)
#         skill_preds = discrim.apply(discrim_params, next_obs_embeddings)
#         skill_pred_idxs = jnp.argmax(skill_preds, axis=1)

#         loss = optax.softmax_cross_entropy(skill_preds, skills).mean()
#         acc = (skill_pred_idxs == skill_idxs).mean()
#         pplx = 2. ** jax.scipy.special.entr(nn.activation.softmax(skill_preds)).sum(axis=1).mean()

#         return loss, (acc, pplx)
    
#     (discrim_loss, (discrim_acc, discrim_pplx)), discrim_grads = jax.value_and_grad(discrim_loss_fn, has_aux=True)(discrim_params, batch)
#     discrim_updates, discrim_opt_state = discrim_opt.update(discrim_grads, discrim_opt_state)
#     discrim_params = optax.apply_updates(discrim_params, discrim_updates)

#     model_metrics = {
#         'discrim_loss': discrim_loss,
#         'discrim_acc': discrim_acc,
#         'discrim_pplx': discrim_pplx,
#         'discrim_grad_norm': grad_norm(discrim_grads)
#     }

#     return discrim_params, discrim_opt_state, model_metrics

def metra_repfn_update(phi, phi_params, phi_opt, phi_opt_state, lambda_, lambda_opt, lambda_opt_state, metra_eps, batch):
    def phi_loss_fn(phi_params, batch, lambda_):
        obs, next_obs, skills = batch['obs'], batch['next_obs'], batch['skill']

        phi_next_obs = phi.apply(phi_params, next_obs)
        phi_obs = phi.apply(phi_params, obs)

        term1 = jnp.dot((phi_next_obs - phi_obs).T, skills)
        term2 = jnp.minimum(
            metra_eps,
            1 - (jnp.linalg.norm(phi_next_obs - phi_obs, axis=1) ** 2)
        )

        loss = (term1 + (lambda_ * term2)).mean()
        return loss

    phi_loss, phi_grads = jax.value_and_grad(phi_loss_fn)(phi_params, batch, lambda_)
    phi_updates, phi_opt_state = phi_opt.update(phi_grads, phi_opt_state)
    phi_params = optax.apply_updates(phi_params, phi_updates)

    def lambda_loss_fn(lambda_, batch):
        obs, next_obs = batch['obs'], batch['next_obs']

        phi_next_obs = phi.apply(phi_params, next_obs)
        phi_obs = phi.apply(phi_params, obs)

        loss = lambda_ * jnp.minimum(
            metra_eps,
            1 - (jnp.linalg.norm(phi_next_obs - phi_obs, axis=1) ** 2)
        ).mean()
        return loss

    lambda_loss, lambda_grads = jax.value_and_grad(lambda_loss_fn)(lambda_, batch)
    lambda_updates, lambda_opt_state = lambda_opt.update(lambda_grads, lambda_opt_state)
    lambda_ = optax.apply_updates(lambda_, lambda_updates)

    model_metrics = {
        'phi_loss': phi_loss,
        'lambda_loss': lambda_loss,
        'lambda_': lambda_
    }

    return phi_params, phi_opt_state, lambda_, lambda_opt_state, model_metrics
