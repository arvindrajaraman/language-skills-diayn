from agents.base_agent import Agent
import jax
from jax import jit
import jax.numpy as jnp

class APTAgent(Agent):
    def __init__(self, config):
        super(APTAgent, self).__init__(config)
        
    def init_skill_networks(self, key):
        key, discrim_params, discrim_opt_state = super(APTAgent, self).init_skill_networks()
        
        key, forward_model_key, backward_model_key = jax.random.split(key, num=3)
        self.forward_model = APTForwardDynamicsModel(
            action_size=self.config.action_size,
            skill_size=self.config.state_size,
            hidden_size=self.config.discrim_units
        )
        self.forward_model_opt = optax.adam(learning_rate=self.config.discrim_lr)
        
        forward_model_params = forward_model.init(forward_model_key, dummy_skill, jnp.zeros((1,)))
        forward_model_opt_state = forward_model_opt.init(forward_model_params)
        
        self.backward_model = APTBackwardDynamicsModel(
            action_size=self.config.action_size,
            hidden_size=self.config.discrim_units
        )
        self.backward_model_opt = optax.adam(learning_rate=self.config.discrim_lr)
        
        backward_model_params = backward_model.init(backward_model_key, dummy_skill, dummy_skill)
        backward_model_opt_state = backward_model_opt.init(backward_model_params)
        
        return key, (discrim_params, forward_model_params, backward_model_params), (discrim_opt_state, forward_model_opt_state, backward_model_opt_state)
    
    def pseudo_reward_fn(self, discrim_params, batch):
        next_obs_embeddings = batch['next_obs_embedding']
        
        reps = self.discrim.apply(discrim_params, next_obs_embeddings)
        sum_squares = jnp.square(reps).sum(axis=1)
        dists = jnp.outer(sum_squares, sum_squares) - 2 * jnp.dot(reps, reps.T)
        dists = jnp.sqrt(jnp.maximum(dists, 0.0))
        
        # Within each column of this matrix, find the k smallest values
        dists = jnp.sort(dists, axis=1)
        knns = dists[:, :self.config.apt_k]
        knns_avg_dist = knns.mean(axis=1)
        
        return jnp.log(knns_avg_dist + 1.0)  # 1.0 for numerical stability
    
    def update_skill_networks(self, params, opt_state, batch):
        discrim_params, forward_model_params, backward_model_params = params
        discrim_opt_state, forward_model_opt_state, backward_model_opt_state = opt_state
        
        def loss_fn(discrim_params, forward_model_params, backward_model_params, batch):
            obs_embeddings, actions, next_obs_embeddings = batch['obs_embedding'], batch['action'], batch['next_obs_embedding']
            
            obs_apt_encoding = self.discrim.apply(discrim_params, obs_embeddings)
            next_obs_apt_encoding = self.discrim.apply(discrim_params, next_obs_embeddings)
            
            next_obs_hat = self.forward_model.apply(forward_model_params, obs_apt_encoding, actions)
            action_hat = self.backward_model.apply(backward_model_params, obs_apt_encoding, next_obs_apt_encoding)
            
            forward_error = optax.l2_loss(next_obs_hat, next_obs_embeddings).mean()
            backward_error = optax.softmax_cross_entropy_with_integer_labels(action_hat, actions).mean()
            loss = forward_error + backward_error
            return loss, (forward_error, backward_error)
        
        (loss, (forward_error, backward_error)), (discrim_grads, forward_model_grads, backward_model_grads) = jax.value_and_grad(loss_fn, has_aux=True, argnums=(0, 1, 2))(discrim_params, forward_model_params, backward_model_params, batch)
        
        discrim_updates, discrim_opt_state = discrim_opt.update(discrim_grads, discrim_opt_state)
        discrim_params = optax.apply_updates(discrim_params, discrim_updates)
        
        forward_model_updates, forward_model_opt_state = self.forward_model_opt.update(forward_model_grads, forward_model_opt_state)
        forward_model_params = optax.apply_updates(forward_model_params, forward_model_updates)
        
        backward_model_updates, backward_model_opt_state = self.backward_model_opt.update(backward_model_grads, backward_model_opt_state)
        backward_model_params = optax.apply_updates(backward_model_params, backward_model_updates)
        
        skill_learning_metrics = {
            'apt_loss': loss,
            'forward_error': forward_error,
            'backward_error': backward_error
        }
        
        params = discrim_params, forward_model_params, backward_model_params
        opt_state = discrim_opt_state, forward_model_opt_state, backward_model_opt_state
        return params, opt_state, skill_learning_metrics
