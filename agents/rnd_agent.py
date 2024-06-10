from agents.base_agent import Agent
import jax
from jax import jit
import jax.numpy as jnp

class RNDAgent(Agent):
    def __init__(self, config):
        super(RNDAgent, self).__init__(config)
        
    def init_skill_networks(self, key):
        key, discrim_params, discrim_opt_state = super(RNDAgent, self).init_skill_networks()
        
        key, random_fn_key = jax.random.split(key)
        self.random_fn = DiscriminatorModel(
            skill_size=self.config.skill_size,
            hidden_size=self.config.discrim_units
        )
        self.random_fn_params = random_fn.init(random_fn_key, dummy_embedding)
        
        return key, discrim_params, discrim_opt_state
    
    @jit(static_argnums=0)
    def pseudo_reward_fn(self, discrim_params, batch):
        next_obs_embeddings = batch['next_obs_embedding']
        
        rnd_rep_target = self.discrim.apply(discrim_params, next_obs_embeddings)
        rnd_rep_pred = jax.lax.stop_gradient(self.random_fn.apply(self.random_fn_params, next_obs_embeddings))
        
        return jnp.sqrt(jnp.square(rnd_rep_target - rnd_rep_pred).sum(axis=1))
    
    def update_skill_networks(self, params, opt_state, batch):
        def loss_fn(params, batch):
            next_obs_embeddings, skills = batch['next_obs_embedding'], batch['skill']

            skill_idxs = jnp.argmax(skills, axis=1)
            skill_preds = self.discrim.apply(params, next_obs_embeddings)
            skill_pred_idxs = jnp.argmax(skill_preds, axis=1)

            loss = optax.softmax_cross_entropy(skill_preds, skills).mean()
            acc = (skill_pred_idxs == skill_idxs).mean()
            pplx = 2. ** jax.scipy.special.entr(nn.activation.softmax(skill_preds)).sum(axis=1).mean()

            return loss, (acc, pplx)
        
        (loss, (discrim_acc, discrim_pplx)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, batch)
        discrim_updates, opt_state = self.discrim_opt.update(grads, opt_state)
        params = optax.apply_updates(params, discrim_updates)

        skill_learning_metrics = {
            'discrim_loss': loss,
            'discrim_acc': discrim_acc,
            'discrim_pplx': discrim_pplx,
            'discrim_grad_norm': grad_norm(grads)
        }

        return params, opt_state, skill_learning_metrics
