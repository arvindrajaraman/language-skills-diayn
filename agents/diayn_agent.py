from agents.base_agent import Agent
import jax
from jax import jit
import jax.numpy as jnp

class DIAYNAgent(Agent):
    def __init__(self, config):
        super(DIAYNAgent, self).__init__(config)
    
    def pseudo_reward_fn(self, discrim_params, batch):
        skills, next_obs_embeddings = batch['skill'], batch['next_obs_embedding']

        skill_idxs = jnp.argmax(skills, axis=1)
        skill_preds = jax.nn.log_softmax(discrim.apply(discrim_params, next_obs_embeddings), axis=1)
        
        selected_logits = skill_preds[jnp.arange(self.config.batch_size), skill_idxs]
        reward_prs = selected_logits + jnp.log(self.config.skill_size)
        return reward_prs
    
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
