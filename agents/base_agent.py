from abc import ABC, abstractmethod
import jax
from jax import jit
import jax.numpy as jnp

class Agent(ABC):
    def __init__(self, config):
        self.config = config
        
    def init_policy(self, key):
        key, qlocal_key, qtarget_key = random.split(key, num=3)
        dummy_state = jnp.zeros((1, self.config.state_size))
        dummy_skill = jnp.zeros((1, self.config.skill_size))

        self.qlocal = QNetCraftax(
            action_size=self.config.action_size,
            hidden_size=self.config.policy_units
        )
        self.qlocal_opt = optax.adam(learning_rate=self.config.policy_lr)
        
        qlocal_params = qlocal.init(qlocal_key, dummy_state, dummy_skill)
        qlocal_opt_state = qlocal_opt.init(qlocal_params)

        self.qtarget = QNetCraftax(
            action_size=self.config.action_size,
            hidden_size=self.config.policy_units
        )
        qtarget_params = qtarget.init(qtarget_key, dummy_state, dummy_skill)
        return key, qlocal_params, qlocal_opt_state, qtarget_params
        
    def init_skill_networks(self, key):
        key, discrim_key = random.split(key, num=2)
        dummy_embedding = jnp.zeros((1, self.config.embedding_size))

        DiscriminatorModel = DiscriminatorCraftax if self.config.embedding_type == 'identity' else Discriminator
        self.discrim = DiscriminatorModel(
            skill_size=self.config.skill_size,
            hidden_size=self.config.discrim_units
        )
        self.discrim_opt = optax.adam(learning_rate=self.config.discrim_lr)
        
        discrim_params = discrim.init(discrim_key, dummy_embedding)
        discrim_opt_state = discrim_opt.init(discrim_params)
        return key, discrim_params, discrim_opt_state
        
    @abstractmethod
    def pseudo_reward_fn(self, discrim_params, batch):
        pass
    
    @abstractmethod
    def update_skill_networks(self, params, opt_state, batch):
        pass
    
    def act(self, key, qlocal_params, obs, skill, eps=0.0):
        key, explore_key, action_key = random.split(key, num=3)
        action = jnp.where(
            random.uniform(explore_key, shape=(self.config.vectorization,)) > eps,
            jnp.argmax(qlocal.apply(qlocal_params, obs, skill), axis=1),
            random.choice(action_key, jnp.arange(self.config.action_size), shape=(self.config.vectorization,))
        )
        return key, action

    def update_policy(self, qlocal_params, qtarget_params, qlocal_opt_state, discrim_params, reward_pr_coeff, reward_gt_coeff, batch):
        def loss_fn(qlocal_params, reward_pr_coeff, reward_gt_coeff, batch):
            obs, actions, skills, reward_gts, next_obs, dones = batch['obs'], batch['action'], batch['skill'], batch['reward_gt'], batch['next_obs'], batch['done']

            reward_prs = self.pseudo_reward_fn(discrim_params, batch)
            rewards = (reward_pr_coeff * reward_prs) + (reward_gt_coeff * reward_gts)

            Q_targets_next = qtarget.apply(qtarget_params, next_obs, skills).max(axis=1)
            Q_targets = rewards + (self.config.gamma * Q_targets_next * ~dones)
            Q_expected = qlocal.apply(qlocal_params, obs, skills)[jnp.arange(self.config.batch_size), actions]

            loss = optax.squared_error(Q_targets, Q_expected).mean()

            q_value_mean = Q_expected.mean()
            target_value_mean = Q_targets.mean()
            reward_mean = rewards.mean()
            return loss, (q_value_mean, target_value_mean, reward_mean)

        (loss, (q_value_mean, target_value_mean, reward_mean)), grads = jax.value_and_grad(loss_fn, has_aux=True)(qlocal_params, reward_pr_coeff, reward_gt_coeff, batch)
        qlocal_updates, qlocal_opt_state = qlocal_opt.update(grads, qlocal_opt_state)
        qlocal_params = optax.apply_updates(qlocal_params, qlocal_updates)
        qtarget_params = optax.incremental_update(qlocal_params, qtarget_params, self.config.tau)

        policy_metrics = {
            'critic_loss': loss,
            'q_values': q_value_mean,
            'target_values': target_value_mean,
            'qlocal_grad_norm': grad_norm(grads),
            'reward': reward_mean
        }
        return qlocal_params, qlocal_opt_state, qtarget_params, policy_metrics
