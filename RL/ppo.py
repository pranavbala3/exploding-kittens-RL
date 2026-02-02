import torch
import torch.nn as nn


class PPO():
    def __init__(self, actor_critic, args):
        self.actor_critic = actor_critic
        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef_start  # Start with high entropy
        self.max_grad_norm = args.max_grad_norm
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        
        self.optimizer = torch.optim.Adam(
            actor_critic.parameters(), 
            lr=args.lr, 
            eps=args.eps
        )
    
    def update(self, rollout_storage, device='cpu'):
        """Update policy using PPO."""
        from RL.rollout_storage import batch_observations, batch_action_masks, batch_actions
        
        value_loss_epoch = 0
        action_loss_epoch = 0
        entropy_loss_epoch = 0
        
        # Compute advantages once (how good were actions?)
        rollout_storage.compute_advantages(
            self.actor_critic, 
            self.gamma, 
            self.gae_lambda
        )
        
        # PPO: Update multiple times on the same data
        for e in range(self.ppo_epoch):
            # Get mini-batches of data
            data_generator = rollout_storage.get_batches(self.num_mini_batch)
            
            for batch in data_generator:
                obs_batch, actions_batch, action_masks_batch, \
                    value_preds_batch, returns_batch, \
                    old_action_log_probs_batch, adv_batch = batch
                
                # Batch observations and masks
                obs_batched = batch_observations(obs_batch)
                masks_batched = batch_action_masks(action_masks_batch)
                actions_batched = batch_actions(actions_batch)
                
                # Move to device
                obs_batched = {k: v.to(device) for k, v in obs_batched.items()}
                masks_batched = {k: v.to(device) for k, v in masks_batched.items()}
                actions_batched = {k: v.to(device) for k, v in actions_batched.items()}
                value_preds_batch = value_preds_batch.to(device)
                returns_batch = returns_batch.to(device)
                old_action_log_probs_batch = old_action_log_probs_batch.to(device)
                adv_batch = adv_batch.to(device)
                
                # Apply value normalization (from Catan RL)
                if self.actor_critic.use_value_norm:
                    value_preds_batch = self.actor_critic.value_normalizer.normalize(value_preds_batch)
                    returns_batch = self.actor_critic.value_normalizer.normalize(returns_batch)
                
                # Evaluate actions with current policy
                values, action_log_probs, entropy, _ = \
                    self.actor_critic.evaluate_actions(
                        obs_batched, 
                        None,  # No recurrent states
                        None,  # No masks
                        actions_batched, 
                        masks_batched
                    )
                
                # PPO policy loss
                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                surr1 = ratio * adv_batch
                surr2 = torch.clamp(
                    ratio, 
                    1.0 - self.clip_param, 
                    1.0 + self.clip_param
                ) * adv_batch
                action_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss (with clipping)
                value_pred_clipped = value_preds_batch + \
                    (values - value_preds_batch).clamp(
                        -self.clip_param, 
                        self.clip_param
                    )
                value_losses = (values - returns_batch).pow(2)
                value_losses_clipped = (value_pred_clipped - returns_batch).pow(2)
                value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
                
                # Total loss
                total_loss = (
                    value_loss * self.value_loss_coef + 
                    action_loss - 
                    entropy * self.entropy_coef
                )
                
                # Update
                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.actor_critic.parameters(), 
                    self.max_grad_norm
                )
                self.optimizer.step()
                
                # Track losses
                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                entropy_loss_epoch += entropy.item()
        
        # Average losses
        num_updates = self.ppo_epoch * self.num_mini_batch
        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        entropy_loss_epoch /= num_updates
        
        return {
            'value_loss': value_loss_epoch,
            'action_loss': action_loss_epoch,
            'entropy': entropy_loss_epoch,
        }