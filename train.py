"""
Training script for Exploding Kittens RL agent using PPO.
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import Dict
import time

from env.environment import ExplodingKittensEnv
from core.game_config import GameConfig
from networks.policy import ExplodingKittensPolicy
from RL.ppo import PPO
from RL.rollout_storage import RolloutStorage
from utils.observation_utils import process_observation


@dataclass
class TrainingArgs:
    # PPO hyperparameters (optimized based on Catan RL)
    lr: float = 3e-4
    eps: float = 1e-5
    clip_param: float = 0.2
    ppo_epoch: int = 10  # Increased from 4 (Catan uses 10)
    num_mini_batch: int = 8  # Increased from 4 (better gradient estimates)
    value_loss_coef: float = 1.0  # Increased from 0.5 (Catan uses 1.0)
    entropy_coef_start: float = 0.04  # High exploration early
    entropy_coef_end: float = 0.005  # Low exploration late
    entropy_anneal_start: int = 100  # Start annealing at update 100
    entropy_anneal_end: int = 500  # Finish annealing at update 500
    max_grad_norm: float = 0.5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    
    # Training parameters
    num_steps: int = 10000  # Increased from 2048 (more data per update)
    num_updates: int = 1000
    num_players: int = 4
    
    # Value normalization (from Catan RL)
    use_value_norm: bool = True
    
    # Logging
    log_interval: int = 10
    save_interval: int = 50


def collect_rollout(env, policy, rollout_storage, num_steps, learning_player_id=0, device='cpu'):
    """
    Collect experience by playing games.
    
    Args:
        env: Game environment
        policy: Policy network
        rollout_storage: Storage for experiences
        num_steps: Number of steps to collect
        learning_player_id: Which player is learning (default: 0)
        device: Device to run policy on ('cpu' or 'cuda')
    
    Returns:
        (games_played, wins) tuple
    """
    obs = env.reset()
    done = False
    steps_collected = 0
    games_played = 0
    wins = 0
    
    while steps_collected < num_steps:
        # Only collect data when it's the learning player's turn
        if env.state.current_player == learning_player_id:
            # Process observation
            obs_tensor = process_observation(obs)
            # Move to device
            obs_tensor = {k: v.to(device) for k, v in obs_tensor.items()}
            
            # Get action masks
            masks_dict = env.get_action_mask(learning_player_id)
            masks_tensor = {
                'head1': torch.BoolTensor(masks_dict['head1']).unsqueeze(0).to(device),
                'head2': torch.BoolTensor(masks_dict['head2']).unsqueeze(0).to(device),
                'head3': torch.BoolTensor(masks_dict['head3']).unsqueeze(0).to(device),
                'head4': torch.BoolTensor(masks_dict['head4']).unsqueeze(0).to(device),
                'head5': torch.BoolTensor(masks_dict['head5']).unsqueeze(0).to(device),
            }
            
            # Get action from policy
            with torch.no_grad():
                output = policy.forward(obs_tensor, masks_tensor)
                
                # Sample actions (handle empty masks by using uniform distribution)
                def safe_categorical(logits, mask):
                    """Create categorical distribution, handling all-masked case."""
                    if not mask.any():
                        # All masked - use uniform distribution over all actions
                        logits = torch.zeros_like(logits)
                    return torch.distributions.Categorical(logits=logits)
                
                dist1 = safe_categorical(output['logits1'], masks_tensor['head1'])
                dist2 = safe_categorical(output['logits2'], masks_tensor['head2'])
                dist3 = safe_categorical(output['logits3'], masks_tensor['head3'])
                dist4 = safe_categorical(output['logits4'], masks_tensor['head4'])
                dist5 = safe_categorical(output['logits5'], masks_tensor['head5'])
                
                action1 = dist1.sample()
                action2 = dist2.sample()
                action3 = dist3.sample()
                action4 = dist4.sample()
                action5 = dist5.sample()
                
                # Compute log probs
                log_prob = (dist1.log_prob(action1) + dist2.log_prob(action2) + 
                           dist3.log_prob(action3) + dist4.log_prob(action4) + 
                           dist5.log_prob(action5))
                
                value = output['value']
            
            # Convert to action dict
            action = {
                'head1': action1.item(),
                'head2': action2.item(),
                'head3': action3.item(),
                'head4': action4.item(),
                'head5': action5.item(),
            }
            
            # Take step
            next_obs, reward, done, info = env.step(action)
            
            # Store experience
            rollout_storage.insert(
                obs=obs,  # Store raw obs, not tensor
                action=action,
                action_log_prob=log_prob.cpu(),
                value_pred=value.cpu(),
                reward=reward,
                mask=0.0 if done else 1.0,
                action_mask=masks_dict  # Store raw masks
            )
            
            steps_collected += 1
            obs = next_obs
            
            if done:
                games_played += 1
                if 'winner' in info and info['winner'] == learning_player_id:
                    wins += 1
                obs = env.reset()
                done = False
        else:
            # Other players take random actions
            masks_dict = env.get_action_mask(env.state.current_player)
            
            # Random action
            valid_actions = np.where(masks_dict['head1'])[0]
            action = {
                'head1': int(np.random.choice(valid_actions)),
                'head2': int(np.random.choice(np.where(masks_dict['head2'])[0])) if masks_dict['head2'].any() else 0,
                'head3': int(np.random.choice(np.where(masks_dict['head3'])[0])) if masks_dict['head3'].any() else 0,
                'head4': int(np.random.choice(np.where(masks_dict['head4'])[0])) if masks_dict['head4'].any() else 0,
                'head5': int(np.random.choice(np.where(masks_dict['head5'])[0])) if masks_dict['head5'].any() else 0,
            }
            
            obs, reward, done, info = env.step(action)
            
            if done:
                games_played += 1
                if 'winner' in info and info['winner'] == learning_player_id:
                    wins += 1
                obs = env.reset()
                done = False
    
    return games_played, wins


def train():
    """Main training loop."""
    args = TrainingArgs()
    
    # Setup device (GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Setup
    config = GameConfig(num_players=args.num_players)
    env = ExplodingKittensEnv(config)
    
    policy = ExplodingKittensPolicy(
        embed_dim=128,
        hidden_dim=128,
        num_attention_heads=4,
        use_value_norm=args.use_value_norm
    ).to(device)  # Move to GPU
    
    ppo = PPO(policy, args)
    
    print("Starting training...")
    print(f"Total updates: {args.num_updates}")
    print(f"Steps per update: {args.num_steps}")
    print(f"Total parameters: {sum(p.numel() for p in policy.parameters()):,}")
    print(f"Value normalization: {args.use_value_norm}")
    print(f"Entropy annealing: {args.entropy_coef_start} → {args.entropy_coef_end}")
    
    start_time = time.time()
    total_wins = 0
    total_games = 0
    
    for update in range(args.num_updates):
        # Entropy annealing (from Catan RL)
        if update < args.entropy_anneal_start:
            ppo.entropy_coef = args.entropy_coef_start
        elif update < args.entropy_anneal_end:
            progress = (update - args.entropy_anneal_start) / (args.entropy_anneal_end - args.entropy_anneal_start)
            ppo.entropy_coef = args.entropy_coef_start - progress * (args.entropy_coef_start - args.entropy_coef_end)
        else:
            ppo.entropy_coef = args.entropy_coef_end
        
        # Collect rollout
        rollout_storage = RolloutStorage(num_steps=args.num_steps)
        games_played, wins = collect_rollout(env, policy, rollout_storage, args.num_steps, device=device)
        
        total_games += games_played
        total_wins += wins
        
        # Update policy
        losses = ppo.update(rollout_storage, device=device)
        
        # Logging
        if update % args.log_interval == 0:
            elapsed = time.time() - start_time
            win_rate = total_wins / max(total_games, 1) * 100
            
            print(f"\nUpdate {update}/{args.num_updates} | Time: {elapsed:.1f}s")
            print(f"  Games: {total_games} | Wins: {total_wins} ({win_rate:.1f}%)")
            print(f"  Value Loss: {losses['value_loss']:.4f}")
            print(f"  Policy Loss: {losses['action_loss']:.4f}")
            print(f"  Entropy: {losses['entropy']:.4f} (coef: {ppo.entropy_coef:.4f})")
            
            # Reset counters
            total_wins = 0
            total_games = 0
        
        # Save checkpoint
        if update % args.save_interval == 0 and update > 0:
            torch.save({
                'policy_state_dict': policy.state_dict(),
                'optimizer_state_dict': ppo.optimizer.state_dict(),
                'update': update,
            }, f'checkpoint_{update}.pt')
            print(f"Saved checkpoint at update {update}")
    
    # Final save
    torch.save({
        'policy_state_dict': policy.state_dict(),
        'optimizer_state_dict': ppo.optimizer.state_dict(),
        'update': args.num_updates,
    }, 'final_model.pt')
    
    print("\nTraining complete!")
    print(f"Model saved to final_model.pt")


if __name__ == '__main__':
    train()
