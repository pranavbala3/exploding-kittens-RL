"""
Training script for Exploding Kittens RL agent using PPO.
"""

import torch
from dataclasses import dataclass
import time

from env.environment import ExplodingKittensEnv
from core.game_config import GameConfig
from networks.policy import ExplodingKittensPolicy
from RL.ppo import PPO
from RL.rollout_storage import RolloutStorage
from RL.collect import collect_rollout, random_masked_action


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


def collect_rollout_random(env, policy, rollout_storage, num_steps, device='cpu'):
    """Learner (random seat) vs. uniformly-random masked opponents."""
    return collect_rollout(
        env, policy, rollout_storage, num_steps,
        opponent_action_fn=lambda seat, idx, obs, masks: random_masked_action(masks),
        device=device,
    )


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
        games_played, wins = collect_rollout_random(env, policy, rollout_storage, args.num_steps, device=device)
        
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
                'value_norm': policy.value_normalizer.state_dict(),
                'model_config': {'embed_dim': 128, 'hidden_dim': 128, 'num_attention_heads': 4},
                'optimizer_state_dict': ppo.optimizer.state_dict(),
                'update': update,
            }, f'checkpoint_{update}.pt')
            print(f"Saved checkpoint at update {update}")
    
    # Final save
    torch.save({
        'policy_state_dict': policy.state_dict(),
                'value_norm': policy.value_normalizer.state_dict(),
                'model_config': {'embed_dim': 128, 'hidden_dim': 128, 'num_attention_heads': 4},
        'optimizer_state_dict': ppo.optimizer.state_dict(),
        'update': args.num_updates,
    }, 'final_model.pt')
    
    print("\nTraining complete!")
    print(f"Model saved to final_model.pt")


if __name__ == '__main__':
    train()
