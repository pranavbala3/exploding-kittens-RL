import torch
import numpy as np
from collections import deque
import copy
import time

from env.environment import ExplodingKittensEnv
from core.game_config import GameConfig
from networks.policy import ExplodingKittensPolicy
from RL.ppo import PPO
from RL.rollout_storage import RolloutStorage
from utils.observation_utils import process_observation
from train import TrainingArgs

"""
Self-play training with policy pool (Catan RL approach).

Key differences from basic training:
1. Maintains pool of old policies (last 100 checkpoints)
2. Opponents sample from policy pool (biased toward recent)
3. Only learning player's experience is used for training
"""


def sample_opponent_policies(policy_pool, num_opponents=3, recency_bias=0.7):
    """
    Sample opponent policies from pool with recency bias.
    
    Args:
        policy_pool: Deque of policy state dicts
        num_opponents: Number of opponents to sample
        recency_bias: Probability of sampling from recent half (0.7 = 70% recent)
    
    Returns:
        List of policy state dicts
    """
    if len(policy_pool) == 0:
        return []
    
    opponents = []
    pool_size = len(policy_pool)
    recent_cutoff = pool_size // 2
    
    for _ in range(num_opponents):
        if np.random.random() < recency_bias and pool_size > 1:
            # Sample from recent half
            idx = np.random.randint(recent_cutoff, pool_size)
        else:
            # Sample from entire pool
            idx = np.random.randint(0, pool_size)
        
        opponents.append(policy_pool[idx])
    
    return opponents


def collect_rollout_selfplay(env, policy, opponent_nets, opponent_policies, rollout_storage, num_steps, device='cpu'):
    """
    Collect experience with self-play opponents.
    
    Args:
        env: Game environment
        policy: Current policy (learning player)
        opponent_nets: List of 3 persistent opponent networks
        opponent_policies: List of 3 opponent policy state dicts to load
        rollout_storage: Storage for experiences
        num_steps: Number of steps to collect
        device: Device to run on
    
    Returns:
        (games_played, wins) tuple
    """
    # Update opponent networks with sampled policies
    for i, opp_state_dict in enumerate(opponent_policies):
        opponent_nets[i].load_state_dict(opp_state_dict)
        opponent_nets[i].eval()
    
    obs = env.reset()
    done = False
    steps_collected = 0
    games_played = 0
    wins = 0
    learning_player_id = 0
    
    while steps_collected < num_steps:
        current_player = env.state.current_player
        
        if current_player == learning_player_id:
            # Learning player uses current policy
            obs_tensor = process_observation(obs)
            obs_tensor = {k: v.to(device) for k, v in obs_tensor.items()}
            
            masks_dict = env.get_action_mask(learning_player_id)
            masks_tensor = {
                'head1': torch.BoolTensor(masks_dict['head1']).unsqueeze(0).to(device),
                'head2': torch.BoolTensor(masks_dict['head2']).unsqueeze(0).to(device),
                'head3': torch.BoolTensor(masks_dict['head3']).unsqueeze(0).to(device),
                'head4': torch.BoolTensor(masks_dict['head4']).unsqueeze(0).to(device),
                'head5': torch.BoolTensor(masks_dict['head5']).unsqueeze(0).to(device),
            }
            
            with torch.no_grad():
                output = policy.forward(obs_tensor, masks_tensor)
                
                def safe_categorical(logits, mask):
                    if not mask.any():
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
                
                log_prob = (dist1.log_prob(action1) + dist2.log_prob(action2) + 
                           dist3.log_prob(action3) + dist4.log_prob(action4) + 
                           dist5.log_prob(action5))
                
                value = output['value']
            
            action = {
                'head1': action1.item(),
                'head2': action2.item(),
                'head3': action3.item(),
                'head4': action4.item(),
                'head5': action5.item(),
            }
            
            next_obs, reward, done, info = env.step(action)
            
            rollout_storage.insert(
                obs=obs,
                action=action,
                action_log_prob=log_prob.cpu(),
                value_pred=value.cpu(),
                reward=reward,
                mask=0.0 if done else 1.0,
                action_mask=masks_dict
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
            # Opponent uses old policy
            opponent_idx = (current_player - 1) % 3  # Map player 1,2,3 to opponent 0,1,2
            opponent_net = opponent_nets[opponent_idx]
            
            obs_tensor = process_observation(obs)
            obs_tensor = {k: v.to(device) for k, v in obs_tensor.items()}
            
            masks_dict = env.get_action_mask(current_player)
            masks_tensor = {
                'head1': torch.BoolTensor(masks_dict['head1']).unsqueeze(0).to(device),
                'head2': torch.BoolTensor(masks_dict['head2']).unsqueeze(0).to(device),
                'head3': torch.BoolTensor(masks_dict['head3']).unsqueeze(0).to(device),
                'head4': torch.BoolTensor(masks_dict['head4']).unsqueeze(0).to(device),
                'head5': torch.BoolTensor(masks_dict['head5']).unsqueeze(0).to(device),
            }
            
            with torch.no_grad():
                action = opponent_net.get_action(obs_tensor, masks_tensor, deterministic=False)
            
            obs, reward, done, info = env.step(action)
            
            if done:
                games_played += 1
                if 'winner' in info and info['winner'] == learning_player_id:
                    wins += 1
                obs = env.reset()
                done = False
    
    return games_played, wins


def train_selfplay():
    """Main self-play training loop."""
    args = TrainingArgs()
    args.num_steps = 2000
    args.num_updates = 500  # Self-play needs more updates
    args.log_interval = 5
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🎮 Self-Play Training (Catan RL approach)")
    print(f"Using device: {device}")
    print(f"Updates: {args.num_updates}")
    print(f"Policy pool size: 100 (last 100 checkpoints)")
    print()
    
    config = GameConfig(num_players=args.num_players)
    env = ExplodingKittensEnv(config)
    policy = ExplodingKittensPolicy(embed_dim=128, hidden_dim=128, use_value_norm=True).to(device)
    ppo = PPO(policy, args)

    # Create persistent opponent networks
    opponent_nets = [
        ExplodingKittensPolicy(embed_dim=128, hidden_dim=128, use_value_norm=True).to(device)
        for _ in range(3)
    ]
    
    # Policy pool (store last 100 policies)
    policy_pool = deque(maxlen=100)
    policy_pool.append(copy.deepcopy(policy.state_dict()))
    
    start_time = time.time()
    total_wins = 0
    total_games = 0
    
    for update in range(args.num_updates):
        # Entropy annealing
        if update < args.entropy_anneal_start:
            ppo.entropy_coef = args.entropy_coef_start
        elif update < args.entropy_anneal_end:
            progress = (update - args.entropy_anneal_start) / (args.entropy_anneal_end - args.entropy_anneal_start)
            ppo.entropy_coef = args.entropy_coef_start - progress * (args.entropy_coef_start - args.entropy_coef_end)
        else:
            ppo.entropy_coef = args.entropy_coef_end
        
        # Sample opponent policies from pool
        opponent_policies = sample_opponent_policies(policy_pool, num_opponents=3, recency_bias=0.7)
        
        # Collect rollout with self-play
        rollout_storage = RolloutStorage(num_steps=args.num_steps)
        games_played, wins = collect_rollout_selfplay(
            env, policy, opponent_nets,opponent_policies, rollout_storage, args.num_steps, device
        )
        
        total_games += games_played
        total_wins += wins
        
        # Update policy
        losses = ppo.update(rollout_storage, device=device)
        
        # Add current policy to pool every 5 updates
        if update % 5 == 0:
            policy_pool.append(copy.deepcopy(policy.state_dict()))
        
        # Logging
        if update % args.log_interval == 0:
            elapsed = time.time() - start_time
            win_rate = total_wins / max(total_games, 1) * 100
            
            print(f'Update {update}/{args.num_updates} | {elapsed/60:.1f}min | Wins: {win_rate:.1f}%')
            print(f'  Policy pool size: {len(policy_pool)} | Entropy: {ppo.entropy_coef:.4f}')
            
            total_wins = 0
            total_games = 0
        
        # Save checkpoint
        if update % 50 == 0 and update > 0:
            torch.save({
                'policy_state_dict': policy.state_dict(),
                'policy_pool': list(policy_pool),
                'update': update,
            }, f'selfplay_checkpoint_{update}.pt')
            print(f'💾 Saved checkpoint with policy pool')
    
    # Final save
    torch.save({
        'policy_state_dict': policy.state_dict(),
        'policy_pool': list(policy_pool),
        'update': args.num_updates,
    }, 'selfplay_final.pt')
    
    print('✅ Self-play training complete!')


if __name__ == '__main__':
    train_selfplay()
