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
from RL.collect import collect_rollout, masks_to_tensors
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
    Collect experience with self-play opponents drawn from the policy pool.

    Args:
        opponent_nets: persistent opponent networks (one per opponent seat, up to 4)
        opponent_policies: state dicts to load into the opponent networks this rollout
    Returns:
        (games_played, wins) tuple
    """
    for net, state_dict in zip(opponent_nets, opponent_policies):
        net.load_state_dict(state_dict)
        net.eval()

    def opponent_action(seat, opp_idx, obs, masks_dict):
        net = opponent_nets[opp_idx % len(opponent_nets)]
        obs_tensor = {k: v.to(device) for k, v in process_observation(obs).items()}
        masks_tensor = masks_to_tensors(masks_dict, device)
        with torch.no_grad():
            action = net.get_action(obs_tensor, masks_tensor, deterministic=False)
        return {k: int(v) for k, v in action.items()}

    return collect_rollout(env, policy, rollout_storage, num_steps, opponent_action, device=device)


def train_selfplay(argv=None):
    """Main self-play training loop."""
    import argparse
    import os
    import random as _random
    from evaluate import win_rate_vs_random

    parser = argparse.ArgumentParser(description='Self-play PPO for Exploding Kittens')
    parser.add_argument('--updates', type=int, default=500)
    parser.add_argument('--steps', type=int, default=2048, help='learner decisions per update')
    parser.add_argument('--embed-dim', type=int, default=128)
    parser.add_argument('--snapshot-every', type=int, default=5, help='add policy to the pool every N updates')
    parser.add_argument('--eval-every', type=int, default=25)
    parser.add_argument('--eval-games', type=int, default=400)
    parser.add_argument('--save-every', type=int, default=50)
    parser.add_argument('--out', default='checkpoints')
    parser.add_argument('--init', default=None, help='warm-start from a checkpoint')
    parser.add_argument('--threads', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    cli = parser.parse_args(argv)

    if cli.threads:
        torch.set_num_threads(cli.threads)
    torch.manual_seed(cli.seed)
    np.random.seed(cli.seed)
    _random.seed(cli.seed)

    args = TrainingArgs()
    args.num_steps = cli.steps
    args.num_updates = cli.updates
    model_config = {'embed_dim': cli.embed_dim, 'hidden_dim': cli.embed_dim, 'num_attention_heads': 4}
    os.makedirs(cli.out, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Self-play training | device={device} | updates={args.num_updates} | "
          f"steps/update={args.num_steps} | model={model_config}", flush=True)

    env = ExplodingKittensEnv(GameConfig(num_players=args.num_players))
    policy = ExplodingKittensPolicy(use_value_norm=True, **model_config).to(device)
    if cli.init:
        ckpt = torch.load(cli.init, map_location=device)
        policy.load_state_dict(ckpt['policy_state_dict'])
        if 'value_norm' in ckpt:
            policy.value_normalizer.load_state_dict(ckpt['value_norm'])
        print(f"Warm-started from {cli.init}", flush=True)
    ppo = PPO(policy, args)

    num_opponents = args.num_players - 1
    opponent_nets = [ExplodingKittensPolicy(use_value_norm=True, **model_config).to(device)
                     for _ in range(num_opponents)]

    policy_pool = deque(maxlen=100)
    policy_pool.append(copy.deepcopy(policy.state_dict()))

    def save(path, update):
        torch.save({
            'policy_state_dict': policy.state_dict(),
            'optimizer_state_dict': ppo.optimizer.state_dict(),
            'value_norm': policy.value_normalizer.state_dict(),
            'model_config': model_config,
            'policy_pool': list(policy_pool),
            'update': update,
        }, path)

    start_time = time.time()
    window_wins = window_games = 0
    best = -1.0

    for update in range(1, args.num_updates + 1):
        if update < args.entropy_anneal_start:
            ppo.entropy_coef = args.entropy_coef_start
        elif update < args.entropy_anneal_end:
            progress = (update - args.entropy_anneal_start) / (args.entropy_anneal_end - args.entropy_anneal_start)
            ppo.entropy_coef = args.entropy_coef_start - progress * (args.entropy_coef_start - args.entropy_coef_end)
        else:
            ppo.entropy_coef = args.entropy_coef_end

        opponent_policies = sample_opponent_policies(policy_pool, num_opponents=num_opponents, recency_bias=0.7)
        rollout_storage = RolloutStorage(num_steps=args.num_steps)
        games_played, wins = collect_rollout_selfplay(
            env, policy, opponent_nets, opponent_policies, rollout_storage, args.num_steps, device
        )
        window_games += games_played
        window_wins += wins

        losses = ppo.update(rollout_storage, device=device)

        if update % cli.snapshot_every == 0:
            policy_pool.append(copy.deepcopy(policy.state_dict()))

        if update % args.log_interval == 0:
            print(f"update {update:4d} | {(time.time() - start_time) / 60:5.1f} min | "
                  f"self-play win {100 * window_wins / max(window_games, 1):5.1f}% ({window_games} games) | "
                  f"vloss {losses['value_loss']:.3f} | ent {losses['entropy']:.2f} "
                  f"(coef {ppo.entropy_coef:.4f}) | pool {len(policy_pool)}", flush=True)
            window_wins = window_games = 0

        if update % cli.eval_every == 0 or update == args.num_updates:
            rate = win_rate_vs_random(policy, num_games=cli.eval_games, num_players=args.num_players, device=device)
            print(f"  EVAL update {update}: win rate vs random = {100 * rate:.1f}% "
                  f"(baseline {100 / args.num_players:.0f}%, {cli.eval_games} games)", flush=True)
            if rate > best:
                best = rate
                save(os.path.join(cli.out, 'selfplay_best.pt'), update)

        if update % cli.save_every == 0:
            save(os.path.join(cli.out, f'selfplay_{update}.pt'), update)

    save(os.path.join(cli.out, 'selfplay_final.pt'), args.num_updates)
    print(f"Self-play training complete. Best eval win rate vs random: {100 * best:.1f}%", flush=True)


if __name__ == '__main__':
    train_selfplay()
