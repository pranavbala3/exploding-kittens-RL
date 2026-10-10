"""
Evaluation script for trained Exploding Kittens agent.
"""

import torch
import numpy as np
from collections import defaultdict

from env.environment import ExplodingKittensEnv
from core.game_config import GameConfig
from networks.policy import ExplodingKittensPolicy
from utils.observation_utils import process_observation


class RandomAgent:
    """Baseline random agent."""
    def get_action(self, obs, masks):
        valid = np.where(masks['head1'])[0]
        return {
            'head1': int(np.random.choice(valid)),
            'head2': int(np.random.choice(np.where(masks['head2'])[0])) if masks['head2'].any() else 0,
            'head3': int(np.random.choice(np.where(masks['head3'])[0])) if masks['head3'].any() else 0,
            'head4': int(np.random.choice(np.where(masks['head4'])[0])) if masks['head4'].any() else 0,
            'head5': int(np.random.choice(np.where(masks['head5'])[0])) if masks['head5'].any() else 0,
        }


class PolicyAgent:
    """Agent using trained policy."""
    def __init__(self, policy, deterministic=True, device='cpu'):
        self.policy = policy
        self.deterministic = deterministic
        self.device = device
    
    def get_action(self, obs, masks):
        obs_tensor = {k: v.to(self.device) for k, v in process_observation(obs).items()}
        masks_tensor = {k: torch.as_tensor(v, dtype=torch.bool).unsqueeze(0).to(self.device)
                        for k, v in masks.items()}
        
        with torch.no_grad():
            action = self.policy.get_action(obs_tensor, masks_tensor, self.deterministic)
        
        return action


def play_game(env, agents, verbose=False):
    """Play one game and return winner."""
    obs = env.reset()
    done = False
    steps = 0
    
    while not done and steps < 1000:
        player_id = env.state.current_player
        masks = env.get_action_mask(player_id)
        
        action = agents[player_id].get_action(obs, masks)
        obs, reward, done, info = env.step(action)
        
        steps += 1
        
        if verbose and steps % 10 == 0:
            alive = sum(env.state.players_alive)
            print(f"Step {steps}: {alive} players alive")
    
    if done and 'winner' in info:
        return info['winner']
    
    # Timeout - pick random alive player
    alive = [i for i, a in enumerate(env.state.players_alive) if a]
    return alive[0] if alive else 0


def win_rate_vs_random(policy, num_games=200, num_players=4, deterministic=True, device='cpu'):
    """Fraction of games `policy` wins against random opponents, rotating its seat."""
    env = ExplodingKittensEnv(GameConfig(num_players=num_players))
    agent = PolicyAgent(policy, deterministic=deterministic, device=device)
    wins = 0
    was_training = policy.training
    policy.eval()
    for game in range(num_games):
        seat = game % num_players
        agents = [RandomAgent() for _ in range(num_players)]
        agents[seat] = agent
        wins += play_game(env, agents) == seat
    policy.train(was_training)
    return wins / num_games


def load_policy(checkpoint_path, device='cpu'):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = checkpoint.get('model_config', {'embed_dim': 128, 'hidden_dim': 128})
    policy = ExplodingKittensPolicy(**cfg).to(device)
    policy.load_state_dict(checkpoint['policy_state_dict'])
    if 'value_norm' in checkpoint:
        policy.value_normalizer.load_state_dict(checkpoint['value_norm'])
    policy.eval()
    return policy, checkpoint


def evaluate(checkpoint_path, num_games=100):
    """Evaluate trained agent against random opponents."""
    
    # Load policy
    policy, checkpoint = load_policy(checkpoint_path)
    
    print(f"Loaded checkpoint from update {checkpoint.get('update', '?')}")
    
    # Setup
    config = GameConfig(num_players=4)
    env = ExplodingKittensEnv(config)
    
    # The trained agent rotates through every seat (seat 0 always moves first, which is an
    # advantage), against random opponents. wins[0] counts wins of the TRAINED agent.
    trained = PolicyAgent(policy, deterministic=True)
    wins = defaultdict(int)
    
    print(f"\nPlaying {num_games} games...")
    for game in range(num_games):
        seat = game % 4
        agents = [RandomAgent() for _ in range(4)]
        agents[seat] = trained
        winner = play_game(env, agents, verbose=False)
        wins[0 if winner == seat else 1] += 1
        
        if (game + 1) % 10 == 0:
            print(f"Games {game + 1}/{num_games}: Trained agent wins: {wins[0]}/{game + 1} ({100*wins[0]/(game+1):.1f}%)")
    
    # Results
    print("\n" + "="*50)
    print("RESULTS")
    print("="*50)
    print(f"Trained agent (all seats): {wins[0]}/{num_games} wins ({100*wins[0]/num_games:.1f}%)")
    
    print(f"\nRandom baseline: 25% (1/4 players)")
    print(f"Trained agent: {100*wins[0]/num_games:.1f}%")
    
    if wins[0] > num_games * 0.25:
        improvement = (wins[0] / num_games - 0.25) / 0.25 * 100
        print(f"Improvement over random: +{improvement:.1f}%")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python evaluate.py <checkpoint_path> [num_games]")
        print("Example: python evaluate.py final_model.pt 100")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    num_games = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    
    evaluate(checkpoint_path, num_games)
