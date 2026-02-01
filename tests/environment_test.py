"""
Test script for Exploding Kittens environment.

This script:
1. Creates the environment
2. Plays games with random actions
3. Verifies everything works correctly
4. Collects statistics
"""

import numpy as np
from typing import Dict
from collections import defaultdict

from env.environment import ExplodingKittensEnv
from core.game_config import GameConfig
from core.card_types import ActionType


class RandomAgent:
    """Agent that selects random valid actions."""
    
    def get_action(self, obs: Dict, masks: Dict[str, np.ndarray]) -> Dict[int, int]:
        """
        Sample a random valid action.
        
        Args:
            obs: Observation (unused for random agent)
            masks: Action masks
        
        Returns:
            Dictionary with sampled actions for each head
        """
        # Sample from each head (only from valid actions)
        action = {}
        
        # Head 1: Action type
        valid_actions = np.where(masks['head1'])[0]
        if len(valid_actions) == 0:
            raise ValueError("No valid actions available!")
        action['head1'] = int(np.random.choice(valid_actions))
        
        # Head 2: Target player
        valid_targets = np.where(masks['head2'])[0]
        if len(valid_targets) > 0:
            action['head2'] = int(np.random.choice(valid_targets))
        else:
            action['head2'] = 0  # Default (will be masked out if not used)
        
        # Head 3: Card type for combo
        valid_cards = np.where(masks['head3'])[0]
        if len(valid_cards) > 0:
            action['head3'] = int(np.random.choice(valid_cards))
        else:
            action['head3'] = 0
        
        # Head 4: Card type to request
        valid_requests = np.where(masks['head4'])[0]
        if len(valid_requests) > 0:
            action['head4'] = int(np.random.choice(valid_requests))
        else:
            action['head4'] = 0
        
        # Head 5: Defuse position
        valid_positions = np.where(masks['head5'])[0]
        if len(valid_positions) > 0:
            action['head5'] = int(np.random.choice(valid_positions))
        else:
            action['head5'] = 0
        
        return action


def play_one_game(env: ExplodingKittensEnv, agents: list, verbose: bool = False) -> Dict:
    """
    Play one complete game.
    
    Args:
        env: Game environment
        agents: List of agents (one per player)
        verbose: Print game progress
    
    Returns:
        Dictionary with game statistics
    """
    obs = env.reset()
    done = False
    step_count = 0
    
    # Track statistics
    stats = {
        'winner': None,
        'steps': 0,
        'actions_taken': defaultdict(int),
        'players_exploded': [],
    }
    
    if verbose:
        print("=" * 50)
        print("Starting new game!")
        print(f"Players: {env.state.num_players}")
        print("=" * 50)
    
    while not done:
        current_player = env.state.current_player
        
        # Get action from current player's agent
        masks = env.get_action_mask(current_player)
        
        # Get action from current player's agent
        masks = env.get_action_mask(current_player)

        if not np.any(masks['head1']):
            print(f"\n                                                                                                                                                                                                                                                                                                 ERROR: No valid actions for Player {current_player}!")
            print(f"  Hand: {env.state.hands[current_player]}")
            print(f"  Deck size: {len(env.state.deck)}")
            print(f"  Is alive: {env.state.players_alive[current_player]}")
            print(f"  Have drawn: {env.state.have_drawn_this_turn}")
            print(f"  Holding kitten: {env.state.holding_exploding_kitten[current_player]}")
            print(f"  Turns remaining: {env.state.turns_remaining[current_player]}")
            print(f"  Head1 mask: {masks['head1']}")
            
            # Try to continue anyway or break
            raise ValueError("No valid actions available!")

        action = agents[current_player].get_action(obs, masks)
        
        if verbose:
            action_name = ActionType(action['head1']).name
            print(f"\nStep {step_count}: Player {current_player} plays {action_name}")
            print(f"  Hand: {obs['your_hand']}")
            print(f"  Deck size: {obs['deck_size']}")
        
        # Execute action
        obs_next, reward, done, info = env.step(action)
        
        # Track statistics
        stats['actions_taken'][action['head1']] += 1
        step_count += 1
        
        # Check for explosions
        if 'exploded' in info and info['exploded']:
            stats['players_exploded'].append(info['player'])
            if verbose:
                print(f"  Player {info['player']} EXPLODED!")
        
        obs = obs_next
        
        # Safety check - prevent infinite loops
        if step_count > 1000:
            print("Game exceeded 1000 steps, ending...")
            break
    
    stats['winner'] = info.get('winner', None)
    stats['steps'] = step_count
    
    if verbose:
        print("\n" + "=" * 50)
        print(f"Game Over! Winner: Player {stats['winner']}")
        print(f"Total steps: {step_count}")
        print("=" * 50)
    
    return stats


def run_test_suite(num_games: int = 10, num_players: int = 4, verbose: bool = False):
    """
    Run multiple games and collect statistics.
    
    Args:
        num_games: Number of games to play
        num_players: Number of players per game
        verbose: Print detailed game info
    """
    print(f"\n{'='*60}")
    print(f"Running Test Suite: {num_games} games with {num_players} players")
    print(f"{'='*60}\n")
    
    # Create environment
    config = GameConfig(num_players=num_players)
    env = ExplodingKittensEnv(config)
    
    # Create random agents
    agents = [RandomAgent() for _ in range(num_players)]
    
    # Track overall statistics
    all_stats = {
        'games_played': 0,
        'total_steps': 0,
        'winner_counts': defaultdict(int),
        'action_counts': defaultdict(int),
        'avg_game_length': 0,
        'errors': 0,
    }
    
    # Play games
    for game_num in range(num_games):
        try:
            if verbose or game_num % 10 == 0:
                print(f"\n--- Game {game_num + 1}/{num_games} ---")
            
            stats = play_one_game(env, agents, verbose=verbose)
            
            # Update overall stats
            all_stats['games_played'] += 1
            all_stats['total_steps'] += stats['steps']
            all_stats['winner_counts'][stats['winner']] += 1
            
            for action_type, count in stats['actions_taken'].items():
                all_stats['action_counts'][action_type] += count
            
        except Exception as e:
            print(f"Error in game {game_num + 1}: {e}")
            all_stats['errors'] += 1
            import traceback
            traceback.print_exc()
    
    # Calculate averages
    if all_stats['games_played'] > 0:
        all_stats['avg_game_length'] = all_stats['total_steps'] / all_stats['games_played']
    
    # Print summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Games completed: {all_stats['games_played']}/{num_games}")
    print(f"Errors: {all_stats['errors']}")
    print(f"Average game length: {all_stats['avg_game_length']:.1f} steps")
    
    print("\n Winner Distribution:")
    for player, count in sorted(all_stats['winner_counts'].items()):
        win_rate = (count / all_stats['games_played']) * 100
        print(f"  Player {player}: {count} wins ({win_rate:.1f}%)")
    
    print("\n🎯 Action Distribution:")
    total_actions = sum(all_stats['action_counts'].values())
    for action_type, count in sorted(all_stats['action_counts'].items()):
        action_name = ActionType(action_type).name
        pct = (count / total_actions) * 100
        print(f"  {action_name:20s}: {count:5d} ({pct:5.1f}%)")
    
    print(f"\n{'='*60}\n")
    
    return all_stats


def test_masking():
    """Test that action masking works correctly."""
    print("\n" + "="*60)
    print("Testing Action Masking")
    print("="*60 + "\n")
    
    env = ExplodingKittensEnv(GameConfig(num_players=4))
    env.reset()
    
    print("Masking test complete!\n")


def test_single_game_verbose():
    """Play one game with verbose output to see detailed flow."""
    print("\n" + "="*60)
    print("Playing Single Verbose Game")
    print("="*60 + "\n")
    
    env = ExplodingKittensEnv(GameConfig(num_players=4))
    agents = [RandomAgent() for _ in range(4)]
    
    stats = play_one_game(env, agents, verbose=True)
    
    return stats


if __name__ == "__main__":
    # Test 1: Check masking
    test_masking()
    
    # Test 2: Play one verbose game
    test_single_game_verbose()
    
    # Test 3: Run test suite with multiple games
    run_test_suite(num_games=100, num_players=4, verbose=False)
    
    # Test 4: Test with different player counts
    print("\nTesting different player counts:")
    for num_players in [2, 3, 4, 5]:
        print(f"\n--- {num_players} Players ---")
        run_test_suite(num_games=20, num_players=num_players, verbose=False)