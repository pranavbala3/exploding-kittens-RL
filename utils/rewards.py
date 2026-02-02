"""Hybrid reward shaping for Exploding Kittens."""

from core.game_state import GameState
from core.card_types import ActionType


def compute_reward(state: GameState, player_id: int, action: dict, done: bool, info: dict) -> float:
    """
    Hybrid reward: Terminal + small progress rewards.
    
    Returns:
        - Win: +1.0
        - Lose: -1.0
        - Survival: +0.01 per turn
        - Opponent eliminated: +0.2
        - Successfully defuse: +0.1
    """
    # Terminal rewards (dominant)
    if done and 'winner' in info:
        return 1.0 if info['winner'] == player_id else -1.0
    
    reward = 0.0
    
    # Survival reward
    if state.players_alive[player_id]:
        reward += 0.01
    
    # Opponent elimination bonus
    if 'exploded' in info and info.get('player') != player_id:
        reward += 0.2
    
    # Successful defuse
    if action.get('head1') == ActionType.PLAY_DEFUSE and state.players_alive[player_id]:
        reward += 0.1
    
    return reward
