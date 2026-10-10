"""Hybrid reward shaping for Exploding Kittens.

Rewards are computed PER PLAYER for every environment step, because the events that
matter most to a player (being eliminated, winning) are usually caused by somebody
else's action. The old actor-only reward meant the learner never saw the +1 for a win.
"""

from typing import Dict

from core.game_state import GameState
from core.card_types import ActionType

WIN_REWARD = 1.0
LOSE_REWARD = -1.0
SURVIVE_TURN_REWARD = 0.01
OPPONENT_ELIMINATED_REWARD = 0.2
DEFUSE_REWARD = 0.1


def compute_rewards(
    state: GameState,
    actor: int,
    action: dict,
    done: bool,
    info: dict,
    turn_ended: bool,
) -> Dict[int, float]:
    """Return {player_id: reward} for the transition caused by `actor`'s action.

    - Win: +1.0 (replaces shaping for the winner on that step)
    - Exploded: -1.0 (terminal for that player)
    - Opponent eliminated: +0.2 to every other player still alive
    - Successful defuse: +0.1 to the actor
    - Survived a turn: +0.01 to the actor when its turn ends and it is still alive
    """
    rewards: Dict[int, float] = {}

    def add(player: int, value: float):
        rewards[player] = rewards.get(player, 0.0) + value

    if info.get('exploded'):
        add(info['player'], LOSE_REWARD)
        for p in range(state.num_players):
            if p != info['player'] and state.players_alive[p]:
                add(p, OPPONENT_ELIMINATED_REWARD)

    if action.get('head1') == ActionType.PLAY_DEFUSE and state.players_alive[actor]:
        add(actor, DEFUSE_REWARD)

    if turn_ended and state.players_alive[actor]:
        add(actor, SURVIVE_TURN_REWARD)

    if done and 'winner' in info:
        # Terminal reward dominates any shaping collected on the same step.
        rewards[info['winner']] = WIN_REWARD

    return rewards


def compute_reward(state: GameState, player_id: int, action: dict, done: bool, info: dict) -> float:
    """Backward-compatible actor-only reward (prefer `info['rewards']`)."""
    return compute_rewards(state, player_id, action, done, info, turn_ended=False).get(player_id, 0.0)
