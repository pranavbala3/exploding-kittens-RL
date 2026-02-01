from core.game_state import GameState


def get_next_alive_player(state: GameState, current: int) -> int:
    """Get next alive player in turn order."""
    next_player = (current + 1) % state.num_players
    while not state.players_alive[next_player]:
        next_player = (next_player + 1) % state.num_players
    return next_player


def get_opponent_id(state: GameState, player_id: int, opponent_idx: int) -> int:
    """Convert opponent index (0-3) to actual player ID."""
    opponents = [i for i in range(state.num_players) if i != player_id and state.players_alive[i]]
    return opponents[opponent_idx] if opponent_idx < len(opponents) else opponents[0]
