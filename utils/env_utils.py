from core.game_state import GameState


def get_next_alive_player(state: GameState, current: int) -> int:
    """Get next alive player in turn order."""
    next_player = (current + 1) % state.num_players
    while not state.players_alive[next_player]:
        next_player = (next_player + 1) % state.num_players
    return next_player


def get_opponent_id(state: GameState, player_id: int, opponent_idx: int) -> int:
    """Convert a head-2 slot to a player ID.

    Slot k is the opponent k+1 seats after `player_id` -- the same order used for the
    opponent block of the observation, so the target head attends to the right player.
    A slot pointing at an eliminated/absent player falls back to the next alive opponent.
    """
    if 0 <= opponent_idx < state.num_players - 1:
        candidate = (player_id + 1 + opponent_idx) % state.num_players
        if state.players_alive[candidate]:
            return candidate
    return get_next_alive_player(state, player_id)
