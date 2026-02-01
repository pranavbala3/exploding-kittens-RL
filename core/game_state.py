from dataclasses import dataclass
from typing import List, Optional
import numpy as np


@dataclass
class GameState:
    """Complete game state."""
    num_players: int
    hands: List[np.ndarray]  # Card counts per player
    deck: List[int]  # Card types in deck
    discard_pile: List[int]  # Card types in discard
    players_alive: List[bool]
    turns_remaining: List[int]
    current_player: int
    have_drawn_this_turn: bool
    holding_exploding_kitten: List[bool]
    
    # See the Future state
    top_3_visible: List[Optional[int]]
    see_future_active: bool
    
    # Opponent knowledge tracking
    opponent_min_counts: List[np.ndarray]
    opponent_max_counts: List[np.ndarray]
    
    def copy(self) -> 'GameState':
        """Create a deep copy of the game state."""
        return GameState(
            num_players=self.num_players,
            hands=[h.copy() for h in self.hands],
            deck=self.deck.copy(),
            discard_pile=self.discard_pile.copy(),
            players_alive=self.players_alive.copy(),
            turns_remaining=self.turns_remaining.copy(),
            current_player=self.current_player,
            have_drawn_this_turn=self.have_drawn_this_turn,
            holding_exploding_kitten=self.holding_exploding_kitten.copy(),
            top_3_visible=self.top_3_visible.copy(),
            see_future_active=self.see_future_active,
            opponent_min_counts=[m.copy() for m in self.opponent_min_counts],
            opponent_max_counts=[m.copy() for m in self.opponent_max_counts],
        )