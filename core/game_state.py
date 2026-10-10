from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np


@dataclass
class GameState:
    """Complete game state."""
    num_players: int
    hands: List[np.ndarray]  # Card counts per player
    deck: List[int]  # Card types in deck (index 0 = top)
    discard_pile: List[int]  # Card types in discard
    players_alive: List[bool]
    turns_remaining: List[int]
    current_player: int
    have_drawn_this_turn: bool
    holding_exploding_kitten: List[bool]

    # See the Future knowledge, PER PLAYER: top_3_visible[p] is what player p
    # currently knows about deck[0:3] (None = unknown).
    top_3_visible: List[List[Optional[int]]]

    # PUBLIC knowledge about each player's hand, shared by every observer:
    #   known_counts[p][c] -- copies of card c that player p is publicly known to hold
    #   known_zero[p][c]   -- player p is publicly known to hold NO copies of c
    #                         (cleared when p takes an unseen card)
    known_counts: List[np.ndarray]
    known_zero: List[np.ndarray]

    # Number of copies of each card type in this game (deck + hands + discard).
    card_totals: np.ndarray = field(default_factory=lambda: np.zeros(13, dtype=np.int32))

    # Reactive-Nope window. None when no action is waiting to resolve, otherwise a dict:
    #   actor, action (resolved action dict), cost (cards taken from the actor's hand),
    #   nopes (count played so far), last (player who acted last), queue (responders still to ask)
    pending: Optional[Dict] = None

    def copy(self) -> 'GameState':
        """Create a deep copy of the game state."""
        pending = None
        if self.pending is not None:
            pending = dict(self.pending)
            pending['action'] = dict(self.pending['action'])
            pending['cost'] = list(self.pending['cost'])
            pending['queue'] = list(self.pending['queue'])
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
            top_3_visible=[t.copy() for t in self.top_3_visible],
            known_counts=[k.copy() for k in self.known_counts],
            known_zero=[z.copy() for z in self.known_zero],
            card_totals=self.card_totals.copy(),
            pending=pending,
        )
