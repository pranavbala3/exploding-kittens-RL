from abc import ABC, abstractmethod
from typing import Dict, Tuple
from core.game_state import GameState


class ActionHandler(ABC):
    """Base class for action handlers."""

    @abstractmethod
    def can_execute(self, state: GameState, player_id: int, action: Dict) -> bool:
        """Check if action can be executed."""

    @abstractmethod
    def execute(self, state: GameState, player_id: int, action: Dict) -> Tuple[float, bool, Dict]:
        """
        Execute the action.

        Returns:
            reward: Immediate reward for this action (unused; see utils/rewards.py)
            done: Whether the game is over
            info: Additional information dictionary
        """

    # ------------------------------------------------------------------ #
    # PUBLIC KNOWLEDGE TRACKING
    #
    # state.known_counts[p][c] is a lower bound, known to EVERY player, on how many
    # copies of card c player p holds. state.known_zero[p][c] marks "p is known to hold
    # none". Upper bounds are derived per observer in ExplodingKittensEnv.get_observation.
    # ------------------------------------------------------------------ #

    @staticmethod
    def _knowledge_card_left_hand(state: GameState, player_id: int, card_type: int, count: int = 1):
        """Player p publicly played/discarded `count` copies of `card_type`."""
        known = state.known_counts[player_id]
        known[card_type] = max(int(known[card_type]) - count, 0)

    @staticmethod
    def _knowledge_unseen_card_gained(state: GameState, player_id: int):
        """Player p took a card nobody else saw (a draw): 'known to hold none' no longer holds."""
        state.known_zero[player_id][:] = False

    @staticmethod
    def _knowledge_card_transferred(state: GameState, giver_id: int, receiver_id: int, card_type: int):
        """A card moved between two players in view of the table."""
        ActionHandler._knowledge_card_left_hand(state, giver_id, card_type, 1)
        state.known_counts[receiver_id][card_type] += 1
        state.known_zero[receiver_id][card_type] = False

    @staticmethod
    def _knowledge_holds_none(state: GameState, player_id: int, card_type: int):
        """Player p was shown to hold no copies of `card_type` (failed 3-of-a-kind request)."""
        state.known_counts[player_id][card_type] = 0
        state.known_zero[player_id][card_type] = True

    # Backward-compatible names used by older handler code.
    def _update_knowledge_card_played(self, state, player_id, card_type, count=1):
        self._knowledge_card_left_hand(state, player_id, card_type, count)

    def _update_knowledge_combo_played(self, state, player_id, card_type, combo_size):
        self._knowledge_card_left_hand(state, player_id, card_type, combo_size)

    def _update_knowledge_card_given(self, state, giver_id, receiver_id, card_type):
        self._knowledge_card_transferred(state, giver_id, receiver_id, card_type)

    def _update_knowledge_request_failed(self, state, requester_id, target_id, card_type):
        self._knowledge_holds_none(state, target_id, card_type)

    def _update_knowledge_request_succeeded(self, state, requester_id, target_id, card_type):
        self._knowledge_card_transferred(state, target_id, requester_id, card_type)
