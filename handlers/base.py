from abc import ABC, abstractmethod
from typing import Dict, Tuple
from core.game_state import GameState


class ActionHandler(ABC):
    """Base class for action handlers."""
    
    @abstractmethod
    def can_execute(self, state: GameState, player_id: int, action: Dict) -> bool:
        """Check if action can be executed."""
        pass
    
    @abstractmethod
    def execute(self, state: GameState, player_id: int, action: Dict) -> Tuple[float, bool, Dict]:
        """
        Execute the action.
        
        Returns:
            reward: Immediate reward for this action
            done: Whether the game is over
            info: Additional information dictionary
        """
        pass
    
    # KNOWLEDGE TRACKING HELPERS
    # These are not abstract - handlers can use them as needed
    
    def _update_knowledge_card_played(
        self, 
        state: GameState, 
        player_id: int, 
        card_type: int, 
        count: int = 1
    ):
        """
        Update all opponents' knowledge when a card is played.
        
        When a player plays a card publicly, all opponents learn they had
        at least that many of that card type.
        
        Args:
            state: Game state
            player_id: Player who played the card
            card_type: Type of card played
            count: Number of cards played (default 1)
        """
        for observer_id in range(state.num_players):
            if observer_id == player_id:
                continue
            state.opponent_min_counts[observer_id][card_type] = max(
                state.opponent_min_counts[observer_id][card_type],
                count
            )
    
    def _update_knowledge_card_given(
        self,
        state: GameState,
        giver_id: int,
        receiver_id: int,
        card_type: int
    ):
        """
        Update knowledge when a card is publicly transferred.
        
        Used for Favor and stealing (combos). All players can observe
        the transfer and update their beliefs.
        
        Args:
            state: Game state
            giver_id: Player who gave/lost the card
            receiver_id: Player who received the card
            card_type: Type of card transferred
        """
        for observer_id in range(state.num_players):
            if observer_id == giver_id:
                # Giver learns receiver now has at least 1
                state.opponent_min_counts[giver_id][card_type] = max(
                    state.opponent_min_counts[giver_id][card_type], 1
                )
            elif observer_id == receiver_id:
                # Receiver learns giver had at least 1
                # (tracking about the giver from receiver's perspective)
                pass
    
    def _update_knowledge_combo_played(
        self,
        state: GameState,
        player_id: int,
        card_type: int,
        combo_size: int
    ):
        """
        Update knowledge when a combo (2 or 3 of a kind) is played.
        
        Observers know the player had EXACTLY combo_size of this card
        (since they played them all at once).
        
        Args:
            state: Game state
            player_id: Player who played the combo
            card_type: Type of card used in combo
            combo_size: Number of cards (2 or 3)
        """
        for observer_id in range(state.num_players):
            if observer_id == player_id:
                continue
            state.opponent_min_counts[observer_id][card_type] = combo_size
    
    def _update_knowledge_request_failed(
        self,
        state: GameState,
        requester_id: int,
        target_id: int,
        card_type: int
    ):
        """
        Update when 3-of-a-kind request fails (target doesn't have the card).
        
        This is VERY valuable information - we now KNOW with certainty
        that the target has 0 of this card type!
        
        Args:
            state: Game state
            requester_id: Player who requested the card
            target_id: Player who was asked (and doesn't have it)
            card_type: Type of card that was requested
        """
        state.opponent_min_counts[requester_id][card_type] = 0
        state.opponent_max_counts[requester_id][card_type] = 0
    
    def _update_knowledge_request_succeeded(
        self,
        state: GameState,
        requester_id: int,
        target_id: int,
        card_type: int
    ):
        """
        Update when 3-of-a-kind request succeeds (target had the card).
        
        Requester learns target had at least 1 of this card.
        Target learns requester now has at least 1.
        
        Args:
            state: Game state
            requester_id: Player who requested and received the card
            target_id: Player who gave the card
            card_type: Type of card that was transferred
        """
        # Requester learns target had at least 1
        state.opponent_min_counts[requester_id][card_type] = max(
            state.opponent_min_counts[requester_id][card_type], 1
        )
        # Target learns requester now has at least 1
        state.opponent_min_counts[target_id][card_type] = max(
            state.opponent_min_counts[target_id][card_type], 1
        )