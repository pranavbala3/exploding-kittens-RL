from typing import Dict, Tuple
import random
import numpy as np
from handlers.base import ActionHandler
from core.game_state import GameState
from utils.env_utils import get_opponent_id


def resolve_combo_card(hand, requested_card: int, size: int):
    """Return a card type with >= `size` copies, preferring the one the policy asked for.

    The head-3 mask is shared by the 2- and 3-of-a-kind actions (it only knows >=2),
    so a 3-of-a-kind can arrive with a card the player only holds two of. Repair it to
    the first card type that does have `size` copies instead of rejecting a masked action.
    """
    if hand[requested_card] >= size:
        return requested_card
    for card in range(len(hand)):
        if hand[card] >= size:
            return card
    return None


class TwoOfAKindHandler(ActionHandler):
    """Handler for 2-of-a-kind combo."""
    
    def can_execute(self, state: GameState, player_id: int, action: Dict) -> bool:
        has_cards = resolve_combo_card(state.hands[player_id], action.get('head3', 0), 2) is not None
        has_opponents = any(state.players_alive[i] for i in range(state.num_players) if i != player_id)
        return has_cards and has_opponents
    
    def execute(self, state: GameState, player_id: int, action: Dict) -> Tuple[float, bool, Dict]:
        card_type = resolve_combo_card(state.hands[player_id], action['head3'], 2)
        target_id = get_opponent_id(state, player_id, action['head2'])
        
        # Remove 2 cards
        state.hands[player_id][card_type] -= 2
        state.discard_pile.extend([card_type, card_type])
        
        # Update knowledge: played 2-of-a-kind combo
        self._update_knowledge_combo_played(state, player_id, card_type, combo_size=2)
        
        # Steal random card
        if np.sum(state.hands[target_id]) > 0:
            available_cards = [i for i in range(13) if state.hands[target_id][i] > 0]
            if available_cards:
                stolen_card = random.choice(available_cards)
                state.hands[target_id][stolen_card] -= 1
                state.hands[player_id][stolen_card] += 1
                
                # Update knowledge: card transfer is PUBLIC
                self._update_knowledge_card_given(state, target_id, player_id, stolen_card)
                
                return 0, False, {'stolen_card': stolen_card, 'from_player': target_id}
        
        return 0, False, {}


class ThreeOfAKindHandler(ActionHandler):
    """Handler for 3-of-a-kind combo."""
    
    def can_execute(self, state: GameState, player_id: int, action: Dict) -> bool:
        has_cards = resolve_combo_card(state.hands[player_id], action.get('head3', 0), 3) is not None
        has_opponents = any(state.players_alive[i] for i in range(state.num_players) if i != player_id)
        return has_cards and has_opponents
    
    def execute(self, state: GameState, player_id: int, action: Dict) -> Tuple[float, bool, Dict]:
        card_type = resolve_combo_card(state.hands[player_id], action['head3'], 3)
        requested_card = action['head4']
        target_id = get_opponent_id(state, player_id, action['head2'])
        
        # Remove 3 cards
        state.hands[player_id][card_type] -= 3
        state.discard_pile.extend([card_type, card_type, card_type])
        
        # Update knowledge: played 3-of-a-kind combo
        self._update_knowledge_combo_played(state, player_id, card_type, combo_size=3)
        
        # Try to take requested card
        if state.hands[target_id][requested_card] > 0:
            state.hands[target_id][requested_card] -= 1
            state.hands[player_id][requested_card] += 1
            
            # Update knowledge: request succeeded
            self._update_knowledge_request_succeeded(state, player_id, target_id, requested_card)
            
            return 0, False, {'got_card': True, 'card': requested_card, 'from_player': target_id}
        else:
            # CRITICAL: Request failed - we KNOW they have 0!
            self._update_knowledge_request_failed(state, player_id, target_id, requested_card)
            
            return 0, False, {'got_card': False}