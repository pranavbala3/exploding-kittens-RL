from typing import Dict, Tuple
import random
import numpy as np
from handlers.base import ActionHandler
from core.game_state import GameState
from core.card_types import CardType
from utils.env_utils import get_opponent_id


class SeeFutureHandler(ActionHandler):
    """Handler for See the Future card."""
    
    def can_execute(self, state: GameState, player_id: int, action: Dict) -> bool:
        return state.hands[player_id][CardType.SEE_FUTURE] > 0
    
    def execute(self, state: GameState, player_id: int, action: Dict) -> Tuple[float, bool, Dict]:
        state.hands[player_id][CardType.SEE_FUTURE] -= 1
        state.discard_pile.append(CardType.SEE_FUTURE)
        
        # Update knowledge: played See Future
        self._update_knowledge_card_played(state, player_id, CardType.SEE_FUTURE, count=1)
        
        # View top 3 cards
        state.top_3_visible[player_id] = (state.deck[:3] + [None, None, None])[:3]
        return 0, False, {'top_3': list(state.top_3_visible[player_id])}


class FavorHandler(ActionHandler):
    """Handler for Favor card."""
    
    def can_execute(self, state: GameState, player_id: int, action: Dict) -> bool:
        has_card = state.hands[player_id][CardType.FAVOR] > 0
        has_opponents = any(state.players_alive[i] for i in range(state.num_players) if i != player_id)
        return has_card and has_opponents
    
    def execute(self, state: GameState, player_id: int, action: Dict) -> Tuple[float, bool, Dict]:
        state.hands[player_id][CardType.FAVOR] -= 1
        state.discard_pile.append(CardType.FAVOR)
        
        # Update knowledge: played Favor
        self._update_knowledge_card_played(state, player_id, CardType.FAVOR, count=1)
        
        target_id = get_opponent_id(state, player_id, action['head2'])
        
        # Target gives a random card (simplified - should be their choice)
        if np.sum(state.hands[target_id]) > 0:
            available_cards = [i for i in range(13) if state.hands[target_id][i] > 0]
            if available_cards:
                card_to_give = random.choice(available_cards)
                state.hands[target_id][card_to_give] -= 1
                state.hands[player_id][card_to_give] += 1
                
                # Update knowledge: card transfer is PUBLIC
                self._update_knowledge_card_given(state, target_id, player_id, card_to_give)
                
                return 0, False, {'received_card': card_to_give, 'from_player': target_id}
        
        return 0, False, {}


class DefuseHandler(ActionHandler):
    """Handler for Defuse card."""
    
    def can_execute(self, state: GameState, player_id: int, action: Dict) -> bool:
        return (state.hands[player_id][CardType.DEFUSE] > 0 and 
                state.holding_exploding_kitten[player_id])
    
    def execute(self, state: GameState, player_id: int, action: Dict) -> Tuple[float, bool, Dict]:
        state.hands[player_id][CardType.DEFUSE] -= 1
        state.discard_pile.append(CardType.DEFUSE)
        
        # Update knowledge: played Defuse
        self._update_knowledge_card_played(state, player_id, CardType.DEFUSE, count=1)
        
        # Insert Exploding Kitten at specified position
        position = min(action['head5'], len(state.deck))
        state.deck.insert(position, CardType.EXPLODING_KITTEN)
        # Peeked cards at/after the insertion point move down one slot. Only the defuser
        # knows where the kitten went.
        for p, known in enumerate(state.top_3_visible):
            shifted = (known[:position] + [None] + known[position:])[:3]
            if p == player_id and position < 3:
                shifted[position] = CardType.EXPLODING_KITTEN
            state.top_3_visible[p] = shifted
        
        state.holding_exploding_kitten[player_id] = False
        state.have_drawn_this_turn = True
        return 0, False, {'placed_kitten_at': position}


class NopeHandler(ActionHandler):
    """Plays a Nope card during a Nope window.

    The window itself (who may respond, chaining, resolving the pending action) is run by
    ExplodingKittensEnv; this handler only pays for the card.
    """
    
    def can_execute(self, state: GameState, player_id: int, action: Dict) -> bool:
        return state.pending is not None and state.hands[player_id][CardType.NOPE] > 0
    
    def execute(self, state: GameState, player_id: int, action: Dict) -> Tuple[float, bool, Dict]:
        state.hands[player_id][CardType.NOPE] -= 1
        state.discard_pile.append(CardType.NOPE)
        
        # Update knowledge: played Nope
        self._update_knowledge_card_played(state, player_id, CardType.NOPE, count=1)
        
        return 0, False, {}