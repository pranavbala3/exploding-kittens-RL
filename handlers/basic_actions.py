from typing import Dict, Tuple
import random
from handlers.base import ActionHandler
from core.game_state import GameState
from core.card_types import CardType
from utils.env_utils import get_next_alive_player

class DrawHandler(ActionHandler):
    """Handler for drawing a card."""
    
    def can_execute(self, state: GameState, player_id: int, action: Dict) -> bool:
        return not state.have_drawn_this_turn and len(state.deck) > 0
    
    def execute(self, state: GameState, player_id: int, action: Dict) -> Tuple[float, bool, Dict]:
        if not state.deck:
            return 0, False, {'error': 'Deck is empty'}
        
        card = state.deck.pop(0)
        
        if card == CardType.EXPLODING_KITTEN:
            state.holding_exploding_kitten[player_id] = True
            
            if state.hands[player_id][CardType.DEFUSE] == 0:
                # Player explodes!
                state.players_alive[player_id] = False
                
                # Update knowledge
                for observer_id in range(state.num_players):
                    if observer_id != player_id:
                        state.opponent_min_counts[observer_id][CardType.DEFUSE] = 0
                        state.opponent_max_counts[observer_id][CardType.DEFUSE] = 0
                
                # IMPORTANT: Don't continue their turn!
                state.have_drawn_this_turn = True  # ← ADD THIS LINE
                
                return -1.0, False, {'exploded': True, 'player': player_id}
            else:
                # Must play Defuse next
                return 0, False, {'drew_kitten': True}
        else:
            # Normal card
            state.hands[player_id][card] += 1
            state.have_drawn_this_turn = True
            return 0, False, {'drew_card': card}


class SkipHandler(ActionHandler):
    """Handler for Skip card."""
    
    def can_execute(self, state: GameState, player_id: int, action: Dict) -> bool:
        return state.hands[player_id][CardType.SKIP] > 0
    
    def execute(self, state: GameState, player_id: int, action: Dict) -> Tuple[float, bool, Dict]:
        state.hands[player_id][CardType.SKIP] -= 1
        state.discard_pile.append(CardType.SKIP)
        
        # Update knowledge: played Skip
        self._update_knowledge_card_played(state, player_id, CardType.SKIP, count=1)
        
        state.have_drawn_this_turn = True
        return 0, False, {}


class AttackHandler(ActionHandler):
    """Handler for Attack card."""
    
    def can_execute(self, state: GameState, player_id: int, action: Dict) -> bool:
        return state.hands[player_id][CardType.ATTACK] > 0
    
    def execute(self, state: GameState, player_id: int, action: Dict) -> Tuple[float, bool, Dict]:
        state.hands[player_id][CardType.ATTACK] -= 1
        state.discard_pile.append(CardType.ATTACK)
        
        # Update knowledge: played Attack
        self._update_knowledge_card_played(state, player_id, CardType.ATTACK, count=1)
        
        # Next player gets +1 turn
        next_player = get_next_alive_player(state, player_id)
        state.turns_remaining[next_player] += 1
        
        state.have_drawn_this_turn = True
        return 0, False, {'attacked_player': next_player}


class ShuffleHandler(ActionHandler):
    """Handler for Shuffle card."""
    
    def can_execute(self, state: GameState, player_id: int, action: Dict) -> bool:
        return state.hands[player_id][CardType.SHUFFLE] > 0
    
    def execute(self, state: GameState, player_id: int, action: Dict) -> Tuple[float, bool, Dict]:
        state.hands[player_id][CardType.SHUFFLE] -= 1
        state.discard_pile.append(CardType.SHUFFLE)
        
        # Update knowledge: played Shuffle
        self._update_knowledge_card_played(state, player_id, CardType.SHUFFLE, count=1)
        
        random.shuffle(state.deck)
        
        # Invalidate See the Future
        state.see_future_active = False
        state.top_3_visible = [None, None, None]
        return 0, False, {}