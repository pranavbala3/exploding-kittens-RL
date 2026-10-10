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
        # Everyone's peeked knowledge shifts up by one position.
        state.top_3_visible = [known[1:] + [None] for known in state.top_3_visible]
        
        if card == CardType.EXPLODING_KITTEN:
            state.holding_exploding_kitten[player_id] = True
            
            if state.hands[player_id][CardType.DEFUSE] == 0:
                # Player explodes. Their hand (and the kitten) are revealed and go to the
                # discard pile, so every card stays accounted for publicly.
                state.players_alive[player_id] = False
                state.holding_exploding_kitten[player_id] = False
                state.discard_pile.append(CardType.EXPLODING_KITTEN)
                for c in range(len(state.hands[player_id])):
                    state.discard_pile.extend([c] * int(state.hands[player_id][c]))
                state.hands[player_id][:] = 0
                state.known_counts[player_id][:] = 0
                state.known_zero[player_id][:] = True
                state.have_drawn_this_turn = True   # their turn is over
                
                return -1.0, False, {'exploded': True, 'player': player_id}
            else:
                # Must play Defuse next
                return 0, False, {'drew_kitten': True}
        else:
            # Normal card (only the drawer sees it)
            state.hands[player_id][card] += 1
            self._knowledge_unseen_card_gained(state, player_id)
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
        
        # Attack ends ALL of the attacker's remaining turns (not just one) and the
        # next player owes the 2 attacked turns plus whatever the attacker still owed.
        owed = max(state.turns_remaining[player_id], 1)
        next_player = get_next_alive_player(state, player_id)
        state.turns_remaining[player_id] = 1          # _advance_turn will decrement to 0
        state.turns_remaining[next_player] = owed + 1
        
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
        state.top_3_visible = [[None, None, None] for _ in range(state.num_players)]
        return 0, False, {}