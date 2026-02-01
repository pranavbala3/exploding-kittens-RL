from typing import Dict, Tuple, Optional
import numpy as np
import random

from core.card_types import CardType, ActionType
from core.game_config import GameConfig
from core.game_state import GameState
from handlers.basic_actions import DrawHandler, SkipHandler, AttackHandler, ShuffleHandler
from handlers.special_actions import SeeFutureHandler, FavorHandler, DefuseHandler, NopeHandler
from handlers.combo_actions import TwoOfAKindHandler, ThreeOfAKindHandler
from utils.env_utils import get_next_alive_player


class ExplodingKittensEnv:
    """Main Exploding Kittens environment with modular action handlers."""
    
    def __init__(self, config: Optional[GameConfig] = None):
        self.config = config or GameConfig()
        self.config.validate()
        
        # Register action handlers
        self.action_handlers = {
            ActionType.DRAW: DrawHandler(),
            ActionType.PLAY_SKIP: SkipHandler(),
            ActionType.PLAY_ATTACK: AttackHandler(),
            ActionType.PLAY_SHUFFLE: ShuffleHandler(),
            ActionType.PLAY_SEE_FUTURE: SeeFutureHandler(),
            ActionType.PLAY_FAVOR: FavorHandler(),
            ActionType.PLAY_DEFUSE: DefuseHandler(),
            ActionType.PLAY_NOPE: NopeHandler(),
            ActionType.PLAY_2_OF_A_KIND: TwoOfAKindHandler(),
            ActionType.PLAY_3_OF_A_KIND: ThreeOfAKindHandler(),
        }
        
        self.state = None
        self.reset()
    
    def reset(self) -> Dict:
        """Reset the game to initial state."""
        num_players = self.config.num_players
        
        self.state = GameState(
            num_players=num_players,
            hands=[np.zeros(13, dtype=np.int32) for _ in range(num_players)],
            deck=[],
            discard_pile=[],
            players_alive=[True] * num_players,
            turns_remaining=[1] * num_players,
            current_player=0,
            have_drawn_this_turn=False,
            holding_exploding_kitten=[False] * num_players,
            top_3_visible=[None, None, None],
            see_future_active=False,
            opponent_min_counts=[np.zeros(13, dtype=np.int32) for _ in range(num_players)],
            opponent_max_counts=[np.full(13, 10, dtype=np.int32) for _ in range(num_players)],
        )
        
        self._setup_deck()
        return self.get_observation(self.state.current_player)
    
    def _setup_deck(self):
        """Setup deck according to official rules."""
        temp_deck = []
        
        # Add all non-Exploding/Defuse cards
        for card_type, count in self.config.card_counts.items():
            if card_type not in [CardType.EXPLODING_KITTEN, CardType.DEFUSE]:
                temp_deck.extend([card_type] * count)
        
        # Deal 1 Defuse to each player
        for i in range(self.state.num_players):
            self.state.hands[i][CardType.DEFUSE] = 1
        
        # Shuffle and deal cards
        random.shuffle(temp_deck)
        for i in range(self.state.num_players):
            for _ in range(self.config.cards_per_player):
                if temp_deck:
                    card = temp_deck.pop()
                    self.state.hands[i][card] += 1
        
        # Insert Exploding Kittens
        num_kittens = self.state.num_players - 1
        temp_deck.extend([CardType.EXPLODING_KITTEN] * num_kittens)
        
        # Add remaining Defuses
        remaining_defuses = 2 if self.state.num_players < 5 else 1
        temp_deck.extend([CardType.DEFUSE] * remaining_defuses)
        
        random.shuffle(temp_deck)
        self.state.deck = temp_deck
    
    def get_observation(self, player_id: int) -> Dict:
        """Get observation for a specific player."""
        obs = {
            'your_hand': self.state.hands[player_id].copy(),
            'opponents': [],
            'deck_size': len(self.state.deck),
            'exploding_kittens_in_deck': self.state.deck.count(CardType.EXPLODING_KITTEN),
            'discard_counts': self._get_discard_counts(),
            'have_drawn_this_turn': self.state.have_drawn_this_turn,
            'holding_exploding_kitten': self.state.holding_exploding_kitten[player_id],
            'top_3_cards': self.state.top_3_visible.copy() if self.state.see_future_active else [None, None, None],
            'see_future_active': self.state.see_future_active,
        }
        
        # Add opponent information
        for offset in range(1, self.state.num_players):
            opponent_id = (player_id + offset) % self.state.num_players
            
            obs['opponents'].append({
                'num_cards': int(np.sum(self.state.hands[opponent_id])),
                'is_alive': self.state.players_alive[opponent_id],
                'turns_remaining': self.state.turns_remaining[opponent_id],
                'min_counts': self.state.opponent_min_counts[opponent_id].copy(),
                'max_counts': self.state.opponent_max_counts[opponent_id].copy(),
            })
        
        # Pad to 4 opponents
        while len(obs['opponents']) < 4:
            obs['opponents'].append({
                'num_cards': 0,
                'is_alive': False,
                'turns_remaining': 0,
                'min_counts': np.zeros(13, dtype=np.int32),
                'max_counts': np.zeros(13, dtype=np.int32),
            })
        
        return obs
    
    def _get_discard_counts(self) -> np.ndarray:
        """Get counts of each card type in discard pile."""
        counts = np.zeros(13, dtype=np.int32)
        for card in self.state.discard_pile:
            counts[card] += 1
        return counts
    
    def get_action_mask(self, player_id: int) -> Dict[str, np.ndarray]:
        """Get valid action masks for each head."""
        hand = self.state.hands[player_id]
        
        # If holding exploding kitten, then must use defuse. Else, game over :(
        if self.state.holding_exploding_kitten[player_id]:
            head1_mask = np.zeros(10, dtype=bool)
            
            if hand[CardType.DEFUSE] > 0:
                # Has Defuse - must play it
                head1_mask[ActionType.PLAY_DEFUSE] = True
            else:
                # No Defuse - player should have already exploded
                # This shouldn't happen, but return empty mask
                pass
            
            return {
                'head1': head1_mask,
                'head2': np.zeros(4, dtype=bool),
                'head3': np.zeros(13, dtype=bool),
                'head4': np.ones(13, dtype=bool),
                'head5': np.ones(self.config.max_deck_size, dtype=bool)[:self.config.max_deck_size],
            }

        # Head 1: Primary action mask
        head1_mask = np.zeros(10, dtype=bool)
        
        # Simple actions - check directly (no dummy action needed)
        head1_mask[ActionType.DRAW] = (
            not self.state.have_drawn_this_turn and len(self.state.deck) > 0
        )
        head1_mask[ActionType.PLAY_SKIP] = hand[CardType.SKIP] > 0
        head1_mask[ActionType.PLAY_ATTACK] = hand[CardType.ATTACK] > 0
        head1_mask[ActionType.PLAY_SHUFFLE] = hand[CardType.SHUFFLE] > 0
        head1_mask[ActionType.PLAY_SEE_FUTURE] = hand[CardType.SEE_FUTURE] > 0
        head1_mask[ActionType.PLAY_FAVOR] = (
            hand[CardType.FAVOR] > 0 and 
            self._has_alive_opponents(player_id)
        )
        head1_mask[ActionType.PLAY_DEFUSE] = (
            hand[CardType.DEFUSE] > 0 and 
            self.state.holding_exploding_kitten[player_id]
        )
        head1_mask[ActionType.PLAY_NOPE] = hand[CardType.NOPE] > 0
        
        # Combo actions - check if ANY card type works
        head1_mask[ActionType.PLAY_2_OF_A_KIND] = (
            any(hand[i] >= 2 for i in range(13)) and 
            self._has_alive_opponents(player_id)
        )
        head1_mask[ActionType.PLAY_3_OF_A_KIND] = (
            any(hand[i] >= 3 for i in range(13)) and 
            self._has_alive_opponents(player_id)
        )
        
        # Head 2: Target player mask (in turn order!)
        head2_mask = []
        for offset in range(1, self.state.num_players):
            opponent_id = (player_id + offset) % self.state.num_players
            head2_mask.append(self.state.players_alive[opponent_id])
        
        head2_mask.extend([False] * (4 - len(head2_mask)))
        head2_mask = np.array(head2_mask, dtype=bool)
        
        # Head 3: Card type for combo
        head3_mask = np.array([hand[i] >= 2 for i in range(13)], dtype=bool)
        
        # Head 4: Card type to request
        head4_mask = np.ones(13, dtype=bool)
        
        # Head 5: Defuse position
        head5_mask = np.zeros(self.config.max_deck_size, dtype=bool)
        head5_mask[:len(self.state.deck) + 1] = True
        
        return {
            'head1': head1_mask,
            'head2': head2_mask,
            'head3': head3_mask,
            'head4': head4_mask,
            'head5': head5_mask,
        }

    def _has_alive_opponents(self, player_id: int) -> bool:
        """Check if player has any alive opponents."""
        return any(self.state.players_alive[i] for i in range(self.state.num_players) if i != player_id)
        
    def step(self, action: Dict) -> Tuple[Dict, float, bool, Dict]:
        """Execute an action and return next observation."""
        player_id = self.state.current_player
        action_type = action['head1']
        
        handler = self.action_handlers.get(action_type)
        if handler is None:
            return self.get_observation(player_id), 0, False, {'error': 'Invalid action type'}
        
        if not handler.can_execute(self.state, player_id, action):
            return self.get_observation(player_id), 0, False, {'error': 'Cannot execute action'}
        
        reward, done, info = handler.execute(self.state, player_id, action)
        
        # CHECK WIN CONDITION IMMEDIATELY AFTER EXECUTION
        alive_count = sum(self.state.players_alive)
        if alive_count == 1:
            done = True
            winner = [i for i, alive in enumerate(self.state.players_alive) if alive][0]
            reward = 1.0 if player_id == winner else 0
            info['winner'] = winner
            return self.get_observation(winner), reward, done, info
        
        # Advance turn if needed
        if self.state.have_drawn_this_turn or action_type in [ActionType.PLAY_SKIP, ActionType.PLAY_ATTACK, ActionType.PLAY_DEFUSE]:
            self._advance_turn()
        
        # Check win condition again (in case advancing turn revealed winner)
        alive_count = sum(self.state.players_alive)
        if alive_count == 1:
            done = True
            winner = [i for i, alive in enumerate(self.state.players_alive) if alive][0]
            if self.state.players_alive[player_id]:
                reward = 1.0
            info['winner'] = winner
        
        return self.get_observation(self.state.current_player), reward, done, info

    def _advance_turn(self):
        """Advance to next player's turn."""
        
        # Decrement current player's remaining turns
        self.state.turns_remaining[self.state.current_player] -= 1
        
        # Keep moving to next player while current has no turns OR is dead
        while (self.state.turns_remaining[self.state.current_player] == 0 or
            not self.state.players_alive[self.state.current_player]):
            
            self.state.current_player = get_next_alive_player(self.state, self.state.current_player)
            
            # Give new player turns if needed
            if self.state.turns_remaining[self.state.current_player] == 0:
                self.state.turns_remaining[self.state.current_player] = 1
        
        self.state.have_drawn_this_turn = False