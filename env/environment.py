from typing import Dict, List, Optional, Tuple
import numpy as np
import random

from core.card_types import CardType, ActionType, NOPEABLE_ACTIONS, NUM_ACTION_TYPES, NUM_CARD_TYPES
from core.game_config import GameConfig
from core.game_state import GameState
from handlers.basic_actions import DrawHandler, SkipHandler, AttackHandler, ShuffleHandler
from handlers.special_actions import SeeFutureHandler, FavorHandler, DefuseHandler, NopeHandler
from handlers.combo_actions import TwoOfAKindHandler, ThreeOfAKindHandler, resolve_combo_card
from utils.env_utils import get_next_alive_player, get_opponent_id
from utils.rewards import compute_rewards

# Card paid for each single-card action.
ACTION_CARD = {
    ActionType.PLAY_SKIP: CardType.SKIP,
    ActionType.PLAY_ATTACK: CardType.ATTACK,
    ActionType.PLAY_SHUFFLE: CardType.SHUFFLE,
    ActionType.PLAY_SEE_FUTURE: CardType.SEE_FUTURE,
    ActionType.PLAY_FAVOR: CardType.FAVOR,
}


class ExplodingKittensEnv:
    """Main Exploding Kittens environment with modular action handlers.

    Turn flow
    ---------
    The player whose turn it is plays any number of action cards, then ends the turn by
    drawing (or by Skip / Attack, or by Defusing a drawn kitten).

    Reactive Nope
    -------------
    Playing a Noppable action (see NOPEABLE_ACTIONS) pays for the card(s) immediately but
    opens a Nope window before the effect happens. Every other alive player holding a Nope
    is asked in turn order (``current_player`` points at the responder) and chooses
    PLAY_NOPE or PASS. A Nope restarts the round of asking from the player after the
    noper, so Nopes can be Noped. When everyone asked has passed the action resolves: it
    takes effect if an even number of Nopes were played and is cancelled otherwise. Play
    then returns to the actor.
    """

    def __init__(self, config: Optional[GameConfig] = None):
        self.config = config or GameConfig()
        self.config.validate()

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

    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #

    def reset(self) -> Dict:
        """Reset the game to initial state."""
        n = self.config.num_players

        self.state = GameState(
            num_players=n,
            hands=[np.zeros(NUM_CARD_TYPES, dtype=np.int32) for _ in range(n)],
            deck=[],
            discard_pile=[],
            players_alive=[True] * n,
            turns_remaining=[1] + [0] * (n - 1),
            current_player=0,
            have_drawn_this_turn=False,
            holding_exploding_kitten=[False] * n,
            top_3_visible=[[None, None, None] for _ in range(n)],
            known_counts=[np.zeros(NUM_CARD_TYPES, dtype=np.int32) for _ in range(n)],
            known_zero=[np.zeros(NUM_CARD_TYPES, dtype=bool) for _ in range(n)],
        )

        self._setup_deck()
        return self.get_observation(self.state.current_player)

    def _setup_deck(self):
        """Setup deck according to official rules."""
        temp_deck = []

        for card_type, count in self.config.card_counts.items():
            if card_type not in [CardType.EXPLODING_KITTEN, CardType.DEFUSE]:
                temp_deck.extend([card_type] * count)

        # Deal 1 Defuse to each player -- this is public knowledge.
        for i in range(self.state.num_players):
            self.state.hands[i][CardType.DEFUSE] = 1
            self.state.known_counts[i][CardType.DEFUSE] = 1

        random.shuffle(temp_deck)
        for i in range(self.state.num_players):
            for _ in range(self.config.cards_per_player):
                if temp_deck:
                    self.state.hands[i][temp_deck.pop()] += 1

        temp_deck.extend([CardType.EXPLODING_KITTEN] * (self.state.num_players - 1))
        remaining_defuses = 2 if self.state.num_players < 5 else 1
        temp_deck.extend([CardType.DEFUSE] * remaining_defuses)

        random.shuffle(temp_deck)
        self.state.deck = temp_deck

        totals = np.bincount(temp_deck, minlength=NUM_CARD_TYPES).astype(np.int32)
        for hand in self.state.hands:
            totals += hand
        self.state.card_totals = totals

    # ------------------------------------------------------------------ #
    # Observation
    # ------------------------------------------------------------------ #

    def turn_owner(self) -> int:
        """Player whose turn it is (differs from current_player during a Nope window)."""
        return self.state.pending['actor'] if self.state.pending else self.state.current_player

    def _card_upper_bounds(self, observer: int) -> List[np.ndarray]:
        """Upper bound on every player's count of each card type, from `observer`'s view.

        A player holds at most what they are known to hold plus as many as could fit in
        their unknown hand slots, limited by the copies `observer` cannot locate.
        """
        s = self.state
        accounted = self._get_discard_counts() + s.hands[observer]
        for card in s.top_3_visible[observer]:
            if card is not None:
                accounted[card] += 1
        if s.pending is not None:
            for card in s.pending['cost']:
                accounted[card] += 1
        for p in range(s.num_players):
            if p != observer:
                accounted = accounted + s.known_counts[p]
        unaccounted = np.maximum(s.card_totals - accounted, 0)

        bounds = []
        for p in range(s.num_players):
            known = s.known_counts[p]
            unknown_slots = max(int(np.sum(s.hands[p])) - int(np.sum(known)), 0)
            upper = known + np.minimum(unaccounted, unknown_slots)
            upper[s.known_zero[p]] = 0
            upper[CardType.EXPLODING_KITTEN] = 0      # kittens are never kept in hand
            bounds.append(upper.astype(np.int32))
        return bounds

    def _pending_target(self) -> Optional[int]:
        pending = self.state.pending
        if pending is None:
            return None
        a_type = pending['action']['head1']
        if a_type == ActionType.PLAY_ATTACK:
            return get_next_alive_player(self.state, pending['actor'])
        if a_type in (ActionType.PLAY_FAVOR, ActionType.PLAY_2_OF_A_KIND, ActionType.PLAY_3_OF_A_KIND):
            return get_opponent_id(self.state, pending['actor'], pending['action']['head2'])
        return None

    def get_observation(self, player_id: int) -> Dict:
        """Get observation for a specific player."""
        s = self.state
        pending = s.pending
        obs = {
            'your_hand': s.hands[player_id].copy(),
            'opponents': [],
            'deck_size': len(s.deck),
            'exploding_kittens_in_deck': s.deck.count(CardType.EXPLODING_KITTEN),
            'discard_counts': self._get_discard_counts(),
            'have_drawn_this_turn': s.have_drawn_this_turn and player_id == self.turn_owner(),
            'holding_exploding_kitten': s.holding_exploding_kitten[player_id],
            # See-the-Future knowledge is private to the player who played it.
            'top_3_cards': list(s.top_3_visible[player_id]),
            'see_future_active': any(c is not None for c in s.top_3_visible[player_id]),
            # Nope window
            'nope_window': pending is not None,
            'pending_action': int(pending['action']['head1']) if pending else -1,
            'pending_noped': bool(pending and pending['nopes'] % 2 == 1),
            'pending_is_mine': bool(pending and pending['actor'] == player_id),
            'pending_targets_me': self._pending_target() == player_id,
        }

        upper = self._card_upper_bounds(player_id)
        for offset in range(1, s.num_players):
            opp = (player_id + offset) % s.num_players
            obs['opponents'].append({
                'num_cards': int(np.sum(s.hands[opp])),
                'is_alive': s.players_alive[opp],
                'turns_remaining': s.turns_remaining[opp],
                'min_counts': s.known_counts[opp].copy(),
                'max_counts': upper[opp],
            })

        while len(obs['opponents']) < 4:
            obs['opponents'].append({
                'num_cards': 0,
                'is_alive': False,
                'turns_remaining': 0,
                'min_counts': np.zeros(NUM_CARD_TYPES, dtype=np.int32),
                'max_counts': np.zeros(NUM_CARD_TYPES, dtype=np.int32),
            })

        return obs

    def _get_discard_counts(self) -> np.ndarray:
        return np.bincount(self.state.discard_pile, minlength=NUM_CARD_TYPES).astype(np.int32)

    # ------------------------------------------------------------------ #
    # Action masks
    # ------------------------------------------------------------------ #

    def get_action_mask(self, player_id: int) -> Dict[str, np.ndarray]:
        """Get valid action masks for each head."""
        hand = self.state.hands[player_id]
        max_deck = self.config.max_deck_size

        # Responding to a pending action: Nope or pass, nothing else.
        if self.state.pending is not None:
            head1 = np.zeros(NUM_ACTION_TYPES, dtype=bool)
            head1[ActionType.PASS] = True
            head1[ActionType.PLAY_NOPE] = hand[CardType.NOPE] > 0
            return {
                'head1': head1,
                'head2': np.zeros(4, dtype=bool),
                'head3': np.zeros(NUM_CARD_TYPES, dtype=bool),
                'head4': np.zeros(NUM_CARD_TYPES, dtype=bool),
                'head5': np.zeros(max_deck, dtype=bool),
            }

        # Holding an exploding kitten: must Defuse.
        if self.state.holding_exploding_kitten[player_id]:
            head1 = np.zeros(NUM_ACTION_TYPES, dtype=bool)
            head1[ActionType.PLAY_DEFUSE] = hand[CardType.DEFUSE] > 0
            head5 = np.zeros(max_deck, dtype=bool)
            head5[:len(self.state.deck) + 1] = True
            return {
                'head1': head1,
                'head2': np.zeros(4, dtype=bool),
                'head3': np.zeros(NUM_CARD_TYPES, dtype=bool),
                'head4': np.ones(NUM_CARD_TYPES, dtype=bool),
                'head5': head5,
            }

        has_opponents = self._has_alive_opponents(player_id)
        head1 = np.zeros(NUM_ACTION_TYPES, dtype=bool)
        head1[ActionType.DRAW] = not self.state.have_drawn_this_turn and len(self.state.deck) > 0
        head1[ActionType.PLAY_SKIP] = hand[CardType.SKIP] > 0
        head1[ActionType.PLAY_ATTACK] = hand[CardType.ATTACK] > 0
        head1[ActionType.PLAY_SHUFFLE] = hand[CardType.SHUFFLE] > 0
        head1[ActionType.PLAY_SEE_FUTURE] = hand[CardType.SEE_FUTURE] > 0
        head1[ActionType.PLAY_FAVOR] = hand[CardType.FAVOR] > 0 and has_opponents
        head1[ActionType.PLAY_DEFUSE] = False          # only when holding a kitten
        head1[ActionType.PLAY_NOPE] = False            # only inside a Nope window
        head1[ActionType.PLAY_2_OF_A_KIND] = bool(np.any(hand >= 2)) and has_opponents
        head1[ActionType.PLAY_3_OF_A_KIND] = bool(np.any(hand >= 3)) and has_opponents
        head1[ActionType.PASS] = False

        head2 = []
        for offset in range(1, self.state.num_players):
            head2.append(self.state.players_alive[(player_id + offset) % self.state.num_players])
        head2.extend([False] * (4 - len(head2)))
        head2 = np.array(head2, dtype=bool)

        head5 = np.zeros(max_deck, dtype=bool)
        head5[:len(self.state.deck) + 1] = True

        return {
            'head1': head1,
            'head2': head2,
            'head3': hand >= 2,
            'head4': np.ones(NUM_CARD_TYPES, dtype=bool),
            'head5': head5,
        }

    def _has_alive_opponents(self, player_id: int) -> bool:
        return any(self.state.players_alive[i] for i in range(self.state.num_players) if i != player_id)

    # ------------------------------------------------------------------ #
    # Step
    # ------------------------------------------------------------------ #

    def step(self, action: Dict) -> Tuple[Dict, float, bool, Dict]:
        """Execute the current player's action.

        Returns (observation for the NEXT player to act, reward for the acting player,
        done, info). `info['rewards']` maps EVERY player to the reward this transition
        produced for them -- use it to credit a learner that did not act.
        """
        if self.state.pending is not None:
            return self._step_nope_window(action)

        player_id = self.state.current_player
        action_type = action['head1']

        handler = self.action_handlers.get(action_type)
        if handler is None:
            return self._reject(player_id, 'Invalid action type')
        if action_type == ActionType.PLAY_NOPE:
            return self._reject(player_id, 'Nope can only be played in response to an action')
        if not handler.can_execute(self.state, player_id, action):
            return self._reject(player_id, 'Cannot execute action')

        if action_type in NOPEABLE_ACTIONS and self.config.enable_nope:
            action = self._resolve_action(player_id, action)
            responders = self._responders_after(player_id, exclude=player_id)
            if responders:
                self._open_nope_window(player_id, action, responders)
                return self.get_observation(self.state.current_player), 0.0, False, {
                    'nope_window': True, 'rewards': {},
                }

        _, done, info = handler.execute(self.state, player_id, action)
        return self._finish(player_id, action, done, info)

    def _reject(self, player_id, message):
        return self.get_observation(player_id), 0.0, False, {'error': message, 'rewards': {}}

    def _resolve_action(self, player_id: int, action: Dict) -> Dict:
        """Fix up combo card choice so the action pays for the cards it will actually use."""
        action = dict(action)
        a_type = action['head1']
        if a_type == ActionType.PLAY_2_OF_A_KIND:
            action['head3'] = resolve_combo_card(self.state.hands[player_id], action.get('head3', 0), 2)
        elif a_type == ActionType.PLAY_3_OF_A_KIND:
            action['head3'] = resolve_combo_card(self.state.hands[player_id], action.get('head3', 0), 3)
        return action

    def _action_cost(self, action: Dict) -> List[int]:
        a_type = action['head1']
        if a_type in ACTION_CARD:
            return [ACTION_CARD[a_type]]
        if a_type == ActionType.PLAY_2_OF_A_KIND:
            return [action['head3']] * 2
        if a_type == ActionType.PLAY_3_OF_A_KIND:
            return [action['head3']] * 3
        return []

    def _responders_after(self, last: int, exclude: int) -> List[int]:
        """Alive players holding a Nope, in turn order after `last`, excluding `exclude`."""
        s = self.state
        order = [(last + k) % s.num_players for k in range(1, s.num_players + 1)]
        return [p for p in order
                if p != exclude and s.players_alive[p] and s.hands[p][CardType.NOPE] > 0]

    def _open_nope_window(self, actor: int, action: Dict, responders: List[int]):
        s = self.state
        cost = self._action_cost(action)
        # The cards are paid now and sit on the table until the action resolves.
        for card in cost:
            s.hands[actor][card] -= 1
        self.action_handlers[action['head1']]._knowledge_card_left_hand(s, actor, cost[0], len(cost))
        s.pending = {
            'actor': actor, 'action': action, 'cost': cost,
            'nopes': 0, 'last': actor, 'queue': responders,
        }
        s.current_player = responders[0]

    def _step_nope_window(self, action: Dict):
        s = self.state
        pending = s.pending
        responder = s.current_player
        a_type = action['head1']

        if a_type == ActionType.PLAY_NOPE:
            if s.hands[responder][CardType.NOPE] <= 0:
                return self._reject(responder, 'No Nope card to play')
            self.action_handlers[ActionType.PLAY_NOPE].execute(s, responder, action)
            pending['nopes'] += 1
            pending['last'] = responder
            pending['queue'] = self._responders_after(responder, exclude=responder)
            info = {'noped_by': responder}
        elif a_type == ActionType.PASS:
            pending['queue'].pop(0)
            info = {}
        else:
            return self._reject(responder, 'Only Nope or Pass is allowed while a Nope window is open')

        if pending['queue']:
            s.current_player = pending['queue'][0]
            info['nope_window'] = True
            info['rewards'] = {}
            return self.get_observation(s.current_player), 0.0, False, info

        return self._resolve_pending(info)

    def _resolve_pending(self, window_info: Dict):
        s = self.state
        pending = s.pending
        s.pending = None
        actor, action, cost = pending['actor'], pending['action'], pending['cost']
        s.current_player = actor

        if pending['nopes'] % 2 == 1:
            s.discard_pile.extend(cost)
            info = dict(window_info, noped=True, cancelled_action=int(action['head1']))
            return self._finish(actor, {'head1': ActionType.PASS}, False, info)

        # Hand the paid cards back so the handler can pay and resolve as usual.
        for card in cost:
            s.hands[actor][card] += 1
        s.known_counts[actor][cost[0]] += len(cost)
        _, done, info = self.action_handlers[action['head1']].execute(s, actor, action)
        info.update(window_info)
        info['nopes'] = pending['nopes']
        return self._finish(actor, action, done, info)

    def _finish(self, player_id: int, action: Dict, done: bool, info: Dict):
        """Win check, rewards and turn advancement after an action actually resolved."""
        turn_ended = self.state.have_drawn_this_turn

        if sum(self.state.players_alive) == 1:
            done = True
            info['winner'] = self.state.players_alive.index(True)

        rewards = compute_rewards(self.state, player_id, action, done, info, turn_ended)
        info['rewards'] = rewards

        if not done and turn_ended:
            self._advance_turn()

        return (
            self.get_observation(self.state.current_player),
            rewards.get(player_id, 0.0),
            done,
            info,
        )

    def _advance_turn(self):
        """Consume one of the current player's owed turns and pass play on if none are left.

        Invariant: turns_remaining[p] is 0 for every player except the one whose turn it
        is, who owes >= 1.
        """
        state = self.state
        current = state.current_player

        state.turns_remaining[current] = max(state.turns_remaining[current] - 1, 0)
        if not state.players_alive[current]:
            state.turns_remaining[current] = 0

        if state.turns_remaining[current] == 0:
            nxt = get_next_alive_player(state, current)
            state.current_player = nxt
            if state.turns_remaining[nxt] == 0:
                state.turns_remaining[nxt] = 1

        state.have_drawn_this_turn = False
