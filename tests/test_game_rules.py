"""Rule-level regression tests for the Exploding Kittens environment.

Run with:  python -m pytest tests/test_game_rules.py   (or python tests/test_game_rules.py)
"""
import os
import sys
import random

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.card_types import ActionType, CardType
from core.game_config import GameConfig
from env.environment import ExplodingKittensEnv


def act(head1, head2=0, head3=0, head4=0, head5=0):
    return {'head1': int(head1), 'head2': head2, 'head3': head3, 'head4': head4, 'head5': head5}


def make_env(num_players=4, seed=0):
    random.seed(seed)
    np.random.seed(seed)
    env = ExplodingKittensEnv(GameConfig(num_players=num_players))
    # Deterministic, kitten-free deck of harmless cards so draws are predictable.
    for h in env.state.hands:
        h[:] = 0
        h[CardType.DEFUSE] = 1
    env.state.deck = [CardType.TACOCAT] * 30 + [CardType.EXPLODING_KITTEN] * (num_players - 1)
    return env


def give(env, player, card, n=1):
    env.state.hands[player][card] += n


def test_normal_turn_passes_to_next_player():
    env = make_env()
    env.step(act(ActionType.DRAW))
    assert env.state.current_player == 1
    assert env.state.turns_remaining[1] == 1


def test_attack_gives_next_player_two_turns_in_every_round():
    env = make_env()
    # Round 1: everyone just draws until we are back at player 0.
    for _ in range(4):
        env.step(act(ActionType.DRAW))
    assert env.state.current_player == 0
    # Round 2: player 0 attacks.
    give(env, 0, CardType.ATTACK)
    env.step(act(ActionType.PLAY_ATTACK))
    assert env.state.current_player == 1
    assert env.state.turns_remaining[1] == 2
    env.step(act(ActionType.DRAW))
    assert env.state.current_player == 1, "attacked player must take a second turn"
    env.step(act(ActionType.DRAW))
    assert env.state.current_player == 2


def test_attack_ends_all_of_the_attackers_remaining_turns():
    env = make_env()
    give(env, 0, CardType.ATTACK, 2)
    env.step(act(ActionType.PLAY_ATTACK))          # 0 attacks -> 1 owes 2 turns
    give(env, 1, CardType.ATTACK)
    env.step(act(ActionType.PLAY_ATTACK))          # 1 attacks with 2 turns owed
    assert env.state.current_player == 2
    assert env.state.turns_remaining[1] == 0, "attacker must not keep owing turns"
    assert env.state.turns_remaining[2] >= 2


def test_skip_uses_up_only_one_of_several_turns():
    env = make_env()
    give(env, 0, CardType.ATTACK)
    give(env, 3, CardType.ATTACK)
    # Hand the turn around so that player 1 is attacked and owes 2 turns.
    env.state.current_player = 3
    env.state.turns_remaining = [0, 0, 0, 1]
    env.step(act(ActionType.PLAY_ATTACK))          # 3 attacks -> 0 owes 2
    assert env.state.current_player == 0 and env.state.turns_remaining[0] == 2
    give(env, 0, CardType.SKIP)
    env.step(act(ActionType.PLAY_SKIP))
    assert env.state.current_player == 0, "skip only consumes one turn"
    assert env.state.turns_remaining[0] == 1
    assert not env.state.have_drawn_this_turn


def test_dead_player_is_skipped_and_ends_their_turns():
    env = make_env()
    env.state.hands[0][CardType.DEFUSE] = 0
    env.state.deck = [CardType.EXPLODING_KITTEN] + env.state.deck
    _, _, done, info = env.step(act(ActionType.DRAW))
    assert info.get('exploded') and not env.state.players_alive[0]
    assert not done
    assert env.state.current_player == 1


def test_defuse_flow_reinserts_kitten_and_ends_turn():
    env = make_env()
    env.state.deck = [CardType.EXPLODING_KITTEN] + env.state.deck
    env.step(act(ActionType.DRAW))
    assert env.state.current_player == 0 and env.state.holding_exploding_kitten[0]
    mask = env.get_action_mask(0)
    assert mask['head1'].nonzero()[0].tolist() == [ActionType.PLAY_DEFUSE]
    env.step(act(ActionType.PLAY_DEFUSE, head5=2))
    assert env.state.deck[2] == CardType.EXPLODING_KITTEN
    assert env.state.current_player == 1


def test_last_survivor_wins_and_receives_the_reward():
    env = make_env(num_players=2)
    env.state.hands[0][CardType.DEFUSE] = 0
    env.state.deck = [CardType.EXPLODING_KITTEN] + env.state.deck
    _, _, done, info = env.step(act(ActionType.DRAW))
    assert done and info['winner'] == 1
    assert info['rewards'][1] > 0.9, "winner must be paid even though they did not act"
    assert info['rewards'][0] < -0.9, "exploded player must be penalised"


def test_masked_actions_are_always_executable():
    """Every head1 action the mask allows must change the state, whatever sub-heads are sampled."""
    rng = random.Random(1)
    for players in (2, 3, 4, 5):
        for g in range(60):
            random.seed(g)
            np.random.seed(g)
            env = ExplodingKittensEnv(GameConfig(num_players=players))
            for _ in range(2000):
                p = env.state.current_player
                m = env.get_action_mask(p)
                valid = np.where(m['head1'])[0]
                assert len(valid) > 0
                a = {'head1': int(rng.choice(list(valid)))}
                for h in ('head2', 'head3', 'head4', 'head5'):
                    v = np.where(m[h])[0]
                    a[h] = int(rng.choice(list(v))) if len(v) else 0
                _, _, done, info = env.step(a)
                assert 'error' not in info, (ActionType(a['head1']).name, info)
                if done:
                    break
            else:
                raise AssertionError('game did not finish')


def test_turn_invariants_hold_in_random_play():
    rng = random.Random(7)
    for players in (2, 3, 4, 5):
        for g in range(40):
            random.seed(g)
            np.random.seed(g)
            env = ExplodingKittensEnv(GameConfig(num_players=players))
            for _ in range(2000):
                s = env.state
                p = s.current_player
                owner = env.turn_owner()
                assert s.players_alive[p] and s.players_alive[owner]
                assert s.turns_remaining[owner] >= 1
                assert all(t == 0 for i, t in enumerate(s.turns_remaining) if i != owner)
                if s.pending is not None:
                    assert p != s.pending['last'] and s.hands[p][CardType.NOPE] > 0
                check_knowledge_is_sound(env)
                m = env.get_action_mask(p)
                a = {'head1': int(rng.choice(list(np.where(m['head1'])[0])))}
                for h in ('head2', 'head3', 'head4', 'head5'):
                    v = np.where(m[h])[0]
                    a[h] = int(rng.choice(list(v))) if len(v) else 0
                _, _, done, _ = env.step(a)
                if done:
                    break


def test_see_future_is_private_and_goes_stale_on_draw():
    env = make_env()
    give(env, 0, CardType.SEE_FUTURE)
    env.step(act(ActionType.PLAY_SEE_FUTURE))
    assert env.get_observation(0)['see_future_active']
    assert not env.get_observation(1)['see_future_active'], "opponents must not see player 0's peek"
    env.step(act(ActionType.DRAW))
    assert not env.get_observation(0)['see_future_active'] or \
        env.get_observation(0)['top_3_cards'][0] == env.state.deck[0]


def check_knowledge_is_sound(env):
    """Public lower bounds and per-observer upper bounds must bracket every true hand."""
    s = env.state
    total = sum(h for h in s.hands) + np.bincount(s.discard_pile, minlength=13) \
        + np.bincount(s.deck, minlength=13).astype(np.int32)
    if s.pending is not None:
        total = total + np.bincount(s.pending['cost'], minlength=13)
    total[CardType.EXPLODING_KITTEN] += sum(s.holding_exploding_kitten)   # drawn, not yet defused
    assert (total == s.card_totals).all(), 'card conservation violated'
    for observer in range(s.num_players):
        upper = env._card_upper_bounds(observer)
        for p in range(s.num_players):
            if p == observer:
                continue
            assert (s.known_counts[p] <= s.hands[p]).all(), (p, s.known_counts[p], s.hands[p])
            assert (s.hands[p] <= upper[p]).all(), (observer, p, s.hands[p], upper[p])


def nope_env():
    """Player 0 to move; players 1 and 2 hold one Nope each; 3 holds none."""
    env = make_env()
    give(env, 1, CardType.NOPE)
    give(env, 2, CardType.NOPE)
    return env


def test_nope_cancels_an_attack():
    env = nope_env()
    give(env, 0, CardType.ATTACK)
    _, _, _, info = env.step(act(ActionType.PLAY_ATTACK))
    assert info.get('nope_window') and env.state.current_player == 1
    assert env.get_action_mask(1)['head1'].nonzero()[0].tolist() == [ActionType.PLAY_NOPE, ActionType.PASS]
    env.step(act(ActionType.PLAY_NOPE))              # 1 nopes
    assert env.state.current_player == 2             # 2 may nope the nope; 0 holds none
    _, _, _, info = env.step(act(ActionType.PASS))
    assert info.get('noped') and env.state.pending is None
    assert env.state.current_player == 0, 'noped attack: attacker keeps the turn'
    assert env.state.hands[0][CardType.ATTACK] == 0 and CardType.ATTACK in env.state.discard_pile
    assert env.get_action_mask(0)['head1'][ActionType.DRAW]


def test_nope_of_a_nope_lets_the_action_through():
    env = nope_env()
    give(env, 0, CardType.SKIP)
    env.step(act(ActionType.PLAY_SKIP))
    env.step(act(ActionType.PLAY_NOPE))              # 1 nopes
    _, _, _, info = env.step(act(ActionType.PLAY_NOPE))   # 2 nopes the nope
    # 1 has no Nope left and 0 never had one, so the window closes and Skip resolves.
    assert env.state.pending is None and info.get('nopes') == 2
    assert env.state.current_player == 1, 'skip went through'


def test_everyone_passing_resolves_the_action():
    env = nope_env()
    give(env, 0, CardType.SHUFFLE)
    env.step(act(ActionType.PLAY_SHUFFLE))
    env.step(act(ActionType.PASS))
    assert env.state.current_player == 2
    env.step(act(ActionType.PASS))
    assert env.state.pending is None and env.state.current_player == 0
    assert env.state.hands[0][CardType.SHUFFLE] == 0


def test_no_window_when_nobody_else_holds_nope():
    env = make_env()
    give(env, 0, CardType.NOPE)
    give(env, 0, CardType.SKIP)
    assert not env.get_action_mask(0)['head1'][ActionType.PLAY_NOPE], 'Nope is reactive only'
    env.step(act(ActionType.PLAY_SKIP))
    assert env.state.pending is None and env.state.current_player == 1


def test_defuse_and_draw_cannot_be_noped():
    env = nope_env()
    env.step(act(ActionType.DRAW))
    assert env.state.pending is None and env.state.current_player == 1


def test_failed_three_of_a_kind_reveals_none_and_draw_clears_it():
    env = make_env()
    give(env, 0, CardType.TACOCAT, 3)
    env.state.hands[1][CardType.FAVOR] = 0
    env.step(act(ActionType.PLAY_3_OF_A_KIND, head2=0, head3=CardType.TACOCAT, head4=CardType.FAVOR))
    assert env.state.known_zero[1][CardType.FAVOR]
    assert env.get_observation(0)['opponents'][0]['max_counts'][CardType.FAVOR] == 0
    env.step(act(ActionType.DRAW))   # player 0 ends turn
    env.step(act(ActionType.DRAW))   # player 1 draws an unseen card
    assert not env.state.known_zero[1][CardType.FAVOR]


def test_initial_defuse_is_public_knowledge():
    env = ExplodingKittensEnv(GameConfig(num_players=4))
    for opp in env.get_observation(0)['opponents'][:3]:
        assert opp['min_counts'][CardType.DEFUSE] == 1


if __name__ == '__main__':
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            try:
                fn()
                print(f'PASS {name}')
            except Exception as e:  # noqa: BLE001
                failed += 1
                print(f'FAIL {name}: {type(e).__name__}: {e}')
    sys.exit(1 if failed else 0)
