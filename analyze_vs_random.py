"""
Behavioural breakdown of a trained agent vs 3 random opponents.

Runs the same games twice: once with the trained agent in the test seat, once with a
random agent in that seat (control), so every number has a like-for-like baseline.

Usage: python analyze_vs_random.py <checkpoint> [num_games] [num_players]
"""
import sys
import random
from collections import Counter

import numpy as np
import torch

from core.card_types import ActionType, CardType
from core.game_config import GameConfig
from env.environment import ExplodingKittensEnv
from evaluate import PolicyAgent, RandomAgent, load_policy

NUM_PLAYERS = 4


def run(agent, num_games, seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    env = ExplodingKittensEnv(GameConfig(num_players=NUM_PLAYERS))
    st = Counter()
    seat_wins, seat_games = Counter(), Counter()
    actions = Counter()            # normal-turn action choices of the test seat
    finish = Counter()             # finishing place: 1 = winner, 4 = first to die
    defuse_pos = Counter()
    for g in range(num_games):
        seat = g % NUM_PLAYERS
        agents = [RandomAgent() for _ in range(NUM_PLAYERS)]
        agents[seat] = agent
        obs = env.reset()
        done, steps, info = False, 0, {}
        deaths = []
        while not done and steps < 1000:
            s = env.state
            p = s.current_player   # during a Nope window this is the responder
            masks = env.get_action_mask(p)
            act = agents[p].get_action(obs, masks)
            if p == seat:
                a = ActionType(int(act['head1']))
                if s.pending is not None:
                    if masks['head1'][ActionType.PLAY_NOPE]:
                        aimed = env._pending_target() == seat
                        key = 'aimed_at_me' if aimed else 'not_aimed'
                        st[f'nope_window_{key}'] += 1
                        st[f'nope_used_{key}'] += a == ActionType.PLAY_NOPE
                elif s.holding_exploding_kitten[seat]:
                    pos = int(act['head5']); n = len(s.deck)
                    pos = min(pos, n)
                    defuse_pos['top (next draw)' if pos == 0 else
                               'pos 1-2' if pos <= 2 else
                               'bottom' if pos == n else 'middle'] += 1
                else:
                    actions[a.name] += 1
                    peek = s.top_3_visible[seat] if s.top_3_visible else None
                    if peek and peek[0] == CardType.EXPLODING_KITTEN:
                        tag = 'with_defuse' if s.hands[seat][CardType.DEFUSE] > 0 else 'no_defuse'
                        escape = any(masks['head1'][x] for x in (
                            ActionType.PLAY_SKIP, ActionType.PLAY_ATTACK, ActionType.PLAY_SHUFFLE))
                        tag += '_could_escape' if escape else '_forced'
                        st[f'knew_kitten_{tag}'] += 1
                        st[f'drew_known_kitten_{tag}'] += a == ActionType.DRAW
            alive_before = list(s.players_alive)
            obs, _, done, info = env.step(act)
            steps += 1
            for i in range(NUM_PLAYERS):
                if alive_before[i] and not env.state.players_alive[i]:
                    deaths.append(i)
        winner = info.get('winner', -1)
        seat_games[seat] += 1
        seat_wins[seat] += winner == seat
        st['games'] += 1; st['steps'] += steps
        finish[1 if winner == seat else NUM_PLAYERS - deaths.index(seat) if seat in deaths else 0] += 1
    return dict(st=st, seat_wins=seat_wins, seat_games=seat_games,
                actions=actions, finish=finish, defuse_pos=defuse_pos)


def pct(a, b):
    return f'{100 * a / b:5.1f}%' if b else '   n/a'


def report(name, r):
    st = r['st']
    print(f'\n=== {name} ===')
    wins = sum(r['seat_wins'].values())
    print(f"win rate {pct(wins, st['games'])}  | by seat: " + '  '.join(
        f"s{i} {pct(r['seat_wins'][i], r['seat_games'][i])}" for i in range(NUM_PLAYERS)))
    print('finishing place: ' + '  '.join(
        f"{k}:{pct(r['finish'][k], st['games'])}" for k in range(1, NUM_PLAYERS + 1)))
    tot = sum(r['actions'].values())
    print('normal-turn actions: ' + '  '.join(
        f'{k} {pct(v, tot)}' for k, v in r['actions'].most_common()))
    for tag in ('no_defuse_could_escape', 'no_defuse_forced',
                'with_defuse_could_escape', 'with_defuse_forced'):
        print(f"knew kitten was on top ({tag}): {st[f'knew_kitten_{tag}']} times, drew it anyway "
              f"{pct(st[f'drew_known_kitten_{tag}'], st[f'knew_kitten_{tag}'])}")
    dtot = sum(r['defuse_pos'].values())
    print('defused kitten put back at: ' + '  '.join(
        f'{k} {pct(v, dtot)}' for k, v in r['defuse_pos'].most_common()) + f'  (n={dtot})')
    for k in ('aimed_at_me', 'not_aimed'):
        print(f'Nope used when holding one, action {k}: '
              f"{pct(st[f'nope_used_{k}'], st[f'nope_window_{k}'])} (n={st[f'nope_window_{k}']})")


if __name__ == '__main__':
    ckpt = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    NUM_PLAYERS = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    print(f"{NUM_PLAYERS}-player games, {n} per agent")
    policy, _ = load_policy(ckpt)
    report('RANDOM agent in test seat (control)', run(RandomAgent(), n, seed=123))
    report(f'TRAINED agent ({ckpt})', run(PolicyAgent(policy, deterministic=True), n, seed=123))
