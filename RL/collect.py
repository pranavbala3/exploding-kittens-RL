"""Shared rollout collection for single-learner PPO (random opponents or self-play pool).

Why this exists: train.py and train_selfplay.py each had their own copy of this loop and
both had the same credit-assignment bugs:

* The learner only received `reward` from ITS OWN env.step. Winning (or being eliminated) is
  almost always caused by someone else's action, so the learner never saw +1 / -1.
* A game ending during an opponent's step never marked the learner's last transition as done,
  so GAE bootstrapped across game boundaries.
* The learner was always seat 0 (always first to move).

Here the learner's reward for a decision is everything that happens to it until its next
decision (or the end of its game), taken from `info['rewards']`. The learner's episode ends
when it explodes or the game ends, whichever is first.
"""

import random
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch

from utils.observation_utils import process_observation

MAX_CONSECUTIVE_REJECTIONS = 50
MAX_STEPS_PER_GAME = 3000


def masks_to_tensors(masks_dict: Dict[str, np.ndarray], device) -> Dict[str, torch.Tensor]:
    return {k: torch.as_tensor(v, dtype=torch.bool).unsqueeze(0).to(device) for k, v in masks_dict.items()}


def random_masked_action(masks: Dict[str, np.ndarray]) -> Dict[str, int]:
    action = {}
    for head in ('head1', 'head2', 'head3', 'head4', 'head5'):
        valid = np.flatnonzero(masks[head])
        action[head] = int(np.random.choice(valid)) if len(valid) else 0
    return action


def _sample_head(logits: torch.Tensor, mask: torch.Tensor):
    """Categorical over masked logits; an all-masked head (unused by the action) is uniform."""
    if not mask.any():
        logits = torch.zeros_like(logits)
    return torch.distributions.Categorical(logits=logits)


def sample_learner_action(policy, obs, masks_dict, device):
    """Sample an action from `policy`; returns (action_dict, joint_log_prob, raw_value)."""
    obs_tensor = {k: v.to(device) for k, v in process_observation(obs).items()}
    masks_tensor = masks_to_tensors(masks_dict, device)
    with torch.no_grad():
        output = policy.forward(obs_tensor, masks_tensor)
        heads = ('head1', 'head2', 'head3', 'head4', 'head5')
        dists = [_sample_head(output[f'logits{i + 1}'], masks_tensor[h]) for i, h in enumerate(heads)]
        samples = [d.sample() for d in dists]
        log_prob = sum(d.log_prob(a) for d, a in zip(dists, samples))
        value = output['value']
        if getattr(policy, 'use_value_norm', False):
            value = policy.value_normalizer.denormalize(value)   # store RAW-scale values for GAE
    action = {h: int(a.item()) for h, a in zip(heads, samples)}
    return action, log_prob.reshape(-1)[0].cpu(), value.reshape(-1)[0].cpu()


def collect_rollout(
    env,
    policy,
    rollout_storage,
    num_steps: int,
    opponent_action_fn: Callable[[int, int, Dict, Dict], Dict],
    device='cpu',
    randomize_seat: bool = True,
    on_game_end: Optional[Callable[[Dict], None]] = None,
) -> Tuple[int, int]:
    """Collect `num_steps` learner decisions.

    opponent_action_fn(seat, opponent_index, obs, masks_dict) -> action dict.
    `opponent_index` is 0..num_players-2 and is stable for a seat within one game.

    Returns (games_played, wins_for_learner).
    """
    games = wins = 0
    collected = 0

    def new_game():
        env.reset()
        seat = random.randrange(env.state.num_players) if randomize_seat else 0
        others = [p for p in range(env.state.num_players) if p != seat]
        return seat, {p: i for i, p in enumerate(others)}

    learner, opp_index = new_game()
    pending = None          # the learner decision still waiting for its reward
    game_steps = 0
    rejections = 0

    def flush(done: bool):
        nonlocal pending, collected
        if pending is None:
            return
        rollout_storage.insert(
            obs=pending['obs'], action=pending['action'], action_log_prob=pending['log_prob'],
            value_pred=pending['value'], reward=pending['reward'], mask=0.0 if done else 1.0,
            action_mask=pending['masks'],
        )
        pending = None
        collected += 1

    while collected < num_steps:
        seat = env.state.current_player
        masks_dict = env.get_action_mask(seat)
        obs = env.get_observation(seat)

        if seat == learner:
            flush(done=False)                      # previous decision is now fully rewarded
            if collected >= num_steps:
                break
            action, log_prob, value = sample_learner_action(policy, obs, masks_dict, device)
            pending = dict(obs=obs, action=action, log_prob=log_prob, value=value, reward=0.0, masks=masks_dict)
        else:
            action = opponent_action_fn(seat, opp_index[seat], obs, masks_dict)

        _, _, done, info = env.step(action)
        game_steps += 1

        if 'error' in info:
            rejections += 1
            if rejections > MAX_CONSECUTIVE_REJECTIONS:
                raise RuntimeError(
                    f"{rejections} consecutive rejected actions (last: {action}, seat {seat}). "
                    "The action mask allowed an action the handler refuses."
                )
        else:
            rejections = 0

        if pending is not None:
            pending['reward'] += info.get('rewards', {}).get(learner, 0.0)

        learner_out = not env.state.players_alive[learner]
        truncated = game_steps >= MAX_STEPS_PER_GAME
        if done or learner_out or truncated:
            flush(done=True)
            games += 1
            if done and info.get('winner') == learner:
                wins += 1
            if on_game_end is not None:
                on_game_end({'winner': info.get('winner'), 'learner': learner, 'steps': game_steps,
                             'truncated': truncated and not done})
            learner, opp_index = new_game()
            pending, game_steps, rejections = None, 0, 0

    return games, wins
