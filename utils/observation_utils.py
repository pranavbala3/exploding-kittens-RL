"""Observation -> tensor featurization (single source of truth for train/eval/PPO)."""

from typing import Dict, List

import numpy as np
import torch

from core.card_types import NUM_ACTION_TYPES, NUM_CARD_TYPES

# [have_drawn, holding_kitten, nope_window, pending_noped, pending_is_mine,
#  pending_targets_me] + one-hot(pending action)
GAME_STATE_DIM = 6 + NUM_ACTION_TYPES
SEE_FUTURE_DIM = 3 * NUM_CARD_TYPES + 1
OPPONENT_FEATURE_DIM = 3 + 2 * NUM_CARD_TYPES


def observation_features(obs: Dict) -> Dict[str, np.ndarray]:
    """Unbatched float32 feature arrays for one observation."""
    game_state = np.zeros(GAME_STATE_DIM, dtype=np.float32)
    game_state[0] = float(obs['have_drawn_this_turn'])
    game_state[1] = float(obs['holding_exploding_kitten'])
    game_state[2] = float(obs.get('nope_window', False))
    game_state[3] = float(obs.get('pending_noped', False))
    game_state[4] = float(obs.get('pending_is_mine', False))
    game_state[5] = float(obs.get('pending_targets_me', False))
    pending = obs.get('pending_action', -1)
    if pending is not None and pending >= 0:
        game_state[6 + pending] = 1.0

    see_future = np.zeros(SEE_FUTURE_DIM, dtype=np.float32)
    if obs['see_future_active']:
        for i, card in enumerate(obs['top_3_cards']):
            if card is not None:
                see_future[i * NUM_CARD_TYPES + card] = 1.0
        see_future[-1] = 1.0

    opponents = np.stack([
        np.concatenate([
            [opp['num_cards'], float(opp['is_alive']), opp['turns_remaining']],
            opp['min_counts'],
            opp['max_counts'],
        ]).astype(np.float32)
        for opp in obs['opponents']
    ])

    return {
        'hand': np.asarray(obs['your_hand'], dtype=np.float32),
        'opponents': opponents,
        'deck_info': np.array([obs['deck_size'], obs['exploding_kittens_in_deck']], dtype=np.float32),
        'discard': np.asarray(obs['discard_counts'], dtype=np.float32),
        'game_state': game_state,
        'see_future': see_future,
    }


def batch_observation_features(obs_list: List[Dict]) -> Dict[str, torch.Tensor]:
    feats = [observation_features(o) for o in obs_list]
    return {k: torch.from_numpy(np.stack([f[k] for f in feats])) for k in feats[0]}


def process_observation(obs: Dict) -> Dict[str, torch.Tensor]:
    """Convert one environment observation to a batch-of-1 tensor dict."""
    return batch_observation_features([obs])


def process_masks(masks: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
    """Convert action masks to tensors."""
    return {k: torch.as_tensor(v, dtype=torch.bool).unsqueeze(0) for k, v in masks.items()}
