"""End-to-end smoke tests for the training stack (collector, GAE, PPO, self-play).

Run with:  python tests/test_training_smoke.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.game_config import GameConfig
from env.environment import ExplodingKittensEnv
from networks.policy import ExplodingKittensPolicy
from RL.collect import collect_rollout, random_masked_action
from RL.ppo import PPO
from RL.rollout_storage import RolloutStorage
from train import TrainingArgs
from train_selfplay import collect_rollout_selfplay
from utils.observation_utils import process_observation


def small_args(**kw):
    args = TrainingArgs()
    args.num_steps = 256
    args.ppo_epoch = 2
    args.num_mini_batch = 2
    for k, v in kw.items():
        setattr(args, k, v)
    return args


def test_policy_get_action_returns_plain_ints():
    env = ExplodingKittensEnv(GameConfig(num_players=4))
    policy = ExplodingKittensPolicy(embed_dim=32, hidden_dim=32)
    obs = {k: v for k, v in process_observation(env.get_observation(0)).items()}
    m = env.get_action_mask(0)
    masks = {k: torch.as_tensor(v, dtype=torch.bool).unsqueeze(0) for k, v in m.items()}
    for det in (False, True):
        action = policy.get_action(obs, masks, deterministic=det)
        assert all(type(v) is int for v in action.values()), action


def test_rewards_reach_the_learner_and_episodes_terminate():
    torch.manual_seed(0)
    np.random.seed(0)
    env = ExplodingKittensEnv(GameConfig(num_players=4))
    policy = ExplodingKittensPolicy(embed_dim=32, hidden_dim=32)
    storage = RolloutStorage(num_steps=600)
    games, wins = collect_rollout(
        env, policy, storage, 600,
        opponent_action_fn=lambda seat, idx, obs, masks: random_masked_action(masks),
    )
    rewards = np.array(storage.rewards)
    masks = np.array(storage.masks)
    assert games > 5
    assert (masks == 0).sum() >= games - 1            # every finished game closes an episode
    assert (rewards >= 0.99).sum() == wins, "each win is paid exactly once"
    assert (rewards <= -0.99).sum() > 0, "eliminations must be penalised"
    assert 0 < wins < games


def test_gae_does_not_leak_across_episode_boundaries():
    s = RolloutStorage(num_steps=4)
    zero = torch.tensor(0.0)
    for r, m in [(0.0, 1.0), (1.0, 0.0), (0.0, 1.0), (0.0, 1.0)]:
        s.insert({}, {}, zero, zero, r, m, {})
    s.compute_advantages(None, gamma=1.0, gae_lambda=1.0)
    # The first episode (steps 0-1) must only see its own +1; later steps must see nothing.
    raw_returns = s.returns
    assert raw_returns[0] == 1.0 and raw_returns[1] == 1.0
    assert raw_returns[2] == 0.0 and raw_returns[3] == 0.0


def test_ppo_update_runs_and_changes_weights():
    torch.manual_seed(0)
    args = small_args()
    env = ExplodingKittensEnv(GameConfig(num_players=4))
    policy = ExplodingKittensPolicy(embed_dim=32, hidden_dim=32)
    ppo = PPO(policy, args)
    storage = RolloutStorage(num_steps=args.num_steps)
    collect_rollout(env, policy, storage, args.num_steps,
                    lambda seat, idx, obs, masks: random_masked_action(masks))
    before = [p.clone() for p in policy.parameters()]
    losses = ppo.update(storage)
    assert all(np.isfinite(v) for v in losses.values()), losses
    assert any(not torch.equal(a, b) for a, b in zip(before, policy.parameters()))


def test_selfplay_rollout_finishes():
    torch.manual_seed(0)
    np.random.seed(0)
    env = ExplodingKittensEnv(GameConfig(num_players=4))
    policy = ExplodingKittensPolicy(embed_dim=32, hidden_dim=32)
    opp_nets = [ExplodingKittensPolicy(embed_dim=32, hidden_dim=32) for _ in range(3)]
    pool = [policy.state_dict()] * 3
    storage = RolloutStorage(num_steps=200)
    games, wins = collect_rollout_selfplay(env, policy, opp_nets, pool, storage, 200)
    assert len(storage) == 200 and games > 0


if __name__ == '__main__':
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            try:
                fn()
                print(f'PASS {name}')
            except Exception as e:  # noqa: BLE001
                import traceback
                failed += 1
                print(f'FAIL {name}: {type(e).__name__}: {e}')
                traceback.print_exc()
    sys.exit(1 if failed else 0)
