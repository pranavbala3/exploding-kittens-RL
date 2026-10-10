"""
Quick test to verify the setup works before training.
"""

import torch
import numpy as np

from env.environment import ExplodingKittensEnv
from core.game_config import GameConfig
from networks.policy import ExplodingKittensPolicy
from utils.observation_utils import process_observation


def test_environment():
    """Test that environment works."""
    print("Testing environment...")
    config = GameConfig(num_players=4)
    env = ExplodingKittensEnv(config)
    
    obs = env.reset()
    print(f"✓ Environment reset successful")
    print(f"  - Hand size: {len(obs['your_hand'])}")
    print(f"  - Deck size: {obs['deck_size']}")
    print(f"  - Opponents: {len(obs['opponents'])}")
    
    # Test action masks
    masks = env.get_action_mask(0)
    print(f"✓ Action masks generated")
    print(f"  - Valid actions: {np.sum(masks['head1'])}")
    
    # Take a random action
    valid_actions = np.where(masks['head1'])[0]
    action = {
        'head1': int(np.random.choice(valid_actions)),
        'head2': 0,
        'head3': 0,
        'head4': 0,
        'head5': 0,
    }
    
    obs, reward, done, info = env.step(action)
    print(f"✓ Step executed successfully")
    print(f"  - Reward: {reward}")
    print(f"  - Done: {done}")
    
    return True


def test_policy():
    """Test that policy network works."""
    print("\nTesting policy network...")
    
    policy = ExplodingKittensPolicy(
        embed_dim=128,
        hidden_dim=128,
        num_attention_heads=4
    )
    
    print(f"✓ Policy created")
    print(f"  - Parameters: {sum(p.numel() for p in policy.parameters()):,}")
    
    # Create dummy observation
    config = GameConfig(num_players=4)
    env = ExplodingKittensEnv(config)
    obs = env.reset()
    masks = env.get_action_mask(0)
    
    # Process observation
    obs_tensor = process_observation(obs)
    masks_tensor = {
        'head1': torch.BoolTensor(masks['head1']).unsqueeze(0),
        'head2': torch.BoolTensor(masks['head2']).unsqueeze(0),
        'head3': torch.BoolTensor(masks['head3']).unsqueeze(0),
        'head4': torch.BoolTensor(masks['head4']).unsqueeze(0),
        'head5': torch.BoolTensor(masks['head5']).unsqueeze(0),
    }
    
    print(f"✓ Observation processed")
    
    # Forward pass
    with torch.no_grad():
        output = policy.forward(obs_tensor, masks_tensor)
    
    print(f"✓ Forward pass successful")
    print(f"  - Value: {output['value'].item():.3f}")
    
    # Sample action
    action = policy.get_action(obs_tensor, masks_tensor, deterministic=False)
    print(f"✓ Action sampled")
    print(f"  - Action type: {action['head1']}")
    
    return True


def test_training_components():
    """Test that training components work together."""
    print("\nTesting training components...")
    
    from RL.rollout_storage import RolloutStorage
    from RL.ppo import PPO
    from dataclasses import dataclass
    
    @dataclass
    class Args:
        lr = 3e-4
        eps = 1e-5
        clip_param = 0.2
        ppo_epoch = 1
        num_mini_batch = 2
        value_loss_coef = 0.5
        entropy_coef_start = 0.01
        max_grad_norm = 0.5
        gamma = 0.99
        gae_lambda = 0.95
    
    policy = ExplodingKittensPolicy(embed_dim=128, hidden_dim=128)
    ppo = PPO(policy, Args())
    storage = RolloutStorage(num_steps=10)
    
    print(f"✓ PPO and storage created")
    
    # Collect some dummy experience
    env = ExplodingKittensEnv(GameConfig(num_players=4))
    obs = env.reset()
    
    for _ in range(10):
        masks = env.get_action_mask(0)
        valid = np.where(masks['head1'])[0]
        action = {
            'head1': int(np.random.choice(valid)),
            'head2': 0,
            'head3': 0,
            'head4': 0,
            'head5': 0,
        }
        
        storage.insert(
            obs=obs,
            action=action,
            action_log_prob=torch.tensor(-1.0),
            value_pred=torch.tensor(0.5),
            reward=0.0,
            mask=1.0,
            action_mask=masks
        )
        
        obs, _, done, _ = env.step(action)
        if done:
            obs = env.reset()
    
    print(f"✓ Experience collected: {len(storage)} steps")
    
    # Try update (this will test the full pipeline)
    try:
        losses = ppo.update(storage)
        print(f"✓ PPO update successful")
        print(f"  - Value loss: {losses['value_loss']:.4f}")
        print(f"  - Policy loss: {losses['action_loss']:.4f}")
    except Exception as e:
        print(f"✗ PPO update failed: {e}")
        return False
    
    return True


def main():
    """Run all tests."""
    print("="*60)
    print("EXPLODING KITTENS RL - SETUP VERIFICATION")
    print("="*60)
    
    tests = [
        ("Environment", test_environment),
        ("Policy Network", test_policy),
        ("Training Components", test_training_components),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            success = test_fn()
            results.append((name, success))
        except Exception as e:
            print(f"\n✗ {name} test failed with error:")
            print(f"  {e}")
            results.append((name, False))
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(success for _, success in results)
    
    if all_passed:
        print("\n🎉 All tests passed! Ready to train.")
        print("\nRun: python train.py")
    else:
        print("\n⚠️  Some tests failed. Please fix errors before training.")
    
    return all_passed


if __name__ == '__main__':
    main()
