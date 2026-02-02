import torch
import numpy as np
from typing import Dict, List

"""
RolloutStorage - Experience buffer for PPO training.

This class stores experiences (observations, actions, rewards, etc.) collected
during gameplay and computes advantages using GAE (Generalized Advantage Estimation).
"""


class RolloutStorage:
    """
    Storage for rollouts (game episodes) used in PPO training.
    
    For multi-player games like Exploding Kittens, we only store experience
    from ONE player (the learning player), discarding the other players' data.
    """
    
    def __init__(self, num_steps: int, num_processes: int = 1):
        """
        Initialize storage.
        
        Args:
            num_steps: Maximum number of steps to store before update
            num_processes: Number of parallel environments (start with 1)
        """
        self.num_steps = num_steps
        self.num_processes = num_processes
        
        # Lists to store experience (will be converted to tensors later)
        self.observations = []
        self.actions = []  # Dict with head1-head5
        self.action_log_probs = []  # Combined log prob from all heads
        self.value_preds = []  # Value estimates V(s)
        self.rewards = []
        self.masks = []  # 0 if episode done, 1 if continuing
        self.action_masks = []  # Which actions were valid
        
        # Computed during advantage calculation
        self.returns = None  # Discounted future rewards
        self.advantages = None  # How good was action vs. expected
        
        self.step_count = 0
    
    def insert(
        self,
        obs: Dict,
        action: Dict,
        action_log_prob: torch.Tensor,
        value_pred: torch.Tensor,
        reward: float,
        mask: float,
        action_mask: Dict
    ):
        """
        Insert one step of experience.
        
        Args:
            obs: Observation dictionary from environment
            action: Action dictionary {'head1': ..., 'head2': ..., etc.}
            action_log_prob: Log probability of the action
            value_pred: Value prediction V(s)
            reward: Reward received
            mask: 0 if episode done, 1 if continuing
            action_mask: Valid action masks
        """
        self.observations.append(obs)
        self.actions.append(action)
        self.action_log_probs.append(action_log_prob.cpu())
        self.value_preds.append(value_pred.cpu())
        self.rewards.append(reward)
        self.masks.append(mask)
        self.action_masks.append(action_mask)
        
        self.step_count += 1
    
    def compute_advantages(
        self,
        actor_critic,
        gamma: float = 0.99,
        gae_lambda: float = 0.95
    ):
        """
        Compute returns and advantages using GAE (Generalized Advantage Estimation).
        
        This is the key step that determines "how good were our actions?"
        
        Args:
            actor_critic: Policy network (to get final value estimate)
            gamma: Discount factor (how much we value future rewards)
            gae_lambda: GAE parameter (bias-variance tradeoff)
        """
        num_steps = len(self.rewards)
        
        # Convert to tensors
        rewards = torch.tensor(self.rewards, dtype=torch.float32)
        values = torch.stack(self.value_preds).squeeze()
        masks = torch.tensor(self.masks, dtype=torch.float32)
        
        # Initialize
        advantages = torch.zeros(num_steps)
        returns = torch.zeros(num_steps)
        
        # Get value of final state (or 0 if terminal)
        if masks[-1] == 1:  # Episode not done
            # Estimate value of next state
            with torch.no_grad():
                # You'd need to get the next observation here
                # For simplicity, assume 0 (conservative)
                next_value = 0
        else:
            next_value = 0
        
        # Compute advantages using GAE (backwards through time)
        gae = 0
        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                next_value_t = next_value
                next_mask = 1.0 if masks[-1] == 1 else 0.0
            else:
                next_value_t = values[t + 1]
                next_mask = masks[t + 1]
            
            # TD error: reward + discounted next value - current value
            delta = rewards[t] + gamma * next_value_t * next_mask - values[t]
            
            # GAE: exponentially weighted average of TD errors
            gae = delta + gamma * gae_lambda * next_mask * gae
            
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
        
        self.advantages = advantages
        self.returns = returns
        
        # Normalize advantages (important for stable training!)
        self.advantages = (self.advantages - self.advantages.mean()) / (
            self.advantages.std() + 1e-8
        )
    
    def get_batches(self, num_mini_batch: int):
        """
        Generate random mini-batches for training.
        
        Args:
            num_mini_batch: Number of mini-batches to split data into
        
        Yields:
            Batches of (obs, actions, action_masks, values, returns, old_log_probs, advantages)
        """
        num_steps = len(self.observations)
        batch_size = num_steps // num_mini_batch
        
        # Shuffle indices for random sampling
        indices = torch.randperm(num_steps)
        
        for start_idx in range(0, num_steps, batch_size):
            end_idx = min(start_idx + batch_size, num_steps)
            batch_indices = indices[start_idx:end_idx]
            
            # Gather batch data
            obs_batch = [self.observations[i] for i in batch_indices]
            actions_batch = [self.actions[i] for i in batch_indices]
            action_masks_batch = [self.action_masks[i] for i in batch_indices]
            
            # Stack tensors
            value_preds_batch = torch.stack([self.value_preds[i] for i in batch_indices])
            returns_batch = self.returns[batch_indices]
            old_log_probs_batch = torch.stack([self.action_log_probs[i] for i in batch_indices])
            advantages_batch = self.advantages[batch_indices]
            
            yield (
                obs_batch,
                actions_batch,
                action_masks_batch,
                value_preds_batch,
                returns_batch,
                old_log_probs_batch,
                advantages_batch,
            )
    
    def clear(self):
        """Clear all stored data after update."""
        self.observations = []
        self.actions = []
        self.action_log_probs = []
        self.value_preds = []
        self.rewards = []
        self.masks = []
        self.action_masks = []
        
        self.returns = None
        self.advantages = None
        
        self.step_count = 0
    
    def is_full(self) -> bool:
        """Check if storage is full."""
        return self.step_count >= self.num_steps
    
    def __len__(self) -> int:
        """Return number of steps stored."""
        return self.step_count


# ==================== HELPER FUNCTIONS ====================

def compute_returns_simple(rewards: List[float], gamma: float = 0.99) -> torch.Tensor:
    """
    Simple return computation (no GAE).
    
    Returns are discounted cumulative rewards:
    R_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ...
    
    Args:
        rewards: List of rewards
        gamma: Discount factor
    
    Returns:
        Tensor of returns
    """
    returns = []
    R = 0
    
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    
    return torch.tensor(returns, dtype=torch.float32)


def batch_observations(obs_list: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Convert list of observation dictionaries to batched tensors.
    
    Args:
        obs_list: List of observation dicts from environment
    
    Returns:
        Dictionary with batched tensors
    """
    # Stack simple tensors
    batched = {}
    batched['hand'] = torch.stack([torch.FloatTensor(o['your_hand']) for o in obs_list])
    batched['deck_info'] = torch.stack([
        torch.FloatTensor([o['deck_size'], o['exploding_kittens_in_deck']]) 
        for o in obs_list
    ])
    batched['discard'] = torch.stack([torch.FloatTensor(o['discard_counts']) for o in obs_list])
    batched['game_state'] = torch.stack([
        torch.FloatTensor([
            float(o['have_drawn_this_turn']), 
            float(o['holding_exploding_kitten'])
        ]) 
        for o in obs_list
    ])
    
    # See the future encoding
    see_future_list = []
    for o in obs_list:
        see_future = np.zeros(40)
        if o['see_future_active']:
            for i, card in enumerate(o['top_3_cards']):
                if card is not None:
                    see_future[i * 13 + card] = 1.0
            see_future[39] = 1.0
        see_future_list.append(see_future)
    batched['see_future'] = torch.FloatTensor(np.array(see_future_list))
    
    # Stack opponent data
    opponent_data = []
    for o in obs_list:
        opp_features = []
        for opp in o['opponents']:
            features = np.concatenate([
                [opp['num_cards']],
                [float(opp['is_alive'])],
                [opp['turns_remaining']],
                opp['min_counts'],
                opp['max_counts'],
            ])
            opp_features.append(features)
        opponent_data.append(np.array(opp_features))
    
    batched['opponents'] = torch.FloatTensor(np.array(opponent_data))
    
    return batched


def batch_action_masks(mask_list: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Convert list of action mask dictionaries to batched tensors.
    
    Args:
        mask_list: List of action mask dicts
    
    Returns:
        Dictionary with batched mask tensors
    """
    return {
        'head1': torch.stack([torch.BoolTensor(m['head1']) for m in mask_list]),
        'head2': torch.stack([torch.BoolTensor(m['head2']) for m in mask_list]),
        'head3': torch.stack([torch.BoolTensor(m['head3']) for m in mask_list]),
        'head4': torch.stack([torch.BoolTensor(m['head4']) for m in mask_list]),
        'head5': torch.stack([torch.BoolTensor(m['head5']) for m in mask_list]),
    }


def batch_actions(action_list: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Convert list of action dictionaries to batched tensors.
    
    Args:
        action_list: List of action dicts
    
    Returns:
        Dictionary with batched action tensors
    """
    return {
        'head1': torch.LongTensor([a['head1'] for a in action_list]),
        'head2': torch.LongTensor([a['head2'] for a in action_list]),
        'head3': torch.LongTensor([a['head3'] for a in action_list]),
        'head4': torch.LongTensor([a['head4'] for a in action_list]),
        'head5': torch.LongTensor([a['head5'] for a in action_list]),
    }


# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    """Example of how to use RolloutStorage."""
    
    # Create storage
    storage = RolloutStorage(num_steps=2048)
    
    # Simulate collecting experience
    print("Simulating experience collection...")
    for step in range(100):
        # Dummy data
        obs = {
            'hand': np.random.rand(13),
            'opponents': [{
                'num_cards': 5,
                'is_alive': True,
                'turns_remaining': 1,
                'min_counts': np.zeros(13),
                'max_counts': np.ones(13) * 4,
            }] * 4,
            'deck_size': 20,
            'exploding_kittens_in_deck': 2,
            'discard_counts': np.zeros(13),
            'have_drawn_this_turn': False,
            'holding_exploding_kitten': False,
            'top_3_cards': [None, None, None],
            'see_future_active': False,
        }
        
        action = {
            'head1': 0,
            'head2': 0,
            'head3': 0,
            'head4': 0,
            'head5': 0,
        }
        
        action_log_prob = torch.tensor(-1.5)
        value_pred = torch.tensor(0.5)
        reward = 0.0 if step < 99 else 1.0  # Win at the end
        mask = 1.0 if step < 99 else 0.0
        action_mask = {
            'head1': np.ones(10, dtype=bool),
            'head2': np.ones(4, dtype=bool),
            'head3': np.ones(13, dtype=bool),
            'head4': np.ones(13, dtype=bool),
            'head5': np.ones(50, dtype=bool),
        }
        
        storage.insert(obs, action, action_log_prob, value_pred, reward, mask, action_mask)
    
    print(f"Collected {len(storage)} steps")
    
    # Compute advantages
    print("\nComputing advantages...")
    storage.compute_advantages(None, gamma=0.99, gae_lambda=0.95)
    
    print(f"Advantages: mean={storage.advantages.mean():.3f}, std={storage.advantages.std():.3f}")
    print(f"Returns: mean={storage.returns.mean():.3f}, std={storage.returns.std():.3f}")
    
    # Get batches
    print("\nGenerating mini-batches...")
    num_mini_batch = 4
    for i, batch in enumerate(storage.get_batches(num_mini_batch)):
        obs_batch, actions_batch, masks_batch, values_batch, returns_batch, \
            old_log_probs_batch, adv_batch = batch
        
        print(f"Batch {i+1}: {len(obs_batch)} samples")
    
    print("\n✅ RolloutStorage test complete!")