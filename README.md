# Exploding Kittens RL

Reinforcement Learning agent for the board game Exploding Kittens using PPO (Proximal Policy Optimization).

## Project Structure

```
├── core/               # Game logic
│   ├── card_types.py   # Card and action enums
│   ├── game_config.py  # Game configuration
│   └── game_state.py   # Game state representation
├── env/                # Environment
│   └── environment.py  # Main game environment
├── handlers/           # Action handlers
│   ├── base.py         # Base handler class
│   ├── basic_actions.py
│   ├── special_actions.py
│   └── combo_actions.py
├── networks/           # Neural networks
│   ├── policy.py       # Main policy network
│   ├── encoders.py     # State encoders
│   └── action_heads.py # Action output heads
├── RL/                 # RL algorithms
│   ├── ppo.py          # PPO implementation
│   └── rollout_storage.py  # Experience buffer
├── utils/              # Utilities
│   ├── observation_utils.py
│   └── env_utils.py
├── tests/              # Tests
│   └── environment_test.py
├── train.py            # Training script
└── evaluate.py         # Evaluation script
```

## Architecture

The agent uses a multi-head policy network inspired by the Settlers of Catan RL implementation:

1. **Encoders**: Separate encoders for hand, opponents, and game state
2. **Action Heads**: 5 specialized heads for different action dimensions:
   - Head 1: Action type (draw, play card, etc.)
   - Head 2: Target player
   - Head 3: Card type for combos
   - Head 4: Card type to request
   - Head 5: Defuse position

3. **Training**: PPO with GAE (Generalized Advantage Estimation)

## Installation

```bash
pip install torch numpy
```

## Usage

**IMPORTANT:** Always activate the virtual environment first or use `python3`:

```bash
# Option 1: Activate venv (recommended)
source .venv/bin/activate
python train.py

# Option 2: Use python3 directly
python3 train.py

# Option 3: Use venv python directly
.venv/bin/python train.py
```

### Training

```bash
python train.py
```

This will:
- Train for 1000 updates (configurable in `TrainingArgs`)
- Save checkpoints every 50 updates
- Log progress every 10 updates
- Save final model to `final_model.pt`

### Evaluation

```bash
python evaluate.py final_model.pt 100
```

This plays 100 games with the trained agent against 3 random opponents and reports win rate.

### Testing

```bash
python tests/test_game_rules.py        # turn order, Attack/Skip, Defuse, rewards, mask/handler agreement
python tests/test_training_smoke.py    # collector, GAE episode boundaries, PPO update, self-play rollout
python tests/environment_test.py       # random-play statistics
```

> The checked-in `.venv` was created on macOS and will not run on Linux. Recreate it with
> `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`.

## Key Features

- **Modular action handlers**: Easy to extend with new card types
- **Opponent modeling**: Tracks min/max card counts for opponents
- **Action masking**: Only valid actions are considered
- **Multi-player support**: 2-5 players
- **See the Future tracking**: Remembers visible cards

## Training Tips

1. **Start small**: Begin with 2-3 players for faster training
2. **Adjust hyperparameters**: See `TrainingArgs` in `train.py`
3. **Monitor win rate**: Should exceed 25% (random baseline) after training
4. **Self-play**: For stronger agents, implement self-play (all players use policy)

## Next Steps

- [x] Implement self-play training (`train_selfplay.py`)
- [ ] Reactive Nope (currently a no-op that burns a card)
- [ ] Fix opponent-knowledge tracking (`opponent_min/max_counts`, see handlers/base.py)
- [ ] Add curriculum learning (start with 2 players, increase gradually)
- [ ] Tune hyperparameters
- [ ] Add more sophisticated reward shaping
- [ ] Implement opponent modeling in the policy
- [ ] Add support for expansion packs

## References

- [Settlers of Catan RL](https://settlers-rl.github.io/)
- [PPO Paper](https://arxiv.org/abs/1707.06347)
- [Exploding Kittens Rules](https://www.explodingkittens.com/pages/rules)
