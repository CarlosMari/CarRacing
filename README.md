# CarRacing

Deep Q-learning on Gymnasium's [CarRacing-v2](https://gymnasium.farama.org/environments/box2d/car_racing/)
with discrete actions, learning directly from pixels.

- **Model** (`DQN.py`): a LeNet-style CNN (two conv + pool layers, three fully connected layers) that
  maps a 96×96 RGB frame to one Q-value per action, plus a replay memory.
- **Training** (`main.py`): DQN with a soft-updated target network (τ = 0.005), exponential ε-decay
  (1.0 → 0.05), AdamW (lr 1e-4) and batch size 256. It trains for 700 episodes on a GPU (50 on CPU),
  saves the policy and target weights as `.pth` files, and plots episode results.

`LeNet like Structure.png` and `Figure_1.png` are training plots. `logs/best_model.zip` is from an
earlier PPO attempt that did not train correctly; the DQN agent is the one that learns.

## Run

```bash
pip install "gymnasium[box2d]" torch matplotlib tqdm
python main.py
```

Status: work in progress (2023).
