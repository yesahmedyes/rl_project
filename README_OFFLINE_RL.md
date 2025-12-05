# Offline RL Training for Ludo

This project implements and compares four offline RL/imitation learning algorithms for training Ludo game agents:

1. **Behavior Cloning (BC)** - Pure imitation learning
2. **Conservative Q-Learning (CQL)** - Conservative offline RL
3. **Implicit Q-Learning (IQL)** - Expectile-based offline RL
4. **MARWIL** - Hybrid imitation learning and policy gradient

## Overview

The offline RL approach allows training agents from pre-collected expert demonstrations without needing online interaction with the environment. This is particularly useful for:
- Bootstrapping policies before self-play
- Learning from strong heuristic policies
- Reducing training time and computational costs

## Project Structure

```
.
├── collect_offline_data.py      # Collect expert demonstrations
├── train_bc.py                  # Train Behavior Cloning
├── train_cql.py                 # Train Conservative Q-Learning
├── train_iql.py                 # Train Implicit Q-Learning
├── train_marwil.py              # Train MARWIL
├── evaluate_policies.py         # Evaluate and compare trained policies
├── run_all_experiments.py       # Master script to run complete pipeline
├── requirements_offline_rl.txt  # Python dependencies
└── README_OFFLINE_RL.md        # This file
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements_offline_rl.txt
```

2. Verify Ray RLlib installation:
```bash
python -c "import ray; print(ray.__version__)"
```

## Quick Start

### Option 1: Run Complete Pipeline (Recommended)

Run everything with one command:

```bash
python run_all_experiments.py \
    --num_data_episodes 1000 \
    --encoding_type handcrafted \
    --expert_type heuristic \
    --num_train_iterations 100 \
    --num_eval_episodes 100
```

This will:
1. Collect 1000 episodes of offline data from heuristic expert
2. Train BC, CQL, IQL, and MARWIL for 100 iterations each
3. Evaluate all trained policies against Random, Heuristic, and Milestone2 opponents
4. Save results to `evaluation_results_heuristic_handcrafted.json`

### Option 2: Step-by-Step

#### Step 1: Collect Offline Data

Collect expert demonstrations from a heuristic policy:

```bash
python collect_offline_data.py \
    --num_episodes 1000 \
    --encoding_type handcrafted \
    --expert_type heuristic \
    --opponent_type heuristic \
    --output_dir offline_data
```

**Parameters:**
- `--num_episodes`: Number of episodes to collect (default: 1000)
- `--encoding_type`: State encoding (`handcrafted` or `onehot`)
- `--expert_type`: Expert policy (`random`, `heuristic`, or `milestone2`)
- `--opponent_type`: Opponent policy type
- `--output_dir`: Directory to save collected data

**Output:** Data saved to `offline_data/{expert_type}_{encoding_type}/`

#### Step 2: Train Each Algorithm

**Behavior Cloning (BC):**
```bash
python train_bc.py \
    --data_dir offline_data/heuristic_handcrafted \
    --encoding_type handcrafted \
    --num_iterations 100 \
    --lr 1e-4 \
    --train_batch_size 512
```

**Conservative Q-Learning (CQL):**
```bash
python train_cql.py \
    --data_dir offline_data/heuristic_handcrafted \
    --encoding_type handcrafted \
    --num_iterations 100 \
    --lr 3e-4 \
    --train_batch_size 256 \
    --bc_iters 20000 \
    --temperature 1.0
```

**Implicit Q-Learning (IQL):**
```bash
python train_iql.py \
    --data_dir offline_data/heuristic_handcrafted \
    --encoding_type handcrafted \
    --num_iterations 100 \
    --lr 3e-4 \
    --train_batch_size 256 \
    --expectile 0.7 \
    --temperature 3.0
```

**MARWIL:**
```bash
python train_marwil.py \
    --data_dir offline_data/heuristic_handcrafted \
    --encoding_type handcrafted \
    --num_iterations 100 \
    --lr 1e-4 \
    --train_batch_size 512 \
    --beta 1.0 \
    --vf_coeff 1.0
```

**Output:** Checkpoints saved to `checkpoints/{algorithm_name}/`

#### Step 3: Evaluate All Policies

```bash
python evaluate_policies.py \
    --bc_checkpoint checkpoints/bc/BC_heuristic_handcrafted/final \
    --cql_checkpoint checkpoints/cql/CQL_heuristic_handcrafted/final \
    --iql_checkpoint checkpoints/iql/IQL_heuristic_handcrafted/final \
    --marwil_checkpoint checkpoints/marwil/MARWIL_heuristic_handcrafted_beta1.0/final \
    --encoding_type handcrafted \
    --num_episodes 100
```

**Output:** Results saved to `evaluation_results.json`

## Algorithm Comparison

### Behavior Cloning (BC)
- **Type:** Pure imitation learning
- **How it works:** Directly learns to mimic expert actions via supervised learning
- **Pros:** Simple, stable, fast training
- **Cons:** Cannot improve beyond expert, no value function learning
- **Best for:** When expert is very strong and consistent

### Conservative Q-Learning (CQL)
- **Type:** Offline RL with conservative Q-function
- **How it works:** Learns Q-function that penalizes out-of-distribution actions
- **Pros:** Can potentially improve beyond expert, handles distribution shift
- **Cons:** More complex, requires careful tuning
- **Best for:** When expert data has some suboptimal actions

### Implicit Q-Learning (IQL)
- **Type:** Offline RL with expectile regression
- **How it works:** Learns value function using expectile regression to avoid bootstrapping errors
- **Pros:** Stable, doesn't require explicit conservatism
- **Cons:** May be slower to converge
- **Best for:** When data quality varies

### MARWIL
- **Type:** Hybrid imitation learning + policy gradient
- **How it works:** Weights expert actions by estimated advantages
- **Pros:** Balances imitation and RL, adjustable via beta parameter
- **Cons:** Requires good advantage estimation
- **Best for:** When you want to interpolate between BC and RL

## Hyperparameter Tuning

### BC Hyperparameters
- `lr`: Learning rate (try: 1e-4, 1e-3, 1e-5)
- `train_batch_size`: Batch size (try: 256, 512, 1024)

### CQL Hyperparameters
- `lr`: Learning rate (try: 1e-4, 3e-4, 1e-3)
- `bc_iters`: BC warmup iterations (try: 10000, 20000, 50000)
- `temperature`: CQL temperature (try: 0.5, 1.0, 2.0)
- `min_q_weight`: Conservative penalty weight (try: 1.0, 5.0, 10.0)

### IQL Hyperparameters
- `lr`: Learning rate (try: 1e-4, 3e-4, 1e-3)
- `expectile`: Value function expectile (try: 0.7, 0.8, 0.9)
- `temperature`: AWR temperature (try: 1.0, 3.0, 10.0)

### MARWIL Hyperparameters
- `lr`: Learning rate (try: 1e-4, 1e-3, 1e-5)
- `beta`: Advantage weighting (0=pure BC, higher=more RL) (try: 0.0, 0.5, 1.0, 2.0)
- `vf_coeff`: Value function coefficient (try: 0.5, 1.0, 2.0)

## Expected Results

After training on ~1000 episodes of heuristic expert data, you should expect:

| Algorithm | vs Random | vs Heuristic | vs Milestone2 |
|-----------|-----------|--------------|---------------|
| BC        | ~90-95%   | ~45-55%      | ~30-40%       |
| CQL       | ~85-95%   | ~50-60%      | ~35-45%       |
| IQL       | ~85-95%   | ~50-60%      | ~35-45%       |
| MARWIL    | ~90-95%   | ~48-58%      | ~32-42%       |

**Note:** These are rough estimates. Actual results depend on:
- Data quality and quantity
- Hyperparameter tuning
- Encoding type used
- Training iterations

## Next Steps: Self-Play

After pretraining with offline RL, you can:

1. Use the pretrained policy as initialization for PPO/DQN
2. Fine-tune with online RL in the environment
3. Implement self-play for further improvement

Example self-play setup:
```python
from env.ludo_gym_env import LudoGymEnv
from ray.rllib.algorithms.ppo import PPOConfig

# Load pretrained BC policy
pretrained_checkpoint = "checkpoints/bc/BC_heuristic_handcrafted/final"

# Create PPO config with pretrained weights
config = PPOConfig()
config.environment(env=LudoGymEnv, env_config={
    "encoding_type": "handcrafted",
    "opponent_type": "self"  # Self-play!
})

# Load pretrained weights and continue training
algo = config.build()
algo.restore(pretrained_checkpoint)

# Continue training with PPO
for i in range(100):
    result = algo.train()
    print(f"Iteration {i}: reward={result['episode_reward_mean']}")
```

## Troubleshooting

### Issue: Out of Memory
- **Solution:** Reduce `train_batch_size` or `num_data_episodes`

### Issue: Training is too slow
- **Solution:** 
  - Use `handcrafted` encoding instead of `onehot` (946 dims → 70 dims)
  - Reduce `num_iterations`
  - Use GPU if available

### Issue: Poor performance
- **Solution:**
  - Collect more expert data (try 2000-5000 episodes)
  - Use better expert (e.g., milestone2 instead of heuristic)
  - Tune hyperparameters
  - Try different encoding type

### Issue: Ray errors
- **Solution:**
  - Restart Ray: `ray stop` then try again
  - Check Ray version: `pip install -U ray[rllib]`
  - Initialize Ray explicitly: `ray.init(ignore_reinit_error=True)`

## References

- **BC:** Pomerleau, D. A. (1991). Efficient training of artificial neural networks for autonomous navigation.
- **CQL:** Kumar et al. (2020). Conservative Q-Learning for Offline Reinforcement Learning. NeurIPS.
- **IQL:** Kostrikov et al. (2021). Offline Reinforcement Learning with Implicit Q-Learning. ICLR.
- **MARWIL:** Wang et al. (2018). Exponentially Weighted Imitation Learning for Batched Historical Data. NeurIPS.
- **RLlib Documentation:** https://docs.ray.io/en/latest/rllib/

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{ludo_offline_rl_2025,
  title={Offline RL Training for Ludo Game},
  author={Your Name},
  year={2025},
  howpublished={\url{https://github.com/yourusername/ludo-offline-rl}}
}
```

## License

MIT License - See LICENSE file for details

