# Quick Start Guide: Offline RL for Ludo

## 🚀 30-Second Start

```bash
# 1. Install dependencies
pip install -r requirements_offline_rl.txt

# 2. Run everything
python run_all_experiments.py \
    --num_data_episodes 1000 \
    --num_train_iterations 100
```

That's it! The script will:
- ✅ Collect 1000 episodes from heuristic expert
- ✅ Train BC, CQL, IQL, and MARWIL
- ✅ Evaluate all 4 algorithms
- ✅ Save results to `evaluation_results_heuristic_handcrafted.json`

## 📊 What You'll Get

After ~30-60 minutes (depending on hardware), you'll have:

1. **Trained Models** in `checkpoints/`
   - `checkpoints/bc/` - Behavior Cloning
   - `checkpoints/cql/` - Conservative Q-Learning  
   - `checkpoints/iql/` - Implicit Q-Learning
   - `checkpoints/marwil/` - MARWIL

2. **Offline Dataset** in `offline_data/heuristic_handcrafted/`

3. **Evaluation Results** showing win rates against:
   - Random opponent
   - Heuristic opponent
   - Milestone2 opponent

## 🎯 Algorithm Overview

| Algorithm | Type | Best For |
|-----------|------|----------|
| **BC** | Pure Imitation | Fast baseline, mimics expert exactly |
| **CQL** | Offline RL | Learning from imperfect data |
| **IQL** | Offline RL | Stable training, handles data variance |
| **MARWIL** | Hybrid IL+RL | Balance between imitation and improvement |

## ⚙️ Common Options

### Quick Test (5 minutes)
```bash
python run_all_experiments.py \
    --num_data_episodes 100 \
    --num_train_iterations 10 \
    --num_eval_episodes 20
```

### High Quality (2-3 hours)
```bash
python run_all_experiments.py \
    --num_data_episodes 5000 \
    --num_train_iterations 500 \
    --num_eval_episodes 500
```

### Use Different Expert
```bash
# Train from milestone2 expert
python run_all_experiments.py \
    --expert_type milestone2 \
    --num_data_episodes 1000

# Train from random baseline (for comparison)
python run_all_experiments.py \
    --expert_type random \
    --num_data_episodes 1000
```

### Use Different Encoding
```bash
# Use one-hot encoding (946 dims instead of 70)
python run_all_experiments.py \
    --encoding_type onehot \
    --num_data_episodes 1000
```

## 🔍 Viewing Results

Results are saved as JSON:

```python
import json

with open('evaluation_results_heuristic_handcrafted.json', 'r') as f:
    results = json.load(f)

# Print BC results
print("BC Results:")
for opponent, metrics in results['BC'].items():
    print(f"  vs {opponent}: {metrics['win_rate']:.2f}% win rate")

# Compare all algorithms vs Heuristic
print("\nAll Algorithms vs Heuristic:")
for algo in ['BC', 'CQL', 'IQL', 'MARWIL']:
    win_rate = results[algo]['Heuristic']['win_rate']
    print(f"  {algo}: {win_rate:.2f}%")
```

## 📈 Expected Performance

With 1000 episodes of heuristic data and 100 training iterations:

```
Algorithm vs Random vs Heuristic vs Milestone2
BC        ~90-95%  ~45-55%      ~30-40%
CQL       ~85-95%  ~50-60%      ~35-45%
IQL       ~85-95%  ~50-60%      ~35-45%
MARWIL    ~90-95%  ~48-58%      ~32-42%
```

## 🔧 Step-by-Step (For More Control)

### 1. Collect Data Only
```bash
python collect_offline_data.py \
    --num_episodes 1000 \
    --expert_type heuristic
```

### 2. Train Specific Algorithm
```bash
# Train just BC
python train_bc.py \
    --data_dir offline_data/heuristic_handcrafted \
    --num_iterations 100

# Train just CQL
python train_cql.py \
    --data_dir offline_data/heuristic_handcrafted \
    --num_iterations 100
```

### 3. Evaluate Specific Checkpoint
```bash
python evaluate_policies.py \
    --bc_checkpoint checkpoints/bc/BC_heuristic_handcrafted/final \
    --cql_checkpoint checkpoints/cql/CQL_heuristic_handcrafted/final \
    --iql_checkpoint checkpoints/iql/IQL_heuristic_handcrafted/final \
    --marwil_checkpoint checkpoints/marwil/MARWIL_heuristic_handcrafted_beta1.0/final \
    --num_episodes 500
```

## 🐛 Troubleshooting

### "Out of Memory"
```bash
# Use smaller batch sizes
python train_bc.py --data_dir ... --train_batch_size 128
python train_cql.py --data_dir ... --train_batch_size 64
```

### "Ray is already running"
```bash
ray stop
# Then run your command again
```

### "Data directory not found"
Make sure you collected data first:
```bash
python collect_offline_data.py --num_episodes 1000
```

## 🎓 Next Steps

1. **Hyperparameter Tuning:** See `README_OFFLINE_RL.md` for tuning tips
2. **Self-Play:** Use trained policies as initialization for PPO/DQN
3. **Ensemble:** Combine multiple algorithms for better performance
4. **Visualization:** Add TensorBoard logging to monitor training

## 📚 More Information

- Full documentation: `README_OFFLINE_RL.md`
- Algorithm details: See individual `train_*.py` files
- RLlib docs: https://docs.ray.io/en/latest/rllib/

## ❓ Questions?

Common questions:

**Q: Which algorithm should I use?**  
A: Start with BC for a simple baseline, then try CQL or IQL for potential improvements.

**Q: How much data do I need?**  
A: 1000 episodes is a good start. Try 2000-5000 for better results.

**Q: How long does training take?**  
A: On a modern CPU: ~30-60 min for 1000 episodes + 100 iterations per algorithm.

**Q: Can I use GPU?**  
A: Yes! PyTorch will automatically use GPU if available. Install with:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

**Q: Why are my results different?**  
A: Training is stochastic. Run multiple seeds or increase data/training iterations.

