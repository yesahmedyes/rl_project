"""
Master script to run all offline RL experiments:
1. Collect offline data
2. Train BC, CQL, IQL, and MARWIL
3. Evaluate all trained policies
"""

import subprocess
import argparse
from pathlib import Path


def run_command(cmd, description):
    """Run a shell command and handle errors."""
    print("\n" + "=" * 70)
    print(f"RUNNING: {description}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 70 + "\n")
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"\nERROR: {description} failed with return code {result.returncode}")
        return False
    
    print(f"\nSUCCESS: {description} completed")
    return True


def main(
    num_data_episodes=1000,
    encoding_type='handcrafted',
    expert_type='heuristic',
    opponent_type='heuristic',
    num_train_iterations=100,
    num_eval_episodes=100,
    skip_data_collection=False,
    skip_training=False
):
    """
    Run the complete offline RL experiment pipeline.
    
    Args:
        num_data_episodes: Number of episodes to collect for offline data
        encoding_type: State encoding type
        expert_type: Expert policy for data collection
        opponent_type: Opponent policy type
        num_train_iterations: Number of training iterations for each algorithm
        num_eval_episodes: Number of episodes for evaluation
        skip_data_collection: Skip data collection if data already exists
        skip_training: Skip training if checkpoints already exist
    """
    # Define paths
    data_dir = f"offline_data/{expert_type}_{encoding_type}"
    
    # Step 1: Collect offline data
    if not skip_data_collection:
        collect_cmd = [
            'python', 'collect_offline_data.py',
            '--num_episodes', str(num_data_episodes),
            '--encoding_type', encoding_type,
            '--expert_type', expert_type,
            '--opponent_type', opponent_type,
            '--output_dir', 'offline_data'
        ]
        
        if not run_command(collect_cmd, "Data Collection"):
            return False
    else:
        print(f"\nSkipping data collection. Using existing data in {data_dir}")
    
    # Check if data exists
    if not Path(data_dir).exists():
        print(f"\nERROR: Data directory {data_dir} does not exist!")
        print("Run without --skip_data_collection to collect data first.")
        return False
    
    # Define checkpoint paths
    checkpoints = {}
    
    # Step 2: Train each algorithm
    algorithms = [
        ('BC', 'train_bc.py', {
            'lr': '1e-4',
            'train_batch_size': '512'
        }),
        ('CQL', 'train_cql.py', {
            'lr': '3e-4',
            'train_batch_size': '256',
            'bc_iters': '20000',
            'temperature': '1.0'
        }),
        ('IQL', 'train_iql.py', {
            'lr': '3e-4',
            'train_batch_size': '256',
            'expectile': '0.7'
        }),
        ('MARWIL', 'train_marwil.py', {
            'lr': '1e-4',
            'train_batch_size': '512',
            'beta': '1.0',
            'vf_coeff': '1.0'
        })
    ]
    
    if not skip_training:
        for algo_name, script_name, params in algorithms:
            print(f"\n{'='*70}")
            print(f"Training {algo_name}")
            print(f"{'='*70}")
            
            train_cmd = [
                'python', script_name,
                '--data_dir', data_dir,
                '--encoding_type', encoding_type,
                '--num_iterations', str(num_train_iterations),
                '--output_dir', f'checkpoints/{algo_name.lower()}'
            ]
            
            # Add algorithm-specific parameters
            for param_name, param_value in params.items():
                train_cmd.extend([f'--{param_name}', param_value])
            
            if not run_command(train_cmd, f"{algo_name} Training"):
                print(f"\nWARNING: {algo_name} training failed. Continuing with other algorithms...")
                continue
            
            # Store checkpoint path
            checkpoints[algo_name] = f"checkpoints/{algo_name.lower()}/{algo_name}_{expert_type}_{encoding_type}/final"
    else:
        print("\nSkipping training. Using existing checkpoints.")
        # Assume checkpoints exist at standard locations
        for algo_name, _, _ in algorithms:
            checkpoints[algo_name] = f"checkpoints/{algo_name.lower()}/{algo_name}_{expert_type}_{encoding_type}/final"
    
    # Step 3: Evaluate all trained policies
    print(f"\n{'='*70}")
    print("Evaluating All Trained Policies")
    print(f"{'='*70}")
    
    # Check if all checkpoints exist
    missing_checkpoints = []
    for algo_name, checkpoint_path in checkpoints.items():
        if not Path(checkpoint_path).exists():
            missing_checkpoints.append(algo_name)
            print(f"WARNING: Checkpoint for {algo_name} not found at {checkpoint_path}")
    
    if missing_checkpoints:
        print(f"\nERROR: Missing checkpoints for: {', '.join(missing_checkpoints)}")
        print("Train the algorithms first or provide correct checkpoint paths.")
        return False
    
    eval_cmd = [
        'python', 'evaluate_policies.py',
        '--bc_checkpoint', checkpoints['BC'],
        '--cql_checkpoint', checkpoints['CQL'],
        '--iql_checkpoint', checkpoints['IQL'],
        '--marwil_checkpoint', checkpoints['MARWIL'],
        '--encoding_type', encoding_type,
        '--num_episodes', str(num_eval_episodes),
        '--output_file', f'evaluation_results_{expert_type}_{encoding_type}.json'
    ]
    
    if not run_command(eval_cmd, "Policy Evaluation"):
        return False
    
    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"\nResults saved to: evaluation_results_{expert_type}_{encoding_type}.json")
    
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run complete offline RL experiment pipeline"
    )
    parser.add_argument(
        '--num_data_episodes',
        type=int,
        default=1000,
        help='Number of episodes to collect for offline data'
    )
    parser.add_argument(
        '--encoding_type',
        type=str,
        default='handcrafted',
        choices=['handcrafted', 'onehot'],
        help='State encoding type'
    )
    parser.add_argument(
        '--expert_type',
        type=str,
        default='heuristic',
        choices=['random', 'heuristic', 'milestone2'],
        help='Expert policy for data collection'
    )
    parser.add_argument(
        '--opponent_type',
        type=str,
        default='heuristic',
        choices=['random', 'heuristic', 'milestone2'],
        help='Opponent policy type'
    )
    parser.add_argument(
        '--num_train_iterations',
        type=int,
        default=100,
        help='Number of training iterations for each algorithm'
    )
    parser.add_argument(
        '--num_eval_episodes',
        type=int,
        default=100,
        help='Number of episodes for evaluation'
    )
    parser.add_argument(
        '--skip_data_collection',
        action='store_true',
        help='Skip data collection if data already exists'
    )
    parser.add_argument(
        '--skip_training',
        action='store_true',
        help='Skip training if checkpoints already exist'
    )
    
    args = parser.parse_args()
    
    success = main(
        num_data_episodes=args.num_data_episodes,
        encoding_type=args.encoding_type,
        expert_type=args.expert_type,
        opponent_type=args.opponent_type,
        num_train_iterations=args.num_train_iterations,
        num_eval_episodes=args.num_eval_episodes,
        skip_data_collection=args.skip_data_collection,
        skip_training=args.skip_training
    )
    
    exit(0 if success else 1)

