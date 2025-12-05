"""
Evaluate and compare trained offline RL policies against baseline opponents.
Tests BC, CQL, IQL, and MARWIL trained policies.
"""

import ray
import numpy as np
import argparse
import json
from pathlib import Path
from env.ludo_gym_env import LudoGymEnv
from policies.policy_random import Policy_Random
from policies.policy_heuristic import Policy_Heuristic
from policies.milestone2 import Policy_Milestone2
from ray.rllib.algorithms.algorithm import Algorithm


class RLlibPolicy:
    """Wrapper to use RLlib trained policy in the Ludo environment."""
    
    def __init__(self, checkpoint_path, encoding_type='handcrafted'):
        self.encoding_type = encoding_type
        self.checkpoint_path = checkpoint_path
        
        # Load the trained algorithm
        print(f"Loading policy from {checkpoint_path}")
        self.algo = Algorithm.from_checkpoint(checkpoint_path)
        
    def get_action(self, state, action_space):
        """
        Get action from the RLlib policy.
        
        Args:
            state: Ludo game state
            action_space: List of valid action tuples
            
        Returns:
            Action tuple
        """
        from misc.state_encoding import encode_handcrafted_state, encode_onehot_state
        
        # Encode state
        if self.encoding_type == 'handcrafted':
            _, _, encoded_state = encode_handcrafted_state(state)
        elif self.encoding_type == 'onehot':
            _, _, encoded_state = encode_onehot_state(state)
        else:
            raise ValueError(f"Unknown encoding_type: {self.encoding_type}")
        
        # Get action from policy (without exploration)
        action = self.algo.compute_single_action(
            encoded_state,
            explore=False
        )
        
        # Convert discrete action to tuple
        dice_index = action // 4
        goti_index = action % 4
        action_tuple = (dice_index, goti_index)
        
        # Check if action is valid
        if action_tuple in action_space:
            return action_tuple
        
        # If not valid, return first valid action (fallback)
        return action_space[0] if action_space else None


def evaluate_policy(policy, opponent_policy, env, num_episodes=100):
    """
    Evaluate a policy against an opponent.
    
    Args:
        policy: Policy to evaluate
        opponent_policy: Opponent policy
        env: Ludo environment
        num_episodes: Number of episodes to run
        
    Returns:
        Dictionary with evaluation results
    """
    wins = 0
    total_reward = 0
    episode_lengths = []
    
    for _ in range(num_episodes):
        state = env.reset()
        terminated = False
        episode_reward = 0
        episode_length = 0
        
        # Determine which player is the agent (0 or 1)
        agent_player = env.agent_player
        
        while not terminated:
            player_turn = state[4]
            action_space = env.get_action_space()
            
            # Choose action based on whose turn it is
            if player_turn == agent_player:
                action = policy.get_action(state, action_space)
            else:
                action = opponent_policy.get_action(state, action_space)
            
            state = env.step(action)
            terminated = state[3]
            episode_length += 1
        
        # Check if agent won
        winner = state[4]
        if winner == agent_player:
            wins += 1
        
        episode_lengths.append(episode_length)
    
    win_rate = (wins / num_episodes) * 100
    avg_episode_length = np.mean(episode_lengths)
    
    return {
        'win_rate': win_rate,
        'wins': wins,
        'total_episodes': num_episodes,
        'avg_episode_length': avg_episode_length
    }


def evaluate_all_algorithms(
    checkpoint_dirs,
    encoding_type='handcrafted',
    num_episodes=100,
    output_file='evaluation_results.json'
):
    """
    Evaluate all trained algorithms against baseline opponents.
    
    Args:
        checkpoint_dirs: Dictionary mapping algorithm names to checkpoint directories
        encoding_type: State encoding type
        num_episodes: Number of episodes per evaluation
        output_file: Output file for results
    """
    # Define opponent policies
    opponents = {
        'Random': Policy_Random(),
        'Heuristic': Policy_Heuristic(),
        'Milestone2': Policy_Milestone2()
    }
    
    # Create environment for evaluation
    from env.ludo import Ludo
    env = Ludo()
    
    results = {}
    
    print("=" * 70)
    print("EVALUATING OFFLINE RL ALGORITHMS")
    print("=" * 70)
    
    for algo_name, checkpoint_path in checkpoint_dirs.items():
        print(f"\n{'='*70}")
        print(f"Evaluating {algo_name}")
        print(f"Checkpoint: {checkpoint_path}")
        print(f"{'='*70}")
        
        # Load trained policy
        try:
            trained_policy = RLlibPolicy(checkpoint_path, encoding_type)
        except Exception as e:
            print(f"Error loading {algo_name}: {e}")
            continue
        
        results[algo_name] = {}
        
        for opponent_name, opponent_policy in opponents.items():
            print(f"\n{algo_name} vs {opponent_name}:")
            
            eval_results = evaluate_policy(
                trained_policy,
                opponent_policy,
                env,
                num_episodes=num_episodes
            )
            
            results[algo_name][opponent_name] = eval_results
            
            print(f"  Win Rate: {eval_results['win_rate']:.2f}%")
            print(f"  Wins: {eval_results['wins']}/{eval_results['total_episodes']}")
            print(f"  Avg Episode Length: {eval_results['avg_episode_length']:.2f}")
    
    # Print summary comparison
    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON")
    print("=" * 70)
    print(f"\n{'Algorithm':<15} {'vs Random':<12} {'vs Heuristic':<15} {'vs Milestone2':<15}")
    print("-" * 70)
    
    for algo_name in results.keys():
        vs_random = results[algo_name].get('Random', {}).get('win_rate', 0)
        vs_heuristic = results[algo_name].get('Heuristic', {}).get('win_rate', 0)
        vs_milestone2 = results[algo_name].get('Milestone2', {}).get('win_rate', 0)
        
        print(f"{algo_name:<15} {vs_random:>6.2f}%      {vs_heuristic:>6.2f}%         {vs_milestone2:>6.2f}%")
    
    # Save results to file
    output_path = Path(output_file)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate offline RL algorithms")
    parser.add_argument(
        '--bc_checkpoint',
        type=str,
        required=True,
        help='Path to BC checkpoint'
    )
    parser.add_argument(
        '--cql_checkpoint',
        type=str,
        required=True,
        help='Path to CQL checkpoint'
    )
    parser.add_argument(
        '--iql_checkpoint',
        type=str,
        required=True,
        help='Path to IQL checkpoint'
    )
    parser.add_argument(
        '--marwil_checkpoint',
        type=str,
        required=True,
        help='Path to MARWIL checkpoint'
    )
    parser.add_argument(
        '--encoding_type',
        type=str,
        default='handcrafted',
        choices=['handcrafted', 'onehot'],
        help='State encoding type'
    )
    parser.add_argument(
        '--num_episodes',
        type=int,
        default=100,
        help='Number of episodes for evaluation'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default='evaluation_results.json',
        help='Output file for results'
    )
    
    args = parser.parse_args()
    
    checkpoint_dirs = {
        'BC': args.bc_checkpoint,
        'CQL': args.cql_checkpoint,
        'IQL': args.iql_checkpoint,
        'MARWIL': args.marwil_checkpoint
    }
    
    evaluate_all_algorithms(
        checkpoint_dirs=checkpoint_dirs,
        encoding_type=args.encoding_type,
        num_episodes=args.num_episodes,
        output_file=args.output_file
    )

