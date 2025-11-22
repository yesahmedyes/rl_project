"""
Test script to verify parallel training works correctly.
Runs a short training session with parallel workers.
"""

from train_dqn import train_dqn
import multiprocessing as mp

if __name__ == "__main__":
    print("=" * 60)
    print("TESTING PARALLEL TRAINING")
    print("=" * 60)
    print(f"\nDetected {mp.cpu_count()} CPU cores")
    print("\nRunning short test with 4 workers, 100 episodes...")
    print()

    try:
        agent = train_dqn(
            n_episodes=100,
            num_workers=4,
            episodes_per_batch=2,
            learning_steps_per_batch=8,
            eval_interval=50,
            save_interval=100,
            snapshot_interval=25,
            log_interval=25,
            save_path="test_parallel_dqn.pth",
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay=0.99,
            batch_size=32,
            buffer_size=10000,
        )

        print("\n" + "=" * 60)
        print("✅ TEST PASSED: Parallel training completed successfully!")
        print("=" * 60)

    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌ TEST FAILED: {e}")
        print("=" * 60)
        import traceback

        traceback.print_exc()
