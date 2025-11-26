"""
Script to verify the PPO training setup is working correctly.
"""

import sys
import warnings

warnings.filterwarnings("ignore")


def check_imports():
    """Check if all required packages are installed."""
    print("Checking imports...")

    required_packages = [
        ("gymnasium", "gym"),
        ("stable_baselines3", "sb3"),
        ("sb3_contrib", "sb3_contrib"),
        ("torch", "PyTorch"),
        ("numpy", "NumPy"),
        ("tqdm", "tqdm"),
    ]

    missing = []
    for module, name in required_packages:
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} (missing)")
            missing.append(name)

    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False

    print("\nAll required packages installed!")
    return True


def check_cuda():
    """Check CUDA availability."""
    print("\nChecking CUDA...")
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        if cuda_available:
            n_gpus = torch.cuda.device_count()
            print(f"  ✓ CUDA available")
            print(f"  ✓ {n_gpus} GPU(s) detected")
            for i in range(n_gpus):
                gpu_name = torch.cuda.get_device_name(i)
                print(f"    - GPU {i}: {gpu_name}")
        else:
            print("  ✗ CUDA not available (will use CPU)")
            print("    Training will be slower without GPU")

        return cuda_available
    except Exception as e:
        print(f"  ✗ Error checking CUDA: {e}")
        return False


def test_environment():
    """Test if the Ludo Gym environment works."""
    print("\nTesting Ludo Gym environment...")
    try:
        from env.ludo_gym_env import LudoGymEnv

        # Test handcrafted encoding
        env = LudoGymEnv(encoding_type="handcrafted", opponent_type="random")
        obs, info = env.reset(seed=42)
        print(f"  ✓ Handcrafted encoding environment created")
        print(f"    - Observation shape: {obs.shape}")
        print(f"    - Action space: {env.action_space}")

        # Test one-hot encoding
        env = LudoGymEnv(encoding_type="onehot", opponent_type="random")
        obs, info = env.reset(seed=42)
        print(f"  ✓ One-hot encoding environment created")
        print(f"    - Observation shape: {obs.shape}")

        # Test step
        action_mask = info["action_mask"]
        valid_actions = [i for i in range(len(action_mask)) if action_mask[i] == 1]
        if valid_actions:
            action = valid_actions[0]
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"  ✓ Environment step successful")

        env.close()
        print("\nEnvironment tests passed!")
        return True

    except Exception as e:
        print(f"  ✗ Error testing environment: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_vectorized_env():
    """Test vectorized environment creation."""
    print("\nTesting vectorized environment...")
    try:
        from training.vec_env_factory import make_vec_env

        # Create small vectorized environment
        vec_env = make_vec_env(
            n_envs=2,
            encoding_type="handcrafted",
            opponent_type="random",
            use_subprocess=False,  # Use DummyVecEnv for testing
        )

        obs = vec_env.reset()
        print(f"  ✓ Vectorized environment created")
        print(f"    - Number of envs: 2")
        print(f"    - Observation shape: {obs.shape}")

        vec_env.close()
        print("\nVectorized environment test passed!")
        return True

    except Exception as e:
        print(f"  ✗ Error testing vectorized environment: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_ppo_model():
    """Test PPO model creation."""
    print("\nTesting PPO model creation...")
    try:
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
        from training.vec_env_factory import make_vec_env
        import torch

        # Create environment
        vec_env = make_vec_env(
            n_envs=2,
            encoding_type="handcrafted",
            opponent_type="random",
            use_subprocess=False,
        )

        # Create model
        model = MaskablePPO(
            policy=MaskableActorCriticPolicy,
            env=vec_env,
            learning_rate=3e-4,
            n_steps=128,
            batch_size=64,
            verbose=0,
        )

        print(f"  ✓ MaskablePPO model created")
        print(f"    - Policy: MaskableActorCriticPolicy")
        print(f"    - Device: {model.device}")

        # Test short training
        print("  Testing short training run (256 steps)...")
        model.learn(total_timesteps=256, progress_bar=False)
        print(f"  ✓ Training successful")

        vec_env.close()
        print("\nPPO model test passed!")
        return True

    except Exception as e:
        print(f"  ✗ Error testing PPO model: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all verification tests."""
    print("=" * 60)
    print("Ludo PPO Training Setup Verification")
    print("=" * 60)

    all_passed = True

    # Check imports
    if not check_imports():
        all_passed = False
        print("\n❌ Setup incomplete: Missing dependencies")
        print("   Run: pip install -r requirements.txt")
        return

    # Check CUDA
    check_cuda()

    # Test environment
    if not test_environment():
        all_passed = False

    # Test vectorized environment
    if not test_vectorized_env():
        all_passed = False

    # Test PPO model
    if not test_ppo_model():
        all_passed = False

    # Final summary
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ All tests passed! Setup is ready for training.")
        print("=" * 60)
        print("\nNext steps:")
        print("  1. To start training: bash training/launch_training.sh")
        print("  2. To monitor training: tensorboard --logdir=./tensorboard_logs")
        print("  3. Read TRAINING_README.md for detailed instructions")
    else:
        print("❌ Some tests failed. Please fix the issues above.")
        print("=" * 60)


if __name__ == "__main__":
    main()
