import numpy as np
import torch
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
import os

from policy_dqn import Policy_DQN
from policy_heuristic import Policy_Heuristic
from policy_random import Policy_Random
from milestone2 import Policy_Milestone2
from ludo import Ludo


def collect_expert_demonstrations(num_episodes=5000, opponent_policy=None, device=None):
    heuristic_policy = Policy_Heuristic()

    # Create a temporary DQN instance just for state encoding
    temp_dqn = Policy_DQN(training_mode=False, device=device)

    expert_data = []

    iterator = tqdm(range(num_episodes), desc="Collecting expert data")

    for episode in iterator:
        env = Ludo()
        state = env.reset()
        player_turn = state[4]

        # Randomly assign which player uses heuristic (for diversity)
        heuristic_player = episode % 2

        while not env.terminated:
            action_space = env.get_action_space()

            if not action_space:
                state = env.step(None)
                player_turn = state[4]
                continue

            # Get action from heuristic if it's heuristic player's turn
            if player_turn == heuristic_player:
                action = heuristic_policy.get_action(state, action_space)

                if action is not None:
                    # Encode state using DQN's encoding method
                    state_encoded = temp_dqn.encode_state(state)

                    # Convert action to action index
                    dice_idx, goti_idx = action
                    action_idx = dice_idx * 4 + goti_idx
                    action_idx = max(0, min(action_idx, temp_dqn.max_actions - 1))

                    expert_data.append((state_encoded, action_idx, action_space))
            else:
                # Opponent's turn
                action = opponent_policy.get_action(state, action_space)

            state = env.step(action)
            player_turn = state[4]

    return expert_data


def behavioral_cloning_train(
    dqn_agent,
    expert_data,
    batch_size=512,
    epochs=10,
    learning_rate=0.001,
    use_scheduler=True,
):
    # Extract states and actions
    expert_states = np.array([data[0] for data in expert_data])
    expert_actions = np.array([data[1] for data in expert_data])

    # Convert to tensors
    expert_states = torch.FloatTensor(expert_states).to(dqn_agent.device)
    expert_actions = torch.LongTensor(expert_actions).to(dqn_agent.device)

    # Create a temporary optimizer with the specified learning rate
    temp_optimizer = torch.optim.Adam(
        dqn_agent.policy_net.parameters(), lr=learning_rate
    )

    # Create learning rate scheduler if requested
    if use_scheduler:
        eta_min = learning_rate * 0.01

        scheduler = lr_scheduler.CosineAnnealingLR(
            temp_optimizer, T_max=epochs, eta_min=eta_min
        )
    else:
        scheduler = None

    # Supervised learning: train network to predict heuristic's actions
    dqn_agent.policy_net.train()

    dataset_size = len(expert_states)
    num_batches = (dataset_size + batch_size - 1) // batch_size

    for epoch in range(epochs):
        total_loss = 0.0
        correct_predictions = 0
        current_lr = temp_optimizer.param_groups[0]["lr"]

        # Shuffle data
        indices = torch.randperm(dataset_size)
        expert_states_shuffled = expert_states[indices]
        expert_actions_shuffled = expert_actions[indices]

        iterator = tqdm(range(num_batches), desc=f"Epoch {epoch + 1}/{epochs}")

        for batch_idx in iterator:
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, dataset_size)

            batch_states = expert_states_shuffled[start_idx:end_idx]
            batch_actions = expert_actions_shuffled[start_idx:end_idx]

            # Get Q-values from network
            q_values = dqn_agent.policy_net(batch_states)

            # Use cross-entropy loss: maximize Q-value for expert action
            # Convert to log-probabilities via softmax
            log_probs = F.log_softmax(q_values, dim=1)
            loss = F.nll_loss(log_probs, batch_actions)

            temp_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dqn_agent.policy_net.parameters(), 1.0)
            temp_optimizer.step()

            # Track accuracy
            predicted_actions = q_values.argmax(dim=1)
            correct_predictions += (predicted_actions == batch_actions).sum().item()
            total_loss += loss.item()

        # Update learning rate scheduler
        if scheduler is not None:
            scheduler.step()

        avg_loss = total_loss / num_batches
        accuracy = correct_predictions / dataset_size

        print(
            f"   Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2%}, LR: {current_lr:.6f}"
        )

    # Update target network
    dqn_agent.target_net.load_state_dict(dqn_agent.policy_net.state_dict())

    # Final evaluation
    with torch.no_grad():
        dqn_agent.policy_net.eval()
        q_values = dqn_agent.policy_net(expert_states)
        predicted_actions = q_values.argmax(dim=1)
        final_accuracy = (predicted_actions == expert_actions).float().mean().item()

        print(f"\n   ✅ Training complete! Final accuracy: {final_accuracy:.2%}\n")

        dqn_agent.policy_net.train()


def pretrain_dqn(
    num_episodes=5000,
    batch_size=512,
    epochs=10,
    learning_rate=0.001,
    save_path="models/pretrained_model.pth",
    device="cuda:1",
    use_scheduler=True,
):
    # Set device
    if device is None:
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    print(f"Using device: {device}")

    # Create DQN agent
    dqn_agent = Policy_DQN(
        training_mode=True,
        learning_rate=learning_rate,
        epsilon_start=0.0,  # No exploration during pre-training
        epsilon_end=0.0,
        device=device,
    )

    # Step 1: Collect expert demonstrations
    print("Step 1: Collecting expert demonstrations...")

    expert_data = collect_expert_demonstrations(
        num_episodes=num_episodes,
        opponent_policy=Policy_Random(),
        device=device,
    )

    expert_data += collect_expert_demonstrations(
        num_episodes=num_episodes,
        opponent_policy=Policy_Heuristic(),
        device=device,
    )

    expert_data += collect_expert_demonstrations(
        num_episodes=num_episodes,
        opponent_policy=Policy_Milestone2(),
        device=device,
    )

    print(f"✅ Collected {len(expert_data)} expert state-action pairs\n")

    # Step 2: Train DQN using behavioral cloning
    print("Step 2: Training DQN...")

    behavioral_cloning_train(
        dqn_agent,
        expert_data,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        use_scheduler=use_scheduler,
    )

    # Step 3: Save the pretrained model
    print("Step 3: Saving pretrained model...")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    dqn_agent.save(save_path)

    print(f"✅ Model saved to {save_path}\n")

    return dqn_agent


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Pre-train DQN using behavioral cloning"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=500000,
        help="Number of episodes to collect from heuristic",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1000,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Batch size for training (default: 512)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="models/pretrained_model.pth",
        help="Path to save pretrained model (default: models/pretrained_model.pth)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:1",
        help="Device to use (default: cuda:1)",
    )
    parser.add_argument(
        "--no-scheduler",
        action="store_true",
        help="Disable learning rate scheduler (default: enabled with cosine annealing)",
    )

    args = parser.parse_args()

    pretrain_dqn(
        num_episodes=args.episodes,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        save_path=args.save_path,
        device=args.device,
        use_scheduler=not args.no_scheduler,
    )
