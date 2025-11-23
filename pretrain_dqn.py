import numpy as np
import torch
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import pickle
import glob

from policy_dqn import Policy_DQN


class DemonstrationDataset(Dataset):
    def __init__(self, states, actions):
        self.states = torch.FloatTensor(states)
        self.actions = torch.LongTensor(actions)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]


def load_demonstrations(demonstrations_dir="demonstrations"):
    all_states = []
    all_actions = []

    # Find all .pkl files in the demonstrations directory
    pattern = os.path.join(demonstrations_dir, "*.pkl")
    files = glob.glob(pattern)

    print(f"Loading demonstrations from {len(files)} file(s)...")

    for filepath in files:
        with open(filepath, "rb") as f:
            data_dict = pickle.load(f)

        states = data_dict["states"]
        actions = data_dict["actions"]

        all_states.append(states)
        all_actions.append(actions)

    # Concatenate all data
    all_states = np.concatenate(all_states, axis=0)
    all_actions = np.concatenate(all_actions, axis=0)

    return all_states, all_actions


def behavioral_cloning_train(
    dqn_agent,
    train_loader,
    epochs=10,
    learning_rate=0.001,
    use_scheduler=True,
    checkpoint_dir=None,
):
    # Create a temporary optimizer with the specified learning rate
    temp_optimizer = torch.optim.Adam(
        dqn_agent.policy_net.parameters(), lr=learning_rate
    )

    # Create learning rate scheduler if requested
    if use_scheduler:
        eta_min = learning_rate * 0.01

        scheduler = lr_scheduler.CosineAnnealingLR(
            temp_optimizer,
            T_max=epochs,
            eta_min=eta_min,
        )
    else:
        scheduler = None

    dqn_agent.policy_net.train()

    dataset_size = len(train_loader.dataset)

    for epoch in range(epochs):
        total_loss = 0.0
        correct_predictions = 0
        num_batches = 0
        current_lr = temp_optimizer.param_groups[0]["lr"]

        iterator = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")

        for batch_states, batch_actions in iterator:
            # Move to device
            batch_states = batch_states.to(dqn_agent.device)
            batch_actions = batch_actions.to(dqn_agent.device)

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
            num_batches += 1

        # Update learning rate scheduler
        if scheduler is not None:
            scheduler.step()

        avg_loss = total_loss / num_batches
        accuracy = correct_predictions / dataset_size

        print(
            f"   Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2%}, LR: {current_lr:.6f}"
        )

        # Save checkpoint every 10 epochs
        if checkpoint_dir is not None and (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(
                checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pth"
            )
            # Clear CUDA cache before saving to free up memory
            if dqn_agent.device.type == "cuda":
                torch.cuda.empty_cache()
            dqn_agent.save(checkpoint_path)
            print(f"   💾 Checkpoint saved: {checkpoint_path}")

    # Update target network
    dqn_agent.target_net.load_state_dict(dqn_agent.policy_net.state_dict())

    with torch.no_grad():
        dqn_agent.policy_net.eval()
        correct_predictions = 0
        total_samples = 0

        for batch_states, batch_actions in train_loader:
            batch_states = batch_states.to(dqn_agent.device)
            batch_actions = batch_actions.to(dqn_agent.device)

            q_values = dqn_agent.policy_net(batch_states)
            predicted_actions = q_values.argmax(dim=1)
            correct_predictions += (predicted_actions == batch_actions).sum().item()
            total_samples += batch_actions.size(0)

        final_accuracy = correct_predictions / total_samples
        print(f"\n   ✅ Training complete! Final accuracy: {final_accuracy:.2%}\n")

        dqn_agent.policy_net.train()


def pretrain_dqn(
    batch_size=512,
    epochs=10,
    learning_rate=0.001,
    save_path="pretrained/pretrained_model.pth",
    device="cuda:1",
    use_scheduler=True,
    demonstrations_dir="demonstrations",
    num_workers=0,
    use_noisy=False,
):
    # Set device
    if device is None:
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    print(f"Using device: {device}")

    # Step 1: Load demonstrations
    print("Step 1: Loading expert demonstrations...")

    states, actions = load_demonstrations(demonstrations_dir)

    # Create dataset and dataloader
    dataset = DemonstrationDataset(states, actions)

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if device.type == "cuda" else False,
    )

    print(f"✅ Created DataLoader with {len(dataset)} samples\n")

    # Step 2: Create DQN agent
    print("Step 2: Creating DQN agent...")

    dqn_agent = Policy_DQN(
        training_mode=True,
        learning_rate=learning_rate,
        epsilon_start=0.0,  # No exploration during pre-training
        epsilon_end=0.0,
        device=device,
        use_noisy=use_noisy,
    )

    print("✅ DQN agent created\n")

    # Step 3: Train DQN using behavioral cloning
    print("Step 3: Training DQN using behavioral cloning...")

    # Set up checkpoint directory
    checkpoint_dir = os.path.dirname(save_path)
    os.makedirs(checkpoint_dir, exist_ok=True)

    behavioral_cloning_train(
        dqn_agent,
        train_loader,
        epochs=epochs,
        learning_rate=learning_rate,
        use_scheduler=use_scheduler,
        checkpoint_dir=checkpoint_dir,
    )

    # Step 4: Save the pretrained model
    print("Step 4: Saving pretrained model...")

    # Clear CUDA cache before saving to free up memory
    if device.type == "cuda":
        torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    dqn_agent.save(save_path)

    print(f"✅ Model saved to {save_path}\n")

    return dqn_agent


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Pre-train DQN using behavioral cloning on saved demonstrations"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
        help="Batch size for training",
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
        default="pretrained/pretrained_model.pth",
        help="Path to save pretrained model",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:5",
        help="Device to use)",
    )
    parser.add_argument(
        "--demonstrations-dir",
        type=str,
        default="demonstrations",
        help="Directory containing demonstration files",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=48,
        help="Number of DataLoader workers",
    )
    parser.add_argument(
        "--no-scheduler",
        action="store_true",
        help="Disable learning rate scheduler",
    )
    parser.add_argument(
        "--noisy",
        action="store_true",
        help="Enable noisy linear layers",
    )

    args = parser.parse_args()

    pretrain_dqn(
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        save_path=args.save_path,
        device=args.device,
        use_scheduler=not args.no_scheduler,
        demonstrations_dir=args.demonstrations_dir,
        num_workers=args.num_workers,
        use_noisy=args.noisy,
    )
