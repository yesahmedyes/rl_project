import argparse
import glob
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from policy_ppo import PolicyPPO


class DemonstrationDataset(Dataset):
    def __init__(self, states: np.ndarray, actions: np.ndarray):
        self.states = torch.from_numpy(states).float()
        self.actions = torch.from_numpy(actions).long()

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]


def load_demonstrations(demonstrations_dir: str):
    pattern = os.path.join(demonstrations_dir, "*.pkl")
    files = glob.glob(pattern)

    if not files:
        raise FileNotFoundError(
            f"No demonstration files found in {demonstrations_dir}. "
            "Run collect_demonstrations.py first."
        )

    all_states, all_actions = [], []
    for filepath in files:
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        all_states.append(data["states"])
        all_actions.append(data["actions"])

    states = np.concatenate(all_states, axis=0)
    actions = np.concatenate(all_actions, axis=0)
    return states, actions


def behavioral_cloning_train(
    agent: PolicyPPO,
    dataloader: DataLoader,
    epochs: int = 50,
    learning_rate: float = 1e-3,
    device: str = "cpu",
    checkpoint_dir: str | None = None,
):
    optimizer = torch.optim.Adam(agent.model.parameters(), lr=learning_rate)
    agent.set_training(True)

    for epoch in range(epochs):
        running_loss = 0.0
        running_acc = 0.0

        iterator = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
        for batch_states, batch_actions in iterator:
            batch_states = batch_states.to(device)
            batch_actions = batch_actions.to(device)

            logits, _ = agent.model(batch_states)
            loss = F.cross_entropy(logits, batch_actions)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.model.parameters(), 1.0)
            optimizer.step()

            with torch.no_grad():
                preds = logits.argmax(dim=1)
                accuracy = (preds == batch_actions).float().mean()

            running_loss += loss.item()
            running_acc += accuracy.item()
            iterator.set_postfix(
                {"loss": f"{loss.item():.4f}", "acc": f"{accuracy.item():.3f}"}
            )

        avg_loss = running_loss / len(dataloader)
        avg_acc = running_acc / len(dataloader)
        print(f"Epoch {epoch + 1}: loss={avg_loss:.4f}, acc={avg_acc:.3f}")

        if checkpoint_dir and (epoch + 1) % 10 == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f"bc_epoch_{epoch + 1}.pth")
            agent.save(checkpoint_path)
            print(f"   💾 Saved checkpoint to {checkpoint_path}")


def pretrain_ppo(
    demonstrations_dir: str = "demonstrations",
    batch_size: int = 2048,
    epochs: int = 50,
    learning_rate: float = 1e-3,
    device: str = "cuda",
    save_path: str = "models/pretrained_ppo.pth",
    num_workers: int = 0,
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("📥 Loading demonstrations...")
    states, actions = load_demonstrations(demonstrations_dir)
    dataset = DemonstrationDataset(states, actions)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    print(f"✅ Loaded {len(dataset)} samples\n")

    agent = PolicyPPO(
        learning_rate=learning_rate,
        policy_path=save_path,
        training_mode=True,
    )
    agent.model.to(device)

    checkpoint_dir = os.path.join(os.path.dirname(save_path), "bc_checkpoints")
    behavioral_cloning_train(
        agent,
        dataloader,
        epochs=epochs,
        learning_rate=learning_rate,
        device=device,
        checkpoint_dir=checkpoint_dir,
    )

    agent.save(save_path)
    print(f"\n✅ Pretrained PPO model saved to {save_path}")

    return save_path


def main():
    parser = argparse.ArgumentParser(
        description="Behavioral cloning pre-training for PPO"
    )
    parser.add_argument("--demonstrations-dir", type=str, default="demonstrations")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-path", type=str, default="models/pretrained_ppo.pth")
    parser.add_argument("--num-workers", type=int, default=0)

    args = parser.parse_args()

    pretrain_ppo(
        demonstrations_dir=args.demonstrations_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        device=args.device,
        save_path=args.save_path,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
