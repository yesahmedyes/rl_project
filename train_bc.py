import argparse
import json
from pathlib import Path
from typing import List, Optional, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from misc.utils import OfflineTransitionDataset, save_checkpoint
from policies.policy_bc import BCPolicyNet

ACTION_DIM = 12


def train_bc(
    data_dir: str,
    encoding_type: str = "handcrafted",
    num_iterations: int = 10,
    checkpoint_freq: int = 5,
    output_dir: str = "bc_results",
    model_layers: Optional[Sequence[int]] = None,
    batch_size: int = 256,
    lr: float = 1e-4,
    weight_decay: float = 0.0,
    num_workers: int = 0,
    device: Optional[str] = None,
):
    data_path = Path(data_dir)
    output_path = Path(output_dir)

    if not data_path.exists():
        raise ValueError(f"Data directory does not exist: {data_path}")

    json_files = [
        p.resolve() for p in data_path.glob("*.json") if p.name != "dataset_info.json"
    ]

    if not json_files:
        raise ValueError(f"No episode JSON files found in {data_path}")

    obs_dim = 70 if encoding_type == "handcrafted" else 946
    hidden_layers = list(model_layers) if model_layers else [256, 256]
    torch_device = torch.device(
        device
        if device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    print(f"Training BC (Torch) on data from: {data_path}")
    print(f"Encoding type: {encoding_type}")
    print(f"Observation dim: {obs_dim}, Action dim: {ACTION_DIM}")

    print("\nModel/optimizer config:")
    print(f"  Hidden layers: {hidden_layers}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")
    print(f"  Weight decay: {weight_decay}")
    print(f"  Device: {torch_device}")
    print(f"  Training epochs: {num_iterations}")
    print(f"  Checkpoint frequency: {checkpoint_freq}")

    dataset = OfflineTransitionDataset(json_files=json_files, obs_dim=obs_dim)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch_device.type == "cuda",
    )

    model = BCPolicyNet(
        obs_dim=obs_dim, hidden_layers=hidden_layers, action_dim=ACTION_DIM
    )
    model.to(torch_device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_loss = float("inf")
    best_checkpoint: Optional[Path] = None
    loss_history: List[dict] = []

    metadata = {
        "encoding_type": encoding_type,
        "obs_dim": obs_dim,
        "action_dim": ACTION_DIM,
        "hidden_layers": hidden_layers,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
    }

    print("\nStarting training...")

    for epoch in range(1, num_iterations + 1):
        model.train()
        running_loss = 0.0
        sample_count = 0

        progress = tqdm(dataloader, desc=f"Epoch {epoch}/{num_iterations}", leave=False)

        for obs_batch, action_batch in progress:
            obs_batch = obs_batch.to(torch_device, non_blocking=True)
            action_batch = action_batch.to(torch_device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(obs_batch)
            loss = criterion(logits, action_batch)
            loss.backward()
            optimizer.step()

            batch_size_effective = action_batch.shape[0]
            running_loss += loss.item() * batch_size_effective
            sample_count += batch_size_effective

            progress.set_postfix({"loss": loss.item()})

        avg_loss = running_loss / max(sample_count, 1)
        loss_history.append({"epoch": epoch, "loss": float(avg_loss)})

        print(f"Epoch {epoch}/{num_iterations} - avg loss: {avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_checkpoint = save_checkpoint(
                output_dir=output_path,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                metadata={**metadata, "best_loss": best_loss},
            )
            print(f"  New best checkpoint -> {best_checkpoint}")

        if epoch % checkpoint_freq == 0:
            ckpt_path = save_checkpoint(
                output_dir=output_path,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                metadata={**metadata, "epoch_loss": avg_loss},
            )
            print(f"  Checkpoint saved to: {ckpt_path}")

    final_checkpoint = save_checkpoint(
        output_dir=output_path,
        epoch=num_iterations,
        model=model,
        optimizer=optimizer,
        metadata={**metadata, "final_loss": loss_history[-1]["loss"]},
    )

    history_path = output_path / "loss_history.jsonl"

    with open(history_path, "w") as f:
        for entry in loss_history:
            f.write(json.dumps(entry) + "\n")

    print("\nTraining complete!")

    if best_checkpoint:
        print(f"Best checkpoint saved to: {best_checkpoint}")

    print(f"Final checkpoint saved to: {final_checkpoint}")
    print(f"Loss history logged to: {history_path}")
    print(f"Best loss achieved: {best_loss:.6f}")

    return str(final_checkpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Behavior Cloning model using PyTorch (no Ray/RLlib dependency)"
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing offline data",
    )
    parser.add_argument(
        "--encoding_type",
        type=str,
        default="handcrafted",
        choices=["handcrafted", "onehot"],
        help="State encoding type",
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--checkpoint_freq",
        type=int,
        default=5,
        help="Epoch interval for checkpointing",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="bc_results",
        help="Output directory for checkpoints and results",
    )
    parser.add_argument(
        "--model_layers",
        type=json.loads,
        default="[256, 256]",
        help='JSON list of hidden sizes, e.g. "[256, 256]"',
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for training",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="L2 weight decay",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of dataloader workers",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help='Torch device string, e.g. "cuda:0" or "cpu". Defaults to auto.',
    )

    args = parser.parse_args()

    train_bc(
        data_dir=args.data_dir,
        encoding_type=args.encoding_type,
        num_iterations=args.num_iterations,
        checkpoint_freq=args.checkpoint_freq,
        output_dir=args.output_dir,
        model_layers=args.model_layers,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        device=args.device,
    )
