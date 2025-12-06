import argparse
import json
from pathlib import Path
from typing import List, Optional, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader
from tensordict import TensorDict
from torchrl.trainers import CountFramesLog, LogScalar, Trainer
from torchrl.record.loggers import get_logger

from misc.utils import OfflineTransitionDataset
from policies.policy_bc import BCPolicyNet

ACTION_DIM = 12


class BCLossModule(nn.Module):
    def __init__(self, model: nn.Module, criterion: nn.Module):
        super().__init__()
        self.model = model
        self.criterion = criterion

    def forward(self, batch: TensorDict):
        logits = self.model(batch["observation"])
        loss = self.criterion(logits, batch["action"])
        batch.set("loss", loss.detach())

        return TensorDict({"loss": loss, "logits": logits}, batch_size=batch.batch_size)


class LossTracker:
    def __init__(self):
        self.history: List[float] = []
        self.best_loss: float = float("inf")

    def register(self, trainer: Trainer, name: str = "loss_tracker"):
        trainer.register_module(self, name)
        trainer.register_op("post_optim_log", self)

    def state_dict(self):
        return {"history": self.history, "best_loss": self.best_loss}

    def load_state_dict(self, state_dict):
        self.history = state_dict.get("history", [])
        self.best_loss = state_dict.get("best_loss", float("inf"))

    def __call__(self, batch: TensorDict):
        if "loss" not in batch:
            return None
        loss_value = float(batch["loss"].mean().detach().cpu().item())
        self.history.append(loss_value)
        if loss_value < self.best_loss:
            self.best_loss = loss_value
        return {"loss": loss_value, "log_pbar": True}


def train_bc(
    data_dir: str,
    encoding_type: str = "handcrafted",
    num_iterations: int = 1000,
    checkpoint_freq: int = 100,
    output_dir: str = "results/bc",
    model_layers: Optional[Sequence[int]] = None,
    batch_size: int = 16384,
    lr: float = 1e-4,
    weight_decay: float = 0.0,
    num_workers: int = 16,
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

    print(f"Training BC (TorchRL) on data from: {data_path}")
    print(f"Encoding type: {encoding_type}")
    print(f"Observation dim: {obs_dim}, Action dim: {ACTION_DIM}")

    print("\nModel/optimizer config:")
    print(f"  Hidden layers: {hidden_layers}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")
    print(f"  Weight decay: {weight_decay}")
    print(f"  Device: {torch_device}")
    print(f"  Training epochs (approx via frames): {num_iterations}")
    print(f"  Checkpoint frequency: {checkpoint_freq}")

    dataset = OfflineTransitionDataset(json_files=json_files, obs_dim=obs_dim)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch_device.type == "cuda",
        collate_fn=lambda batch: TensorDict.stack(batch, dim=0),
    )

    model = BCPolicyNet(
        obs_dim=obs_dim, hidden_layers=hidden_layers, action_dim=ACTION_DIM
    )
    model.to(torch_device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    loss_module = BCLossModule(model=model, criterion=criterion)

    logger = get_logger(
        "stdout",
        experiment_name="bc_trainer",
        logger_name="bc_trainer",
        log_dir=output_path / "logs",
    )
    log_loss = LogScalar(key="loss", logname="train/loss", log_pbar=True)
    frames_log = CountFramesLog(frame_skip=1)
    loss_tracker = LossTracker()

    total_frames = num_iterations * len(dataset)

    def infinite_collector():
        while True:
            for batch in dataloader:
                yield batch.to(device=torch_device, non_blocking=True)

    trainer = Trainer(
        collector=infinite_collector(),
        total_frames=total_frames,
        loss_module=loss_module,
        optimizer=optimizer,
        optim_steps_per_batch=1,
        save_trainer_file=output_path / "trainer_state.pt",
        logger=logger,
        progress_bar=True,
        log_interval=checkpoint_freq,
    )
    trainer.register_module(model, "model")
    trainer.register_module(optimizer, "optimizer")
    loss_tracker.register(trainer)
    log_loss.register(trainer)
    frames_log.register(trainer)

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
    trainer.train()

    final_loss = loss_tracker.history[-1] if loss_tracker.history else float("nan")
    best_loss = loss_tracker.best_loss if loss_tracker.history else float("inf")
    metadata["final_loss"] = final_loss
    metadata["best_loss"] = best_loss

    trainer_state = trainer.state_dict(full_state=True)
    trainer_state["metadata"] = metadata
    trainer_state_path = output_path / "trainer_state.pt"
    torch.save(trainer_state, trainer_state_path)

    inference_state = {"metadata": metadata, "state_dict": model.state_dict()}
    inference_path = output_path / "bc_policy.pt"
    torch.save(inference_state, inference_path)

    history_path = output_path / "loss_history.jsonl"

    with open(history_path, "w") as f:
        for step, loss_val in enumerate(loss_tracker.history, start=1):
            f.write(json.dumps({"step": step, "loss": loss_val}) + "\n")

    print("\nTraining complete!")
    print(f"Loss history logged to: {history_path}")
    print(f"Trainer state saved to: {trainer_state_path}")
    print(f"Inference checkpoint saved to: {inference_path}")
    print(f"Best loss achieved: {best_loss:.6f}")

    return str(trainer_state_path)


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
        default=1,
        help="Epoch interval for checkpointing",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/bc",
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
        default=1024,
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
        default=1e-5,
        help="L2 weight decay",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=16,
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
