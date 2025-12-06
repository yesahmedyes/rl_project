import json
import random
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info


class OfflineTransitionDataset(IterableDataset):
    def __init__(self, json_files: Sequence[Path], obs_dim: int):
        self.json_files = [Path(p) for p in json_files]
        self.obs_dim = obs_dim

    def _iter_file(self, path: Path) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
        with open(path, "r") as f:
            for line in f:
                if not line.strip():
                    continue

                parsed = json.loads(line)
                episodes = parsed if isinstance(parsed, list) else [parsed]

                for episode in episodes:
                    if not isinstance(episode, dict):
                        raise ValueError(
                            f"Episode entry must be a dict, got {type(episode)}"
                        )

                    obs_list = episode.get("obs", [])
                    actions = episode.get("actions", [])

                    for obs, action in zip(obs_list, actions):
                        obs_arr = np.asarray(obs, dtype=np.float32).reshape(-1)
                        if obs_arr.shape[0] != self.obs_dim:
                            raise ValueError(
                                f"Observation dim mismatch: expected {self.obs_dim}, "
                                f"got {obs_arr.shape[0]} from {path}"
                            )

                        yield (
                            torch.from_numpy(obs_arr),
                            torch.tensor(int(action), dtype=torch.long),
                        )

    def __iter__(self):
        worker = get_worker_info()

        if worker is None:
            files = list(self.json_files)
        else:
            files = list(self.json_files[worker.id :: worker.num_workers])

        rng = random.Random()
        rng.seed(torch.initial_seed() % 2**32)
        rng.shuffle(files)

        for path in files:
            yield from self._iter_file(path)


def save_checkpoint(
    output_dir: Path,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    metadata: dict,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / f"model_epoch_{epoch:05d}.pt"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metadata": metadata,
        },
        ckpt_path,
    )

    return ckpt_path
