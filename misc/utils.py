import json
import random
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from tensordict import TensorDict
from torch.utils.data import IterableDataset, get_worker_info


class OfflineTransitionDataset(IterableDataset):
    def __init__(self, json_files: Sequence[Path], obs_dim: int):
        self.json_files = [Path(p) for p in json_files]
        self.obs_dim = obs_dim
        self._length = None

    def _iter_file(self, path: Path) -> Iterable[TensorDict]:
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

                    # TorchRL Episode Data (TED) format preferred.
                    if "steps" in episode:
                        for step in episode.get("steps", []):
                            obs = step.get("observation")
                            action = step.get("action")

                            if obs is None or action is None:
                                continue

                            obs_arr = np.asarray(obs, dtype=np.float32).reshape(-1)
                            if obs_arr.shape[0] != self.obs_dim:
                                raise ValueError(
                                    f"Observation dim mismatch: expected {self.obs_dim}, "
                                    f"got {obs_arr.shape[0]} from {path}"
                                )

                            yield TensorDict(
                                {
                                    "observation": torch.from_numpy(obs_arr),
                                    "action": torch.tensor(
                                        int(action), dtype=torch.long
                                    ),
                                },
                                batch_size=[],
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

    def _compute_length(self) -> int:
        count = 0

        for path in self.json_files:
            with open(path, "r") as f:
                for line in f:
                    if not line.strip():
                        continue
                    parsed = json.loads(line)
                    episodes = parsed if isinstance(parsed, list) else [parsed]

                    for episode in episodes:
                        if not isinstance(episode, dict):
                            continue
                        steps = episode.get("steps", [])
                        count += len(steps)
        return count

    def __len__(self) -> int:
        if self._length is None:
            self._length = self._compute_length()
        return self._length
