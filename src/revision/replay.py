from __future__ import annotations

import random
from collections import deque
from functools import lru_cache
from typing import Any

import numpy as np


class IndexedPanelReplayBuffer:
    """Replay buffer that stores panel indices instead of duplicated state windows."""

    def __init__(self, dataset: Any, capacity: int = 10_000, state_cache_size: int = 512):
        self.dataset = dataset
        self.capacity = int(capacity)
        self.buffer = deque(maxlen=self.capacity)
        # Binding the configured size to the cache at construction keeps the
        # implementation pickle-free while allowing efficient repeated samples.
        self._state = lru_cache(maxsize=int(state_cache_size))(self._materialize_state)

    def _materialize_state(self, index: int) -> np.ndarray:
        return self.dataset[int(index)]["features"].numpy().astype(np.float16, copy=True)

    def push(self, state_index: int, action: np.ndarray, reward: float, next_index: int, done: float) -> None:
        self.buffer.append(
            (
                int(state_index),
                np.asarray(action, dtype=np.float32),
                float(reward),
                int(next_index),
                float(done),
            )
        )

    def state(self, index: int) -> np.ndarray:
        """Return a float32 working copy while retaining a compact float16 cache."""
        return self._state(int(index)).astype(np.float32)

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, int(batch_size))
        state_indices, actions, rewards, next_indices, dones = zip(*batch)
        states = np.stack([self._state(index) for index in state_indices])
        next_states = np.stack([self._state(index) for index in next_indices])
        return (
            states,
            np.stack(actions),
            np.asarray(rewards, dtype=np.float32).reshape(-1, 1),
            next_states,
            np.asarray(dones, dtype=np.float32).reshape(-1, 1),
        )

    def __len__(self) -> int:
        return len(self.buffer)

    def is_full(self) -> bool:
        return len(self.buffer) >= self.capacity
