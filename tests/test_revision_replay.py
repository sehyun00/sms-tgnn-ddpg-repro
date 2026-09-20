from __future__ import annotations

import numpy as np
import torch

from src.revision.replay import IndexedPanelReplayBuffer


class TinyDataset:
    def __getitem__(self, index):
        return {"features": torch.full((3, 4, 2), float(index))}


def test_indexed_replay_materializes_without_storing_state_copies():
    buffer = IndexedPanelReplayBuffer(TinyDataset(), capacity=4, state_cache_size=2)
    buffer.push(1, np.array([0.2, 0.3, 0.5]), 0.01, 2, 0.0)
    buffer.push(2, np.array([0.1, 0.4, 0.5]), 0.02, 3, 1.0)
    assert isinstance(buffer.buffer[0][0], int)
    states, actions, rewards, next_states, dones = buffer.sample(2)
    assert states.shape == (2, 3, 4, 2)
    assert states.dtype == np.float16
    assert actions.shape == (2, 3)
    assert rewards.shape == dones.shape == (2, 1)
    assert set(np.unique(next_states)).issubset({2.0, 3.0})
