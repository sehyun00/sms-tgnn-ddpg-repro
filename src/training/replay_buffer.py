"""
Replay Buffer for Experience Replay
경험 재생 버퍼
"""

import random
import numpy as np
from collections import deque


class ReplayBuffer:
    """
    Experience Replay Buffer (FIFO with Capacity Limit)

    - 학습 데이터를 저장하고 랜덤 샘플링하여 데이터 간 상관관계를 깨고 학습 안정성을 향상시킵니다.
    - 용량 초과 시 가장 오래된 경험을 자동 제거 (FIFO)
    - README 기준: 10,000 경험 저장
    """

    def __init__(self, capacity=10000):
        """
        Args:
            capacity: 버퍼 최대 용량 (README 기준: 10,000)
                     deque의 maxlen 파라미터를 사용하여 FIFO 자동 처리
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        # Note: deque with maxlen automatically discards oldest items
        # when new items are added beyond capacity (FIFO behavior)

    def push(self, state, action, reward, next_state, done):
        """
        새로운 경험 저장 (FIFO)

        버퍼가 가득 차면 가장 오래된 경험을 자동으로 제거하고
        새 경험을 추가합니다.

        Args:
            state: 현재 상태
            action: 수행한 행동
            reward: 받은 보상
            next_state: 다음 상태
            done: 에피소드 종료 여부
        """
        self.buffer.append((state, action, reward, next_state, done))
        # If len(buffer) >= capacity, oldest item is automatically removed

    def push_batch(self, states, actions, rewards, next_states, dones):
        """
        배치 단위 경험 저장
        """
        # Convert to list of tuples and extend
        # This is faster than loop push
        batch_data = zip(states, actions, rewards, next_states, dones)
        self.buffer.extend(batch_data)

    def sample(self, batch_size):
        """
        학습용 미니배치 랜덤 샘플링

        저장된 경험에서 무작위로 batch_size개를 선택하여
        시간적 상관관계를 깨고 학습 안정성을 향상시킵니다.

        Args:
            batch_size: 샘플링할 배치 크기

        Returns:
            states: (batch_size, state_dim) numpy array
            actions: (batch_size, action_dim) numpy array
            rewards: (batch_size, 1) numpy array
            next_states: (batch_size, state_dim) numpy array
            dones: (batch_size, 1) numpy array
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards).reshape(-1, 1),
            np.array(next_states),
            np.array(dones).reshape(-1, 1),
        )

    def __len__(self):
        """버퍼에 저장된 경험의 개수"""
        return len(self.buffer)

    def is_full(self):
        """버퍼가 최대 용량에 도달했는지 확인"""
        return len(self.buffer) >= self.capacity
