import numpy as np
from collections import deque
from typing import Tuple, List

class PrioritizedReplayBuffer:
    def __init__(self, max_size: int, alpha: float = 0.6, beta: float = 0.4, beta_increment: float = 0.001):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.priorities = deque(maxlen=max_size)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.max_priority = 1.0
        self.eps = 1e-6

    def add(self, state: np.ndarray, action: int, reward: float, 
            next_state: np.ndarray, done: bool) -> None:
        """Add a new experience to memory with max priority."""
        # Ensure state and next_state are numpy arrays
        state = np.array(state, dtype=np.float32)
        next_state = np.array(next_state, dtype=np.float32)
        
        self.buffer.append((state, action, reward, next_state, done))
        # Store priority as a simple float
        self.priorities.append(float(self.max_priority))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                              np.ndarray, np.ndarray, np.ndarray, List[int]]:
        """Sample a batch of experiences based on their priorities."""
        buffer_len = len(self.buffer)
        if buffer_len == 0:
            raise ValueError("Buffer is empty!")

        # Handle case where buffer is smaller than batch size
        batch_size = min(buffer_len, batch_size)

        # Ensure priorities are converted to float32 numpy array
        priorities = np.array([float(p) for p in self.priorities], dtype=np.float32)
        
        # Calculate sampling probabilities
        probabilities = (priorities + self.eps) ** self.alpha
        probabilities /= probabilities.sum()

        # Sample indices based on probabilities
        indices = np.random.choice(buffer_len, batch_size, p=probabilities)

        # Calculate importance sampling weights
        weights = (buffer_len * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        # Get samples
        samples = [self.buffer[idx] for idx in indices]
        
        # Carefully unpack and convert samples
        states = np.vstack([np.array(exp[0], dtype=np.float32) for exp in samples])
        actions = np.array([exp[1] for exp in samples], dtype=np.int64)
        rewards = np.array([exp[2] for exp in samples], dtype=np.float32)
        next_states = np.vstack([np.array(exp[3], dtype=np.float32) for exp in samples])
        dones = np.array([exp[4] for exp in samples], dtype=np.bool_)

        # Increment beta
        self.beta = min(1.0, self.beta + self.beta_increment)

        return (states, actions, rewards, next_states, dones, weights, indices)

    def update_priorities(self, indices: List[int], td_errors: np.ndarray) -> None:
        """Update priorities based on TD errors."""
        for idx, td_error in zip(indices, td_errors):
            if idx < len(self.priorities):  # Safety check
                # Ensure priority is stored as a float
                priority = float(abs(td_error[0]) + self.eps)  # Use [0] since td_errors are 2D
                self.priorities[idx] = priority
                self.max_priority = max(self.max_priority, priority)

    def __len__(self) -> int:
        """Return the current size of internal buffer."""
        return len(self.buffer)