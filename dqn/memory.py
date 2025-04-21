# https://github.com/pluebcke/dqn_experiments/blob/master/LICENSE

from collections import deque, namedtuple

import numpy as np
import torch
import typing

from dqn.segment_tree import SumTree, MinTree

Experience = namedtuple('Experience', 'state action reward next_state done')


class ReplayMemory:
    """
    ReplayMemory stores experience in form of tuples  (last_state last_action reward state action done)
    in a deque of maximum length buffer_size.

    Methods:
        add(self, sample): add a sample to the buffer
        sample(self, batch_size): return an experience batch of size batch_size
        update_priorities(self, indices, weights): not implemented, needed for prioritizied replay buffer
        number_samples(self): returns the number of samples currently stored.
    """

    def __init__(self,
                 device: torch.device,
                 memory_size: int,
                 gamma: float) -> None:
        """
        Initializes the memory buffer

        Args:
            device(str): "gpu" or "cpu"
            memory_size(int): maximum number of elements in the ReplayMemory
            gamma(float): decay factor
        """
        self.gamma = gamma
        self.device = device

        self.data = deque(maxlen=memory_size)
        return

    def add(self, sample: Experience) -> None:

        """
            Adds experience to the memory after calculating the n-step returns.
        Args:

        Returns:

        """
        self.data.append(sample)
        return

    def sample_not_needed(self, batch_size: int) -> tuple:
        """
        Samples a batch of size batch_size and returns a tuple of PyTorch tensors.
        Args:
            batch_size(int):  number of elements for the batch

        Returns:
            tuple of tensors
        """
        number_elements = len(self.data)
        indices = np.random.randint(0, number_elements, batch_size)
        states, actions, rewards, next_states, dones = self.get_batch(indices)
        return tuple((states, actions, rewards, next_states, dones, None, None))

    def get_batch(self, indices: np.array) -> tuple:
        """

        Args:
            indices: indices of the data that should be returned

        Returns:

        """
        states, actions = [], []
        next_states = []
        rewards, dones = [], []
        for index in indices:
            experience = self.data[index]
            state, action, reward, next_state, done = experience
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

        states = torch.from_numpy(np.array(states)).float().to(self.device)
        actions = torch.from_numpy(np.array(actions)).int().to(self.device)
        rewards = torch.from_numpy(np.array(rewards)).float().to(self.device)
        next_states = torch.from_numpy(np.array(next_states)).float().to(self.device)
        dones = torch.from_numpy(np.array(dones)).bool().to(self.device)
        return tuple((states, actions, rewards, next_states, dones))

    def update_priorities(self, indices: typing.Optional[np.array], priorities: typing.Optional[np.array]):
        """
        This method later needs to be implemented for prioritized experience replay.
        Args:
            indices(list(int)): list of integers with the indices of the experience tuples in the batch
            priorities(list(float)): priorities of the samples in the batch

        Returns:
            None
        """
        return

    def number_samples(self):
        """
        Returns:
              Number of elements in the Replay Memory
        """
        return len(self.data)


class PrioritizedReplayMemory(ReplayMemory):
    """
    Implemented the prioritized replay buffer according to "Schaul, Tom, et al. "Prioritized experience replay."
    arXiv preprint arXiv:1511.05952 (2015)."

    I also read through https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py before implementing
    this class.

    Methods:
        add(self, sample): add a sample to the buffer
        sample_batch(self, batch_size): return an experience batch of size batch_size
        update_priorities(self, indices, weights): not implemented, needed for prioritizied replay buffer
        number_samples(self): returns the number of samples currently stored.
    """

    def __init__(self,
                 device: torch.device,
                 memory_size: int,
                 gamma: float,
                 alpha: float,
                 beta: float,
                 beta_increment: float) -> None:
        """
        Initializes the memory buffer

        Args:
            device(str): "gpu" or "cpu"
            memory_size(int): maximum number of elements in the ReplayMemory
            gamma(float): decay factor
        """
        super(PrioritizedReplayMemory, self).__init__(device, memory_size, gamma)
        self.memory_size = memory_size

        self.index = 0
        self.data = []

        self.max_prio = 1
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment

        self.sum_tree = SumTree(memory_size)
        self.min_tree = MinTree(memory_size)
        return


    def __len__(self):
        return len(self.data)

    def add(self, exp: Experience) -> None:
        """
            Adds experience to the memory after calculating the n-step return.
        Args:

        Returns:

        """
        if self.number_samples() < self.memory_size:
            self.data.append(exp)
        else:
            self.data[self.index] = exp

        weight = self.max_prio ** self.alpha
        self.sum_tree.add(self.index, weight)
        self.min_tree.add(self.index, weight)
        self.index = (self.index + 1) % self.memory_size
        return

    def sample(self, batch_size: int) -> tuple:
        """
        Samples a batch of size batch_size according to their priority.
        Args:
            batch_size(int):  number of elements for the batch

        Returns:
            tuple of tensors with experiences
        """

        interval_prio = self.sum_tree.root / batch_size
        indices = []
        for i in range(batch_size):
            low = i * interval_prio
            high = (i + 1) * interval_prio
            prio = low + (high - low) * np.random.rand()
            indices.append(self.sum_tree.get_index(prio))

        min_prio = self.min_tree.root / self.sum_tree.root
        max_weight = (self.number_samples() * min_prio) ** -self.beta
        weights = (self.number_samples() * self.sum_tree.get_elements(indices)) ** -self.beta / max_weight
        weights = torch.tensor(weights).to(self.device)

        if self.beta < 1.0:
            self.beta = self.beta + self.beta_increment

        states, actions, rewards, next_states, dones = self.get_batch(np.array(indices))
        return (states, actions, rewards, next_states, dones), indices, weights

    def update_priorities(self, indices: typing.Optional[np.array], priorities: typing.Optional[np.array]):
        """
        Updates the priorities of the replay buffer.
        Args:
            indices(list(int)): list of integers with the indices of the experience tuples in the batch
            priorities(list(float)): priorities of the samples in the batch

        Returns:
            None
        """
        for index, prio in zip(indices, priorities):
            if self.max_prio < prio:
                self.max_prio = prio
            self.sum_tree.update(index, prio ** self.alpha)
            self.min_tree.update(index, prio ** self.alpha)
        return
