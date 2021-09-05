import numpy as np
import torch


class Buffer(object):
    """saving data collected from interaction with the MDP."""
    def __init__(self, dim_state, dim_action, capacity):
        """
        :param dim_state: dim of the state
        :param dim_action: dim of the action
        :param capacity: size of the buffer
        """
        self.dim_state, self.dim_action = dim_state, dim_action
        self.pos, self.capacity, self.normalizer = 0, capacity, None

        self.states = torch.zeros(capacity, dim_state).float()
        self.actions = torch.zeros(capacity, dim_action).float()
        self.states_delta = torch.zeros(capacity, dim_state).float()

    def setup_normalizer(self, normalizer):
        self.normalizer = normalizer

    def push(self, state, action, next_state):
        idx = self.pos % self.capacity

        state = torch.from_numpy(state).float().clone()
        action = torch.from_numpy(action).float().clone()
        next_state = torch.from_numpy(next_state).float().clone()
        state_delta = next_state - state

        self.states[idx], self.actions[idx], self.states_delta[idx] = state, action, state_delta
        self.pos += 1

        if self.normalizer is not None:
            self.normalizer.update(state, action, state_delta)

    def random_iterator(self, batch_size):
        """
        get random data for training model, each ensemble have different data.
        :param batch_size: size of each iterator

        :return: batch data (batch_size, dim)
        """
        num = min(self.pos, self.capacity)
        indices = torch.from_numpy(np.stack([np.random.permutation(range(num))]).T)
        for i in range(0, num, batch_size):
            j = min(num, i + batch_size)
            if j - i < batch_size and i != 0:
                return  # drop last incomplete last batch
            batch_size = j - i
            idx = indices[i:i + batch_size].flatten()
            states = self.states.index_select(dim=0, index=idx).reshape(batch_size, self.dim_state)
            actions = self.actions.index_select(dim=0, index=idx).reshape(batch_size, self.dim_action)
            states_delta = self.states_delta.index_select(dim=0, index=idx).reshape(batch_size, self.dim_state)

            yield states, actions, states_delta

    def __len__(self):
        return min(self.pos, self.capacity)
