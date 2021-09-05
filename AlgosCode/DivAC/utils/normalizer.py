import torch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GaussianNormalizer:
    def __init__(self, dim_data, eps=1e-7, decay=1.0):
        self.eps, self.decay = eps, decay

        self.mean, self.std, self.n = torch.zeros(dim_data).float().to(device), torch.ones(dim_data).float().to(device), torch.zeros(1).to(device)

    def normal_data(self, x):
        return (x - self.mean) / (self.std + self.eps)

    def unnormal_data(self, x):
        return x * self.std + self.mean

    def update(self, samples):
        old_mean, old_std, old_n = self.mean, self.std, self.n
        old_n *= self.decay
        samples = samples - old_mean
        n, delta = samples.size(0), samples.mean(dim=0)
        new_n = old_n + n
        new_mean = old_mean + delta * n / new_n
        new_std = ((old_std ** 2 * old_n + samples.var(dim=0) * n + delta ** 2 * old_n * n / new_n) / new_n).sqrt()

        self.mean, self.std, self.n = new_mean, new_std, new_n


class Normalizer:
    def __init__(self, dim_state, dim_action, decay=1.0):
        self.state = GaussianNormalizer(dim_state, decay=decay)
        self.action = GaussianNormalizer(dim_action, decay=decay)
        self.state_delta = GaussianNormalizer(dim_state, decay=decay)

    def update(self, state, action, state_delta):
        self.state.update(state.clone().to(device))
        self.action.update(action.clone().to(device))
        self.state_delta.update(state_delta.clone().to(device))

    def normal_state(self, state):
        return self.state.normal_data(state.clone().to(device))

    def normal_action(self, action):
        return self.action.normal_data(action.clone().to(device))

    def normal_state_delta(self, state_delta):
        return self.state_delta.normal_data(state_delta.clone().to(device))

    def unnormal_state_delta(self, state_delta):
        return self.state_delta.unnormal_data(state_delta.clone().to(device))
