class EpsilonGreedy:
    """Epsilon greedy exploration"""
    def __init__(self, eps_steps, eps_min=0.01):
        self.eps_min = eps_min
        self.value = 1
        self.dec = (self.value - self.eps_min) / eps_steps

    def decrease(self):
        self.value = self.value - self.dec if self.value > self.eps_min else self.eps_min
