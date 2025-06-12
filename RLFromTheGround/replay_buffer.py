import numpy as np
from collections import deque
import torch as T


class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, device, discrete=False):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.discrete = discrete
        self.state_memory = np.zeros((self.mem_size, input_shape), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, input_shape), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=int)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)
        self.device = device
        self.STATE_NORMALIZATION = 255

    def get_state_normalization(self):
        return self.STATE_NORMALIZATION

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state / self.STATE_NORMALIZATION
        self.new_state_memory[index] = state_ / self.STATE_NORMALIZATION

        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        states = T.tensor(self.state_memory[batch]).to(self.device)
        actions = T.tensor(self.action_memory[batch]).to(self.device)
        rewards = T.tensor(self.reward_memory[batch]).to(self.device)
        states_ = T.tensor(self.new_state_memory[batch]).to(self.device)
        terminal = T.tensor(self.terminal_memory[batch]).to(self.device)
        return states, actions, rewards, states_, terminal


class NStepReplayBuffer:
    """replay buffer for n-step learning"""

    def __init__(self, n, gamma, max_size, input_shape, device):
        self.n = n
        self.gamma = gamma

        self.state_memory, self.action_memory, self.reward_memory, self.next_state_memory, self.terminal_memory = self.restart()
        self.replay_buffer = ReplayBuffer(max_size, input_shape, device)

    def get_state_normalization(self):
        return self.replay_buffer.STATE_NORMALIZATION

    def restart(self):
        state_memory = deque([], maxlen=self.n)
        next_state_memory = deque([], maxlen=self.n)
        action_memory = deque([], maxlen=self.n)
        reward_memory = deque([], maxlen=self.n)
        terminal_memory = deque([], maxlen=self.n)
        return state_memory, action_memory, reward_memory, next_state_memory, terminal_memory

    def sample_buffer(self, batch_size):
        return self.replay_buffer.sample_buffer(batch_size)

    def store_transition(self, state, action, reward, state_, done):
        stored = False
        if len(self.state_memory) == self.n:
            state_0 = self.state_memory[0]
            action_0 = self.action_memory[0]
            done_0 = False
            reward_0 = 0
            for i, (past_reward, past_done) in enumerate(zip(self.reward_memory, self.terminal_memory)):
                reward_0 += (self.gamma ** i) * past_reward
                done_0 = past_done
                if done_0:
                    self.restart()
                    break
            self.replay_buffer.store_transition(state=state_0, action=action_0, reward=reward_0,
                                                state_=self.next_state_memory[-1], done=done_0)
            stored = True

        self.state_memory.append(state)
        self.next_state_memory.append(state_)
        self.action_memory.append(action)
        self.reward_memory.append(reward)
        self.terminal_memory.append(done)
        return stored
