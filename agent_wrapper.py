import torch as T
import time
import traci
import numpy as np

from RLFromTheGround.agent import NoisyNStep3DQN
from RLFromTheGround.replay_buffer import NStepReplayBuffer

class AgentWrapper:
    """Wrapper between agents from RLFromGround and env"""
    def __init__(self, model_name, model_checkpoint_path, eval=False):
        self.model_name = model_name
        device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.initial_episode_num = 0
        self.initial_time = time.time()
        self.episode_num = 0
        self.last_scores = []
        self.last_finish_times = []
        self.timestep_num = 0
        self.epsilon = 1
        self.vehicle_id = "1.0"
        self.MODEL_CHECKPOINT_PATH = model_checkpoint_path

        self.replay_buffer = NStepReplayBuffer(n=3, gamma=0.99,max_size=2000000,input_shape=44,device=device)
        self.agent = NoisyNStep3DQN(gamma=0.99, epsilon=self.epsilon, eps_min=0.01, eps_steps=100000, lr=0.0003, replay_buffer=self.replay_buffer, input_dims=44, batch_size=64, n_actions=11)

        self.agent.replay_buffer.STATE_NORMALIZATION = 1
        self.agent.min_sample_size = 10000

    def learn(self):
        self.agent.learn()

    def _convert_action_to_speed(self, action):
        '''
        the agent picks an acceleration and we convert it to speed in km/h
        '''
        ACTION_TO_INCREMENT = {
            0: 1.8,
            1: 1.44,
            2: 1.08,
            3: 0.72,
            4: 0.36,
            5: 0,
            6: -0.66,
            7: -1.32,
            8: -1.98,
            9: -2.64,
            10: -3.24
        } # in km/h
        acceleration = ACTION_TO_INCREMENT[action]
        current_speed = traci.vehicle.getSpeed(self.vehicle_id)*3.6 # in km/h
        updated_speed = current_speed + acceleration
        if updated_speed < 0:
            updated_speed = 0
        return updated_speed

    def choose_action(self, state):
        state = self._normalize_state(state)
        state = np.array(state)
        action = self.agent.choose_action(state)
        return self._convert_action_to_speed(action), action

    def _normalize_state(self, state):
        '''
        Normalize the received state
        '''
        # "iNS","iNLx","iNLy","iNTx","iNTy","oNS","oNLx", "oNLy", "oNTx", "oNTy","iES","iELx", "iELy", "iETx", "iETy","oES","oELx", "oELy", "oETx", "oETy","iSS","iSLx", "iSLy", "iSTx", "iSTy","oSS","oSLx", "oSLy", "oSTx", "oSTy","iWS","iWLx", "iWLy", "iWTx", "iWTy","oWS","oWLx", "oWLy", "oWTx", "oWTy", "agentWeights", "agentX", "agentY", "agentSpeed"
        normalizer = [50,200,200,200,200,50,200,200,200,200,50,200,200,200,200,50,200,200,200,200,50,200,200,200,200,50,200,200,200,200,50,200,200,200,200,50,200,200,200,200,1,200,200,50]
        normalized_state = [b / m for b,m in zip(state, normalizer)]

        return normalized_state

    def _normalize_reward(self, reward):
        return reward / 60

    def remember(self, state, action, reward, new_state, done):
        state = self._normalize_state(state)
        new_state = self._normalize_state(new_state)
        reward = self._normalize_reward(reward)
        state = np.array(state)
        new_state = np.array(new_state)
        self.agent.store_transition(state=state, action=action, reward=reward,state_=new_state, terminal=done)

    def get_epsilon(self):
        return self.agent.epsilon.value

    def save_model(self):
        T.save({
            'online_net_state_dict': self.agent.online_net.state_dict(),
            'target_net_state_dict': self.agent.target_net.state_dict(),
            'optimizer_state_dict': self.agent.optimizer.state_dict(),
            'replay_buffer': self.replay_buffer,
            'epsilon_value': self.agent.epsilon.value,
            'episode_num': self.episode_num,
            'initial_time': self.initial_time,
            'last_scores': self.last_scores,
            'last_finish_times': self.last_finish_times
            }, f"{self.MODEL_CHECKPOINT_PATH}/model.tar")

    def load_model(self):
        checkpoint = T.load(f"{self.MODEL_CHECKPOINT_PATH}/model.tar", weights_only=False)
        self.agent.online_net.load_state_dict(checkpoint['online_net_state_dict'])
        self.agent.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.agent.epsilon.value = checkpoint['epsilon_value']
        self.initial_episode_num = checkpoint['episode_num'] + 1
        self.initial_time = checkpoint['initial_time']
        self.last_scores = checkpoint['last_scores']
        self.last_finish_times = checkpoint['last_finish_times']
        self.replay_buffer = checkpoint['replay_buffer']
        self.agent.replay_buffer = self.replay_buffer
