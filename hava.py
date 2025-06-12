import os
import math

from pandas import DataFrame

from utils import round_agents_x_and_speed, round_goals, round_state_space, turn_iw_to_negative_if_leader
import numpy as np
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ["KERAS_BACKEND"] = "torch"
import logging
from keras.models import load_model


class AlignmentValue:
    """ HAVA's alignment value that is responsible for computing the agent's reputation. It should hold the rule-based norms, however, since those are taken care of by the SUMO simulator, we only focus on computing the socially normative actions here and the reputation."""
    def __init__(self, alpha, tau, datadriven_social_norms_filename, name):
        self.social_norms_net = self._load_nn(datadriven_social_norms_filename)
        self.name = name
        self.alpha = alpha # essentially number of steps before forgiven
        self.tau = tau # distance (tolerance) from norm satisfaction
        self.reputations = [1.0]

    def reset(self):
        self.reputations = [1.0]

    def _load_nn(self, filename):
        '''
        load neural network version of the ns
        '''
        return load_model(filename)

    def get_last_reputation(self):
        return self.reputations[-1]

    def calculate_weighted_reward(self, reward):
        '''
        eq 4 in the paper
        '''
        if reward >= 0:
            return reward * self.get_last_reputation()
        else:
            return reward*(1+(1-self.get_last_reputation()))

    def append_new_reputation_based_on_delta(self, action_alignment_delta):
        ''' equations 2 and 3 in the paper'''
        last_weight = self.get_last_reputation()
        weight_inc = self.alpha*(math.exp(last_weight)-1)+0.001 # eq 2
        new_weight = last_weight + weight_inc
        if action_alignment_delta >= new_weight: # eq 3
            # grow relatively slowly at speed of alpha
            self.reputations.append(new_weight)
        else:
            # fall fast
            self.reputations.append(action_alignment_delta)

    def calculate_new_reputation_safe_only(self, rb_distance=0):
        action_alignment = (self.tau - rb_distance) / self.tau

        if action_alignment < 0:
            action_alignment = 0

        self.append_new_reputation_based_on_delta(action_alignment_delta=action_alignment)

        return self.reputations # ratios between good and bad actions so far in the episode

    def calculate_new_reputation(self, current_state, ddqn_action, actual_action, rb_distance=0):
        # equation 1 in the paper computing alignment with safe and legal (rule-based) norms coming from SUMO
        action_alignment_with_N = (self.tau - rb_distance) / self.tau
        if action_alignment_with_N < 0:
            action_alignment_with_N = 0

        prediction_curr = self.predict(current_state)
        lowest_allowed_speed = prediction_curr[0]
        highest_allowed_speed = prediction_curr[1]

        if ddqn_action <= highest_allowed_speed and ddqn_action >= lowest_allowed_speed:
            action_alignment_with_S=1
        else:
            distance_to_S_actions = min(abs(ddqn_action - lowest_allowed_speed), abs(ddqn_action - highest_allowed_speed))
            # equation 1 in the paper for the social norms
            action_alignment_with_S = (self.tau - distance_to_S_actions) / self.tau
            if action_alignment_with_S < 0:
                action_alignment_with_S = 0

        logging.warning(f"RB alignment: {float(f'{action_alignment_with_N:.1f}')}, DD alignment {float(f'{action_alignment_with_S:1f}')}; NN says {float(f'{lowest_allowed_speed:.1f}')} - {float(f'{highest_allowed_speed:.1f}')}; RB says action {float(f'{actual_action:.1f}')}; RB difference {float(f'{rb_distance:.1f}')};")

        # compute delta (pick the smallest of the two alignments)
        if action_alignment_with_N <= action_alignment_with_S:
            self.append_new_reputation_based_on_delta(action_alignment_delta=action_alignment_with_N)
        else:
            self.append_new_reputation_based_on_delta(action_alignment_delta=action_alignment_with_S)

        return self.reputations

    def predict(self, state):
        '''
        given a state predict speeds according to the social norms (here learned as a NN)
        '''

        def _create_context_from_state(state):
            '''
            Given an observation from the environment turn it into a context the NN can understand
            '''
            state = DataFrame([state],columns=["iNS","iNLx","iNLy","iNTx","iNTy","oNS","oNLx", "oNLy", "oNTx", "oNTy","iES","iELx", "iELy", "iETx", "iETy","oES","oELx", "oELy", "oETx", "oETy","iSS","iSLx", "iSLy", "iSTx", "iSTy","oSS","oSLx", "oSLy", "oSTx", "oSTy","iWS","iWLx", "iWLy", "iWTx", "iWTy","oWS","oWLx", "oWLy", "oWTx", "oWTy", "currentWeight", "agentX", "agentY", "agentSpeed"])
            state = turn_iw_to_negative_if_leader(state)
            state = round_state_space(state)
            state = round_goals(state)
            state = round_agents_x_and_speed(state)
            # agentX iNS iNLy oNS oNLy iES iELx oES oELx iSS iSLy oSS oSLy iWS iWLx oWS oWLx iNGoal iEGoal iWGoal iSGoal

            context = state[["agentX", "iNLy", "iWGoal", "iNGoal", "iEGoal", "iSGoal", "oNLy", "iELx", "oELx", "iSLy", "oSLy", "iWLx", "oWLx"]].astype(int)
            return context

        context = _create_context_from_state(state)
        nn_context = np.array([np.array(context.iloc[0].to_list())])

        social_actions = self.social_norms_net.predict(nn_context)
        lowest_allowed_speed = social_actions[0][0]
        highest_allowed_speed = social_actions[0][1]
        speed_difference = abs(highest_allowed_speed - lowest_allowed_speed) / 2
        suggested_speed = lowest_allowed_speed + speed_difference
        social_actions = [suggested_speed-4, suggested_speed+3]
        return social_actions
