import copy
from logging import getLogger
import os

import numpy as np
import chainer
from chainer import serializers
from chainer import functions as F

import async_rl.utils.copy_param as copy_param


class Agent(object):
    def __init__(self, process_idx, model, phi):
        self.model = model
        self.process_idx = process_idx
        self.phi = phi

    def get_action(self, state, keep_same_state=False):
        statevar = chainer.Variable(np.expand_dims(self.phi(state), 0))
        agent_info = {}
        # For LSTM implementation, set keep_same_state=True for pi_and_v
        pout, vout = self.model.pi_and_v(statevar)
        agent_info['action_log_prob'] = pout.sampled_actions_log_probs
        agent_info['action_entropy'] = pout.entropy
        agent_info['values'] = vout
        return pout.action_indices[0], agent_info

    def lookup(self, obs):
        """
        look up agent_info for ``t+1``.
        """
        _, agent_info = self.get_action(obs, keep_same_state=True)
        return agent_info
