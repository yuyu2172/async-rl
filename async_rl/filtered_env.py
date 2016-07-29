from gym import Env, spaces
import numpy as np
from async_rl.utils import imresize

class FilteredEnv(Env):

    def __init__(self, env, ob_filter=None, act_filter=None, rew_filter=None, skiprate=0):
        self.env = env
        self.ob_filter = ob_filter
        self.act_filter = act_filter
        self.rew_filter = rew_filter
        self.skiprate = skiprate
        self.metadata = self.env.metadata
        self.spec = self.env.spec

        if ob_filter is not None:
            ob_space = self.env.observation_space
            shape = self.ob_filter.output_shape(ob_space)
            self.observation_space = spaces.Box(-np.inf, np.inf, shape)
        else:
            self.observation_space = self.env.observation_space

        if act_filter is not None:
            self.action_space = spaces.Discrete(self.act_filter.n)
        else:
            self.action_space = self.env.action_space

    def _step(self, ac):
        nac = self.act_filter(ac) if self.act_filter else ac
        total_nrew = 0.0
        total_rew = 0.0
        if self.skiprate > 0:
            num_steps = np.random.randint(self.skiprate) + 1
        else:
            num_steps = 1
        nob = None
        done = False
        for _ in range(num_steps):
            ob, rew, done, info = self.env.step(nac)
            nob = self.ob_filter(ob) if self.ob_filter else ob
            nrew = self.rew_filter(rew) if self.rew_filter else rew
            total_nrew += nrew
            total_rew += rew
            if done:
                info["raw_reward"] = total_rew
                return (nob, total_nrew, done, info)
        info["raw_reward"] = total_rew
        return (nob, total_nrew, done, info)

    def _reset(self):
        ob = self.env.reset()
        return self.ob_filter(ob) if self.ob_filter is not None else ob

    def _render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def _configure(self, *args, **kwargs):
        return self.env.configure(*args, **kwargs)

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

class Identity(object):
    def __init__(self):
        pass
    def __call(self, target):
        return target
    def output_shape(self, target):
        return target.shape

class RGBImageToVector(object):
    def __init__(self, out_width=40, out_height=40):
        self.out_width = out_width
        self.out_height = out_height

    def __call__(self, obs):
        # obs is an M x N x 3 rgb image, want an (out_width x out_height,) vector
        grayscale = rgb2gray(obs)
        downsample = imresize(grayscale, (self.out_width, self.out_height))
        flatten = downsample.reshape(self.out_width * self.out_height)
        return flatten

    def output_shape(self, x):
        return self.out_width * self.out_height

class Resize(object):
    def __init__(self, out_width=40, out_height=40):
        self.out_width = out_width
        self.out_height = out_height

    def __call__(self, obs):
        downsample = imresize(obs, (self.out_width, self.out_height, 3))
        return downsample

    def output_shape(self, x):
        return (self.out_width, self.out_height, 3)

class FlattenToVector(object):
    def __init__(self, obs):
        if len(obs.shape) == 2:
            self.out_width = obs.shape[0]
            self.out_height = obs.shape[1]
        else:
            self.out_width = obs.shape[1]
            self.out_height = obs.shape[2]

    def __call__(self, obs):
        try:
            flatten = obs.reshape(self.out_width * self.out_height)
            return flatten
        except:
            return [0] * (self.out_width * self.out_height)

    def output_shape(self, x):
        return self.out_width * self.out_height

class DiscreteToHighLow(object):
    """
    converts to discrete (can only press one button at a time)
    """
    def __init__(self, prev_act_space, allowed_actions=None):
        self.high_low = prev_act_space
        self.n = self.high_low.shape
        if allowed_actions is None:
            self.actions = list(range(self.high_low.shape))
        else:
            self.actions = allowed_actions
        self.mapping = {i: self.actions[i] for i in range(len(self.actions))}

    def __call__(self, act):
        action_list = [0] * self.n
        if act in self.mapping:
            action_list[self.mapping[act]] = 1
        return action_list

class HighLowMatrix(object):
    """
    do this in the simplest possible way where we only send max and min actions
    """
    def __init__(self, high_low):
        self.actions = high_low.matrix.shape[0]
        self.matrix = high_low.matrix
        self.n = 2 ** self.actions

    def __call__(self, ac):
        # This converts an integer to a bitmask of which actions should be maxed
        bitmask = "{0:b}".format(ac).ljust(self.actions, "0")
        return [self.matrix[i, int(bit)] for i, bit in enumerate(bitmask)]
