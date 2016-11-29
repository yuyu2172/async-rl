import argparse
import multiprocessing as mp
import shutil
import os
import time

import gym
import chainer
from chainer import links as L
from gym.envs.doom.doom_env import DOOM_SETTINGS
import cv2

import async_rl.a3c as a3c
import async_rl.dqn_head as dqn_head
import async_rl.models.policy as policy
import async_rl.rmsprop_async as rmsprop_async
import async_rl.models.v_function as v_function
import async_rl.random_seed as random_seed
from async_rl.envs.filtered_env import *
from async_rl.models.init_like_torch import init_like_torch
from async_rl import a3c_runner
from async_rl.utils import imresize



def phi(obs):
    resized = cv2.resize(obs, (84, 84))
    return resized.transpose(2, 0, 1).astype(np.float32) / 255

class A3CFF(chainer.Chain):

    def __init__(self, n_actions):
        self.head = dqn_head.NIPSDQNHead(n_input_channels=3)
        self.pi = policy.FCSoftmaxPolicy(self.head.n_output_channels, n_actions)
        self.v = v_function.FCVFunction(self.head.n_output_channels)
        super(A3CFF, self).__init__(self.head, self.pi, self.v)
        init_like_torch(self)

    def pi_and_v(self, state, keep_same_state=False):
        out = self.head(state)
        return self.pi(out), self.v(out)

class A3CLSTM(chainer.Chain):

    def __init__(self, n_actions):
        self.head = dqn_head.NIPSDQNHead(n_input_channels=3)
        self.pi = policy.FCSoftmaxPolicy(self.head.n_output_channels, n_actions)
        self.v = v_function.FCVFunction(self.head.n_output_channels)
        self.lstm = L.LSTM(self.head.n_output_channels,
                           self.head.n_output_channels)
        super(A3CLSTM, self).__init__(self.head, self.lstm, self.pi, self.v)
        init_like_torch(self)

    def pi_and_v(self, state, keep_same_state=False):
        out = self.head(state)
        if keep_same_state:
            prev_h, prev_c = self.lstm.h, self.lstm.c
            out = self.lstm(out)
            self.lstm.h, self.lstm.c = prev_h, prev_c
        else:
            out = self.lstm(out)
        return self.pi(out), self.v(out)

    def reset_state(self):
        self.lstm.reset_state()

    def unchain_backward(self):
        self.lstm.h.unchain_backward()
        self.lstm.c.unchain_backward()

def main():
    import logging
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default=None, required=True, help="Gym environment to run")
    parser.add_argument('--agent', type=str, default='a3c.lstm', help="Agent to use a3c.ff or a3c.lstm")
    parser.add_argument('--threads', type=int, required=True, help="Number of parallel threads to use")
    parser.add_argument('--outdir', type=str, default=None, required=True, help="Output directory")
    parser.add_argument('--seed', type=int, default=None, help="Seed to use")
    parser.add_argument('--t-max', type=int, default=5, help="Number of threads steps between gradient update")
    parser.add_argument('--beta', type=float, default=1e-2, help="Parameter that controls the strength of entropy regularization")
    parser.add_argument('--profile', action='store_true', help="Run with profiler enabled")
    parser.add_argument('--steps', type=int, default=8 * 10 ** 7, help="Total number of steps to run algo")
    parser.add_argument('--lr', type=float, default=7e-4, help="Learning rate for RMSprop")
    parser.add_argument('--eval-frequency', type=int, default=10 ** 5, help="How often to calculate the agent performance")
    parser.add_argument('--eval-n-runs', type=int, default=10, help="How many runs to use to calculate the agent performance")
    parser.add_argument('--render', action='store_true', help="Render simulation (Window visible)")
    parser.add_argument('--skiprate', type=int, default=0, help="Runs an action and repeats the same action for x steps")
    parser.add_argument('--obs-filter', type=str, default=None, help="Observation space filter")
    parser.add_argument('--act-filter', type=str, default=None, help="Action space filter")
    parser.add_argument('--height', type=int, default=0, help="For resize filters, the resized height")
    parser.add_argument('--width', type=int, default=0, help="For resize filters, the resized width")
    parser.set_defaults(profile=False)
    parser.set_defaults(render=False)
    args = parser.parse_args()

    if args.seed is not None:
        random_seed.set_random_seed(args.seed)

    # Use lock to avoid processes getting stuck on launch
    env_lock = mp.Lock()
    env = gym.make(args.env)

    # Parsing observation space filters
    if args.obs_filter is None:
        obs_filter = None

    elif args.obs_filter == 'grey-vector':
        # Converts to grayscale, downscales to 40x40, and flattens to a vector
        if min(args.height, args.width) <= 0:
            raise RuntimeError('--width and --height params are required with this filter.')
        obs_filter = RGBImageToVector(args.width, args.height)

    elif args.obs_filter == 'resize':
        # Downscales an image, without flattening to vector
        if min(args.height, args.width) <= 0:
            raise RuntimeError('--width and --height params are required with this filter.')
        obs_filter = Resize(args.width, args.height)

    elif args.obs_filter == 'flatten':
        # Flattens RGB to a vector without downsampling
        obs_filter = FlattenToVector(env.observation_space.sample())

    else:
        raise RuntimeError('--obs-filter not recognized.')

    # Parsing action space filters
    if args.act_filter is None:
        act_filter = None

    elif args.act_filter == 'doom-minimal':
        # Doom with only the allowed actions for the level (Discrete - max 1 button at a time)
        allowed_actions = DOOM_SETTINGS[env.level][4]
        act_filter = DiscreteToHighLow(env.action_space, allowed_actions)

    elif args.act_filter == 'doom-small-constant':
        # Doom with the minimum constant actions to complete all levels (Discrete - max 1 button at a time)
        allowed_actions = [0, 10, 11, 13, 14, 15, 31]
        act_filter = DiscreteToHighLow(env.action_space, allowed_actions)

    elif args.act_filter == 'high-low-matrix':
        # Converts HighLow to a matrix with binary mask (Discrete - Multiple actions allowed at the same time)
        act_filter = HighLowMatrix(env.action_space)

    else:
        raise RuntimeError('--act-filter not recognized.')

    # Applying filters
    env = FilteredEnv(env, ob_filter=obs_filter, act_filter=act_filter, skiprate=args.skiprate)

    # Checking if hdf5 (h5py) is installed
    try:
        chainer.serializers.save_hdf5(None, None)
    except AttributeError:
        pass

    # Creating directories
    # Resuming model if model.h5 is found in output directory
    if os.path.isdir(os.path.join(args.outdir)) and not os.path.isfile(os.path.join(args.outdir, 'model.h5')):
        shutil.rmtree(os.path.join(args.outdir))
    if os.path.isdir(os.path.join(args.outdir, 'gym-monitor')):
        shutil.rmtree(os.path.join(args.outdir, 'gym-monitor'))
    os.makedirs(os.path.join(args.outdir, 'gym-monitor'))

    def make_env(process_idx):
        with env_lock:
            if process_idx == 0:
                env = gym.make(args.env)
                env = FilteredEnv(env, ob_filter=obs_filter, act_filter=act_filter, skiprate=args.skiprate)
                env.monitor.start(os.path.join(args.outdir, 'gym-monitor'))
                return env
            else:
                env = gym.make(args.env)
                env = FilteredEnv(env, ob_filter=obs_filter, act_filter=act_filter, skiprate=args.skiprate)
                return env

    # Getting number of output nodes
    if not isinstance(env.action_space, gym.spaces.discrete.Discrete):
        raise NotImplementedError('Only "discrete" action space implemented. Use an action space filter to convert to "discrete".')
    n_actions = env.action_space.n

    def model_opt():
        if args.agent == 'a3c.lstm':
            model = A3CLSTM(n_actions)
        else:
            model = A3CFF(n_actions)
        opt = rmsprop_async.optimizers.RMSpropAsync(lr=args.lr, eps=1e-1, alpha=0.99)
        opt.setup(model)
        opt.add_hook(chainer.optimizer.GradientClipping(40))
        return model, opt

    a3c_runner.run_a3c(args.threads, make_env, model_opt, phi, t_max=args.t_max,
                       beta=args.beta, profile=args.profile, steps=args.steps,
                       eval_frequency=args.eval_frequency,
                       eval_n_runs=args.eval_n_runs, args=args)


if __name__ == '__main__':
    main()
