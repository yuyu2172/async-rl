import argparse
import copy
import multiprocessing as mp
import os
import sys
import statistics
import time
import random

import chainer
from chainer import links as L
from chainer import functions as F
import numpy as np


import async_rl.workers.a3c as a3c
import async_rl.envs.ale as ale

import optimizers.rmsprop_async as rmsprop_async

from async_rl.models.a3c_models import A3CFF, A3CLSTM
import async_rl.models.policy as policy
import async_rl.models.v_function as v_function
import async_rl.models.dqn_head as dqn_head
from async_rl.models.init_like_torch import init_like_torch
from async_rl.models.dqn_phi import dqn_phi

from async_rl.workers.worker import WorkerProcess

from async_rl.utils.random_seed import set_random_seed
from async_rl.utils.prepare_output_dir import prepare_output_dir
import async_rl.utils.async as async


def main():

    # Prevent numpy from using multiple threads
    os.environ['OMP_NUM_THREADS'] = '1'

    import logging
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument('processes', type=int)
    parser.add_argument('rom', type=str)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--outdir', type=str, default='out')
    parser.add_argument('--use-sdl', action='store_true')
    parser.add_argument('--t-max', type=int, default=5)
    parser.add_argument('--beta', type=float, default=1e-2)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--steps', type=int, default=8 * 10 ** 7)
    parser.add_argument('--lr', type=float, default=7e-4)
    parser.add_argument('--eval-frequency', type=int, default=10 ** 6)
    parser.add_argument('--eval-n-runs', type=int, default=10)
    parser.add_argument('--use-lstm', action='store_true')
    parser.set_defaults(use_sdl=False)
    parser.set_defaults(use_lstm=False)
    args = parser.parse_args()

    if args.seed is not None:
        random_seed.set_random_seed(args.seed)

    args.outdir = prepare_output_dir(args, args.outdir)

    print('Output files are saved in {}'.format(args.outdir))

    n_actions = ale.ALE(args.rom).number_of_actions

    def model_opt():
        if args.use_lstm:
            model = A3CLSTM(n_actions)
        else:
            model = A3CFF(n_actions)
        opt = rmsprop_async.RMSpropAsync(lr=7e-4, eps=1e-1, alpha=0.99)
        opt.setup(model)
        opt.add_hook(chainer.optimizer.GradientClipping(40))
        return model, opt

    model, opt = model_opt()

    shared_params = async.share_params_as_shared_arrays(model)
    shared_states = async.share_states_as_shared_arrays(opt)

    max_score = mp.Value('f', np.finfo(np.float32).min)
    counter = mp.Value('l', 0)
    start_time = time.time()

    # Write a header line first
    with open(os.path.join(args.outdir, 'scores.txt'), 'a+') as f:
        column_names = ('steps', 'elapsed', 'mean', 'median', 'stdev')
        print('\t'.join(column_names), file=f)

    def run_func(process_idx):
        env = ale.ALEGymWrapper(args.rom, process_idx)
        #ale.ALE(args.rom, use_sdl=args.use_sdl)
        model, opt = model_opt()
        async.set_shared_params(model, shared_params)
        async.set_shared_states(opt, shared_states)

        agent = a3c.A3C(model, opt, args.t_max, 0.99, beta=args.beta,
                        process_idx=process_idx, phi=dqn_phi)
        worker = WorkerProcess(process_idx, counter, max_score,
                               args, agent, env, start_time)
        worker.train_loop(args.profile)

    async.run_async(args.processes, run_func)


if __name__ == '__main__':
    main()
