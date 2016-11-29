from __future__ import print_function
import copy
import multiprocessing as mp
import numpy as np
import os
import sys
import time
import chainer
import statistics
from gym import error

import async_rl.a3c as a3c
import async_rl.async as async
from async_rl.prepare_output_dir import prepare_output_dir

env_lock = mp.Lock()

def eval_performance(process_idx, env, model, phi, n_runs):
    assert n_runs > 1, 'Computing stdev requires at least two runs'
    scores = []
    for i in range(n_runs):
        model.reset_state()
        obs = env.reset()
        done = False
        test_r = 0
        while not done:
            s = chainer.Variable(np.expand_dims(phi(obs), 0))
            pout, _ = model.pi_and_v(s)
            a = pout.action_indices[0]
            obs, r, done, info = env.step(a)
            test_r += r
        scores.append(test_r)
        print('test_{}:'.format(i), test_r)
    mean = statistics.mean(scores)
    median = statistics.median(scores)
    stdev = statistics.stdev(scores)
    return mean, median, stdev


def train_loop(process_idx, counter, make_env, max_score, args, agent, env,
               start_time, outdir):
    try:

        # Locking to avoid resetting all envs at once
        with env_lock:
            total_r = 0
            episode_r = 0
            global_t = 0
            local_t = 0
            if args.seed is not None:
                env.seed(args.seed + process_idx)
            else:
                env.seed()
            obs = env.reset()
            r = 0
            done = False
            last_time = time.time()
            last_global_t = 0

        while True:

            # Get and increment the global counter
            with counter.get_lock():
                counter.value += 1
                global_t = counter.value
            local_t += 1

            if global_t > args.steps:
                env.close()
                break

            agent.optimizer.lr = (
                args.steps - global_t - 1) / args.steps * args.lr

            total_r += r
            episode_r += r

            a = agent.act(obs, r, done)

            if done:
                if process_idx == 0:
                    current_time = time.time()
                    time_since_last = current_time - last_time
                    steps_since_last = global_t - last_global_t
                    last_time = current_time
                    last_global_t = global_t
                    current_daily_speed  = 86400 * steps_since_last / (time_since_last + 1e-10)
                    eta = round(24 * (args.steps - global_t) / current_daily_speed, 2)
                    print(
                        '{} global_t:{} local_t:{} lr:{} r:{} speed:{}M '
                        'step/day eta:{} hrs'.format(
                        outdir, global_t, local_t, agent.optimizer.lr, episode_r,
                        round(current_daily_speed/1000000, 2), eta))
                episode_r = 0
                obs = env.reset()
                r = 0
                done = False
            else:
                try:
                    obs, r, done, info = env.step(a)
                except error.ResetNeeded:
                    # Monitor is very picky about never stepping after reset
                    if process_idx == 0:
                        current_time = time.time()
                        time_since_last = current_time - last_time
                        steps_since_last = global_t - last_global_t
                        last_time = current_time
                        last_global_t = global_t
                        current_daily_speed = 86400 * steps_since_last / (time_since_last + 1e-10)
                        eta = round(24 * (args.steps - global_t) / current_daily_speed, 2)
                        print(
                            '{} global_t:{} local_t:{} lr:{} r:{} speed:{}M'
                            'step/day eta:{} hrs'.format(
                                outdir, global_t, local_t, agent.optimizer.lr, episode_r,
                                round(current_daily_speed / 1000000, 2), eta))
                    episode_r = 0
                    obs = env.reset()
                    r = 0
                    done = False

            if global_t % args.eval_frequency == 0:
                # Evaluation

                # We must use a copy of the model because test runs can change
                # the hidden states of the model
                test_model = copy.deepcopy(agent.model)
                test_model.reset_state()

                mean, median, stdev = eval_performance(
                    process_idx, env, test_model, agent.phi,
                    args.eval_n_runs)
                with open(os.path.join(outdir, 'scores.txt'), 'a+') as f:
                    elapsed = time.time() - start_time
                    record = (global_t, elapsed, mean, median, stdev,
                              round(current_daily_speed / 1000000, 2), eta)
                    print('\t'.join(str(x) for x in record), file=f)
                with max_score.get_lock():
                    if mean > max_score.value:
                        # Save the best model so far
                        print('The best score is updated {} -> {}'.format(
                            max_score.value, mean))
                        filename = os.path.join(
                            outdir, '{}.h5'.format(global_t))
                        agent.save_model(filename)
                        print('Saved the current best model to {}'.format(
                            filename))
                        max_score.value = mean

    except KeyboardInterrupt:
        if process_idx == 0:
            # Save the current model before being killed
            agent.save_model(os.path.join(
                outdir, '{}_keyboardinterrupt.h5'.format(global_t)))
            print('Saved the current model to {}'.format(
                outdir), file=sys.stderr)
            env.monitor.close()
            print('Exiting due to KeyboardInterrupt')
        env.close()


    if global_t == args.steps + 1:
        # Save the final model
        agent.save_model(
            os.path.join(args.outdir, '{}_finish.h5'.format(args.steps)))
        print('Saved the final model to {}'.format(args.outdir))


def train_loop_with_profile(process_idx, counter, make_env, max_score, args,
                            agent, env, start_time, outdir):
    import cProfile
    cmd = 'train_loop(process_idx, counter, make_env, max_score, args, agent, env, start_time, outdir)'
    cProfile.runctx(cmd, globals(), locals(),
                    'profile-{}.out'.format(os.getpid()))


def run_a3c(processes, make_env, model_opt, phi, t_max=1, beta=1e-2,
            profile=False, steps=8 * 10 ** 7, eval_frequency=10 ** 6,
            eval_n_runs=10, args={}):

    # Prevent numpy from using multiple threads
    os.environ['OMP_NUM_THREADS'] = '1'

    outdir = prepare_output_dir(args, args.outdir)

    print('Output files are saved in {}'.format(outdir))

    model, opt = model_opt()

    shared_params = async.share_params_as_shared_arrays(model)
    shared_states = async.share_states_as_shared_arrays(opt)

    max_score = mp.Value('f', np.finfo(np.float32).min)
    counter = mp.Value('l', 0)
    start_time = time.time()

    # Write a header line first
    with open(os.path.join(outdir, 'scores.txt'), 'a+') as f:
        column_names = ('steps', 'elapsed', '\tmean', 'median', 'stdev', 'speed', 'eta_hrs')
        print('\t'.join(column_names), file=f)

    def run_func(process_idx):
        env = make_env(process_idx)
        model, opt = model_opt()
        async.set_shared_params(model, shared_params)
        async.set_shared_states(opt, shared_states)

        agent = a3c.A3C(model, opt, t_max, 0.99, beta=beta,
                        process_idx=process_idx, phi=phi)

        # Loading model if model.h5 exists in output dir
        if os.path.isfile(os.path.join(args.outdir, 'model.h5')):
            if process_idx == 0:
                print('Found model.h5 in output directory. Resuming computation of this model.')
            agent.load_model(os.path.join(args.outdir, 'model.h5'))
        else:
            if process_idx == 0:
                print('Exiting model not found. Save model.h5 and model.h5.opt in the output directory to resume computation of a model.')
                print('Building new model from scratch.')

        if profile:
            train_loop_with_profile(process_idx, counter, make_env, max_score,
                       args, agent, env, start_time, outdir=outdir)
        else:
            train_loop(process_idx, counter, make_env, max_score,
                       args, agent, env, start_time, outdir=outdir)

    async.run_async(processes, run_func)
