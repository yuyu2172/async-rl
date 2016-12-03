import copy
import os
import sys
import time
import numpy as np
import statistics
import chainer
import multiprocessing as mp

import async_rl.envs.ale as ale
from async_rl.models.dqn_phi import dqn_phi
from async_rl.analysis.do_line_profile import do_line_profile


env_lock = mp.Lock()


# TODO: fix this
def eval_performance(rom, p_func, n_runs):
    assert n_runs > 1, 'Computing stdev requires at least two runs'
    scores = []
    for i in range(n_runs):
        env = ale.ALE(rom, treat_life_lost_as_terminal=False)
        test_r = 0
        while not env.is_terminal:
            s = chainer.Variable(np.expand_dims(dqn_phi(env.state), 0))
            pout = p_func(s)
            a = pout.action_indices[0]
            test_r += env.receive_action(a)
        scores.append(test_r)
        print('test_{}:'.format(i), test_r)
    mean = statistics.mean(scores)
    median = statistics.median(scores)
    stdev = statistics.stdev(scores)
    return mean, median, stdev


class WorkerProcess(object):
    def __init__(self, process_idx, counter, max_score,
                 args, agent, env, start_time):
        self.process_idx = process_idx
        self.counter = counter
        self.max_score = max_score
        self.args = args
        self.agent = agent
        self.env = env
        self.start_time = start_time

    def train_loop(self, profile=False):
        args = [self.process_idx, self.counter, self.max_score, self.args, self.agent,
                self.env, self.start_time]
        try:
            if profile and self.process_idx == 0:
                do_line_profile(self._train_loop)(*args)
            else:
                self._train_loop(*args)
        except KeyboardInterrupt:
            if self.process_idx == 0:
                # Save the current model before being killed
                self.agent.save_model(os.path.join(
                    args.outdir, '{}_keyboardinterrupt.h5'.format(self.counter.value)))
                print('Saved the current model to {}'.format(
                      self.args.outdir, file=sys.stderr))
            raise

    def _train_loop(self, process_idx, counter, max_score, args, agent, env, start_time):
        # Locking to avoid resetting all envs at once
        with env_lock:
            total_r = 0
            episode_r = 0
            global_t = 0
            local_t = 0

            obs = env.reset()
            r = 0
            done = False

        while True:

            # Get and increment the global counter
            with counter.get_lock():
                counter.value += 1
                global_t = counter.value
            local_t += 1

            if global_t > args.steps:
                break

            agent.optimizer.lr = (
                args.steps - global_t - 1) / args.steps * args.lr

            total_r += r
            episode_r += r

            a = agent.act(obs, r, done)

            if done:
                if process_idx == 0:
                    print('{} global_t:{} local_t:{} lr:{} episode_r:{}'.format(
                        args.outdir, global_t, local_t, agent.optimizer.lr, episode_r))
                episode_r = 0
                obs = env.reset()
                r = 0
                done = False
            else:
                obs, r, done, info = env.step(a)

            if global_t % args.eval_frequency == 0:
                self.evaluation(agent, args, start_time, global_t, max_score)

            # Save the final model
            if global_t == args.steps + 1:
                agent.save_model(
                    os.path.join(args.outdir, '{}_finish.h5'.format(args.steps)))
                print('Saved the final model to {}'.format(args.outdir))

    
    def evaluation(self, agent, args, start_time, global_t, max_score):
        # Evaluation

        # We must use a copy of the model because test runs can change
        # the hidden states of the model
        test_model = copy.deepcopy(agent.model)
        test_model.reset_state()

        def p_func(s):
            pout, _ = test_model.pi_and_v(s)
            test_model.unchain_backward()
            return pout
        mean, median, stdev = eval_performance(
            args.rom, p_func, args.eval_n_runs)
        with open(os.path.join(args.outdir, 'scores.txt'), 'a+') as f:
            elapsed = time.time() - start_time
            record = (global_t, elapsed, mean, median, stdev)
            print('\t'.join(str(x) for x in record), file=f)
        with max_score.get_lock():
            if mean > max_score.value:
                # Save the best model so far
                print('The best score is updated {} -> {}'.format(
                    max_score.value, mean))
                filename = os.path.join(
                    args.outdir, '{}.h5'.format(global_t))
                agent.save_model(filename)
                print('Saved the current best model to {}'.format(
                    filename))
                max_score.value = mean
