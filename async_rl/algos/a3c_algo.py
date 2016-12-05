import copy
from logging import getLogger
import os
import sys
import time
import numpy as np
import statistics
import chainer
import multiprocessing as mp

import chainer
from chainer import serializers
from chainer import functions as F
import async_rl.utils.copy_param as copy_param

import async_rl.envs.ale as ale
from async_rl.models.dqn_phi import dqn_phi
from async_rl.analysis.do_line_profile import do_line_profile


env_lock = mp.Lock()
logger = getLogger(__name__)


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


class A3CAlgo(object):
    """A3C: Asynchronous Advantage Actor-Critic.

    See http://arxiv.org/abs/1602.01783
    """
    def __init__(self, process_idx, counter, max_score,
                 args, agent, env, start_time,
                 # previously in Agent
                 shared_model, thread_model, optimizer, t_max, gamma, beta,
                 phi, pi_loss_coef=1.0, v_loss_coef=0.5):
        self.process_idx = process_idx
        self.counter = counter
        self.max_score = max_score
        self.args = args
        self.agent = agent
        self.env = env
        self.start_time = start_time

        ################## FROM agent ########################
        # Globally shared model
        self.shared_model = shared_model

        # Thread specific model
        self.model = thread_model

        self.optimizer = optimizer

        self.gamma = gamma
        self.beta = beta
        self.phi = phi
        self.pi_loss_coef = pi_loss_coef
        self.v_loss_coef = v_loss_coef
        #self.keep_loss_scale_same = keep_loss_scale_same

        self.t_max = t_max

    def train(self, profile=False):
        args = [self.process_idx, self.counter, self.max_score, self.args, self.agent,
                self.env, self.start_time]
        try:
            if profile and self.process_idx == 0:
                do_line_profile(self._train_core)(*args)
            else:
                self._train_core(*args)
        except KeyboardInterrupt:
            if self.process_idx == 0:
                # Save the current model before being killed
                self.save_model(os.path.join(
                    self.args.outdir, '{}_keyboardinterrupt.h5'.format(self.counter.value)))
                print('Saved the current model to {}'.format(
                      self.args.outdir, file=sys.stderr))
            raise

    def _train_core(self, process_idx, counter, max_score, args, agent, env, start_time):
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
            if global_t > args.steps:
                break
            self.optimizer.lr = (
                args.steps - global_t - 1) / args.steps * args.lr

            total_r += r
            episode_r += r

            next_obs, done, samples_data, lookup_agent_info = self.proceed(
                env, agent, obs, self.t_max)
            self.optimize(samples_data, lookup_agent_info, done)

            if done:
                episode_r = 0
                obs = env.reset()
                r = 0
                done = False
            else:
                obs = next_obs

            # update counter
            with counter.get_lock():
                counter.value += len(samples_data)
                global_t = counter.value
            prev_t = global_t - len(samples_data)  # this is little tricky
            local_t += len(samples_data)

            #print('global_t {}'.format(global_t))
            if abs(global_t % args.eval_frequency -\
                   prev_t % args.eval_frequency) > self.t_max:
                self.evaluation(agent, args, start_time, global_t, max_score)

    def proceed(self, env, agent, obs, steps, do_lookup=True):
        """Proceed agent by steps

        If environment terminates in the middle of advancing the episode,
            return the truncated episode info.

        Returns:
            obs:
            done: Whether there was termination from ``t`` to ``t + steps``.
            samples_data: info gathered through the course of episode advance
        """
        samples_data = []
        path_length = 0
        while path_length < steps:
            a, agent_info = agent.get_action(obs)
            next_obs, r, done, env_info = env.step(a)
            # clip reward
            r = np.clip(r, -1, 1)
            sample_data = {
                'observation': obs,
                'reward': r,
                'action': a,
                'agent_info': agent_info,
                'env_info': env_info}
            samples_data.append(sample_data)
            path_length += 1
            if done:
                obs = None
                break
            obs = next_obs

        lookup_agent_info = {}
        if do_lookup:
            lookup_agent_info = agent.lookup(next_obs)
        return obs, done, samples_data, lookup_agent_info

    def optimize(self, samples_data, lookup_agent_info, done):
        """Calculate loss from collected samples and update model with it.
        """
        #print('start optimize ', done)
        if done:
            R = 0
        else:
            R = float(lookup_agent_info['values'].data)

        pi_loss = 0
        v_loss = 0
        # from muupan's code
        # if (is_state_terminal and self.t_start == self.t)
        if len(samples_data) == 1:
            return
        for i in reversed(range(len(samples_data))):
            R *= self.gamma
            R += samples_data[i]['reward']
            v = samples_data[i]['agent_info']['values']
            advantage = R - v
            # Accumulate gradients of policy
            log_prob = samples_data[i]['agent_info']['action_log_prob']
            entropy = samples_data[i]['agent_info']['action_entropy']

            # Log probability is increased proportionally to advantage
            pi_loss -= log_prob * float(advantage.data)
            # Entropy is maximized
            pi_loss -= self.beta * entropy
            # Accumulate gradients of value function

            v_loss += (v - R) ** 2 / 2

        pi_loss *= self.pi_loss_coef
        v_loss *= self.v_loss_coef

        # Normalize the loss of sequences truncated by terminal states
        # TODO
        #if self.keep_loss_scale_same and \
        #        self.t - self.t_start < self.t_max:
        #    factor = self.t_max / (self.t - self.t_start)
        #    pi_loss *= factor
        #    v_loss *= factor

        total_loss = pi_loss + F.reshape(v_loss, pi_loss.data.shape)

        # Compute gradients using thread-specific model
        self.model.zerograds()
        total_loss.backward()
        # Copy the gradients to the globally shared model
        self.shared_model.zerograds()
        copy_param.copy_grad(
            target_link=self.shared_model, source_link=self.model)
        #if self.process_idx == 0:
        #    norm = self.optimizer.compute_grads_norm()
        #    logger.debug('grad norm:%s', norm)
        self.optimizer.update()

        self.sync_parameters()
        self.model.unchain_backward()

    def sync_parameters(self):
        copy_param.copy_param(target_link=self.model,
                              source_link=self.shared_model)

    def load_model(self, model_filename):
        """Load a network model form a file
        """
        serializers.load_hdf5(model_filename, self.model)
        copy_param.copy_param(target_link=self.model,
                              source_link=self.shared_model)
        opt_filename = model_filename + '.opt'
        if os.path.exists(opt_filename):
            print('WARNING: {0} was not found, so loaded only a model'.format(
                opt_filename))
            serializers.load_hdf5(model_filename + '.opt', self.optimizer)

    def save_model(self, model_filename):
        """Save a network model to a file
        """
        serializers.save_hdf5(model_filename, self.model)
        serializers.save_hdf5(model_filename + '.opt', self.optimizer)
        
    
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
            #print('\t'.join(str(x) for x in record), file=f)
            print >>f, '\t'.join(str(x) for x in record)
        with max_score.get_lock():
            if mean > max_score.value:
                # Save the best model so far
                print('The best score is updated {} -> {}'.format(
                    max_score.value, mean))
                filename = os.path.join(
                    args.outdir, '{}.h5'.format(global_t))
                self.save_model(filename)
                print('Saved the current best model to {}'.format(
                    filename))
                max_score.value = mean
