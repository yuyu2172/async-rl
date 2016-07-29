# Async-RL

Adapted from [muupan/async-rl](https://github.com/muupan/async-rl) to work with OpenAI Gym.
Based on original paper: [Asynchronous Methods for Deep Reinforcement Learning](http://arxiv.org/abs/1602.01783).

## Requirements:

Same as OpenAI gym.
pip: see requirements.txt

## How to run

```
python run_a3c.py --env Breakout-v0 --threads 16 --outdir ~/outdir
```

```
python run_a3c.py --env DoomBasic-v0 --threads 16 --outdir ~/outdir --act-filter doom-minimal
```

Note: Environments are automatically rescaled to 80x80 as per the original paper, so you should not need to apply an observation space filter.

## Similar Projects

- https://github.com/muupan/async-rl
- https://github.com/miyosuda/async_deep_reinforce
