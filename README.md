## Overview

This is the record of assignment4 of Reinforcement Learning Course. Code is inheritted from [cs260r-assignment-2023fall](https://github.com/ucla-rlcourse/cs260r-assignment-2023fall). 

In this project, I elaborately implemented a PPO algorithm and optimize its performance from architeture desgin, regularization and training option. Details are in [report](report.pdf).

## Code 

Only files I modified widely are listed. 

+ `core.buffer`: A buffer for training PPO, with function of scale rewards, calculate return (GAE or UPGO) and generate batched data for next state prediction and PPO training.
+ `core.network`: All neural networks (`Actor`, `Critic`, `StateEncoder`, `Predictor`) stores in this file. There is also a simple wrapper class `Policy` help you pass this object to the multi-agent environment instead of a whole trainer (which may create unexpected files or records).
+ `core.ppo_trainer`: Trainer and config for PPO training, well designed to handle unstableness when training PPO in multi-agent environment. Detailed hyper-parameters and its descriptions are in `PPOConfig`.
+ `train_sa.py`: A simple toy code to train PPO with all code above. Just pass config and environment to trainer and call `trainer.train()`!
+ `train_ma.py`: This file implements prioritized fictitious self-play in `Player`. Also log and record some training curve when training.

