from math import isinf
import os
from dataclasses import dataclass, asdict
import select
import time
from typing import Callable, Type, Union, Tuple
from collections import deque
import datetime
from glob import glob

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
import pdb
# torch.autograd.set_detect_anomaly(True)

from .buffer import PPORolloutStorage
from .network import BaseModel, PPOModel, SEPPOModel
from .utils import pretty_print, step_envs, verify_log_dir, seed_everything
from .envs import VecEnv, make_envs

@dataclass
class PPOConfig:
    # Identifier
    trainer_name: str = "default_trainer"
    save_dir: str = None
    pretrained_model_path: str = None
    
    # Env
    device: torch.device = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 0
    
    # Data
    num_agents: int = 2
    num_steps: int = 3000
    num_processes: int = 20
    num_epochs: int = 10
    mini_batch_size: int = 400
    obs_dim: int = 161
    act_dim: int = 2
    discrete: int = False
    actor_logstd: float = None
    
    # Train
    training_steps: int = 60_000
    epsilon: float = 1e-8
    hidden_size: int = 1024
    lr: float = 5e-5 
    lr_schedule: str = "fix" # "fix" or "annealing"
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.02
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    calculate_return: str = "upgo" # "gae" or "upgo"
    sample_dist: str = "normal"
    
    # Game
    use_opponent_obs: bool = False 
    use_representation_learning: bool = False # TODO
    state_embedding_size: int = 32 
    
    def __post_init__(self):
        seed_everything(self.seed)
        torch.set_num_threads(1)
    
    @classmethod
    def parse_args(cls):
        from simple_parsing import ArgumentParser
        parser = ArgumentParser()
        parser.add_arguments(PPOConfig, dest="config")
        args = parser.parse_args()
        config = args.config
        return config
    

class PPOTrainer:
    
    def __init__(self,
                 config: PPOConfig,
                 envs: gym.Env = None,
                 single_env_factory: Callable = None,
    ) -> None:
        self.config = config
        for k, v in asdict(config).items():
            setattr(self, k, v)
        self.rollouts = PPORolloutStorage(
            num_steps=self.num_steps, 
            num_processes=self.num_processes, 
            act_dim=self.act_dim, 
            obs_dim=self.obs_dim, 
            device=self.device, 
            mini_batch_size=self.mini_batch_size,
            gamma=self.gamma,
            calculate_return=self.calculate_return,
            gae_lambda=self.gae_lambda,
            num_agents=self.num_agents,
        )
        self.step = 0 # times loss backward and update
        self.epoch = 0 
        self.iteration = 0 # self.epoch // self.num_epochs
        self._setup_model_and_optimizer()
        
        if self.pretrained_model_path:
            self.load_w(self.pretrained_model_path)
        
        self.envs = None
        self.single_env_factory = single_env_factory
        if envs is not None:
            config.obs_dim = envs.observation_space.shape[0]
            config.act_dim = envs.action_space.shape[0]
            config.discrete = isinstance(envs.action_space, gym.spaces.Discrete)
            self.set_envs(envs)
        elif single_env_factory is not None:
            self._make_envs()
        else:
            print("Do not specify envs when init trainer. Please call trainer.set_envs manually.")
    
    def _init_buffer(self):
        obs = self.envs.reset()
        with torch.no_grad():
            obs = self._to_tensor(obs)
            value = self.compute_values(obs)
            self.rollouts.reset(
                observation=obs, 
                value_pred=value
            )
            
    def _make_envs(self):
        print(f"===== Create Envs For {self.trainer_name} agent! =====")
        envs: VecEnv = make_envs(
            single_env_factory=self.single_env_factory,
            num_envs=self.num_processes,
            asynchronous=True,
        )
        self.set_envs(envs)
        time.sleep(10)
        return True
        
    def _del_envs(self):
        del self.envs
        self.envs = None
        self.rollouts.initialized = False
    
    def set_envs(self, envs: gym.Env):
        self.envs = envs
        self.rollouts.initialized = False

    def reinit_envs(self):
        self._del_envs()
        self._make_envs()
        self._init_buffer()

    def _setup_model_and_optimizer(self):
        self.model: BaseModel = None
        if self.use_representation_learning:
            self.model = SEPPOModel(
                obs_dim=self.obs_dim,
                act_dim=self.act_dim,
                hidden_size=self.hidden_size,
                state_embedding_size=self.state_embedding_size,
                discrete=self.discrete,
                actor_logstd=None,
                use_opponent_obs=self.use_opponent_obs,
                num_agents=self.num_agents,
                device=self.device,
                sample_dist=self.sample_dist,
            )
            self._backup_model = SEPPOModel(
                obs_dim=self.obs_dim,
                act_dim=self.act_dim,
                hidden_size=self.hidden_size,
                state_embedding_size=self.state_embedding_size,
                discrete=self.discrete,
                actor_logstd=None,
                use_opponent_obs=self.use_opponent_obs,
                num_agents=self.num_agents,
                device=self.device,
                sample_dist=self.sample_dist,
            )
            self.se_optimizer = optim.Adam([
                {"params": self.model.state_encoder.parameters(), "lr": self.lr/10},
                {"params": self.model.state_predictor.parameters(), "lr": self.lr/10},
            ])
        else:
            self.model = PPOModel(
                obs_dim=self.obs_dim,
                act_dim=self.act_dim,
                hidden_size=self.hidden_size,
                discrete=self.discrete,
                actor_logstd=None,
                use_opponent_obs=self.use_opponent_obs,
                num_agents=self.num_agents,
                device=self.device,
                sample_dist=self.sample_dist,
            )
            self._backup_model = PPOModel(
                obs_dim=self.obs_dim,
                act_dim=self.act_dim,
                hidden_size=self.hidden_size,
                discrete=self.discrete,
                actor_logstd=None,
                use_opponent_obs=self.use_opponent_obs,
                num_agents=self.num_agents,
                device=self.device,
                sample_dist=self.sample_dist,
            )
        self.model.train()
        self._backup_step = self.step
        self._backup_model.eval()
        self.optimizer = optim.Adam([
                {"params": self.model.actor.parameters(), "lr": self.lr},
                {"params": self.model.critic.parameters(), "lr": self.lr},
            ])
        # For more info on LrSchedular, see https://towardsdatascience.com/a-visual-guide-to-learning-rate-schedulers-in-pytorch-24bbb262c863
        def lr_lambda(x):
            if x < 2000:
                return x/2000
            else:
                if self.lr_schedule == "annealing":
                    return 1-x/self.training_steps
                else:
                    return 1. 
        self.schedular = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)
    
    def _to_tensor(self, arr):
        return self.model._to_tensor(arr)
    
    def compute_actions(self, observations: Union[torch.Tensor, np.ndarray]):
        return self.model.compute_actions(observations)
    
    def evaluate_actions(self, 
                         observations: Union[torch.Tensor, np.ndarray],
                         actions: Union[torch.Tensor, np.ndarray],
    ):
        return self.model.evaluate_actions(observations, actions)

    def compute_values(self, agent_observations: Union[torch.Tensor, np.ndarray]):
        # [..., num_agents, obs_dim]
        return self.model.compute_values(agent_observations)
    
    def _save_backup(self):
        self._backup_model.load_state_dict(self.model.state_dict())
        self._backup_step = self.step
        
    def _recover_backup(self):
        self.model.load_state_dict(self._backup_model.state_dict())
    
    def _bug_trigger(self, error_message, info_dict):
        debug_dir = "./experiment/.debug"
        os.makedirs(debug_dir, exist_ok=True)
        time_stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        ckpt_path = os.path.join(debug_dir, 
            f"checkpoint-{self.trainer_name}-{time_stamp}-step{self.step}-debug.pkl")
        print(f"[Fatal] Find error in training: [{error_message}]. Save model and info for debug:")
        print(info_dict)
        torch.save({
            "config": asdict(self.config),
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "info": info_dict,
        }, ckpt_path)
        print(f"Recover to backup model saved at step {self._backup_step}. Please re-run the procedure.")
        self._recover_backup()
        
    def save_w(self, save_path):
        save_dir = os.path.dirname(save_path)
        os.makedirs(save_dir, exist_ok=True)
        torch.save({
            "config": asdict(self.config),
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": self.step,
            "epoch": self.epoch,
            "iteration": self.iteration,
        }, save_path)
        print("Successfully saved weights to {}!".format(save_path))
    
    def load_w(self, load_path, only_model: bool = False):
        if os.path.isfile(load_path):
            state_dict = torch.load(load_path, self.device)
            config = state_dict["config"]
            self.model.load_state_dict(state_dict["model"])
            if not only_model:
                self.optimizer.load_state_dict(state_dict["optimizer"])
                self.step = state_dict.get("step", 0)
                self.epoch = state_dict.get("epoch", 0)
                self.iteration = state_dict.get("iteration", 0)
            self._save_backup()
            print("Successfully loaded weights from {}!".format(load_path))
        else:
            raise ValueError("Failed to load weights from {}! File does not exist!".format(load_path))
    
    def _train_se_per_mini_batch(self, sample: Tuple[torch.Tensor], return_dict=True):
        # TODO: This function is out-of-date. please check.
        observations_batch, next_observations_batch, actions_batch, rewards_batch = sample
        # observation: [mini_batch_size, num_agents, obs_dim]
    
        pred_next_observations, pred_rewards = self.model.predict_states_and_rewards(
            observations_batch[:, 0, :], actions_batch
        )
        obs_loss = F.huber_loss(input=pred_next_observations, target=next_observations_batch[:, 0, :], reduction="mean")
        rew_loss = F.huber_loss(input=pred_rewards, target=rewards_batch, reduction="mean")
        loss: torch.Tensor = obs_loss + rew_loss
        if loss.isnan() or loss.isinf() or abs(loss.item()) > 60:
            print("Find loss to be nan or inf or too big (>60), quit this step for continuing training.")
            loss.backward()
            self.se_optimizer.zero_grad()
            return {} if return_dict else None
        
        self.se_optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm:
            norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        else:
            norm = None
        self.se_optimizer.step()
        
        if return_dict:
            return dict(
                se_loss=loss.item(),
                se_norm=norm.item()
            )
        return loss.item(), norm.item()

    _error_return = {"bad_flag": True}

    def _train_ppo_per_mini_batch(self, sample: Tuple[torch.Tensor], return_dict=True):
        _error_return = self._error_return
        observations_batch, old_actions_batch, value_preds_batch, returns_batch, \
        masks_batch, old_action_log_probs_batch, advantages_batch = sample
        # observation: [mini_batch_size, num_agents, obs_dim]
        
        assert old_action_log_probs_batch.shape == (self.mini_batch_size, 1)
        assert advantages_batch.shape == (self.mini_batch_size, 1)
        assert returns_batch.shape == (self.mini_batch_size, 1)
        
        try:
            _, action_log_probs, entropy = self.evaluate_actions(observations_batch[:, 0, :], old_actions_batch)
        except:
            info_dict = {
                "observations": observations_batch[:, 0, :],
                "masks": masks_batch,
                "old_actions": old_actions_batch,
                "old_action_log_probs_batch": old_action_log_probs_batch,
                "is_model_nan": sum([p.isnan().long().sum() for p in self.model.parameters()]),
            }
            self._bug_trigger("Actions Eval Error", info_dict)
            return _error_return
            
        values = self.compute_values(observations_batch)
        entropy_loss = -torch.clamp(entropy.mean(), -10, 10)
        
        assert values.shape == (self.mini_batch_size, 1)
        assert action_log_probs.shape == (self.mini_batch_size, 1)
        
        assert values.requires_grad
        assert action_log_probs.requires_grad
        assert entropy.requires_grad
        
        # Implement policy loss
        logratio = action_log_probs - old_action_log_probs_batch
        ratio = logratio.exp()
        if torch.logical_or(ratio.isinf().any(), ratio.isnan().any()):    
            info_dict = {
                "logratio": logratio,
                "ratio": logratio.exp(),
                "action_log_probs": action_log_probs,
                "old_action_log_probs_batch": old_action_log_probs_batch,
                "masks": masks_batch,
                "is_model_nan": sum([p.isnan().long().sum() for p in self.model.parameters()]),
            }
            self._bug_trigger("Logratio Inf or Nan", info_dict)
            return _error_return
        approx_kl = ((ratio - 1) - logratio).mean().item()
        clipfracs = ((ratio - 1.0).abs() > self.clip_range).float().mean().item()
        
        # dual-clip ppo https://arxiv.org/pdf/1912.09729.pdf
        origin_policy_objective = ratio * advantages_batch 
        clipped_policy_objective = ratio.clamp(1 - self.clip_range, 1 + self.clip_range) * advantages_batch
        first_min = torch.min(torch.stack([origin_policy_objective, clipped_policy_objective], dim=0), dim=0).values
        # second_max = torch.max(torch.stack([first_min, 2 * advantages_batch], dim=0), dim=0).values
        # policy_objective = torch.where(advantages_batch < 0, second_max, first_min)
        policy_loss = -first_min.mean()
        
        # Implement value loss
        value_loss = F.huber_loss(input=values, target=returns_batch, reduction="mean")
        
        loss: torch.Tensor = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss
        # Backward and update
        self.optimizer.zero_grad()
        loss.backward()
        for name, param in self.model.named_parameters():
            if torch.logical_or(param.grad.isnan().any(), param.grad.isinf().any()):
                info_dict = {
                    "logratio": logratio,
                    "ratio": ratio,
                    "approx_kl": approx_kl,
                    "advantage": advantages_batch,
                    "policy_loss": policy_loss,
                    "value_loss": value_loss,
                    "entropy_loss": entropy_loss,
                }
                self.optimizer.zero_grad()
                self._bug_trigger("Gradient Inf or Nan", info_dict)
                return _error_return
        
        if self.max_grad_norm:
            norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        else:
            norm = None
        self.optimizer.step()
        self.schedular.step()
        self.step += 1
        
        if return_dict:
            return dict(
                loss=loss.item(),
                policy_loss=policy_loss.item(),
                value_loss=value_loss.item(),
                mean_entropy=entropy.mean().item(),
                mean_ratio=ratio.mean().item(),
                norm=norm.item(),
                approx_kl=approx_kl,
                clipfracs=clipfracs,
            )
        return loss.item(), policy_loss.item(), value_loss.item(), entropy.mean().item(), ratio.mean().item(), norm.item(), approx_kl, clipfracs

    def _train_per_rollout(self):
        results = []
        
        self.rollouts.compute_returns_and_advantages()
        for epoch in range(self.num_epochs):
            result = {}
            
            if self.use_representation_learning:
                se_data_generator = self.rollouts.se_feed_forward_generator()
                for sample in se_data_generator:
                    result.update(self._train_se_per_mini_batch(sample=sample, return_dict=True))
                    
            data_generator = self.rollouts.ppo_feed_forward_generator()
            try:
                for sample in data_generator:
                    ppo_result = self._train_ppo_per_mini_batch(sample=sample, return_dict=True)
                    result.update(ppo_result)
            except:
                result = self._error_return
                
            if result.get("bad_flag", False):
                return self._error_return
            
            results.append(result)
            self.epoch += 1
            
        self._save_backup()
            
        summary = {}
        if results:
            summary = {k: np.nanmean([r.get(k, np.nan) for r in results]) for k in results[0].keys()}
        self.iteration += 1
        
        approx_kl = summary.get("approx_kl", np.nan)
        if not np.isnan(approx_kl):
            if approx_kl > 0.04:
                self.clip_range = max(0.1, self.clip_range-0.05)
            elif approx_kl < 0.02:
                self.clip_range = min(0.3, self.clip_range+0.05)
        
        return summary

    def train(self):
        envs = self.envs
        if envs is None:
            raise ValueError("No envs found in trainer, please call trainer.set_envs")
        
        if not self.rollouts.initialized:
            self._init_buffer()
        
        # Setup some stats helpers
        episode_rewards = np.zeros([self.num_processes, 1], dtype=float)
        total_episodes = total_steps = 0
        result_recorder = dict(
            crash_sidewalk_rate=deque(maxlen=self.num_processes*self.num_epochs*self.num_steps),
            crash_vehicle_rate=deque(maxlen=self.num_processes*self.num_epochs*self.num_steps),
            idle_rate=deque(maxlen=self.num_processes*self.num_epochs*self.num_steps),
            speed_km_h=deque(maxlen=self.num_processes*self.num_epochs*self.num_steps),
            max_step_rate=deque(maxlen=self.num_processes*self.num_epochs),
            success_rate=deque(maxlen=self.num_processes*self.num_epochs),
            episode_reward=deque(maxlen=self.num_processes*self.num_epochs),
            episode_length=deque(maxlen=self.num_processes*self.num_epochs)
        )
        # ===== Sample Data =====
        for index in range(self.num_steps):
            # in torch
            with torch.no_grad():
                action, action_log_prob, _ = self.compute_actions(self.rollouts.observations[index, :, 0, :])
            # in np
            obs, reward, done, info, mask, total_episodes, \
            total_steps, episode_rewards = step_envs(
                cpu_actions=action.cpu().numpy(),
                envs=envs,
                episode_rewards=episode_rewards,
                result_recorder=result_recorder,
                total_steps=total_steps,
                total_episodes=total_episodes,
            )
            # in torch
            with torch.no_grad():
                obs = self._to_tensor(obs)
                value = self.compute_values(obs)
            mask = self._to_tensor(mask).view(-1, 1)
            reward = self._to_tensor(reward).view(-1, 1)
            # insert to rollout
            self.rollouts.insert(obs, action, action_log_prob, value, reward, mask)
        # ===== Update Policy =====
        stats = self._train_per_rollout()
        self.rollouts.after_update()
        if stats.get("bad_flag", False):
            return self._error_return
        # ===== Log information =====
        stats.update({k: np.mean(r) for k, r in result_recorder.items()})
        
        return stats
        
        
if __name__ == "__main__":
    # TODO
    ...