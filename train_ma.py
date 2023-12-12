import os
from collections import defaultdict
from dataclasses import dataclass, field, asdict
import datetime
from glob import glob
from functools import partial
from abc import ABC, abstractmethod
import random
import logging
import time
from contextlib import contextmanager

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
mp.set_start_method("spawn", force=True)
from tqdm import tqdm
from metadrive.envs.marl_envs.marl_racing_env import MultiAgentRacingEnv
import wandb
import pdb

from core.envs import make_envs, VecEnv
from core.ppo_trainer import PPOTrainer, PPOConfig
from core.utils import pretty_print, seed_everything, verify_log_dir
from core.network import Policy
from eval import eval

@dataclass
class EnvironmentConfig:
    # If you want to extend the number of agents to 12, you need to fill the policies for agent2, ..., agent11 and
    # use these configs:
    # num_agents: int = 12,
    # map_config: dict = dict(lane_num=2, exit_length=60),  # Increasing the space in spawning the agents.
    num_agents: int = 2
    crash_sidewalk_penalty: float = 1
    success_reward: float = 100
    speed_reward: float = 0
    out_of_road_penalty: float = 40
    crash_vehicle_penalty: float = 0 # Default is 10. Set to -10 so env will reward agent +10 for crashing.
    idle_penalty: float = 40
    horizon: int = 2000 # Notice!
    # use_render=True,  # For debug, set num_processes=1

ENV_CONFIG = EnvironmentConfig()

class RacingEnvWithOpponent(MultiAgentRacingEnv):
    """
    MetaDrive provides a MultiAgentRacingEnv class, where all the input/output data is dict.
    There can be multiple vehicles running at the same time in the environment.
    We will consider the "agent0" as the "ego vehicle" and others are the "opponent vehicle".
    This wrapper class will load the opponent vehicle's policy then use this policy to control the opponent
    vehicles. The maneuver of opponent vehicle(s) is conducted inside this wrapper and this wrapper will only expose
    the data for the ego vehicle in `env.step` and `env.reset`. Therefore, this environment still behaves like a
    single-agent RL environment, and thus we can reuse single-agent RL algorithm.
    Though the environment supports up to 12 agents running concurrently, we now only consider the competition between
    two agents.
    """

    EGO_VEHICLE_NAME = "agent0"

    def __init__(self, config, opponent_ckpt_path=None):
        # You can increase the number of agents if you want, but you need to prepare policy for them.
        assert config["num_agents"] == 2
        self.num_agents = config["num_agents"]
        config.update({"target_vehicle_configs": {"agent0": {"use_special_color": True}}})

        # Load policy to control agent1.
        agent1 = Policy(opponent_ckpt_path)
        self.policy_map = {
            "agent1": agent1  # Remember to instantiate the policy.
        }
        self.agent_to_history_obs = defaultdict(list) # defaultdict return a instance of given factory when key is missing.
        self.agent_to_history_rew = defaultdict(list)
        self.agent_is_terminated = dict()
        logging.disable(logging.WARNING)
        super(RacingEnvWithOpponent, self).__init__(config)
        logging.disable(logging.NOTSET)

    @property
    def action_space(self) -> gym.Space:
        return super(RacingEnvWithOpponent, self).action_space[self.EGO_VEHICLE_NAME]

    @property
    def observation_space(self) -> gym.Space:
        return super(RacingEnvWithOpponent, self).observation_space[self.EGO_VEHICLE_NAME]

    def reset_opponent_policy(self, opponent_ckpt_path=None):
        self.policy_map["agent1"].load_checkpoint(opponent_ckpt_path)
        return None
    
    def reset(self, return_all_vehicles: bool = False, *args, **kwargs):
        logging.disable(logging.INFO)
        self.agent_to_history_obs.clear()
        self.agent_to_history_rew.clear()
        self.agent_is_terminated.clear()
        obs, info = super(RacingEnvWithOpponent, self).reset(*args, **kwargs)
        # obs: {"agent0": obs(np.ndarray), "agent1": obs(np.ndarray)}
        # Cache the observation of all vehicles.
        for agent_name, agent_obs in obs.items():
            self.agent_to_history_obs[agent_name].append(agent_obs)
            self.agent_is_terminated[agent_name] = False
            
        if return_all_vehicles:
            return obs, info
        
        obs_arr = []
        for i in range(self.num_agents):
            obs_arr.append(obs.get(f"agent{i}"))
        obs_arr = np.stack(obs_arr, axis=0)
        logging.disable(logging.NOTSET)
        return obs_arr, info[self.EGO_VEHICLE_NAME]

    def step(self, action, return_all_vehicles: bool = False):

        # Form an action dict.
        action_dict = {}
        for agent_name, agent_obs in self.agent_to_history_obs.items():
            if agent_name == self.EGO_VEHICLE_NAME:
                continue
            if self.agent_is_terminated[agent_name]:
                continue
            assert agent_name in self.policy_map.keys(), f"Can not find {agent_name} in policy map {self.policy_map.keys()}."

            # Note that agent_obs is a list containing all history observations of that agent.
            # This is useful when you want to use a recurrent neural network or transformer as the agent's model.
            # But for now we only feed the latest agent observation to the policy.
            # It's absolute OK to extend current codebase to accommodate the usage of recurrent networks.
            # Please notify me (Mark Peng pzh@cs.ucla.edu) if you want to use this.
            opponent_action = self.policy_map[agent_name](agent_obs[-1])
            if opponent_action.ndim == 2:  # Squeeze the batch dim
                opponent_action = np.squeeze(opponent_action, axis=0)
            action_dict[agent_name] = opponent_action
        action_dict[self.EGO_VEHICLE_NAME] = action

        # Forward the environment.
        obs, reward, terminated, truncated, info = super(RacingEnvWithOpponent, self).step(action_dict)

        # Cache data.
        for agent_name in terminated.keys():
            if agent_name == "__all__":
                continue
            done = terminated[agent_name] or truncated[agent_name]
            self.agent_is_terminated[agent_name] = done
            self.agent_to_history_obs[agent_name].append(obs[agent_name])
            self.agent_to_history_rew[agent_name].append(reward[agent_name])
        
        if return_all_vehicles:
            return obs, reward, terminated, truncated, info
        
        # modify obs and reward.
        obs_arr = []
        for i in range(self.num_agents):
            agent_name = f"agent{i}"
            obs_arr.append(obs.get(agent_name, self.agent_to_history_obs[agent_name][-1]))
        obs_arr = np.stack(obs_arr, axis=0)

        rew_adv = []
        for i in range(self.num_agents):
            agent_name = f"agent{i}"
            rew_adv.append(reward.get(agent_name, np.mean(self.agent_to_history_rew[agent_name])))
        rew_adv = rew_adv[0] - np.mean(rew_adv[1:])
        
        return (
            obs_arr,
            rew_adv,
            terminated[self.EGO_VEHICLE_NAME],
            truncated[self.EGO_VEHICLE_NAME],
            info[self.EGO_VEHICLE_NAME]
        )

    def reward_function(self, vehicle_id):
        """
        Reward function copied from metadrive.envs.marl_envs.mark_racing_env
        You can freely adjust the config or add terms.
        """
        # vehicle_id: "agent0" / "agent1"
        vehicle = self.vehicles[vehicle_id]
        is_main = (vehicle_id == self.EGO_VEHICLE_NAME)
        step_info = dict()

        # Reward for moving forward in current lane
        if vehicle.lane in vehicle.navigation.current_ref_lanes:
            current_lane = vehicle.lane
        else:
            current_lane = vehicle.navigation.current_ref_lanes[0]
            current_road = vehicle.navigation.current_road
        longitudinal_last, _ = current_lane.local_coordinates(vehicle.last_position)
        longitudinal_now, lateral_now = current_lane.local_coordinates(vehicle.position)

        self.movement_between_steps[vehicle_id].append(abs(longitudinal_now - longitudinal_last))

        reward = 0.0
        reward += self.config["driving_reward"] * (longitudinal_now - longitudinal_last)
        reward += self.config["speed_reward"] * (vehicle.speed_km_h / vehicle.max_speed_km_h)

        step_info["progress"] = (longitudinal_now - longitudinal_last)
        step_info["speed_km_h"] = vehicle.speed_km_h

        step_info["step_reward"] = reward
        step_info["crash_sidewalk"] = False
        if self._is_arrive_destination(vehicle):
            reward = +self.config["success_reward"]
        elif self._is_out_of_road(vehicle) and is_main:
            reward = -self.config["out_of_road_penalty"]
        elif vehicle.crash_vehicle:
            reward = -self.config["crash_vehicle_penalty"]
        elif vehicle.crash_sidewalk and is_main:
            reward = -self.config["crash_sidewalk_penalty"]
            step_info["crash_sidewalk"] = True
        elif self._is_idle(vehicle_id) and is_main:
            reward = -self.config["idle_penalty"]

        return reward, step_info

def race_env_factory():
    return RacingEnvWithOpponent(config=asdict(ENV_CONFIG))

@dataclass
class PlayerConfig:
    # Identifier
    agent_name: str = "DEFAULT"
    project_name: str = "MetaDrive-Game"
    experiemnt_name: str = "default"
    save_root: str = "./experiment"
    save_dir: str = None
    pretrained_model_path: str = None
    
    # Evaluation
    num_processes_eval: int = 20
    num_episodes_to_eval: int = 10
    num_rollouts_to_save: int = 30
    num_rollouts_to_payoff: int = 10

    def __post_init__(self):
        self.save_dir = verify_log_dir(self.save_root, self.project_name, self.experiemnt_name)
        if self.pretrained_model_path is None:
            main_pretrained_model_path = glob(os.path.join(self.save_dir, f"checkpoint-{self.agent_name}*.pkl"))
            if len(main_pretrained_model_path) > 0:
                main_pretrained_model_path.sort(key=lambda x: -os.path.getmtime(x))
                self.pretrained_model_path = main_pretrained_model_path[0]
                print(f"Automatically find ckpt {self.pretrained_model_path}. Will be re-used.")
        

class Player:
    
    def __init__(self, 
                 config: PlayerConfig,
                 trainer_config: PPOConfig,
    ) -> None:
        self.config = config
        for k, v in asdict(config).items():
            setattr(self, k, v)
        
        trainer_config.trainer_name = config.agent_name
        trainer_config.save_dir = config.save_dir
        trainer_config.pretrained_model_path = config.pretrained_model_path
        self.trainer_config = trainer_config
        wandb.login(key="")
        wandb.init(
            project=self.project_name,
            name=self.experiemnt_name,
            config={
                "player_config": asdict(config),
                "trainer_config": asdict(trainer_config),
                "env_config": asdict(ENV_CONFIG)
            },
            dir=self.save_dir
        )
        
        # Setup Trainer
        self.trainer = PPOTrainer(config=trainer_config, single_env_factory=race_env_factory)
        # Setup Recorder
        self.ckpt_to_rate = defaultdict(lambda :0.)
    
    @property
    def cnt_rollout(self):
        return self.trainer.iteration
    
    def _save_path_format(self, desc: str):
        time_stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        return "checkpoint-{}-{}-{}.pkl".format(self.agent_name, time_stamp, desc)
    
    @contextmanager
    def _save_self(self):
        tmp_dir = os.path.join(self.save_dir, ".tmp")
        os.makedirs(tmp_dir, exist_ok=True)
        self_ckpt_path = os.path.join(tmp_dir, self._save_path_format("self"))
        self.trainer.save_w(self_ckpt_path)
        try:
            yield self_ckpt_path
        finally:
            os.remove(self_ckpt_path)
    
    @staticmethod
    def _pfsp(rates, weighting="linear"):
        weightings = {
            "variance": lambda x: x * (1 - x),
            "linear": lambda x: 1 - x,
            "linear_capped": lambda x: np.minimum(0.5, 1 - x),
            "squared": lambda x: (1 - x)**2,
        }
        fn = weightings[weighting]
        probs = fn(np.asarray(rates))
        norm = probs.sum()
        if norm < 1e-10:
          return np.ones_like(rates) / len(rates)
        return probs / norm
    
    @staticmethod
    def _pfsp_branch(ckpt_to_rate: dict, weighting: str="squared"):
        opponent_ckpt = np.random.choice(
            list(ckpt_to_rate.keys()),
            p=Player._pfsp(list(ckpt_to_rate.values()), weighting)
        )
        return opponent_ckpt
    
    def save_w(self):
        """Save when num_rollouts_to_save or good performance."""
        print(f"===== Rollout[{self.cnt_rollout}] Save weights for {self.agent_name} agent =====")
        save_path = os.path.join(self.save_dir, self._save_path_format(f"rollout[{self.cnt_rollout}]"))
        self.trainer.save_w(save_path)
        return save_path
            
    def payoff(self):
        # 1. Search Opponent from save_dir.
        past_ckpt_paths = glob(os.path.join(self.save_dir, "*.pkl"))
        if len(past_ckpt_paths) == 0:
            past_ckpt_paths = [os.path.join(self.save_dir, self._save_path_format("init"))]
            self.trainer.save_w(past_ckpt_paths[0])
        # 2. Package Them into Evaluation
        print(f"===== Run Payoff for {self.agent_name} agent! =====")
        with self._save_self() as self_ckpt_path:
            policy_infos = []
            for ckpt_path in past_ckpt_paths:
                policy_info = {
                    "agent0": {
                        "name": "Main",
                        "checkpoint": self_ckpt_path,
                    },
                    "agent1": {
                        "name": "Opponent",
                        "checkpoint": ckpt_path,
                    },
                }
                policy_infos.append(policy_info)
            # Evaluation for self and each
            partial_eval = partial(eval, num_episodes=self.num_episodes_to_eval, render_mode=None, do_print=False)
            with mp.Pool(processes=self.num_processes_eval) as pool:
                results = pool.map(partial_eval, policy_infos)
        # 3. Collect Rate to Generate Summary
        ckpt_to_rate = {}
        for result in results:
            opponent_ckpt = result["agent1"]["checkpoint"]
            score_rate = result["agent0"]["score"] / (result["agent0"]["score"] + result["agent1"]["score"])
            win_rate = result["agent0"]["win_rate"]
            ckpt_to_rate[opponent_ckpt] = {
                "score_rate": score_rate,
                "win_rate": win_rate,
            }
        self.ckpt_to_rate.update(ckpt_to_rate)
        # 4. Do Save/Log 
        if min([v["score_rate"] for v in ckpt_to_rate.values()]) >= 0.65 and self.cnt_rollout % self.num_rollouts_to_save != 0:
            self.save_w()
        stats = {
            "win_rate": np.mean([v["win_rate"] for v in ckpt_to_rate.values()]),
            "score_rate": np.mean([v["score_rate"] for v in ckpt_to_rate.values()]),
            "exploiter_existance": np.mean([int(v["score_rate"]<0.3) for v in ckpt_to_rate.values()])
        }
        pretty_print({
            "===== {} Agent Steps {} Payoff Summary =====".format(
                self.agent_name, self.cnt_rollout): stats
        })
        pretty_print({
            "===== Rate Listed Below =====": ckpt_to_rate
        })
        return stats
    
    def choose_opponent(self):
        if not self.ckpt_to_rate:
            self.payoff()
        ckpt_to_score = {k: v["score_rate"] for k, v in self.ckpt_to_rate.items()}
        exploiter_ckpt_to_score = {k: v for k, v in ckpt_to_score.items() if v < 0.3} 
        if np.mean(list(ckpt_to_score.values())) < 0.3:
            opponent_ckpt_path = self._pfsp_branch(ckpt_to_score, "variance")
            print(f"{self.agent_name} Agent: PFSP. Similar level player. {opponent_ckpt_path}!")
        else:
            coin = np.random.random()
            if coin < 0.12:
                opponent_ckpt_path = "self"
                print(f"{self.agent_name} Agent: Self-play.")
            elif coin < 0.3 and len(exploiter_ckpt_to_score) > 0:
                opponent_ckpt_path = self._pfsp_branch(exploiter_ckpt_to_score, "squared")
                print(f"{self.agent_name} Agent: PFSP. Stronger players. {opponent_ckpt_path}!")
            else:
                opponent_ckpt_path = self._pfsp_branch(ckpt_to_score, "squared")
                print(f"{self.agent_name} Agent: PFSP. All players. {opponent_ckpt_path}!")
        return opponent_ckpt_path
            
    def train(self):
        # 1. Choose Opponent and Train
        print(f"===== Rollout[{self.cnt_rollout}] Train for {self.agent_name} Agent! =====")
        opponent_ckpt_path = self.choose_opponent()
        is_sp = opponent_ckpt_path == "self"
        if is_sp:
            with self._save_self() as self_ckpt_path:
                self.trainer.envs.env_method("reset_opponent_policy", opponent_ckpt_path=self_ckpt_path)
                stats = self.trainer.train()
        else:
            self.trainer.envs.env_method("reset_opponent_policy", opponent_ckpt_path=opponent_ckpt_path)
            stats = self.trainer.train()
        # 2. Log & Save & Solve Mem Leak
        pretty_print({
            "===== Rollout[{}] Train for {} Agent =====".format(
                self.cnt_rollout, self.agent_name): stats
        })
        if self.cnt_rollout % self.num_rollouts_to_save == 0:
            self.save_w()
            self.trainer.reinit_envs()
        return stats
        
    def play(self):
        pbar = tqdm(total=self.trainer.training_steps)
        while True:
            stats = {}
            eval_stats = {}
            if self.cnt_rollout % self.num_rollouts_to_payoff == 0:
                eval_stats = self.payoff()
            train_stats = self.train()
            stats = {
                "train": train_stats,
                "eval": eval_stats,
            }
            wandb.log(stats)
            if self.trainer.step > self.trainer.training_steps:
                break
            pbar.update(self.trainer.step - pbar.n)
        
def main():
    
    debug = False
    
    # Identifier
    agent_name: str = "Main"
    project_name: str = "MetaDrive-Game" if not debug else "MetaDrive-Debug"
    experiemnt_name: str = "final" if not debug else "debug"
    save_root: str = "./experiment"
    
    # Evaluation
    num_processes_eval: int = 20 if not debug else 2
    num_episodes_to_eval: int = 6 if not debug else 2
    num_rollouts_to_save: int = 40 if not debug else 2
    num_rollouts_to_payoff: int = 20 if not debug else 2
    
    # Env
    device: torch.device = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 10
    
    # Data
    num_agents: int = 2
    num_steps: int = 2000 if not debug else 100
    num_processes: int = 20 if not debug else 2
    num_epochs: int = 10 if not debug else 2
    mini_batch_size: int = 640 if not debug else 64
    
    # Train
    training_steps: int = 1_000_000
    epsilon: float = 1e-8
    hidden_size: int = 1024
    lr: float = 5e-5 
    lr_schedule: str = "fix" # "fix" or "annealing"
    gamma: float = 0.99
    gae_lambda: float = 0.95 
    clip_range: float = 0.2
    ent_coef: float = 0.
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    calculate_return: str = "upgo" # "gae" or "upgo"
    sample_dist: str = "normal"
    
    # Game
    use_opponent_obs: bool = True 
    use_representation_learning: bool = False
    state_embedding_size: int = 32 
    
    player_config = PlayerConfig(
        agent_name=agent_name,
        project_name=project_name,
        experiemnt_name=experiemnt_name,
        save_root=save_root,
        num_processes_eval=num_processes_eval,
        num_episodes_to_eval=num_episodes_to_eval,
        num_rollouts_to_save=num_rollouts_to_save,
        num_rollouts_to_payoff=num_rollouts_to_payoff,
    )
    trainer_config = PPOConfig(
        trainer_name=agent_name,
        device=device,
        seed=seed,
        num_agents=num_agents,
        num_steps=num_steps,
        num_processes=num_processes,
        num_epochs=num_epochs,
        mini_batch_size=mini_batch_size,
        training_steps=training_steps,
        epsilon=epsilon,
        hidden_size=hidden_size,
        lr=lr,
        lr_schedule=lr_schedule,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        calculate_return=calculate_return,
        sample_dist=sample_dist,
        use_opponent_obs=use_opponent_obs,
        use_representation_learning=use_representation_learning,
        state_embedding_size=state_embedding_size,
    )
    player = Player(player_config, trainer_config)
    player.play()
    
if __name__ == "__main__":
    while True:
        main()
        time.sleep(10)