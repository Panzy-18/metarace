from dataclasses import dataclass, asdict

import gymnasium as gym
import torch
from metadrive.envs.marl_envs.marl_racing_env import MultiAgentRacingEnv
import wandb
import pdb

from core.envs import make_envs
from core.ppo_trainer import PPOTrainer, PPOConfig
from core.utils import pretty_print, seed_everything

@dataclass
class EnvironmentConfig:
    num_agents: int = 1
    crash_sidewalk_penalty: int = 5
    idle_penalty: int = 5
    success_reward: int = 100
    speed_reward: int = 0.01


class SingleAgentRacingEnv(MultiAgentRacingEnv):
    """
    MetaDrive provides a MultiAgentRacingEnv class, where all the input/output data is dict. 
    This wrapper class let the environment "behaves like a single-agent RL environment" 
    by unwrapping the output dicts from the environment and
    wrapping the action to be a dict for feeding to the environment.
    """

    AGENT_NAME = "agent0"

    def __init__(self, config):
        assert config["num_agents"] == 1
        super(SingleAgentRacingEnv, self).__init__(config)

    @property
    def action_space(self) -> gym.Space:
        return super(SingleAgentRacingEnv, self).action_space[self.AGENT_NAME]

    @property
    def observation_space(self) -> gym.Space:
        return super(SingleAgentRacingEnv, self).observation_space[self.AGENT_NAME]

    def reset(self, *args, **kwargs):
        obs, info = super(SingleAgentRacingEnv, self).reset(*args, **kwargs)
        return obs[self.AGENT_NAME][None, :], info[self.AGENT_NAME]

    def step(self, action):
        o, r, tm, tc, i = super(SingleAgentRacingEnv, self).step({self.AGENT_NAME: action})
        return o[self.AGENT_NAME][None, :], r[self.AGENT_NAME], tm[self.AGENT_NAME], tc[self.AGENT_NAME], i[self.AGENT_NAME]

    def reward_function(self, vehicle_id):
        """
        Reward function copied from metadrive.envs.marl_envs.mark_racing_env
        You can freely adjust the config or add terms.
        """
        vehicle = self.vehicles[vehicle_id]
        step_info = dict()

        # Reward for moving forward in current lane
        if vehicle.lane in vehicle.navigation.current_ref_lanes:
            current_lane = vehicle.lane
        else:
            current_lane = vehicle.navigation.current_ref_lanes[0]
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
        elif self._is_out_of_road(vehicle):
            reward = -self.config["out_of_road_penalty"]
        elif vehicle.crash_vehicle:
            reward = -self.config["crash_vehicle_penalty"]
        elif vehicle.crash_sidewalk:
            reward = -self.config["crash_sidewalk_penalty"]
            step_info["crash_sidewalk"] = True
        elif self._is_idle(vehicle_id):
            reward = -self.config["idle_penalty"]

        return reward, step_info


if __name__ == "__main__":
    
    config: PPOConfig = PPOConfig.parse_args()
    env_config = EnvironmentConfig()
    seed = config.seed
    
    # prepare for experiment
    seed_everything(seed)
    torch.set_num_threads(1)
    wandb.login(key="")
    run = wandb.init(
        project=config.project,
        name=config.name,
        config=asdict(config).update(asdict(env_config)),
        dir=config.save_dir
    )
    
    def _make_envs():
        def single_env_factory():
            return SingleAgentRacingEnv(asdict(env_config))
        envs = make_envs(
            single_env_factory=single_env_factory,
            num_envs=config.num_processes,
            asynchronous=True,
        )
        return envs
    
    envs = _make_envs()
    trainer = PPOTrainer(config, envs)
    while True:
        stats = trainer.train()
        wandb.log(stats)
        pretty_print({
            "===== Training Steps {} Train Stats =====".format(
            trainer.step): stats
        })
        if trainer.step > trainer.training_steps:
            break
        if trainer.iteration % 40 == 0:
            trainer.del_envs()
            envs = _make_envs()
            trainer.set_envs(envs)
        