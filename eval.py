from collections import defaultdict
import datetime
import os
import json
import pdb
import logging

import numpy as np
import torch
from tqdm import tqdm
from metadrive.envs.marl_envs.marl_racing_env import MultiAgentRacingEnv
import pygame
import mediapy

from core.utils import pretty_print, seed_everything
from core.network import Policy

class MultiAgentRacingEnvWithSimplifiedReward(MultiAgentRacingEnv):
    """We do not wrap the environment as a single-agent env here."""

    def reward_function(self, vehicle_id):
        """Only the longitudinal movement is in the reward."""
        vehicle = self.vehicles[vehicle_id]
        step_info = dict()
        if vehicle.lane in vehicle.navigation.current_ref_lanes:
            current_lane = vehicle.lane
        else:
            current_lane = vehicle.navigation.current_ref_lanes[0]
        longitudinal_last, _ = current_lane.local_coordinates(vehicle.last_position)
        longitudinal_now, lateral_now = current_lane.local_coordinates(vehicle.position)
        self.movement_between_steps[vehicle_id].append(abs(longitudinal_now - longitudinal_last))
        reward = longitudinal_now - longitudinal_last
        step_info["progress"] = (longitudinal_now - longitudinal_last)
        step_info["speed_km_h"] = vehicle.speed_km_h
        step_info["step_reward"] = reward
        step_info["crash_sidewalk"] = False
        if vehicle.crash_sidewalk:
            step_info["crash_sidewalk"] = True
        return reward, step_info
    
def eval(policy_info: dict,
         num_episodes: int = 10,
         render_mode: str = None,
         do_print: bool = False,
         save_dir: str = None,
         seed: int = 1
) -> dict:
    seed_everything(seed)
    
    policy_map = {}
    for agent_id in list(policy_info.keys()):
        policy_map[agent_id] = Policy(
            policy_info[agent_id]["checkpoint"],
        )
    
    assert render_mode in [None, "window", "video"]
    
    logging.disable(logging.WARNING)
    env = MultiAgentRacingEnvWithSimplifiedReward({
        "num_agents": 2,
        "use_render": True if render_mode is not None else False,
        "target_vehicle_configs": {
            "agent0": {
                "use_special_color": True
            }
        }
    })
    
    video_bev = []
    cnt_episodes = cnt_steps = 0
    result_recorder = defaultdict(lambda: defaultdict(list))
    try:
        obs_dict, _ = env.reset()
        logging.disable(logging.NOTSET)
        terminated_dict = {}
        while True:
            # ===== COLLECT ACTION =====
            action_dict = {}
            for agent_id, agent_obs in obs_dict.items():
                if agent_id in terminated_dict and terminated_dict[agent_id]:
                    continue
                act = policy_map[agent_id](agent_obs)
                action_dict[agent_id] = act
            # ===== STEP ENVIRONMENT =====
            obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict = env.step(action_dict)
            cnt_steps += 1
            # ===== VISUALIZATION =====
            if render_mode == "window":
                env.render(mode="topdown")
            elif render_mode == "video":
                img_bev = env.render(
                    mode="topdown",
                    # show_agent_name=True,
                    # target_vehicle_heading_up=True,
                    draw_target_vehicle_trajectory=False,
                    film_size=(5000, 5000),
                    screen_size=(1000, 1000),
                    crash_vehicle_done=False,
                )
                img_bev = pygame.surfarray.make_surface(img_bev)
                img_bev = pygame.surfarray.array3d(img_bev)
                img_bev = img_bev.swapaxes(0, 1)
                video_bev.append(img_bev)
            # ===== COLLECT INFO =====
            for agent_id, agent_done in terminated_dict.items():
                if agent_id == "__all__":
                    continue
                agent_info = info_dict[agent_id]
                if "crash_vehicle" in agent_info:
                    result_recorder[agent_id]["crash_vehicle_rate"].append(agent_info["crash_vehicle"])
                if "crash_sidewalk" in agent_info:
                    result_recorder[agent_id]["crash_sidewalk_rate"].append(agent_info["crash_sidewalk"])
                if "idle" in agent_info:
                    result_recorder[agent_id]["idle_rate"].append(agent_info["idle"])
                if "speed_km_h" in agent_info:
                    result_recorder[agent_id]["speed_km_h"].append(agent_info["speed_km_h"])
                if agent_done:  # the episode is done
                    # Record the reward of the terminated episode to
                    result_recorder[agent_id]["episode_reward"].append(agent_info["episode_reward"])
                    if "arrive_dest" in agent_info:
                        result_recorder[agent_id]["success_rate"].append(agent_info["arrive_dest"])
                    if "max_step" in agent_info:
                        result_recorder[agent_id]["max_step_rate"].append(agent_info["max_step"])
                    if "episode_length" in agent_info:
                        result_recorder[agent_id]["episode_length"].append(agent_info["episode_length"])
            # ===== CHECK TERMINATE =====
            if terminated_dict["__all__"]:
                cnt_episodes += 1
                obs_dict, _ = env.reset()
                terminated_dict = {}
                for policy in policy_map.values():
                    policy.reset()
                if render_mode and save_dir is not None:
                    video_path = os.path.join(save_dir, f"bev_video_ep[{cnt_episodes}].mp4")
                    mediapy.write_video(video_path, video_bev, fps=60)
                    video_bev = []
                if cnt_episodes >= num_episodes:
                    break
    finally:
        env.close()
    
    stat = {}
    for agent_id in result_recorder.keys():
        agent_result_recorder = result_recorder[agent_id]
        stat[agent_id] = {k: np.mean(v) for k, v in agent_result_recorder.items()}
    if do_print:
        pretty_print(stat)
    
    win_stat = {agent_id: 0 for agent_id in result_recorder.keys()}
    clear_win_stat = {agent_id: 0 for agent_id in result_recorder.keys()}
    score_stat = {agent_id: 0 for agent_id in result_recorder.keys()}
    assert len(win_stat) == 2, "Only support 2 agents for now."
    
    for episode_count in range(num_episodes):
        a0_succ = result_recorder["agent0"]["success_rate"][episode_count]
        a1_succ = result_recorder["agent1"]["success_rate"][episode_count]

        # Only one agent arrives destination:
        if np.logical_xor(a0_succ, a1_succ).item():
            if a0_succ:
                win_stat["agent0"] += 1
                clear_win_stat["agent0"] += 1
                score_stat["agent0"] += 2
            else:
                win_stat["agent1"] += 1
                clear_win_stat["agent1"] += 1
                score_stat["agent1"] += 2

        # If both arrive destination: faster agent winds.
        if a0_succ and a1_succ:
            a0_len = result_recorder["agent0"]["episode_length"][episode_count]
            a1_len = result_recorder["agent1"]["episode_length"][episode_count]
            if a0_len < a1_len:
                win_stat["agent0"] += 1
                score_stat["agent0"] += 1
            elif a0_len > a1_len:
                win_stat["agent1"] += 1
                score_stat["agent1"] += 1
            else:  # If both arrive destination at the same time: higher speed agent wins.
                # I don't believe this case will happen, but just write the code to cover this.
                if result_recorder["agent0"]["speed_km_h"][episode_count] > \
                        result_recorder["agent1"]["speed_km_h"][episode_count]:
                    win_stat["agent0"] += 1
                    score_stat["agent0"] += 1
                else:
                    win_stat["agent1"] += 1
                    score_stat["agent1"] += 1

        # If both fails to arrive destination: higher longitude movement agent wins.
        if not (a0_succ or a1_succ):
            a0_rew = result_recorder["agent0"]["episode_reward"][episode_count]
            a1_rew = result_recorder["agent1"]["episode_reward"][episode_count]
            if a0_rew > a1_rew:
                win_stat["agent0"] += 1
                score_stat["agent0"] += 1
            else:
                win_stat["agent1"] += 1
                score_stat["agent1"] += 1
    
    summary = {}
    for agent_id in win_stat.keys():
        n_wins = win_stat[agent_id]
        summary[agent_id] = {
            "n_wins": n_wins,
            "n_clear_wins": clear_win_stat[agent_id],
            "win_rate": n_wins / num_episodes,
            "score": score_stat[agent_id],
            "checkpoint": policy_info[agent_id]["checkpoint"],
        }
    if do_print:
        pretty_print(summary)
    
    env.close()
    del env
    return summary

    
if __name__ == "__main__":
    policy_info = {
        "agent0": {
            "name": "PandaMain",
            "checkpoint": "./experiment/MetaDrive-Game/final/checkpoint-Main-2023-12-09_20-25-01-rollout[30].pkl"
        },
        "agent1": {
            "name": "PandaOpponent",
            "checkpoint": "./experiment/MetaDrive-Game/final/checkpoint-Main-2023-12-09_20-25-01-rollout[30].pkl"
        },
    }
    eval(
        policy_info=policy_info,
        render_mode="window",
        num_episodes=1,
        seed=0,
    )