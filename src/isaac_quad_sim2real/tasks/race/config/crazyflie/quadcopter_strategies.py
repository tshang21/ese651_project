# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Modular strategy classes for quadcopter environment rewards, observations, and resets."""

from __future__ import annotations
from this import d

import torch
import numpy as np
from typing import TYPE_CHECKING, Dict, Optional, Tuple

from isaaclab.utils.math import subtract_frame_transforms, quat_from_euler_xyz, euler_xyz_from_quat, wrap_to_pi, matrix_from_quat

if TYPE_CHECKING:
    from .quadcopter_env import QuadcopterEnv

D2R = np.pi / 180.0
R2D = 180.0 / np.pi


class DefaultQuadcopterStrategy:
    """Default strategy implementation for quadcopter environment."""

    def __init__(self, env: QuadcopterEnv):
        """Initialize the default strategy.

        Args:
            env: The quadcopter environment instance.
        """
        self.env = env
        self.device = env.device
        self.num_envs = env.num_envs
        self.cfg = env.cfg
        self._max_unlocked_gate = 0

        # Initialize episode sums for logging if in training mode
        if self.cfg.is_train and hasattr(env, 'rew'):
            keys = [key.split("_reward_scale")[0] for key in env.rew.keys() if key != "death_cost"]
            self._episode_sums = {
                key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
                for key in keys
            }

        # Initialize fixed parameters once (no domain randomization)
        # These parameters remain constant throughout the simulation
        # Aerodynamic drag coefficients
        self.env._K_aero[:, :2] = self.env._k_aero_xy_value
        self.env._K_aero[:, 2] = self.env._k_aero_z_value

        # PID controller gains for angular rate control
        # Roll and pitch use the same gains
        self.env._kp_omega[:, :2] = self.env._kp_omega_rp_value
        self.env._ki_omega[:, :2] = self.env._ki_omega_rp_value
        self.env._kd_omega[:, :2] = self.env._kd_omega_rp_value

        # Yaw has different gains
        self.env._kp_omega[:, 2] = self.env._kp_omega_y_value
        self.env._ki_omega[:, 2] = self.env._ki_omega_y_value
        self.env._kd_omega[:, 2] = self.env._kd_omega_y_value

        # Motor time constants (same for all 4 motors)
        self.env._tau_m[:] = self.env._tau_m_value

        # Thrust to weight ratio
        self.env._thrust_to_weight[:] = self.env._twr_value

    def get_rewards(self) -> torch.Tensor:
        """get_rewards() is called per timestep. This is where you define your reward structure and compute them
        according to the reward scales you tune in train_race.py. The following is an example reward structure that
        causes the drone to hover near the zeroth gate. It will not produce a racing policy, but simply serves as proof
        if your PPO implementation works. You should delete it or heavily modify it once you begin the racing task."""

        # TODO ----- START ----- Define the tensors required for your custom reward structure
        # Compute waypoint transitions
        x_curr = self.env._pose_drone_wrt_gate[:, 0]
        x_prev = self.env._prev_x_drone_wrt_gate

        gate_passed = (x_prev > 0) & (x_curr <= 0)
        ids_gate_passed = torch.where(gate_passed)[0]

        self.env._idx_wp[ids_gate_passed] = (self.env._idx_wp[ids_gate_passed] + 1) % self.env._waypoints.shape[0]
        self.env._n_gates_passed[ids_gate_passed] += 1

        # Set next gate target
        self.env._desired_pos_w[ids_gate_passed] = self.env._waypoints[self.env._idx_wp[ids_gate_passed], :3]

        # Distance shaping
        distance_to_gate = torch.linalg.norm(
            self.env._desired_pos_w - self.env._robot.data.root_link_pos_w, dim=1
        )
        distance_to_gate = torch.tanh(distance_to_gate / 3.0)
        distance_to_gate_prev = self.env._last_distance_to_goal

        proximity = torch.clamp(1.0 - distance_to_gate, min=0.0)

        # Corridor and lateral positioning
        lateral_offset = torch.linalg.norm(self.env._pose_drone_wrt_gate[:, 1:], dim=1)
        gate_corridor = torch.clamp(1.0 - lateral_offset, min=0.0)

        inside_opening_reward = torch.clamp(0.5 - lateral_offset, min=0.0)
        in_gate_opening = lateral_offset < 0.5

        # Motion and forward progress
        lin_vel_b = self.env._robot.data.root_com_lin_vel_b
        forward_speed = torch.clamp(lin_vel_b[:, 0], min=0.0)

        lin_vel_w = self.env._robot.data.root_com_lin_vel_w
        vec = self.env._desired_pos_w - self.env._robot.data.root_link_pos_w
        gate_dir = vec / (torch.norm(vec, dim=1, keepdim=True) + 1e-6)
        forward_progress = torch.sum(gate_dir * lin_vel_w, dim=1)

        hover_penalty = (forward_progress < 0.2).float()

        # Crash detection
        contact_forces = self.env._contact_sensor.data.net_forces_w
        crashed = (torch.norm(contact_forces, dim=-1) > 1e-8).squeeze(1).int()

        mask = (self.env.episode_length_buf > 100).int()
        self.env._crashed = self.env._crashed + crashed * mask


        # Heading alignment
        drone_quat_w = self.env._robot.data.root_quat_w
        R = matrix_from_quat(drone_quat_w)
        drone_forward = R[:, :, 0]
        heading_alignment = torch.sum(gate_dir * drone_forward, dim=1)

        # Next gate anticipation
        num_gates = self.env._waypoints.shape[0]
        next_gate_idx = (self.env._idx_wp + 1) % num_gates
        next_gate_pos = self.env._waypoints[next_gate_idx, :3]

        next_gate_dir = (next_gate_pos - self.env._robot.data.root_link_pos_w)
        next_gate_dir = next_gate_dir / (torch.norm(next_gate_dir, dim=1, keepdim=True) + 1e-6)

        anticipation = torch.sum(drone_forward * next_gate_dir, dim=1)

        approach_alignment = heading_alignment * torch.clamp(distance_to_gate_prev - 0.1, min=0.0)

        # Gate pass validation
        valid_gate_pass = gate_passed & (forward_speed > 0.5) & in_gate_opening

        # Update buffers
        self.env._prev_x_drone_wrt_gate = x_curr
        self.env._last_distance_to_goal = distance_to_gate
        # TODO ----- END -----

        if self.cfg.is_train:
            # TODO ----- START ----- Compute per-timestep rewards by multiplying with your reward scales (in train_race.py)
            rewards = {
                # dense shaping reward for being close to the current gate
                "proximity_goal": proximity * self.env.rew['proximity_goal_reward_scale'],
                # bonus for successfully passing through a gate
                "gate_pass": valid_gate_pass.float() * self.env.rew['gate_pass_reward_scale'],
                # penalty for deviating from the gate centerline
                "lateral_deviation": -lateral_offset * self.env.rew['lateral_deviation_reward_scale'],
                # penalty for hovering near the gate
                "hover_penalty": -hover_penalty * self.env.rew['hover_penalty_reward_scale'],
                # reward for moving forward
                "forward_speed": forward_speed * self.env.rew['forward_speed_reward_scale'],
                # small per-step time penalty to discourage standing still
                "time": -torch.ones(self.num_envs, device=self.device) * self.env.rew['time_reward_scale'],
                # penalty for crashing
                "crash": crashed * self.env.rew['crash_reward_scale'],
                # reward for heading towards the gate
                "heading_alignment": heading_alignment * self.env.rew['heading_alignment_reward_scale'],
                # reward for forward progress
                "forward_progress": forward_progress * self.env.rew['forward_progress_reward_scale'],
                # reward for staying in the gate corridor
                "gate_corridor": gate_corridor * self.env.rew['gate_corridor_reward_scale'],
                # reward for looking ahead and anticipating the next gate
                "anticipation": anticipation * self.env.rew['anticipation_reward_scale'],
                # reward for aligning with the approach direction
                "approach_alignment": approach_alignment * self.env.rew['approach_alignment_reward_scale'],
                # reward for staying inside the gate opening
                "inside_opening": inside_opening_reward * self.env.rew['inside_opening_reward_scale'],
            }
            reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
            reward = torch.where(self.env.reset_terminated,
                                torch.ones_like(reward) * self.env.rew['death_cost'], reward)

            # Logging
            for key, value in rewards.items():
                self._episode_sums[key] += value
        else:   # This else condition implies eval is called with play_race.py. Can be useful to debug at test-time
            reward = torch.zeros(self.num_envs, device=self.device)
            # TODO ----- END -----

        return reward

    def get_observations(self) -> Dict[str, torch.Tensor]:
        """Get observations. Read reset_idx() and quadcopter_env.py to see which drone info is extracted from the sim.
        The following code is an example. You should delete it or heavily modify it once you begin the racing task."""

        # TODO ----- START ----- Define tensors for your observation space. Be careful with frame transformations
        #### Basic drone states, modify for your needs)
        drone_pose_w = self.env._robot.data.root_link_pos_w
        drone_lin_vel_b = self.env._robot.data.root_com_lin_vel_b
        drone_quat_w = self.env._robot.data.root_quat_w

        ##### Some example observations you may want to explore using
        # Angular velocities (referred to as body rates)
        # drone_ang_vel_b = self.env._robot.data.root_ang_vel_b  # [roll_rate, pitch_rate, yaw_rate]

        # Current target gate information
        # current_gate_idx = self.env._idx_wp
        # current_gate_pos_w = self.env._waypoints[current_gate_idx, :3]  # World position of current gate
        # current_gate_yaw = self.env._waypoints[current_gate_idx, -1]    # Yaw orientation of current gate

        # Relative position to current gate in gate frame
        drone_pos_gate_frame = self.env._pose_drone_wrt_gate

        # Relative position to current gate in body frame
        # gate_pos_b, _ = subtract_frame_transforms(
        #     self.env._robot.data.root_link_pos_w,
        #     self.env._robot.data.root_quat_w,
        #     current_gate_pos_w
        # )

        # Previous actions
        # prev_actions = self.env._previous_actions  # Shape: (num_envs, 4)

        # Number of gates passed
        # gates_passed = self.env._n_gates_passed.unsqueeze(1).float()

        # TODO ----- END -----

        obs = torch.cat(
            # TODO ----- START ----- List your observation tensors here to be concatenated together
            [
                drone_pose_w,       # position in the world frame (3 dims)
                drone_lin_vel_b,    # velocity in the body frame (3 dims)
                drone_quat_w,       # quaternion in the world frame (4 dims)
                drone_pos_gate_frame
            ],
            # TODO ----- END -----
            dim=-1,
        )
        observations = {"policy": obs}

        return observations

    def reset_idx(self, env_ids: Optional[torch.Tensor]):
        """Reset specific environments to initial states."""
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.env._robot._ALL_INDICES

        # Logging for training mode
        if self.cfg.is_train and hasattr(self, '_episode_sums'):
            extras = dict()
            for key in self._episode_sums.keys():
                episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
                extras["Episode_Reward/" + key] = episodic_sum_avg / self.env.max_episode_length_s
                self._episode_sums[key][env_ids] = 0.0
            self.env.extras["log"] = dict()
            self.env.extras["log"].update(extras)

            extras = dict()
            extras["Episode_Termination/died"] = torch.count_nonzero(self.env.reset_terminated[env_ids]).item()
            extras["Episode_Termination/time_out"] = torch.count_nonzero(self.env.reset_time_outs[env_ids]).item()
            self.env.extras["log"].update(extras)

        # Call robot reset first
        self.env._robot.reset(env_ids)

        # Initialize model paths if needed
        if not self.env._models_paths_initialized:
            num_models_per_env = self.env._waypoints.size(0)
            model_prim_names_in_env = [f"{self.env.target_models_prim_base_name}_{i}" for i in range(num_models_per_env)]

            self.env._all_target_models_paths = []
            for env_path in self.env.scene.env_prim_paths:
                paths_for_this_env = [f"{env_path}/{name}" for name in model_prim_names_in_env]
                self.env._all_target_models_paths.append(paths_for_this_env)

            self.env._models_paths_initialized = True

        n_reset = len(env_ids)
        if n_reset == self.num_envs and self.num_envs > 1:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf,
                high=int(self.env.max_episode_length)
            )
            # use all envs to decide when to unlock the next gate
            global_pass_fraction = (self.env._n_gates_passed >= 1).float().mean()
            if global_pass_fraction > 0.6:
                self._max_unlocked_gate = min(
                    self._max_unlocked_gate + 1,
                    self.env._waypoints.shape[0] - 1
                )



        # Reset action buffers
        self.env._actions[env_ids] = 0.0
        self.env._previous_actions[env_ids] = 0.0
        self.env._previous_yaw[env_ids] = 0.0
        self.env._motor_speeds[env_ids] = 0.0
        self.env._previous_omega_meas[env_ids] = 0.0
        self.env._previous_omega_err[env_ids] = 0.0
        self.env._omega_err_integral[env_ids] = 0.0

        # Reset joints state
        joint_pos = self.env._robot.data.default_joint_pos[env_ids]
        joint_vel = self.env._robot.data.default_joint_vel[env_ids]
        self.env._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        default_root_state = self.env._robot.data.default_root_state[env_ids]

        # TODO ----- START ----- Define the initial state during training after resetting an environment.
        # For now, always initialize the drone a fixed distance behind gate 0 during training.
        # This focuses learning on reliably reaching and passing the first gate before tackling
        # the full track. In play mode, we also start from gate 0 here; play-specific initial
        # gate handling happens in the block below.

        # point drone towards the chosen starting gate
        waypoint_indices = torch.randint(
            low=0,
            high=self._max_unlocked_gate + 1,   # inclusive upper bound
            size=(n_reset,),
            device=self.device,
            dtype=self.env._idx_wp.dtype,
        )


        # get starting poses behind waypoints
        x0_wp = self.env._waypoints[waypoint_indices][:, 0]
        y0_wp = self.env._waypoints[waypoint_indices][:, 1]
        theta = self.env._waypoints[waypoint_indices][:, -1]
        z_wp = self.env._waypoints[waypoint_indices][:, 2]

        x_local = -2.0 * torch.ones(n_reset, device=self.device)
        y_local = torch.zeros(n_reset, device=self.device)
        z_local = torch.zeros(n_reset, device=self.device)

        # rotate local pos to global frame
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        x_rot = cos_theta * x_local - sin_theta * y_local
        y_rot = sin_theta * x_local + cos_theta * y_local
        initial_x = x0_wp - x_rot
        initial_y = y0_wp - y_rot
        initial_z = z_local + z_wp

        default_root_state[:, 0] = initial_x
        default_root_state[:, 1] = initial_y
        default_root_state[:, 2] = initial_z

        # point drone towards the zeroth gate
        initial_yaw = torch.atan2(y0_wp - initial_y, x0_wp - initial_x)
        yaw_noise = torch.empty(n_reset, device=self.device).uniform_(-0.15, 0.15)
        yaw = initial_yaw + yaw_noise
        quat = quat_from_euler_xyz(
            torch.zeros(n_reset, device=self.device),
            torch.zeros(n_reset, device=self.device),
            yaw
        )
        default_root_state[:, 3:7] = quat

        # TODO ----- END -----

        # Handle play mode initial position
        if not self.cfg.is_train:
            # x_local and y_local are randomly sampled
            x_local = torch.empty(1, device=self.device).uniform_(-3.0, -0.5)
            y_local = torch.empty(1, device=self.device).uniform_(-1.0, 1.0)

            x0_wp = self.env._waypoints[self.env._initial_wp, 0]
            y0_wp = self.env._waypoints[self.env._initial_wp, 1]
            theta = self.env._waypoints[self.env._initial_wp, -1]

            # rotate local pos to global frame
            cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)
            x_rot = cos_theta * x_local - sin_theta * y_local
            y_rot = sin_theta * x_local + cos_theta * y_local
            x0 = x0_wp - x_rot
            y0 = y0_wp - y_rot
            z0 = 0.05

            # point drone towards the zeroth gate
            yaw0 = torch.atan2(y0_wp - y0, x0_wp - x0)

            default_root_state = self.env._robot.data.default_root_state[0].unsqueeze(0)
            default_root_state[:, 0] = x0
            default_root_state[:, 1] = y0
            default_root_state[:, 2] = z0

            quat = quat_from_euler_xyz(
                torch.zeros(1, device=self.device),
                torch.zeros(1, device=self.device),
                yaw0
            )
            default_root_state[:, 3:7] = quat
            waypoint_indices = self.env._initial_wp

        # Set waypoint indices and desired positions
        self.env._idx_wp[env_ids] = waypoint_indices

        self.env._desired_pos_w[env_ids, :2] = self.env._waypoints[waypoint_indices, :2].clone()
        self.env._desired_pos_w[env_ids, 2] = self.env._waypoints[waypoint_indices, 2].clone()

        self.env._last_distance_to_goal[env_ids] = torch.linalg.norm(
            self.env._desired_pos_w[env_ids, :2] - self.env._robot.data.root_link_pos_w[env_ids, :2], dim=1
        )
        
        self.env._n_gates_passed[env_ids] = 0

        # Write state to simulation
        self.env._robot.write_root_link_pose_to_sim(default_root_state[:, :7], env_ids)
        self.env._robot.write_root_com_velocity_to_sim(default_root_state[:, 7:], env_ids)

        # Reset variables
        self.env._yaw_n_laps[env_ids] = 0

        self.env._pose_drone_wrt_gate[env_ids], _ = subtract_frame_transforms(
            self.env._waypoints[self.env._idx_wp[env_ids], :3],
            self.env._waypoints_quat[self.env._idx_wp[env_ids], :],
            self.env._robot.data.root_link_state_w[env_ids, :3]
        )

        self.env._prev_x_drone_wrt_gate = self.env._pose_drone_wrt_gate[:, 0].clone()

        self.env._crashed[env_ids] = 0