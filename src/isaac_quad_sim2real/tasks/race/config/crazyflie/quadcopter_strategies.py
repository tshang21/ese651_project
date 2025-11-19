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
from scipy.spatial.transform import Rotation

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

        # Domain randomization ranges
        # TWR
        self.env._twr_min = self.cfg.thrust_to_weight * 0.95
        self.env._twr_max = self.cfg.thrust_to_weight * 1.05
        # Aerodynamics
        self.env._k_aero_xy_min = self.cfg.k_aero_xy * 0.5
        self.env._k_aero_xy_max = self.cfg.k_aero_xy * 2.0
        self.env._k_aero_z_min = self.cfg.k_aero_z * 0.5
        self.env._k_aero_z_max = self.cfg.k_aero_z * 2.0
        # PID gains
        self.env._kp_omega_rp_min = self.cfg.kp_omega_rp * 0.85
        self.env._kp_omega_rp_max = self.cfg.kp_omega_rp * 1.15
        self.env._ki_omega_rp_min = self.cfg.ki_omega_rp * 0.85
        self.env._ki_omega_rp_max = self.cfg.ki_omega_rp * 1.15
        self.env._kd_omega_rp_min = self.cfg.kd_omega_rp * 0.7
        self.env._kd_omega_rp_max = self.cfg.kd_omega_rp * 1.3
        self.env._kp_omega_y_min = self.cfg.kp_omega_y * 0.85
        self.env._kp_omega_y_max = self.cfg.kp_omega_y * 1.15
        self.env._ki_omega_y_min = self.cfg.ki_omega_y * 0.85
        self.env._ki_omega_y_max = self.cfg.ki_omega_y * 1.15
        self.env._kd_omega_y_min = self.cfg.kd_omega_y * 0.7
        self.env._kd_omega_y_max = self.cfg.kd_omega_y * 1.3

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

        # -------------------------------- gate_passed --------------------------------
        x_curr = self.env._pose_drone_wrt_gate[:, 0]
        x_prev = self.env._prev_x_drone_wrt_gate
        # Check if drone crossed the gate plane
        crossed_plane = (x_prev > 0) & (x_curr <= 0)
        
        # Get gate geometry information
        gate_half_size = self.env._gate_model_cfg_data.gate_side / 2.0
        # Check if y and z positions are within gate boundaries
        y_curr = self.env._pose_drone_wrt_gate[:, 1]
        z_curr = self.env._pose_drone_wrt_gate[:, 2]
        within_y_bounds = torch.abs(y_curr) < gate_half_size
        within_z_bounds = torch.abs(z_curr) < gate_half_size
        within_gate_opening = within_y_bounds & within_z_bounds
        
        # Gate is only considered "passed" if drone crossed plane AND was within opening
        gate_passed = crossed_plane & within_gate_opening
        ids_gate_passed = torch.where(gate_passed)[0]
        self.env._n_gates_passed[ids_gate_passed] += 1

        # -------------------------------- velocity_alignment --------------------------------
        # Encourage forward motion toward the next gate (in world frame)
        approaching_gate = (x_prev > 0) & within_gate_opening
        vel_w = self.env._robot.data.root_com_lin_vel_w
        vec = self.env._desired_pos_w - self.env._robot.data.root_link_pos_w
        gate_dir = vec / (torch.norm(vec, dim=1, keepdim=True) + 1e-6)

        # Component of velocity along gate direction (positive when moving toward gate)
        vel_along = torch.sum(vel_w * gate_dir, dim=1)
        vel_along_pos = torch.clamp(vel_along, min=0.0)

        # Bound the shaping term to avoid it dominating other rewards
        velocity_alignment = torch.clamp(vel_along_pos, max=2.0)
        velocity_alignment = torch.where(approaching_gate, velocity_alignment, torch.zeros_like(velocity_alignment))

        # -------------------------------- progress --------------------------------
        self.env._idx_wp[ids_gate_passed] = (self.env._idx_wp[ids_gate_passed] + 1) % self.env._waypoints.shape[0]
        if ids_gate_passed.numel() > 0:
            # recalc pose wrt new current gate
            self.env._pose_drone_wrt_gate[ids_gate_passed], _ = subtract_frame_transforms(
                self.env._waypoints[self.env._idx_wp[ids_gate_passed], :3],
                self.env._waypoints_quat[self.env._idx_wp[ids_gate_passed], :],
                self.env._robot.data.root_link_state_w[ids_gate_passed, :3], self.env._robot.data.root_quat_w[ids_gate_passed, :]
            )

            # reset prev x for the new gate plane
            self.env._prev_x_drone_wrt_gate[ids_gate_passed] = \
                self.env._pose_drone_wrt_gate[ids_gate_passed, 0].clone()

        # set desired positions in the world frame
        self.env._desired_pos_w[ids_gate_passed, :2] = self.env._waypoints[self.env._idx_wp[ids_gate_passed], :2]
        self.env._desired_pos_w[ids_gate_passed, 2] = self.env._waypoints[self.env._idx_wp[ids_gate_passed], 2]

        # calculate progress via change in distance to goal (forward progress only)
        distance_to_goal = torch.linalg.norm(
            self.env._desired_pos_w - self.env._robot.data.root_link_pos_w, dim=1
        )
        prev_distance = self.env._last_distance_to_goal
        delta_distance = prev_distance - distance_to_goal  # >0 when moving toward goal
        progress = torch.clamp(delta_distance, -1.0, 1.0)
        # update stored distance for next step (no gradient needed)
        self.env._last_distance_to_goal = distance_to_goal.detach()
        # -------------------------------- crash detection --------------------------------
        # Crash detection
        contact_forces = self.env._contact_sensor.data.net_forces_w
        crashed = (torch.norm(contact_forces, dim=-1) > 1e-8).squeeze(1).int()

        mask = (self.env.episode_length_buf > 3).int()
        self.env._crashed = self.env._crashed + crashed * mask
        
        # Update x_prev
        self.env._prev_x_drone_wrt_gate = self.env._pose_drone_wrt_gate[:, 0].clone()

        # TODO ----- END -----

        if self.cfg.is_train:
            # TODO ----- START ----- Compute per-timestep rewards by multiplying with your reward scales (in train_race.py)
            rewards = {
                "progress": progress * self.env.rew['progress_reward_scale'],
                "gate_passed": gate_passed.float() * self.env.rew['gate_passed_reward_scale'],
                "crash": -crashed.float() * self.env.rew['crash_reward_scale'],
                "velocity_alignment": velocity_alignment * self.env.rew['velocity_alignment_reward_scale'],
                "time": torch.ones(self.num_envs, device=self.device) * self.env.rew['time_reward_scale'],
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
            if global_pass_fraction > 0.25:
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

        if self.cfg.is_train:
            # ============ Domain Randomization - Per Episode ============
            # Randomize dynamics and control parameters for each resetting environment
            n = len(env_ids)
            
            # Aerodynamic drag coefficients
            k_aero_xy_rand = torch.rand(n, device=self.device) * \
                (self.env._k_aero_xy_max - self.env._k_aero_xy_min) + self.env._k_aero_xy_min
            self.env._K_aero[env_ids, 0] = k_aero_xy_rand
            self.env._K_aero[env_ids, 1] = k_aero_xy_rand  # xy use same value
            self.env._K_aero[env_ids, 2] = torch.rand(n, device=self.device) * \
                (self.env._k_aero_z_max - self.env._k_aero_z_min) + self.env._k_aero_z_min
            
            # Thrust to weight ratio
            self.env._thrust_to_weight[env_ids] = torch.rand(n, device=self.device) * \
                (self.env._twr_max - self.env._twr_min) + self.env._twr_min
            
            # PID gains for roll and pitch
            self.env._kp_omega[env_ids, :2] = (torch.rand(n, 1, device=self.device) * \
                (self.env._kp_omega_rp_max - self.env._kp_omega_rp_min) + self.env._kp_omega_rp_min)
            self.env._ki_omega[env_ids, :2] = (torch.rand(n, 1, device=self.device) * \
                (self.env._ki_omega_rp_max - self.env._ki_omega_rp_min) + self.env._ki_omega_rp_min)
            self.env._kd_omega[env_ids, :2] = (torch.rand(n, 1, device=self.device) * \
                (self.env._kd_omega_rp_max - self.env._kd_omega_rp_min) + self.env._kd_omega_rp_min)
            
            # PID gains for yaw
            self.env._kp_omega[env_ids, 2] = torch.rand(n, device=self.device) * \
                (self.env._kp_omega_y_max - self.env._kp_omega_y_min) + self.env._kp_omega_y_min
            self.env._ki_omega[env_ids, 2] = torch.rand(n, device=self.device) * \
                (self.env._ki_omega_y_max - self.env._ki_omega_y_min) + self.env._ki_omega_y_min
            self.env._kd_omega[env_ids, 2] = torch.rand(n, device=self.device) * \
                (self.env._kd_omega_y_max - self.env._kd_omega_y_min) + self.env._kd_omega_y_min
            # ============================================================

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
        waypoint_indices = torch.zeros(n_reset, device=self.device, dtype=self.env._idx_wp.dtype)

        # get starting poses behind waypoints
        x0_wp = self.env._waypoints[waypoint_indices][:, 0]
        y0_wp = self.env._waypoints[waypoint_indices][:, 1]
        theta = self.env._waypoints[waypoint_indices][:, -1]
        #z_wp = self.env._waypoints[waypoint_indices][:, 2]

        x_local = torch.empty(n_reset, device=self.device).uniform_(-3.0, -0.5)
        y_local = torch.empty(n_reset, device=self.device).uniform_(-1.0, 1.0)

        # rotate local pos to global frame
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        x_rot = cos_theta * x_local - sin_theta * y_local
        y_rot = sin_theta * x_local + cos_theta * y_local
        initial_x = x0_wp - x_rot
        initial_y = y0_wp - y_rot
        #initial_z = z_local + z_wp
        initial_z = torch.zeros(n_reset, device=self.device) + 0.05

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
            self.env._desired_pos_w[env_ids] - self.env._robot.data.root_link_pos_w[env_ids], dim=1
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