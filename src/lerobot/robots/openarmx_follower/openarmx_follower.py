#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import math
import time
from contextlib import suppress
from functools import cached_property
from typing import Any

from lerobot.cameras import make_cameras_from_configs
from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.robstride import RobstrideMotorsBus
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..robot import Robot
from .config_openarmx_follower import CONTROL_JOINTS, OpenArmXFollowerConfig

logger = logging.getLogger(__name__)

ARMS = ("right", "left")
JOINTS_PER_ARM = len(CONTROL_JOINTS)
STATE_STALE_AFTER_S = 0.5


class OpenArmXFollower(Robot):
    """Bimanual OpenArmX follower driven directly over RobStride CAN in MIT mode."""

    config_class = OpenArmXFollowerConfig
    name = "openarmx_follower"

    def __init__(self, config: OpenArmXFollowerConfig):
        super().__init__(config)
        self.config = config

        self.right_bus = self._make_bus(config.can_right)
        self.left_bus = self._make_bus(config.can_left)
        self.cameras = make_cameras_from_configs(config.cameras)
        self._connected = False
        self.hardware_state_valid = False
        self._torque_enabled = False
        self._real_mode_logged = False

        self._feature_keys = [f"{arm}_{joint}.pos" for arm in ARMS for joint in CONTROL_JOINTS]
        self._leader_start: list[float] | None = None
        self._follower_start: list[float] | None = None
        self._current_cmd = [0.0] * len(self._feature_keys)
        self._last_time: float | None = None

        self._last_status_log_time = 0.0
        self._last_observation_log_time = 0.0
        self._last_warning_times: dict[str, float] = {}
        self._last_disabled_sync_time = -math.inf
        self._last_leader_raw: list[float] | None = None
        self._last_leader_now: list[float] | None = None
        self._last_leader_delta: list[float] | None = None
        self._last_target_before_clip: list[float] | None = None
        self._last_target_after_clip: list[float] | None = None
        self._last_max_step: list[float] | None = None
        self._last_jump_warnings: list[str] = []

    def _make_bus(self, port: str) -> RobstrideMotorsBus:
        motors: dict[str, Motor] = {}
        for motor_name, (motor_id, master_id, motor_type) in self.config.motor_config.items():
            motor = Motor(motor_id, motor_type, MotorNormMode.DEGREES)
            motor.motor_type_str = motor_type
            motor.recv_id = motor_id
            motor.master_id = master_id  # type: ignore[attr-defined]
            motors[motor_name] = motor

        return RobstrideMotorsBus(
            port=port,
            motors=motors,
            calibration={},
            can_interface=self.config.can_interface,
            use_can_fd=self.config.can_fd,
            bitrate=self.config.bitrate,
            data_bitrate=self.config.data_bitrate,
        )

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        camera_features = {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }
        return {**{key: float for key in self._feature_keys}, **camera_features}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {key: float for key in self._feature_keys}

    @property
    def is_connected(self) -> bool:
        return self._connected and all(cam.is_connected for cam in self.cameras.values())

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:  # noqa: ARG002
        self._connect_bus("right", self.right_bus)
        self._connect_bus("left", self.left_bus)

        follower_now = self._try_read_follower_positions("connect")
        if follower_now is None:
            self.hardware_state_valid = False
            message = (
                "Could not read OpenArmX follower_start; real MIT control is unavailable. "
                f"Check SocketCAN devices {self.config.can_right}/{self.config.can_left}."
            )
            if self._real_hardware_requested() and self.config.require_hardware_ready:
                self._disconnect_buses(disable_torque=False)
                raise ConnectionError(message)
            logger.warning("%s Forcing dry-run behavior; real MIT control is disabled.", message)
        else:
            self.hardware_state_valid = True
            self._follower_start = follower_now
            self._current_cmd = follower_now.copy()

        self._last_time = time.perf_counter()
        self._connected = True

        for cam in self.cameras.values():
            cam.connect()

        logger.info(
            "%s connected. dry_run=%s enable_teleop=%s hardware_state_valid=%s",
            self,
            self.config.dry_run,
            self.config.enable_teleop,
            self.hardware_state_valid,
        )

    def _connect_bus(self, side: str, bus: RobstrideMotorsBus) -> None:
        logger.info("Opening OpenArmX %s arm CAN bus on %s...", side, bus.port)
        try:
            bus.connect(handshake=False)
            bus.flush_rx_queue()
        except Exception as e:
            logger.warning("Failed to open OpenArmX %s arm CAN bus on %s: %s", side, bus.port, e)

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        logger.info("OpenArmX follower does not use LeRobot motor calibration.")

    def configure(self) -> None:
        logger.info("OpenArmX follower configuration is static; no zero/home command is sent.")

    def setup_motors(self) -> None:
        raise NotImplementedError("OpenArmX motor IDs must be configured with manufacturer tools.")

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        if not self.config.enable_teleop:
            self._sync_follower_when_disabled_if_due(time.perf_counter(), "observation_disabled")
            self._maybe_log_observation(self._current_cmd)
            return self._observation_from_positions(self._current_cmd)

        if not self.config.log_observation:
            return self._observation_from_positions(self._current_cmd)

        follower_now = self._try_read_follower_positions("observation")
        if follower_now is None:
            self._rate_limited_warning(
                "observation_untrusted",
                1.0,
                "OpenArmX CAN state read failed; returning last current_cmd as an untrusted observation.",
            )
            self._maybe_log_observation(self._current_cmd)
            return self._observation_from_positions(self._current_cmd)

        self._maybe_log_observation(follower_now)
        return self._observation_from_positions(follower_now)

    @check_if_not_connected
    def send_action(
        self,
        action: RobotAction,
        custom_kp: dict[str, float] | None = None,
        custom_kd: dict[str, float] | None = None,
    ) -> RobotAction:
        now = time.perf_counter()
        leader_now = self._leader_action_to_radians(action)
        self._last_leader_now = leader_now

        if self._leader_start is None:
            self._leader_start = leader_now.copy()
            self._last_time = now
            self._last_leader_delta = [0.0] * len(self._feature_keys)
            self._last_target_before_clip = None
            self._last_target_after_clip = None
            self._last_max_step = [0.0] * len(self._feature_keys)
            self._maybe_log_status()
            return self._action_from_positions(self._current_cmd)

        if not self.config.enable_teleop:
            self._leader_start = leader_now.copy()
            self._sync_follower_when_disabled_if_due(now, "teleop_disabled")
            self._last_time = now
            self._last_leader_delta = [0.0] * len(self._feature_keys)
            self._last_target_before_clip = self._current_cmd.copy()
            self._last_target_after_clip = self._current_cmd.copy()
            self._last_max_step = [0.0] * len(self._feature_keys)
            self._last_jump_warnings = []
            self._maybe_log_status()
            return self._action_from_positions(self._current_cmd)

        if self._follower_start is None:
            self.hardware_state_valid = False
            self._rate_limited_warning(
                "missing_follower_start",
                1.0,
                "OpenArmX follower_start is unavailable; refusing real MIT control.",
            )
            self._maybe_log_status()
            return self._action_from_positions(self._current_cmd)

        target_before_clip = self._compute_target_before_clip(leader_now)
        target_after_clip = self._clip_positions(target_before_clip)
        current_cmd = self._apply_jump_and_velocity_limits(target_after_clip, now)

        self._last_target_before_clip = target_before_clip
        self._last_target_after_clip = target_after_clip
        self._current_cmd = current_cmd
        self._last_time = now

        if self._should_send_real_hardware():
            try:
                self._ensure_real_hardware_mode()
                self._send_current_cmd(custom_kp=custom_kp, custom_kd=custom_kd)
            except Exception as e:
                self.hardware_state_valid = False
                self._rate_limited_warning(
                    "real_send_failed",
                    1.0,
                    "OpenArmX real MIT control failed; forcing dry-run behavior: %s",
                    e,
                )

        self._maybe_log_status()
        return self._action_from_positions(self._current_cmd)

    def _sync_follower_when_disabled_if_due(self, now: float, context: str) -> None:
        if not self.config.sync_follower_when_disabled:
            return

        if now - self._last_disabled_sync_time < self.config.sync_follower_period_s:
            return

        self._last_disabled_sync_time = now
        follower_now = self._try_read_follower_positions(context)
        if follower_now is None:
            return

        self._follower_start = follower_now.copy()
        self._current_cmd = follower_now.copy()
        self.hardware_state_valid = True

    def _leader_action_to_radians(self, action: RobotAction) -> list[float]:
        missing = [key for key in self._feature_keys if key not in action]
        if missing:
            raise ValueError(f"OpenArmX action is missing required keys: {missing}")

        values = [float(action[key]) for key in self._feature_keys]
        if self.config.leader_action_unit == "degree":
            # OpenArm Mini gripper action is already mapped from 0-100 percent to degrees
            # before send_action(). Convert every command, including gripper, to radians
            # here for RobStride MIT control. Do not use ROS2/URDF 0-0.044 m gripper units.
            values = [value * math.pi / 180.0 for value in values]
        elif self.config.leader_action_unit != "radian":
            raise ValueError(f"Unsupported leader_action_unit: {self.config.leader_action_unit}")

        self._last_leader_raw = values.copy()
        signs = (self.config.joint_signs + [self.config.gripper_sign]) * len(ARMS)
        return [sign * value for sign, value in zip(signs, values, strict=True)]

    def _compute_target_before_clip(self, leader_now: list[float]) -> list[float]:
        if self.config.relative_mode:
            leader_delta = [
                now - start for now, start in zip(leader_now, self._leader_start or leader_now, strict=True)
            ]
            self._last_leader_delta = leader_delta
            target = [
                start + delta * self.config.scale
                for start, delta in zip(self._follower_start or self._current_cmd, leader_delta, strict=True)
            ]
            # Arm joints stay relative as before. Gripper percent is an absolute open/close command:
            # 0/50/100% -> 0/30/60 deg at send_action() -> 0/0.5236/1.0472 rad here.
            for idx in self._gripper_indices():
                target[idx] = leader_now[idx] * self.config.scale
            return target

        self._last_leader_delta = [0.0] * len(self._feature_keys)
        return [value * self.config.scale for value in leader_now]

    def _gripper_indices(self) -> tuple[int, int]:
        return (JOINTS_PER_ARM - 1, (2 * JOINTS_PER_ARM) - 1)

    def _clip_positions(self, positions: list[float]) -> list[float]:
        right_lower = self.config.right_lower + [self.config.gripper_lower]
        right_upper = self.config.right_upper + [self.config.gripper_upper]
        left_lower = self.config.left_lower + [self.config.gripper_lower]
        left_upper = self.config.left_upper + [self.config.gripper_upper]
        lower = right_lower + left_lower
        upper = right_upper + left_upper
        return [
            min(max(position, lo), hi)
            for position, lo, hi in zip(positions, lower, upper, strict=True)
        ]

    def _apply_jump_and_velocity_limits(self, target_after_clip: list[float], now: float) -> list[float]:
        target_for_velocity = target_after_clip.copy()
        jump_warnings: list[str] = []
        for key, target, current in zip(
            self._feature_keys, target_after_clip, self._current_cmd, strict=True
        ):
            jump = abs(target - current)
            if jump > self.config.jump_threshold:
                warning = f"{key}: jump {jump:.4f} rad exceeds threshold {self.config.jump_threshold:.4f}"
                jump_warnings.append(warning)
                if self.config.jump_policy == "freeze":
                    target_for_velocity[self._feature_keys.index(key)] = current

        self._last_jump_warnings = jump_warnings
        if jump_warnings:
            self._rate_limited_warning("jump_threshold", 1.0, "OpenArmX jump threshold: %s", jump_warnings)

        dt = (now - self._last_time) if self._last_time is not None else (1.0 / self.config.rate_hz)
        if dt <= 0.0 or dt > 0.2:
            self._rate_limited_warning(
                "dt_clamped",
                1.0,
                "OpenArmX control dt %.6f is outside expected range; clamping for velocity limit.",
                dt,
            )
            dt = min(max(dt, 0.0), 1.0 / self.config.rate_hz)

        max_velocity = (self.config.max_velocity + [self.config.gripper_max_velocity]) * len(ARMS)
        max_step = [velocity * dt for velocity in max_velocity]
        self._last_max_step = max_step

        next_cmd = []
        if not self.config.velocity_limit_enabled:
            return target_for_velocity

        gripper_indices = set(self._gripper_indices())
        for idx, (target, current, step_limit) in enumerate(
            zip(target_for_velocity, self._current_cmd, max_step, strict=True)
        ):
            if idx in gripper_indices:
                next_cmd.append(target)
                continue
            step = min(max(target - current, -step_limit), step_limit)
            next_cmd.append(current + step)
        return next_cmd

    def _should_send_real_hardware(self) -> bool:
        return self._real_hardware_requested() and self.hardware_state_valid

    def _real_hardware_requested(self) -> bool:
        return (not self.config.dry_run) and self.config.enable_teleop

    def _ensure_real_hardware_mode(self) -> None:
        if not self._torque_enabled:
            self.right_bus.enable_torque(CONTROL_JOINTS)
            self.left_bus.enable_torque(CONTROL_JOINTS)
            self._torque_enabled = True
        if not self._real_mode_logged:
            logger.warning("REAL HARDWARE MODE ENABLED")
            self._real_mode_logged = True

    def _send_current_cmd(
        self,
        *,
        custom_kp: dict[str, float] | None = None,
        custom_kd: dict[str, float] | None = None,
    ) -> None:
        right_cmd = self._commands_for_arm(
            "right", self._current_cmd[:JOINTS_PER_ARM], custom_kp, custom_kd
        )
        left_cmd = self._commands_for_arm(
            "left", self._current_cmd[JOINTS_PER_ARM:], custom_kp, custom_kd
        )
        self.right_bus.mit_control_batch(right_cmd)
        self.left_bus.mit_control_batch(left_cmd)

    def _commands_for_arm(
        self,
        side: str,
        positions: list[float],
        custom_kp: dict[str, float] | None,
        custom_kd: dict[str, float] | None,
    ) -> dict[str, tuple[float, float, float, float, float]]:
        commands = {}
        for idx, (joint, position) in enumerate(zip(CONTROL_JOINTS, positions, strict=True)):
            kp_defaults = self.config.position_kp + [self.config.gripper_position_kp]
            kd_defaults = self.config.position_kd + [self.config.gripper_position_kd]
            kp = self._custom_gain(custom_kp, side, joint, idx, kp_defaults)
            kd = self._custom_gain(custom_kd, side, joint, idx, kd_defaults)
            commands[joint] = (kp, kd, position, 0.0, 0.0)
        return commands

    def _custom_gain(
        self,
        custom: dict[str, float] | None,
        side: str,
        joint: str,
        idx: int,
        defaults: list[float],
    ) -> float:
        if custom is None:
            return defaults[idx]
        for key in (f"{side}_{joint}", f"{side}_{joint}.pos", joint, f"{joint}.pos"):
            if key in custom:
                return float(custom[key])
        return defaults[idx]

    def _try_read_follower_positions(self, context: str) -> list[float] | None:
        try:
            return self._read_follower_positions()
        except Exception as e:
            self.hardware_state_valid = False
            self._rate_limited_warning(
                f"{context}_read_failed",
                1.0,
                "OpenArmX follower state read failed during %s: %s",
                context,
                e,
            )
            return None

    def _read_follower_positions(self) -> list[float]:
        right = self._read_arm_positions("right", self.right_bus)
        left = self._read_arm_positions("left", self.left_bus)
        return right + left

    def _read_arm_positions(self, side: str, bus: RobstrideMotorsBus) -> list[float]:
        if not bus.is_connected:
            raise ConnectionError(f"{side} bus {bus.port} is not connected")

        states = bus.sync_read_all_states(CONTROL_JOINTS)
        now = time.time()
        stale = [
            joint
            for joint in CONTROL_JOINTS
            if bus.last_feedback_time.get(joint) is None
            or now - (bus.last_feedback_time.get(joint) or 0.0) > STATE_STALE_AFTER_S
        ]
        if stale:
            raise ConnectionError(f"{side} bus {bus.port} has stale or missing states for {stale}")

        return [float(states[joint]["position"]) for joint in CONTROL_JOINTS]

    def _action_from_positions(self, positions: list[float]) -> RobotAction:
        return {key: value for key, value in zip(self._feature_keys, positions, strict=True)}

    def _observation_from_positions(self, positions: list[float]) -> RobotObservation:
        observation: RobotObservation = self._action_from_positions(positions)
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            observation[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug("%s read %s: %.1fms", self, cam_key, dt_ms)
        return observation

    def _rate_limited_warning(self, name: str, interval_s: float, message: str, *args: Any) -> None:
        now = time.monotonic()
        if now - self._last_warning_times.get(name, 0.0) >= interval_s:
            logger.warning(message, *args)
            self._last_warning_times[name] = now

    def _maybe_log_status(self) -> None:
        if not self.config.log_observation:
            return

        now = time.monotonic()
        if now - self._last_status_log_time < 1.0:
            return
        self._last_status_log_time = now

        rows = []
        for idx in self._gripper_indices():
            joint_name = self._feature_keys[idx].removesuffix(".pos")
            rows.append(
                f"{joint_name}: "
                f"leader_raw={self._fmt_value(self._last_leader_raw, idx)}, "
                f"leader_now={self._fmt_value(self._last_leader_now, idx)}, "
                f"leader_delta={self._fmt_value(self._last_leader_delta, idx)}, "
                f"target_before_clip={self._fmt_value(self._last_target_before_clip, idx)}, "
                f"target_after_clip={self._fmt_value(self._last_target_after_clip, idx)}, "
                f"current_cmd={self._fmt_value(self._current_cmd, idx)}"
            )
        logger.info("OpenArmX gripper status:\n  %s", "\n  ".join(rows))

    def _fmt_value(self, values: list[float] | None, idx: int) -> str:
        if values is None:
            return "None"
        return f"{float(values[idx]):.4f}"

    def _maybe_log_observation(self, positions_rad: list[float]) -> None:
        if not self.config.log_observation:
            return

        now = time.monotonic()
        if now - self._last_observation_log_time < self.config.log_observation_period_s:
            return
        self._last_observation_log_time = now

        rows = []
        for idx in self._gripper_indices():
            key = self._feature_keys[idx].removesuffix(".pos")
            rad = positions_rad[idx]
            deg = rad * 180.0 / math.pi
            rows.append(f"{key}: {rad:.4f} rad / {deg:.2f} deg")
        logger.info("OpenArmX gripper observation:\n  %s", "\n  ".join(rows))

    @check_if_not_connected
    def disconnect(self) -> None:
        self._disconnect_buses(disable_torque=self.config.disable_torque_on_disconnect)
        for cam in self.cameras.values():
            cam.disconnect()
        self._connected = False
        self._torque_enabled = False
        logger.info("%s disconnected.", self)

    def _disconnect_buses(self, *, disable_torque: bool) -> None:
        for side, bus in (("right", self.right_bus), ("left", self.left_bus)):
            if bus.is_connected:
                if disable_torque:
                    self._disable_bus_on_disconnect(side, bus)
                with suppress(Exception):
                    bus.disconnect(disable_torque=False)
        self._torque_enabled = False

    def _disable_bus_on_disconnect(self, side: str, bus: RobstrideMotorsBus) -> None:
        if not self.config.disable_torque_on_disconnect:
            return

        try:
            logger.info("Disabling all OpenArmX %s arm motors before disconnect...", side)
            bus.disable_torque(num_retry=2)
        except Exception as e:
            logger.warning("Failed to disable all OpenArmX %s arm motors during disconnect: %s", side, e)
