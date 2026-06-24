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

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig

ARM_JOINTS = [f"joint_{idx}" for idx in range(1, 8)]
GRIPPER = "gripper"
CONTROL_JOINTS = ARM_JOINTS + [GRIPPER]
ARM_MOTOR_CONFIG: dict[str, tuple[int, int, str]] = {
    "joint_1": (0x01, 0x11, "RS04"),
    "joint_2": (0x02, 0x12, "RS04"),
    "joint_3": (0x03, 0x13, "RS03"),
    "joint_4": (0x04, 0x14, "RS03"),
    "joint_5": (0x05, 0x15, "RS00"),
    "joint_6": (0x06, 0x16, "RS00"),
    "joint_7": (0x07, 0x17, "RS00"),
    GRIPPER: (0x08, 0x18, "RS00"),
}

RIGHT_LOWER_LIMITS_RAD = [-1.25, -1.57, -1.57, -1.8, -1.5, -0.7, -1.4]
RIGHT_UPPER_LIMITS_RAD = [1.25, 0.15, 1.57, 0.0, 1.5, 0.7, 1.4]
LEFT_LOWER_LIMITS_RAD = [-1.25, -0.15, -1.57, -1.8, -1.5, -0.7, -1.4]
LEFT_UPPER_LIMITS_RAD = [1.25, 1.57, 1.57, 0.0, 1.5, 0.7, 1.4]
GRIPPER_UPPER_LIMIT_RAD = 1.0472
GRIPPER_LOWER_LIMIT_RAD = -GRIPPER_UPPER_LIMIT_RAD


@RobotConfig.register_subclass("openarmx_follower")
@dataclass(kw_only=True)
class OpenArmXFollowerConfig(RobotConfig):
    """Configuration for a bimanual OpenArmX follower driven directly over RobStride CAN."""

    id: str | None = "openarmx_follower"

    can_right: str = "can0"
    can_left: str = "can1"
    can_interface: str = "socketcan"
    bitrate: int = 1000000
    can_fd: bool = False
    data_bitrate: int | None = None

    dry_run: bool = True
    enable_teleop: bool = False
    require_hardware_ready: bool = True
    log_observation: bool = False
    log_observation_period_s: float = 1.0
    disable_torque_on_disconnect: bool = True
    sync_follower_when_disabled: bool = True
    sync_follower_period_s: float = 0.2

    relative_mode: bool = True
    scale: float = 1.0
    leader_action_unit: str = "degree"
    joint_signs: list[float] = field(default_factory=lambda: [-1.0] * 7)
    gripper_sign: float = 1.0

    jump_threshold: float = 0.8
    jump_policy: str = "limit"
    rate_hz: float = 60.0
    velocity_limit_enabled: bool = True
    max_velocity: list[float] = field(
        default_factory=lambda: [1.5708] * 7
    )
    gripper_max_velocity: float = 2.0

    position_kp: list[float] = field(default_factory=lambda: [50.0, 50.0, 50.0, 50.0, 10.0, 10.0, 10.0])
    position_kd: list[float] = field(default_factory=lambda: [2.5, 2.5, 2.5, 2.5, 0.5, 0.5, 0.5])
    gripper_position_kp: float = 10.0
    gripper_position_kd: float = 0.5

    right_lower: list[float] = field(default_factory=lambda: RIGHT_LOWER_LIMITS_RAD.copy())
    right_upper: list[float] = field(default_factory=lambda: RIGHT_UPPER_LIMITS_RAD.copy())
    left_lower: list[float] = field(default_factory=lambda: LEFT_LOWER_LIMITS_RAD.copy())
    left_upper: list[float] = field(default_factory=lambda: LEFT_UPPER_LIMITS_RAD.copy())
    gripper_lower: float = GRIPPER_LOWER_LIMIT_RAD
    gripper_upper: float = GRIPPER_UPPER_LIMIT_RAD

    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    motor_config: dict[str, tuple[int, int, str]] = field(default_factory=lambda: ARM_MOTOR_CONFIG.copy())

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.leader_action_unit not in {"degree", "radian"}:
            raise ValueError("leader_action_unit must be either 'degree' or 'radian'.")
        if self.jump_policy not in {"limit", "freeze"}:
            raise ValueError("jump_policy must be either 'limit' or 'freeze'.")
        if self.rate_hz <= 0:
            raise ValueError("rate_hz must be positive.")
        if self.sync_follower_period_s < 0:
            raise ValueError("sync_follower_period_s must be non-negative.")
        if self.log_observation_period_s < 0:
            raise ValueError("log_observation_period_s must be non-negative.")
        if self.gripper_lower > self.gripper_upper:
            raise ValueError("gripper_lower must be less than or equal to gripper_upper.")

        list_fields = {
            "joint_signs": self.joint_signs,
            "max_velocity": self.max_velocity,
            "position_kp": self.position_kp,
            "position_kd": self.position_kd,
            "right_lower": self.right_lower,
            "right_upper": self.right_upper,
            "left_lower": self.left_lower,
            "left_upper": self.left_upper,
        }
        for name, value in list_fields.items():
            if len(value) != 7:
                raise ValueError(f"{name} must contain exactly 7 values.")

        for side, lower, upper in (
            ("right", self.right_lower, self.right_upper),
            ("left", self.left_lower, self.left_upper),
        ):
            for idx, (lo, hi) in enumerate(zip(lower, upper, strict=True), start=1):
                if lo > hi:
                    raise ValueError(f"{side}_joint_{idx} lower limit {lo} is greater than upper limit {hi}.")


# Kept for older imports. The `openarmx_follower` robot type is now bimanual.
OpenArmXFollowerConfigBase = OpenArmXFollowerConfig
