# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""Configuration tables for Robstride motors (OpenArmX 29-bit extended CAN protocol)."""

from enum import IntEnum


# Motor type definitions
class MotorType(IntEnum):
    O0 = 0
    O1 = 1
    O2 = 2
    O3 = 3
    O4 = 4
    O5 = 5
    ELO5 = 6
    O6 = 7


class CommMode(IntEnum):
    PrivateProtocole = 0
    CANopen = 1
    MIT = 2


# Control modes
class ControlMode(IntEnum):
    MIT = 0
    POS_VEL = 1
    VEL = 2


# Motor limit parameters [PMAX, VMAX, TMAX]
# PMAX: Maximum position (rad)
# VMAX: Maximum velocity (rad/s) — verified from OpenArmX SDK
# TMAX: Maximum torque (N·m)
MOTOR_LIMIT_PARAMS: dict[MotorType, tuple[float, float, float]] = {
    MotorType.O0: (12.57, 33.0, 14.0),    # RS00: SDK verified
    MotorType.O1: (12.57, 44.0, 17.0),
    MotorType.O2: (12.57, 33.0, 20.0),
    MotorType.O3: (12.57, 20.0, 60.0),    # RS03: vMax updated 33→20 (SDK verified)
    MotorType.O4: (12.57, 15.0, 120.0),   # RS04: vMax updated 33→15 (SDK verified)
    MotorType.O5: (12.57, 50.0, 5.5),
    MotorType.ELO5: (12.57, 50.0, 6.0),
    MotorType.O6: (112.5, 50.0, 36.0),
}

# Motor model names
MODEL_NAMES = {
    MotorType.O0: "O0",
    MotorType.O1: "O1",
    MotorType.O2: "O2",
    MotorType.O3: "O3",
    MotorType.O4: "O4",
    MotorType.O5: "O5",
    MotorType.ELO5: "ELO5",
    MotorType.O6: "O6",
}

# Motor resolution table (encoder counts per revolution)
MODEL_RESOLUTION = {
    "O0": 65536,
    "O1": 65536,
    "O2": 65536,
    "O3": 65536,
    "O4": 65536,
    "O5": 65536,
    "ELO5": 65536,
    "O6": 65536,
}

# CAN baudrates supported by Robstride motors
AVAILABLE_BAUDRATES = [
    1000000,  # 1 Mbps (default)
]
DEFAULT_BAUDRATE = 1000000

# Default timeout in milliseconds
DEFAULT_TIMEOUT_MS = 0  # disabled by default

# Data that should be normalized
NORMALIZED_DATA = ["Present_Position", "Goal_Position"]

# RobStride 29-bit extended CAN protocol.
# 29-bit extended CAN protocol (OpenArmX SDK)
#
# Arbitration ID layout (29 bits, stored as a 32-bit value):
#   bits [28:24]  comm_type  (8 bits, top byte)
#   bits [23:8]   param16    (16 bits, middle two bytes)
#   bits [7:0]    motor_id   (8 bits, bottom byte)
#
# Management frames:  arb = (comm_type << 24) | (0x00 << 16) | (MASTER_ID << 8) | motor_id
# MIT control frame:  arb = (CAN_COMM_MIT_CTRL << 24) | (torque_u16 << 8) | motor_id
#
# Response frames from motor:
#   arb = (comm_type << 24) | (0x00 << 16) | (motor_id << 8) | MASTER_ID
#   → motor_id is at (arb_id >> 8) & 0xFF

# Communication type codes (byte 3 of 29-bit arb_id)
CAN_COMM_MIT_CTRL = 0x01       # MIT position/velocity/torque control
CAN_COMM_STATUS_QUERY = 0x02   # Query motor status and clear fault
CAN_COMM_ENABLE = 0x03         # Enable motor (enter Motor mode)
CAN_COMM_DISABLE = 0x04        # Disable motor (enter Reset mode)
CAN_COMM_SET_ZERO = 0x06       # Set current position as zero

# Master (host) node ID embedded in management frame arb_ids
CAN_MASTER_ID = 0xFD

# MIT control KP/KD upper limits per motor type (verified from OpenArmX SDK)
MIT_KP_MAX: dict[MotorType, float] = {
    MotorType.O0: 500.0,
    MotorType.O1: 500.0,
    MotorType.O2: 500.0,
    MotorType.O3: 5000.0,
    MotorType.O4: 5000.0,
    MotorType.O5: 500.0,
    MotorType.ELO5: 500.0,
    MotorType.O6: 5000.0,
}

MIT_KD_MAX: dict[MotorType, float] = {
    MotorType.O0: 5.0,
    MotorType.O1: 5.0,
    MotorType.O2: 5.0,
    MotorType.O3: 100.0,
    MotorType.O4: 100.0,
    MotorType.O5: 5.0,
    MotorType.ELO5: 5.0,
    MotorType.O6: 100.0,
}

# Parameter command IDs (for future parameter read/write support)
CAN_CMD_QUERY_PARAM = 0x33
CAN_CMD_WRITE_PARAM = 0x55
CAN_CMD_SAVE_PARAM = 0xAA
CAN_PARAM_ID = 0x7FF

RUNNING_TIMEOUT = 0.003
HANDSHAKE_TIMEOUT_S = 0.5   # generous timeout for motor status query
PARAM_TIMEOUT = 0.1

STATE_CACHE_TTL_S = 0.02
