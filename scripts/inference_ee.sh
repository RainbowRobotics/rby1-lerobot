#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
    echo "Usage: bash scripts/inference_ee.sh rb10|rb10e SERVER_HOST:PORT ROBOT_IP [client options]" >&2
    echo "Starts paused, dry_run=true. Connect reads robot/cameras; actuator initialization is deferred." >&2
    exit 2
fi

model=$1
server=$2
robot_ip=$3
shift 3
case "$model" in
    rb10|rb10e) ;;
    *) echo "Model must be rb10 or rb10e" >&2; exit 2 ;;
esac

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="$root/lerobot-async-rby1:$root/lerobot-robot-rb${PYTHONPATH:+:$PYTHONPATH}"

exec python -m lerobot_async_inference.robot_client_ee \
    --robot.type=rb10 \
    --robot.ip="$robot_ip" \
    --robot.gripper_type=rby1_dynamixel \
    --robot.control_rate_hz=30 \
    --robot.first_state_timeout_s=0.25 \
    --kinematics_model="$model" \
    --server_address="$server" \
    --task="Control the RB10 arm and gripper using Meta Quest VR." \
    --control_rate_hz=30 \
    --dry_run=true \
    "$@"
