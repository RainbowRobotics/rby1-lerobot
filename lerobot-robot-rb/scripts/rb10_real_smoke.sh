#!/usr/bin/env bash
set -euo pipefail

# First on-hardware smoke test: J6 sine of +/-0.2 deg for 8 s.
# Dry run by default — prints the command without connecting. To execute,
# clear the cell, keep the E-stop in reach, power/activate the arm from the
# pendant, then run:
#   RB10_REAL_TEST_CONFIRM=CELL_CLEAR_ESTOP_READY ./rb10_real_smoke.sh --execute
#
# Joint mode is the default; pass SPACE=ee for the EE (move_servo_l) variant
# (z-axis sine of +/-2 mm) once the joint smoke test has passed.

ROBOT_IP="${RB10_IP:-192.168.0.210}"
DATASET_ROOT="${RB10_DATASET_ROOT:-$HOME/rb10-real-smoke}"
SPACE="${SPACE:-joint}"

if [[ "$SPACE" == "ee" ]]; then
    space_args=(
        --robot.action_space=ee
        --teleop.space=ee
        --teleop.axis=2
        --teleop.amplitude=2.0
    )
else
    space_args=(
        --teleop.space=joint
        --teleop.axis=5
        --teleop.amplitude=0.2
    )
fi

command=(
    conda run -n lerobot
    lerobot-record
    --robot.type=rb10
    --robot.ip="$ROBOT_IP"
    --robot.operation_mode=real
    --robot.real_mode_confirm=true
    --robot.speed_bar=0.1
    --robot.gripper_type=none
    --teleop.type=rb_auto
    --teleop.period_s=12.0
    "${space_args[@]}"
    --dataset.repo_id=local/rb10-real-smoke
    --dataset.root="$DATASET_ROOT"
    --dataset.single_task="Verify gentle RB10 real-mode recording"
    --dataset.fps=30
    --dataset.num_episodes=1
    --dataset.episode_time_s=8
    --dataset.reset_time_s=0
    --dataset.video=false
    --dataset.push_to_hub=false
    --display_data=false
    --play_sounds=false
)

print_command() {
    printf '  %q' "${command[@]}"
    printf '\n'
}

if [[ "${1:-}" != "--execute" ]]; then
    echo "DRY RUN ONLY - no control-box connection was attempted."
    echo "Prepared command:"
    print_command
    exit 0
fi

if [[ "${RB10_REAL_TEST_CONFIRM:-}" != "CELL_CLEAR_ESTOP_READY" ]]; then
    echo "Refusing to execute: set RB10_REAL_TEST_CONFIRM=CELL_CLEAR_ESTOP_READY at the robot." >&2
    exit 2
fi

if [[ -e "$DATASET_ROOT" ]]; then
    echo "Refusing to overwrite existing dataset root: $DATASET_ROOT" >&2
    exit 2
fi

echo "EXECUTING PHYSICAL-ROBOT TEST:"
print_command
exec "${command[@]}"
