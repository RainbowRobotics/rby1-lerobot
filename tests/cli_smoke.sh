#!/usr/bin/env bash
# Hardware-free smoke test of the lerobot CLIs with the RB-Y1 plugins.
#
# Runs the real lerobot-teleoperate / record / replay / calibrate / train /
# rollout entry points against tests/fakes/rby1_sdk (an in-process fake that
# opens no sockets or serial ports) and a synthetic camera.
#
# Requirements in the active Python environment: lerobot[core_scripts,training],
# the plugin packages, and tests/fakes/lerobot-camera-fakecam (all installable
# with pip install -e). Usage: tests/cli_smoke.sh [work_dir]
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
WORK=${1:-$(mktemp -d)}
mkdir -p "$WORK"
export PYTHONPATH="$HERE/fakes${PYTHONPATH:+:$PYTHONPATH}"
export RBY1_FAKE_SUMMARY=1  # the fake prints per-robot command counts at exit
export HF_HOME="$WORK/hf_home" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=disabled
unset DISPLAY WAYLAND_DISPLAY

python -c "import rby1_sdk; assert getattr(rby1_sdk, 'FAKE', False), rby1_sdk.__file__"

CAM='{"front": {"type": "fake", "fps": 30, "width": 320, "height": 240}}'
ROBOT=(--robot.type=rby1 --robot.address=stub.invalid:1)
TELEOP=(--teleop.type=rby1_leader_arm --teleop.robot_address=stub.invalid:1
        --teleop.init_duration=0.5 --teleop.startup_wait_time=0.5)
DS=(--dataset.repo_id=local/rby1_smoke --dataset.root="$WORK/ds")

failed=()
# run <name> <expects_robot_commands:0|1> <cmd...>
run() {
  local name=$1 moves=$2; shift 2
  local log="$WORK/$name.log" rc=0
  timeout 600 "$@" >"$log" 2>&1 || rc=$?
  if ((rc == 0)) && { ((moves == 0)) || grep -q "stream_commands=[1-9]" "$log"; }; then
    echo "PASS $name"
  else
    echo "FAIL $name (rc=$rc, log: $log)"; tail -20 "$log"; failed+=("$name")
  fi
}

run teleoperate 1 lerobot-teleoperate "${ROBOT[@]}" --robot.cameras="$CAM" "${TELEOP[@]}" \
  --teleop_time_s=3 --fps=30
run record 1 lerobot-record "${ROBOT[@]}" --robot.cameras="$CAM" "${TELEOP[@]}" "${DS[@]}" \
  --dataset.push_to_hub=false --dataset.single_task="smoke task" --dataset.num_episodes=1 \
  --dataset.episode_time_s=3 --dataset.reset_time_s=1 --dataset.fps=30
run replay 1 lerobot-replay "${ROBOT[@]}" "${DS[@]}" --dataset.episode=0
run calibrate_robot 0 lerobot-calibrate "${ROBOT[@]}"
run calibrate_teleop 0 lerobot-calibrate "${TELEOP[@]}"
run train 0 lerobot-train "${DS[@]}" --policy.type=act --policy.device=cpu --policy.push_to_hub=false \
  --policy.pretrained_backbone_weights=null --policy.chunk_size=10 --policy.n_action_steps=10 \
  --steps=2 --batch_size=2 --num_workers=0 --save_freq=2 --log_freq=1 --wandb.enable=false \
  --output_dir="$WORK/train"
run rollout 1 lerobot-rollout "${ROBOT[@]}" --robot.cameras="$CAM" \
  --policy.path="$WORK/train/checkpoints/last/pretrained_model" --device=cpu \
  --duration=4 --fps=30 --task="smoke task"

if ((${#failed[@]})); then
  echo "failed: ${failed[*]}"; exit 1
fi
echo "all CLI smoke tests passed"
