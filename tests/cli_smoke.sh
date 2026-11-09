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
PROFILE="$WORK/ds/meta/rby1_profile.json"
export RBY1_PROFILE_DIR="$WORK/profiles"

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
  --robot.profile_out="$PROFILE" --teleop.profile_out="$PROFILE" \
  --dataset.push_to_hub=false --dataset.single_task="smoke task" --dataset.num_episodes=1 \
  --dataset.episode_time_s=3 --dataset.reset_time_s=1 --dataset.fps=30
run replay 1 lerobot-replay "${ROBOT[@]}" "${DS[@]}" --dataset.episode=0
run calibrate_robot 0 lerobot-calibrate "${ROBOT[@]}"
run calibrate_teleop 0 lerobot-calibrate "${TELEOP[@]}"
run train 0 lerobot-train "${DS[@]}" --policy.type=act --policy.device=cpu --policy.push_to_hub=false \
  --policy.pretrained_backbone_weights=null --policy.chunk_size=10 --policy.n_action_steps=10 \
  --steps=2 --batch_size=2 --num_workers=0 --save_freq=2 --log_freq=1 \
  --output_dir="$WORK/train"
run rollout 1 lerobot-rollout "${ROBOT[@]}" --robot.cameras="$CAM" \
  --robot.profile_in="$WORK/ds" --robot.profile_mode=check \
  --policy.path="$WORK/train/checkpoints/last/pretrained_model" --device=cpu \
  --duration=4 --fps=30 --task="smoke task"
# The recording's profile (robot + teleop) must be in the dataset and match at rollout.
check() {
  local name=$1; shift
  if "$@" >>"$WORK/$name.log" 2>&1; then echo "PASS $name"; else echo "FAIL $name"; failed+=("$name"); fi
}
check profile_recorded python -c "import json,sys; d=json.load(open(sys.argv[1])); \
assert d['robot_config'] and d['core_parameters'] and d['teleop']['type']=='rby1_leader_arm'" "$PROFILE"
check profile_matches grep -q "matches the reference profile" "$WORK/rollout.log"
check profile_diff rby1-profile diff "$WORK/ds" "$PROFILE"

# Remote inference (lerobot >= 0.7.0): serve the smoke checkpoint on loopback and run
# lerobot-rollout --inference.type=remote against it. Skipped when lerobot[remote] is absent.
if python -c "import zenoh, msgpack" 2>/dev/null; then
  PORT=$(python -c "import socket; s=socket.socket(); s.bind(('127.0.0.1',0)); print(s.getsockname()[1])")
  check server_config rby1-profile server-config --checkpoint "$WORK/train/checkpoints/last/pretrained_model" \
    --dataset "$WORK/ds" --out "$WORK/server.yaml" --device cpu --listen "tcp/127.0.0.1:$PORT"
  SEMANTICS=$(python -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['semantics'])" "$WORK/server.yaml")
  lerobot-policy-server --config_path="$WORK/server.yaml" >"$WORK/policy_server.log" 2>&1 &
  SERVER_PID=$!
  for _ in $(seq 1 60); do grep -q "Policy server ready" "$WORK/policy_server.log" && break; sleep 1; done
  run rollout_remote 1 lerobot-rollout "${ROBOT[@]}" --robot.cameras="$CAM" \
    --robot.profile_in="$WORK/ds" --inference.type=remote --inference.endpoint="tcp/127.0.0.1:$PORT" \
    --inference.deployment=rby1 --inference.semantics="$SEMANTICS" --inference.mode=chunk \
    --duration=4 --fps=30 --task="smoke task"
  kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null || true
else
  echo "SKIP rollout_remote (lerobot[remote] not installed)"
fi

if ((${#failed[@]})); then
  echo "failed: ${failed[*]}"; exit 1
fi
echo "all CLI smoke tests passed"
