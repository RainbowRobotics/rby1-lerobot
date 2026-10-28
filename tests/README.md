# Tests

Hardware-free tests for the RB-Y1 plugins. `fakes/rby1_sdk` is an in-process
stand-in for `rby1-sdk` (no sockets, no serial ports) that mirrors the subset of
the 0.10.0 API the plugins use; `fakes/lerobot-camera-fakecam` adds a synthetic
`"type": "fake"` camera.

```bash
pip install 'lerobot[core_scripts,training]==0.6.1' pytest
pip install -e lerobot-robot-rby1 -e lerobot-teleoperator-rby1 -e lerobot-async-rby1 \
  -e tests/fakes/lerobot-camera-fakecam

pytest tests                 # unit tests (plugin discovery, config parsing, feature layout, send_action)
tests/cli_smoke.sh [workdir] # real lerobot CLIs: teleoperate, record, replay, calibrate, train, rollout
```

`tests/snapshots/features.json` freezes the observation/action key order. A
change there breaks existing datasets and checkpoints; regenerate it with
`RBY1_UPDATE_SNAPSHOTS=1 pytest tests/test_features.py` only on purpose.

When the plugin starts calling a new `rby1_sdk` API, add it to the fake.
