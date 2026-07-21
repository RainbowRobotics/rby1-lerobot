#!/usr/bin/env bash
# Add the X/Y/Z jog buttons to the Android WebXR page.
#
# The phone UI is served by the third-party `teleop` package (Spes Robotics,
# Apache-2.0). We do not vendor its source; this applies a small patch to the
# copy already installed in your environment. Run once after installing or
# upgrading `teleop`.
#
# Without this patch phone teleoperation still works fully — you just get the
# IMU pose control ("hold to move") and no on-screen jog buttons.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PATCH="$HERE/jog.patch"

TELEOP_DIR="$(python -c 'import teleop, os; print(os.path.dirname(teleop.__file__))' 2>/dev/null)" || {
    echo "teleop is not installed. Run: pip install 'lerobot[phone]'" >&2
    exit 1
}

cd "$TELEOP_DIR"

if grep -q "getJog" assets/teleop-ui.js 2>/dev/null; then
    echo "Jog buttons already present in $TELEOP_DIR - nothing to do."
    exit 0
fi

if ! patch -p0 --dry-run < "$PATCH" >/dev/null 2>&1; then
    echo "The patch does not apply cleanly to teleop in $TELEOP_DIR." >&2
    echo "The upstream frontend has probably changed; re-create jog.patch" >&2
    echo "against the new version (see the README in this directory)." >&2
    exit 1
fi

cp index.html index.html.bak
cp assets/teleop-ui.js assets/teleop-ui.js.bak
patch -p0 < "$PATCH"
echo "Jog buttons applied to: $TELEOP_DIR (originals saved as *.bak)"
echo "Reload the WebXR page on your phone to see the X/Y/Z buttons."
