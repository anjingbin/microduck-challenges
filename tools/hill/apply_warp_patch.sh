#!/bin/bash
# Make MuJoCo Warp's plane-mesh collision emit only the deepest contact, like MuJoCo C (the Arena).
# Active only when ARENA_SINGLE_CONTACT=1 at import time. Run inside the training venv before training.
set -e
SP=$(python -c "import mujoco_warp,os;print(os.path.dirname(os.path.dirname(mujoco_warp.__file__)))")
grep -q _ARENA_SINGLE_CONTACT "$SP/mujoco_warp/_src/collision_primitive.py" || patch -d "$SP" -p1 < "$(dirname "$0")/warp_single_contact.patch"
rm -rf ~/.cache/warp
