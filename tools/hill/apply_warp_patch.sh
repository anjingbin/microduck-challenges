#!/bin/bash
# Make MuJoCo Warp's plane-mesh collision emit only the deepest contact, like MuJoCo C.
# Active only when ARENA_SINGLE_CONTACT=1 at import time.
set -e
if [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
  PY="$VIRTUAL_ENV/bin/python"
elif command -v python >/dev/null 2>&1 && python -c "import mujoco_warp" 2>/dev/null; then
  PY=python
elif command -v python3 >/dev/null 2>&1 && python3 -c "import mujoco_warp" 2>/dev/null; then
  PY=python3
elif command -v uv >/dev/null 2>&1; then
  PY="uv run python"
else
  echo "[warp-patch] no python with mujoco_warp found" >&2; exit 1
fi
SP=$($PY -c "import mujoco_warp,os;print(os.path.dirname(os.path.dirname(mujoco_warp.__file__)))")
TARGET="$SP/mujoco_warp/_src/collision_primitive.py"
if grep -q _ARENA_SINGLE_CONTACT "$TARGET"; then
  echo "[warp-patch] already applied at $TARGET"
else
  PATCH="$(cd "$(dirname "$0")" && pwd)/warp_single_contact.patch"
  patch -d "$SP" -p1 < "$PATCH"
  echo "[warp-patch] applied to $TARGET"
fi
rm -rf "${HOME}/.cache/warp" 2>/dev/null || true
