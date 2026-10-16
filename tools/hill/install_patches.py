"""Install Arena-matching physics patches for hill training/eval.

Called from hill_climb_1m5/env.py make_env_cfg. Safe to call multiple times.
Requires ARENA_SINGLE_CONTACT=1 and/or ARENA_DELAY=1 to take effect.
"""
from __future__ import annotations
import os, subprocess, sys
from pathlib import Path

_DONE = False

def install():
    global _DONE
    if _DONE:
        return
    here = Path(__file__).resolve().parent
    if os.environ.get("ARENA_SINGLE_CONTACT") == "1":
        script = here / "apply_warp_patch.sh"
        if script.exists():
            env = os.environ.copy()
            env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
            # Hint for the shell script.
            venv = Path(sys.executable).resolve().parent.parent
            if (venv / "bin" / "python").exists():
                env["VIRTUAL_ENV"] = str(venv)
            subprocess.run(["bash", str(script)], check=False, env=env)
            print("[hill2] warp single-contact patch applied (ARENA_SINGLE_CONTACT=1)")
    if os.environ.get("ARENA_DELAY") == "1":
        if str(here) not in sys.path:
            sys.path.insert(0, str(here))
        import arena_delay
        if not getattr(arena_delay, "_done", False):
            arena_delay.install()
            arena_delay._done = True
            print("[hill2] Arena command-delay installed (ARENA_DELAY=1)")
    _DONE = True
