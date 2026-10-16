"""Monkeypatch mjlab_microduck.hf_jobs.BOOTSTRAP so the HF job applies Arena physics patches."""
import mjlab_microduck.hf_jobs as H

MARKER = "ARENA_SINGLE_CONTACT"
INSERT = r'''
echo "[bootstrap] applying Arena physics patches for hill2"
export ARENA_SINGLE_CONTACT=1 ARENA_DELAY=1 ARENA_GEOMS=1
# apt patch binary (tiny)
apt-get install -qq -y --no-install-recommends patch >/dev/null 2>&1 || true
bash tools/hill/apply_warp_patch.sh
echo "[bootstrap] ARENA_SINGLE_CONTACT=$ARENA_SINGLE_CONTACT ARENA_DELAY=$ARENA_DELAY"
'''

def apply():
    if MARKER in H.BOOTSTRAP:
        print("[inject] BOOTSTRAP already patched")
        return
    needle = 'echo "[bootstrap] starting training: uv run train $TRAIN_ARGS"'
    if needle not in H.BOOTSTRAP:
        raise SystemExit("BOOTSTRAP needle not found")
    H.BOOTSTRAP = H.BOOTSTRAP.replace(needle, INSERT + "\n" + needle, 1)
    # Also forward env vars into the job env dict via submit wrapper
    _orig = H.submit
    def submit(argv):
        import os
        os.environ.setdefault("ARENA_SINGLE_CONTACT", "1")
        os.environ.setdefault("ARENA_DELAY", "1")
        # Patch the env dict construction by wrapping _pick after... simpler: bake into BOOTSTRAP only.
        return _orig(argv)
    H.submit = submit
    print("[inject] BOOTSTRAP patched for Arena physics")

if __name__ == "__main__":
    apply()
    assert MARKER in H.BOOTSTRAP
    print("ok")
