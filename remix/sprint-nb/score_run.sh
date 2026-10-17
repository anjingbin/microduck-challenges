#!/bin/bash
# usage: score_run.sh <ckpt_repo> <run_name>   (from the repo root)
# Downloads the run's checkpoints, exports each exactly as publish does, scores it on sprint-2m,
# keeps every exported onnx as onnx/<run>_<N>.onnx.
set -u
uv run hf download "$1" --local-dir logs/rsl_rl >/dev/null 2>&1
run=$(ls -d logs/rsl_rl/sprint_2m/*_"$2" | tail -1)
mkdir -p onnx
for n in $(ls $run | sed -n 's/^model_\([0-9]*\)\.pt$/\1/p' | sort -n); do
  uv run publish --run $run --checkpoint $n --repo local/microduck-eval --dry-run --allow-dirty >/dev/null 2>&1 \
    && cp publish-eval/policy.onnx onnx/$2_$n.onnx \
    && echo "$2 $n: $(uv run --group workshop arena score onnx/$2_$n.onnx --event sprint-2m 2>&1 | grep -E '^(scored at|score )' | paste -sd' ' -)"
done
