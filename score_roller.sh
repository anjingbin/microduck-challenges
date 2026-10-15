#!/bin/bash
# usage: score_ckpts.sh <ckpt_repo> <outdir>
set -u; cd /workspace/microduck_train
uv run hf download "$1" --include "*.pt" --local-dir "$2" >/dev/null 2>&1
for f in $(find "$2" -path "*roller_sprint_2m*" -name "model_*.pt" | sort -V); do
  n=$(basename $(dirname $f))_$(basename $f .pt); o="$2/$n.onnx"
  [ -f "$o" ] || uv run python -m mjlab_microduck.export Mjlab-RollerSprint2m-MicroDuck --checkpoint-file "$f" --num-envs 1 --onnx-file "$o" >/dev/null 2>&1
  echo "$n $(uv run --group workshop arena score "$o" --event roller-sprint-2m 2>&1 | grep -E '^score|scored at' | tr '\n' ' ')"
done
