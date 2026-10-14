#!/bin/bash
# usage: run_round.sh <run_name> <warm_pt_path> <commit_msg>
set -u; cd /workspace/microduck_train
RUN=$1; PT=$2; MSG=$3; W=warm_$RUN; CK=$(basename $PT)
busy=$(uv run python -c "
from huggingface_hub import HfApi;print(sum(j.status.stage in ('RUNNING','SCHEDULING') for j in HfApi().list_jobs(namespace='dadudu')))")
[ "$busy" != "0" ] && { echo "BUSY $busy"; exit 1; }
mkdir -p logs/rsl_rl/sprint_2m/$W && cp $PT logs/rsl_rl/sprint_2m/$W/
git add -f logs/rsl_rl/sprint_2m/$W/$CK src/microduck_challenges/sprint_2m/env.py
git -c user.name=anjingbin -c user.email=anjingbin@users.noreply.github.com commit -qm "$RUN: $MSG"; git push -q origin bubu-sprint-fast
MICRODUCK_WARM_START=1 uv run train Mjlab-Sprint2m-MicroDuck --env.scene.num-envs 4096 --agent.max-iterations 300 --agent.seed 1 --agent.logger tensorboard --agent.run-name $RUN --agent.resume True --agent.load-run $W --agent.load-checkpoint $CK --agent.algorithm.learning-rate 0.0001 --agent.algorithm.schedule fixed --hf-jobs --no-wandb --namespace dadudu --flavor l4x1 --timeout 2h > logs/$RUN.submit.log 2>&1 &
sleep 90
uv run python - <<PY > logs/$RUN.wait.log 2>&1
import time
from huggingface_hub import HfApi
a=HfApi(); j=[x for x in a.list_jobs(namespace='dadudu')][0]
repo=[m.id for m in a.list_models(author='dadudu',sort='created_at',limit=1)][0]
print('JOB',j.id,'REPO',repo,flush=True)
t0=time.time(); run0=None
while True:
    s=a.inspect_job(job_id=j.id,namespace='dadudu').status.stage
    if s=='RUNNING' and run0 is None: run0=time.time()
    if s not in ('RUNNING','SCHEDULING'): break
    time.sleep(60)
print('STAGE',s,'RUN_MIN',round((time.time()-(run0 or t0))/60,1),flush=True)
PY
cat logs/$RUN.wait.log
REPO=$(grep -o 'REPO [^ ]*' logs/$RUN.wait.log | cut -d' ' -f2)
./score_ckpts.sh $REPO /workspace/ckpt_$RUN 2>&1 | grep "_${RUN}_" | tee logs/$RUN.scores.txt
