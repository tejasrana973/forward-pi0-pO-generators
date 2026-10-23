#!/bin/bash
# Body of one batch job: NPROC independent serial generator processes (different seeds).
# Usage: m4_job.sh MODEL NEV_PER_PROC SEED0 NPROC
export OMP_NUM_THREADS=1        # serial processes: NPROC x 1 thread = NPROC slots = "bsub -n"
export OMP_STACKSIZE=256M
ulimit -s unlimited 2>/dev/null || true
MODEL=$1; NEV=$2; SEED0=$3; NPROC=${4:-4}

source ${VENV:-$HOME/venvs/lhcf}/bin/activate
cd ${WORKDIR:-$HOME/hu_project} || exit 1
mkdir -p out4 logs4
for i in $(seq 0 $((NPROC-1))); do
  SEED=$((SEED0+i))
  # timeout = backstop against a hung generator; checkpoints (every 10 min) keep the data collected so far
  timeout -s KILL $(( ${MAXSEC:-4200} + 1800 )) python3 m4_generate.py --model $MODEL --nevents $NEV --seed $SEED \
      --max-seconds ${MAXSEC:-4200} --checkpoint-seconds 600 --outdir out4 \
      > logs4/gen_${MODEL}_s${SEED}.log 2>&1 &
done
wait
echo "job finished: $MODEL"
