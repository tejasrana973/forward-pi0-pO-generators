#!/bin/bash
# Body of one batch job: NPROC independent serial generator processes (different seeds).
# Usage: m3_job.sh MODEL TARGET FRAME ENERGY BOOST NEV_PER_PROC SEED0 NPROC
export OMP_NUM_THREADS=1        # serial processes: NPROC x 1 thread = NPROC slots = "bsub -n"
export OMP_STACKSIZE=256M
ulimit -s unlimited 2>/dev/null || true
MODEL=$1; TARGET=$2; FRAME=$3; ENERGY=$4; BOOST=$5; NEV=$6; SEED0=$7; NPROC=${8:-4}

source ${VENV:-$HOME/venvs/lhcf}/bin/activate
cd ${WORKDIR:-$HOME/hu_project} || exit 1
mkdir -p out3 logs3
for i in $(seq 0 $((NPROC-1))); do
  SEED=$((SEED0+i))
  # timeout = backstop against a hung generator (EPOS can hang inside one event); checkpoints keep the data
  timeout -s KILL $(( ${MAXSEC:-4200} + 1800 )) python3 m3_generate.py --model $MODEL --target $TARGET --frame $FRAME --energy $ENERGY --boost $BOOST \
      --forward-only --nevents $NEV --seed $SEED --max-seconds ${MAXSEC:-4200} --checkpoint-seconds 600 --outdir out3 \
      > logs3/gen_${MODEL}_${TARGET}_${FRAME}${ENERGY}_${BOOST}_s${SEED}.log 2>&1 &
done
wait
echo "job finished: $MODEL $TARGET $FRAME $ENERGY $BOOST"