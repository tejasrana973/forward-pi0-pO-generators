#!/bin/bash
# Body of one batch job: runs NPROC independent serial generator processes (different seeds).
# Usage: m2_job.sh MODEL TUNE FRAME ENERGY BOOST NEV_PER_PROC SEED0 NPROC FORWARD_ONLY(1/0)
#   Every chromo process is serial, so total threads = NPROC = the "bsub -n" value.
MODEL=$1; TUNE=$2; FRAME=$3; ENERGY=$4; BOOST=$5; NEV=$6; SEED0=$7; NPROC=${8:-4}; FO=${9:-1}
FOFLAG=""; [ "$FO" = "1" ] && FOFLAG="--forward-only"

export OMP_NUM_THREADS=1        # 1 thread per process x NPROC processes = NPROC slots
export OMP_STACKSIZE=256M
ulimit -s unlimited 2>/dev/null || true

source ${VENV:-$HOME/venvs/lhcf}/bin/activate
cd ${WORKDIR:-$HOME/hu_project} || exit 1
mkdir -p out logs

for i in $(seq 0 $((NPROC-1))); do
  SEED=$((SEED0+i))
  python3 m2_generate.py --model $MODEL --tune $TUNE --frame $FRAME --energy $ENERGY --boost $BOOST \
      --nevents $NEV --seed $SEED --max-seconds ${MAXSEC:-4200} $FOFLAG \
      > logs/gen_${MODEL}_${TUNE}_${FRAME}${ENERGY}_${BOOST}_fo${FO}_s${SEED}.log 2>&1 &
done
wait
echo "job finished: $MODEL $TUNE $FRAME $ENERGY $BOOST"