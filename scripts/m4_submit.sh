#!/bin/bash
# Submit the M4 production jobs (p+16O at 9.62 TeV, ATLAS-like observables) to KEKCC.
#   DRYRUN=1 bash m4_submit.sh fast     only print the bsub commands
#   bash m4_submit.sh fast              Sibyll 2.3e, DPMJET III 2019.1, Pythia 8 Angantyr, QGSJET II-04
#   bash m4_submit.sh slow              EPOS LHC-R, QGSJET III (about 5 to 20 events/s per process)
#   bash m4_submit.sh all
# Each job = 4 slots = 4 serial processes; every process stops after MAXSEC (default 4200 s) and saves.
QUEUE=${QUEUE:-l}
NSLOT=4
WHICH=${1:-fast}
SEED=50000
mkdir -p logs4 out4

sub() {   # name model events_per_process
  cmd=(bsub -q $QUEUE -n $NSLOT -R "span[hosts=1]" -J "$1" -o "logs4/%J_$1.out" -e "logs4/%J_$1.err" \
       bash m4_job.sh "${@:2}" $SEED $NSLOT)
  SEED=$((SEED+10))
  if [ -n "$DRYRUN" ]; then printf "%q " "${cmd[@]}"; echo; else "${cmd[@]}"; fi
}

if [ "$WHICH" = "fast" ] || [ "$WHICH" = "all" ]; then
  sub m4_sibyll   Sibyll23e        1000000
  sub m4_dpmjet   DpmjetIII191      500000
  sub m4_ang1     Pythia8Angantyr   300000
  sub m4_ang2     Pythia8Angantyr   300000
  sub m4_qgs1     QGSJetII04        200000
  sub m4_qgs2     QGSJetII04        200000
fi
if [ "$WHICH" = "slow" ] || [ "$WHICH" = "all" ]; then
  sub m4_eposr1   EposLHCR          200000
  sub m4_eposr2   EposLHCR          200000
  sub m4_qg3a     QGSJetIII         100000
  sub m4_qg3b     QGSJetIII         100000
  sub m4_qg3c     QGSJetIII         100000
fi
