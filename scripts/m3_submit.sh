#!/bin/bash
# Submit the M3 production jobs to KEKCC.
#   DRYRUN=1 bash m3_submit.sh fast     only print the bsub commands
#   bash m3_submit.sh fast              fast models (Sibyll, QGSJET II-04, Pythia): minutes to ~1 h per job
#   bash m3_submit.sh slow              slow models (EPOS-LHC, EPOS LHC-R, QGSJET III): up to ~75 min per job
#   bash m3_submit.sh all               both
# Each job = 4 slots = 4 serial processes with different seeds (OMP_NUM_THREADS=1 each).
# Every process stops after MAXSEC (default 4200 s) and still saves its histograms, so the jobs stay far below
# the 1200 CPU-min limit.  "events per process" below are upper limits; the time limit decides for slow models.
QUEUE=${QUEUE:-l}
NSLOT=4
WHICH=${1:-fast}
SEED=30000
mkdir -p logs3 out3

sub() {   # name model target frame energy boost events_per_process
  cmd=(bsub -q $QUEUE -n $NSLOT -R "span[hosts=1]" -J "$1" -o "logs3/%J_$1.out" -e "logs3/%J_$1.err" \
       bash m3_job.sh "${@:2}" $SEED $NSLOT)
  SEED=$((SEED+10))
  if [ -n "$DRYRUN" ]; then printf "%q " "${cmd[@]}"; echo; else "${cmd[@]}"; fi
}

if [ "$WHICH" = "fast" ] || [ "$WHICH" = "all" ]; then
  # ---- 1 TeV proton on target at rest (Fig. 6): p+16O and the pp reference ----
  sub lab_O_sib   Sibyll23d       O16 lab 1000 none 250000
  sub lab_O_qgs   QGSJetII04      O16 lab 1000 none 250000
  sub lab_O_ang   Pythia8Angantyr O16 lab 1000 none 250000
  sub lab_p_sib   Sibyll23d       p   lab 1000 none 250000
  sub lab_p_qgs   QGSJetII04      p   lab 1000 none 250000
  sub lab_p_pyt   Pythia8         p   lab 1000 none 250000
  # ---- sqrt(s_NN) = 9.62 TeV, CM frame, proton direction +z ----
  sub cm_O_sib    Sibyll23d       O16 cm 9620 plus 250000
  sub cm_O_sibe   Sibyll23e       O16 cm 9620 plus 250000
  sub cm_O_qgs    QGSJetII04      O16 cm 9620 plus 100000
  sub cm_O_ang    Pythia8Angantyr O16 cm 9620 plus 100000
  sub cm_p_sib    Sibyll23d       p   cm 9620 plus 250000
  sub cm_p_sibe   Sibyll23e       p   cm 9620 plus 250000
  sub cm_p_qgs    QGSJetII04      p   cm 9620 plus 100000
  sub cm_p_pyt    Pythia8         p   cm 9620 plus 250000
  # ---- nitrogen check (does 14N differ from 16O?) ----
  sub lab_N_sib   Sibyll23d       N14 lab 1000 none 250000
  sub lab_N_qgs   QGSJetII04      N14 lab 1000 none 250000
  sub cm_N_sib    Sibyll23d       N14 cm 9620 plus 250000
  sub cm_N_qgs    QGSJetII04      N14 cm 9620 plus 100000
fi

if [ "$WHICH" = "slow" ] || [ "$WHICH" = "all" ]; then
  sub lab_O_epos  EposLHC   O16 lab 1000 none 150000
  sub lab_p_epos  EposLHC   p   lab 1000 none 250000
  sub cm_O_epos1  EposLHC   O16 cm 9620 plus 100000
  sub cm_O_epos2  EposLHC   O16 cm 9620 plus 100000
  sub cm_p_epos   EposLHC   p   cm 9620 plus 100000
  sub cm_O_eposr1 EposLHCR  O16 cm 9620 plus 100000
  sub cm_O_eposr2 EposLHCR  O16 cm 9620 plus 100000
  sub cm_p_eposr  EposLHCR  p   cm 9620 plus 100000
  sub cm_O_qg3a   QGSJetIII O16 cm 9620 plus 100000
  sub cm_O_qg3b   QGSJetIII O16 cm 9620 plus 100000
  sub cm_p_qg3    QGSJetIII p   cm 9620 plus 100000
fi
