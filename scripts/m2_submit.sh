#!/bin/bash
# Submit all M2 production jobs to KEKCC.
#   DRYRUN=1 bash m2_submit.sh     only prints the bsub commands
#   bash m2_submit.sh              submits them
# Each job = 4 slots = 4 serial processes with different seeds (so OMP_NUM_THREADS=1 per process).
# --max-seconds (MAXSEC, default 4200 s per process) stops a process early and still saves its histograms,
# so 4 processes stay well below the 1200 CPU-min limit of one job.
QUEUE=${QUEUE:-l}
NSLOT=4
# FORWARD_ONLY=1 (default): keep only CM-forward pi0, as in the paper. FORWARD_ONLY=0 reproduces the first (v1) runs.
ECM1=43.3394149463            # sqrt(s) of 1 TeV protons on protons at rest

mkdir -p logs out
sub() {   # name model tune frame energy boost nevents_per_process seed0
  cmd=(bsub -q $QUEUE -n $NSLOT -R "span[hosts=1]" -J "$1" -o "logs/%J_$1.out" -e "logs/%J_$1.err" \
       bash m2_job.sh "${@:2}" $NSLOT ${FORWARD_ONLY:-1})
  if [ -n "$DRYRUN" ]; then printf "%q " "${cmd[@]}"; echo; else "${cmd[@]}"; fi
}

# ---- Fig. 2 : 1 TeV proton on proton at rest, generated in the lab frame ----
sub fig2_sibyll  Sibyll23d  default lab 1000 none 500000 1000
sub fig2_qgsjet  QGSJetII04 default lab 1000 none 500000 2000
sub fig2_pythia  Pythia8    default lab 1000 none 500000 3000
sub fig2_pyfwd   Pythia8    fwd     lab 1000 none 500000 4000
sub fig2_epos    EposLHC    default lab 1000 none 300000 5000      # EPOS is ~10x slower

# ---- Fig. 3 : same collision in the CM frame + boost (Sibyll 2.3d), and the frame check ----
sub fig3_max     Sibyll23d  default cm $ECM1 max  500000 6000
sub fig3_plus    Sibyll23d  default cm $ECM1 plus 250000 7000

# ---- Fig. 7 : 13.6 TeV pp ----
sub fig7_max     Sibyll23d  default cm 13600 max  500000 8000