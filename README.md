# Forward neutral pions in proton-oxygen collisions: a generator-level study

Air showers need hadronic interaction models, and these models disagree close to the beam direction.
This repository holds a short study of that region with open-source tools.
It reproduces selected forward neutral-pion results for proton-proton collisions, extends them to proton-oxygen (p+O) collisions, and tests the analysis chain on the new ATLAS p+O data.

The full write-up is in [`note/`](note/).

## What was done

| Step | Content |
|---|---|
| M0 | Install Chromo; check energy-momentum conservation; check that every generator runs for pp and p+O |
| M1 | Pseudorapidity window of energetic pi0 versus collision energy, with a generator cross-check |
| M2 | Reproduce pi0 spectra, the pseudorapidity decomposition and the rate of pions with more than 70 % of the beam energy in pp (1 TeV lab, 13.6 TeV) |
| M3 | Repeat for p+O and p+N at 1 TeV (lab) and at 9.62 TeV per nucleon pair; compare with pp |
| M4 | ATLAS-like particle-level analysis of p+O at 9.62 TeV; compare six generators with the ATLAS HEPData tables |

Generators used (through [Chromo](https://github.com/impy-project/chromo)): EPOS-LHC, EPOS LHC-R, QGSJET II-04, QGSJET III, Sibyll 2.3d and 2.3e, DPMJET-III 2019.1, Pythia 8 and Pythia 8 Angantyr.

## Main results

- Pions with at least 70 % of the beam energy lie in a pseudorapidity window of width 2.77. Only its position depends on the collision energy.
- About 96 % of these pions are inside the window. The rest sit just above its upper edge, at very small transverse momentum.
- Only about one inelastic collision in 1000 makes such a pion. The generators differ on this rate by a factor of 3 to 5 in pp, and by a factor of about 9 in p+O at 9.62 TeV.
- The generators also disagree on how the rate changes from pp to p+O (ratio between 0.48 and 1.01 at 9.62 TeV).
- For the two generators tested, replacing oxygen by nitrogen changes the rate by about 5 % or less at an energy fraction above 0.5.
- Against the ATLAS data, Pythia 8 Angantyr describes the pseudorapidity distribution within 2 %. No generator describes all four measured distributions.

## Repository layout

```
.
├── README.md
├── LICENSE
├── requirements.txt
├── scripts/            all code in one flat folder (the scripts call each other by name)
├── data/hepdata/       the four ATLAS HEPData tables used (CSV)
├── results/
│   ├── m0_install_check/
│   ├── m1_window/
│   ├── m2_pp/
│   ├── m3_pO/
│   └── m4_atlas_comparison/
└── note/               LaTeX source and bibliography
```

Each folder in `results/` contains the logs and summary tables (`.log`, `.csv`, `.tex`), the merged histograms (`merged*/`, NumPy `.npz`) and a `figures/` folder with the plots (PDF and PNG).

## Scripts

| Step | Generate and analyse | Batch and plots |
|---|---|---|
| M0 | `m0_check_install.py` | |
| M1 | `m1_kinematics.py` | |
| M2 | `m2_generate.py`, `m2_merge.py` | `m2_submit.sh`, `m2_job.sh`, `m2_collect.sh`, `m2_plot.py` |
| M3 | `m3_generate.py`, `m3_merge.py`, `m3_frame_check.py` | `m3_submit.sh`, `m3_job.sh`, `m3_collect.sh`, `m3_plot.py` |
| M4 | `m4_generate.py`, `m4_merge.py`, `m4_models.py` | `m4_submit.sh`, `m4_job.sh`, `m4_collect.sh`, `m4_compare.py` |

Every script has a header that explains what it does and how to run it.

## Software and data

- Python 3.9, Chromo 0.11.0, NumPy 2.0.2, SciPy 1.13.1, Matplotlib 3.9.4 (see `requirements.txt`).
- Data: ATLAS p+O charged-particle measurement, HEPData record [ins3140590](https://www.hepdata.net/record/ins3140590), DOI [10.17182/hepdata.169575](https://doi.org/10.17182/hepdata.169575). The tables are released under CC0.
- Computing: KEKCC batch system (LSF).

## Reproducibility notes

- Chromo can start each generator only once per Python process. The scripts therefore run one process per generator, system and seed.
- The scripts are meant to be run from the folder that contains them. They create `out*/`, `merged*/` and `logs*/` next to themselves.
- The batch scripts are written for LSF (`bsub`). On other systems, adapt the `*_submit.sh` and `*_job.sh` files. The environment variables `VENV`, `WORKDIR`, `QUEUE` and `MAXSEC` set the Python environment, the working folder, the queue and the time limit per process.
- Random seeds are fixed in the submit scripts. With the same Chromo version, runs that finish all their events give the same histograms.
- Slow generators (EPOS, QGSJET III) stop after a time limit. Their event counts depend on the speed of the machine.
- The raw per-seed histogram files are not stored because of their number. The merged histograms in `results/` are enough to redo all plots and tables.
- Plots use Times New Roman if it is installed. Otherwise the scripts fall back to a similar font.

## Limitations

- The study is at generator level. It has no LHCf acceptance, resolution or reconstruction.
- Generator versions in Chromo are not identical to those used by ATLAS (CRMC 2.2.1, Pythia 8.308).
- The Pythia forward tune is the Monash-based set of Fieg et al. The main forward tune and the ATLAS A3 tune are not included.
- Slow generators have small event counts, so their rates at the highest energy fractions have larger errors.
- HIJING is not available in Chromo.

## References

- C. E. Leitgeb et al., *Probing the cosmic ray background of gamma-ray astronomy with hadron colliders*, Phys. Rev. D 112, 083042 (2025), [doi:10.1103/djx4-qs8m](https://doi.org/10.1103/djx4-qs8m).
- ATLAS Collaboration, *Measurement of charged-particle production in sqrt(s_NN) = 9.62 TeV proton-oxygen collisions as a probe of cosmic-ray air showers with the ATLAS detector*, Phys. Rev. Lett. 137, 121901 (2026), [doi:10.1103/f3nk-5lt9](https://doi.org/10.1103/f3nk-5lt9).
- A. Fedynitch, H. Dembinski, A. Prosekin, *Chromo*, [arXiv:2507.21856](https://arxiv.org/abs/2507.21856).
- LHCf Collaboration (O. Adriani et al.), Phys. Rev. D 86, 092001 (2012), [doi:10.1103/PhysRevD.86.092001](https://doi.org/10.1103/PhysRevD.86.092001).

The full bibliography is in `note/references.bib`.

## License

Code: MIT License (see `LICENSE`). The HEPData tables are CC0.
