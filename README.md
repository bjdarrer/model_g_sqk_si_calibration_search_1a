Date: 10th March 2026 18:05 GMT.

model_g_sqk_si_calibration_search_1a.py

Reference search / calibration harness for Model G-style 1D spherical simulations.

- Written by Brendan Darrer and Solomon Appavoo aided by ChatGPT 5.4 (extended thinking)
- adapted from: @ https://github.com/blue-science/subquantum_kinetics/blob/master/particle.nb and
"model_g_particle_1d_safe__1d.py" etc.
- with ChatGPT 5.4 writing it and Brendan guiding it to produce a clean code.

Purpose
-------
This script is built for the workflow Brendan described:

1. search for stable Model G particle-like solutions,
2. measure the emergent Turing wavelength λ0 in simulation units,
3. map λ0 to the target particle Compton wavelength in SI units,
4. estimate charge-like and active-mass-like source strengths from reaction imbalances,
5. compare those against the elementary charge and Brown / LaViolette
   electrogravitic coupling ratios,
6. rank candidate parameter sets.

Important note
..............

This is a *reference calibration/search framework* built around a plausible

Model G-like reaction-diffusion system. It is not a claim that the exact Chapter 12

book simulation is reproduced here term-for-term. The safest use is either:

  (a) run this as a standalone exploratory search, or
  
  (b) transplant the measurement / calibration / scoring functions into your
  
      existing solver (e.g. your proton batch-search code), which is probably
      
      the best path for your current workflow.

Outputs

.......

Creates an output folder containing:

- search_summary.csv
- best_candidate.json
- best_profiles.npz
- top_candidates.npz

Usage examples

..............

python3 model_g_sqk_si_calibration_search_1a.py --particle proton --ntrials 40

python3 model_g_sqk_si_calibration_search_1a.py --particle electron --ntrials 60

.............................
Brendan: Solomon one of our scientists in the Starburst Foundation, after I published the above on

researchgate, wrote to me saying: "Use particle Compton wavelength, charge of a single particle etc. 

In SI units to try to calibrate SQK constants, along with Paul’s boundary conditions for the relative steady

state concentrations of each etheron type, that are necessary for stable particles and the relative 

concentrations, learned from simulations to get stable particles". He attached "Solomon__IMG_1138.png"

that relates to this and the above work we've done. Solomon also said "Use these but keep in mind they are 

the passive masses not Paul LaViolette’s active masses." See here: "Solomon__IMG_1139.png". Solomon: "Just 

get ChatGPT to do it let me know what it says" Brendan: "Cool! I am using ChatGPT code to do batch runs 

and compare with James Kelly's profiles as plots. But now I will try and do what you have said! See 

@ https://github.com/bjdarrer/model_g_particle_1d_proton_batch_search_1b ". Solomon: "Active masses can 

be gotten from Brown’s electrogravitic experiments as Paul LaViolette has done". Can you work out a 

simulation to do the above?

Thought for 13m 57s

ChatGPT 5.4 (extended thinking):

Yes — I worked out a practical search/calibration simulation for that workflow.

The key idea is to treat this as an inverse-calibration problem:

search for a stable particle-like Model G solution near LaViolette’s marginal-stability regime, using the 

book’s constraints 

G0 > Gc and (G0−Gc)/G0 ≪ 1, with chapter-12-like baseline parameters as the starting point, measure the 

emergent Turing wavelength λ0 in simulation units and map it to the target particle’s Compton wavelength 

in SI, since LaViolette explicitly identifies the particle Turing wavelength with the Compton wavelength, 

estimate a charge-like source from the local Y-source imbalance in the core and an active-mass-like source 

from the G-production imbalance, then calibrate those against the elementary charge and Brown/LaViolette’s 

electrogravitic coupling ratio M/Q.

I also built it with the important limitation in mind: the book says a stable charged-particle branch has not 

yet been cleanly found in published Model G simulations, and that finding such a solution is “like trying to 

find a needle in a haystack.” So this script is best thought of as a search harness plus SI calibration layer, 

not a claim that the full charged-particle problem is already solved.

Download the script

model_g_sqk_si_calibration_search_1a.py

What it does

It starts from a chapter-12-style baseline close to the one described in the book —

A=14, B=29, Dg=1, Dx=1, Dy=12, forward ki=1 except k5=0.9, reverse k−2=0.1, 

and vacuum-style outer boundary conditions — then randomly perturbs around that neighbourhood while 

scoring candidates for stability, outer-boundary relaxation, near-threshold behaviour, measured λ0, 

core size, charge-sign consistency, and Brown-ratio consistency.

Outputs

It writes:

search_summary.csv

best_candidate.json

best_profiles.npz

top_candidates.npz

So you can inspect the best parameter sets and reuse the resulting profiles in your other tools.

How to run

python3 model_g_sqk_si_calibration_search_1a.py --particle proton --ntrials 40

or, for an electron-style search:

python3 model_g_sqk_si_calibration_search_1a.py --particle electron --ntrials 60

Important note

The solver inside this file is a reference Model G-like 1D spherical solver so the whole search/calibration 

workflow is concrete and runnable. The safest research use, though, is probably to transplant the measurement, 

SI calibration, and scoring parts into your existing model_g_particle_1d_proton_batch_search_1b.py pipeline, 

since that is already part of your working search environment.

The next step could be to splice these calibration routines directly into your current batch-search code 

so it scores candidates not only against James Kelly-style profiles, but also against λC, e, and 

Brown/LaViolette active-mass coupling.
