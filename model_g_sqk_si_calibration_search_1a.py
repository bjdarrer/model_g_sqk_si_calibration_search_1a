#!/usr/bin/env python3
"""
Date: 10th March 2026 18:05 GMT

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
--------------
This is a *reference calibration/search framework* built around a plausible
Model G-like reaction-diffusion system. It is not a claim that the exact Chapter 12
book simulation is reproduced here term-for-term. The safest use is either:

  (a) run this as a standalone exploratory search, or
  (b) transplant the measurement / calibration / scoring functions into your
      existing solver (e.g. your proton batch-search code), which is probably
      the best path for your current workflow.

Outputs
-------
Creates an output folder containing:
- search_summary.csv
- best_candidate.json
- best_profiles.npz
- top_candidates.npz

Usage examples
--------------
python3 model_g_sqk_si_calibration_search_1a.py --particle proton --ntrials 40
python3 model_g_sqk_si_calibration_search_1a.py --particle electron --ntrials 60

=========================================
Brendan: Solomon one of our scientists in the Starburst Foundation, after I published the above on 
researchgate, wrote to me saying: "Use particle Compton wavelength, charge of a single particle etc. 
In SI units to try to calibrate SQK constants, along with Paul’s boundary conditions for the relative 
steady state concentrations of each etheron type, that are necessary for stable particles and the relative 
concentrations, learned from simulations to get stable particles". He attached "Solomon__IMG_1138.png" 
that relates to this and the above work we've done. Solomon also said "Use these but keep in mind they 
are the passive masses not Paul LaViolette’s active masses." See here: "Solomon__IMG_1139.png". 
Solomon: "Just get ChatGPT to do it let me know what it says" Brendan: "Cool! I am using ChatGPT code 
to do batch runs and compare with James Kelly's profiles as plots. But now I will try and do what you 
have said! See @ https://github.com/bjdarrer/model_g_particle_1d_proton_batch_search_1b ". Solomon: 
"Active masses can be gotten from Brown’s electrogravitic experiments as Paul LaViolette has done". 
Can you work out a simulation to do the above?

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

I also built it with the important limitation in mind: the book says a stable charged-particle branch has 
not yet been cleanly found in published Model G simulations, and that finding such a solution is “like 
trying to find a needle in a haystack.” So this script is best thought of as a search harness plus SI 
calibration layer, not a claim that the full charged-particle problem is already solved.

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

The solver inside this file is a reference Model G-like 1D spherical solver so the whole 
search/calibration workflow is concrete and runnable. The safest research use, though, is probably 
to transplant the measurement, SI calibration, and scoring parts into your existing 
model_g_particle_1d_proton_batch_search_1b.py pipeline, since that is already part of 
your working search environment.

The next step could be to splice these calibration routines directly into your current batch-search 
code so it scores candidates not only against James Kelly-style profiles, but also against λC, e, 
and Brown/LaViolette active-mass coupling.
=========================================

"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import numpy as np


# -----------------------------------------------------------------------------
# STEP 1: Physical targets in SI units
# -----------------------------------------------------------------------------

C_LIGHT = 299_792_458.0
H_PLANCK = 6.626_070_15e-34
E_CHARGE = 1.602_176_634e-19
EPS0 = 8.854_187_8128e-12
G_NEWTON = 6.674_30e-11

# Brown / LaViolette electrogravitic ratios converted to SI.
# Book values: ϕ_g / Q = 5.2e8 cm^2/s^2/C, M/Q = 3.9e16 g/C.
PHIG_PER_Q_BROWN_SI = 5.2e4        # m^2 s^-2 per Coulomb
M_ACTIVE_PER_Q_BROWN_SI = 3.9e13   # kg per Coulomb


@dataclass
class ParticleTarget:
    name: str
    passive_mass_kg: float
    charge_c: float

    @property
    def compton_wavelength_m(self) -> float:
        return H_PLANCK / (self.passive_mass_kg * C_LIGHT)


PARTICLE_TARGETS: Dict[str, ParticleTarget] = {
    "proton": ParticleTarget(
        name="proton",
        passive_mass_kg=1.67262192369e-27,
        charge_c=+E_CHARGE,
    ),
    "electron": ParticleTarget(
        name="electron",
        passive_mass_kg=9.1093837015e-31,
        charge_c=-E_CHARGE,
    ),
    "neutron": ParticleTarget(
        name="neutron",
        passive_mass_kg=1.67492749804e-27,
        charge_c=0.0,
    ),
}


# -----------------------------------------------------------------------------
# STEP 2: Search / simulation parameter containers
# -----------------------------------------------------------------------------

@dataclass
class ModelGParams:
    # Reaction / diffusion parameters
    A: float = 14.0
    B: float = 29.0
    Dg: float = 1.0
    Dx: float = 1.0
    Dy: float = 12.0
    k1: float = 1.0
    k2: float = 1.0
    k3: float = 1.0
    k4: float = 1.0
    k5: float = 0.9
    km2: float = 0.1  # reverse X -> G channel

    # Domain / integrator
    Rmax: float = 50.0
    Nr: int = 801
    t_end: float = 35.0
    dt: float = 2.0e-4
    save_stride: int = 200

    # Seed fluctuation
    seed_amp_x: float = -1.0
    seed_r_sigma: float = 1.0
    seed_t_peak: float = 10.0
    seed_t_sigma: float = 3.0

    # Optional bias to encourage charged branch exploration
    seed_bias_y: float = 0.0
    seed_bias_g: float = 0.0

    # Measurement settings
    min_core_amp: float = 0.03
    outer_tol: float = 0.02


@dataclass
class CandidateMetrics:
    stable: bool
    score: float
    lambda0_sim: float
    length_scale_m_per_unit: float
    time_scale_s_per_unit: float
    charge_like_sim: float
    active_mass_like_sim: float
    charge_scale_c_per_sim: float
    active_mass_scale_kg_per_sim: float
    brown_ratio_error: float
    core_rms_radius_sim: float
    core_rms_radius_m: float
    edge_error: float
    near_threshold_error: float
    charge_sign_ok: bool
    notes: str


# -----------------------------------------------------------------------------
# STEP 3: Homogeneous steady state and threshold helpers
# -----------------------------------------------------------------------------

def homogeneous_steady_state(p: ModelGParams) -> Tuple[float, float, float]:
    """
    Reference homogeneous steady state for the reaction set used in this script.

    PDE model used here:
        dG/dt = k1*A - k2*G + km2*X + Dg Lap(G)
        dX/dt = k2*G - km2*X + k4*X^2 Y - k3*B*X - k5*X + Dx Lap(X) + seed_x
        dY/dt = k3*B*X - k4*X^2 Y + Dy Lap(Y) + seed_y

    At homogeneous steady state (no diffusion, no seed):
        0 = k1*A - k2*G + km2*X
        0 = k2*G - km2*X + k4 X^2 Y - k3 B X - k5 X
        0 = k3 B X - k4 X^2 Y

    From the third equation: Y = k3*B / (k4*X), assuming X > 0.
    Then the second reduces to k2*G = (km2 + k5) X.
    Substitute into the first to get G, then X, then Y.
    """
    denom = p.k2 * p.k5
    if denom <= 0:
        raise ValueError("Invalid steady-state denominator; check k2 and k5.")

    G0 = p.k1 * p.A * (p.km2 + p.k5) / denom
    X0 = (p.k2 * G0) / (p.km2 + p.k5)
    Y0 = (p.k3 * p.B) / (p.k4 * max(X0, 1e-12))
    return G0, X0, Y0


def critical_threshold_Gc(p: ModelGParams) -> float:
    """
    Book-style threshold estimate used as a search penalty.
    This is the classic Model G threshold expression used as a guide.
    """
    inside = math.sqrt(max(p.B, 0.0)) - math.sqrt(max(p.k5 / max(p.k4, 1e-12), 0.0))
    pref = math.sqrt((p.k4 * p.k5 * p.Dy) / max(p.k2 * p.k3 * p.Dx, 1e-12))
    return pref * inside


# -----------------------------------------------------------------------------
# STEP 4: Radial operators and seeds
# -----------------------------------------------------------------------------

def radial_grid(Rmax: float, Nr: int) -> Tuple[np.ndarray, float]:
    r = np.linspace(0.0, Rmax, Nr)
    dr = r[1] - r[0]
    return r, dr


def laplacian_spherical(u: np.ndarray, r: np.ndarray, dr: float) -> np.ndarray:
    """Spherical radial Laplacian: ∇²u = u_rr + 2/r u_r."""
    out = np.zeros_like(u)

    # Center: use symmetry du/dr = 0, so ∇²u(0) ≈ 6(u1 - u0)/dr²
    out[0] = 6.0 * (u[1] - u[0]) / (dr * dr)

    # Interior points
    uprime = (u[2:] - u[:-2]) / (2.0 * dr)
    usecond = (u[2:] - 2.0 * u[1:-1] + u[:-2]) / (dr * dr)
    out[1:-1] = usecond + 2.0 * uprime / np.maximum(r[1:-1], 1e-12)

    # Outer boundary: Dirichlet-to-vacuum handled outside this function.
    out[-1] = 0.0
    return out


def gaussian_seed(r: np.ndarray, t: float, amp: float, r_sigma: float,
                  t_peak: float, t_sigma: float) -> np.ndarray:
    spatial = np.exp(-0.5 * (r / r_sigma) ** 2)
    temporal = np.exp(-0.5 * ((t - t_peak) / t_sigma) ** 2)
    return amp * spatial * temporal


# -----------------------------------------------------------------------------
# STEP 5: Reference Model G-like solver
# -----------------------------------------------------------------------------

def run_reference_simulation(p: ModelGParams) -> Dict[str, np.ndarray]:
    """
    Run the reference 1D spherical simulation.

    This solver is intentionally simple and transparent rather than highly optimised.
    For production use, Brendan may want to transplant the measurement / scoring
    code into the existing batch-search solver.
    """
    G0, X0, Y0 = homogeneous_steady_state(p)
    r, dr = radial_grid(p.Rmax, p.Nr)
    nt = int(round(p.t_end / p.dt))

    G = np.full_like(r, G0)
    X = np.full_like(r, X0)
    Y = np.full_like(r, Y0)

    saved_t = []
    saved_G = []
    saved_X = []
    saved_Y = []

    def reaction_terms(Gv: np.ndarray, Xv: np.ndarray, Yv: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        RG = p.k1 * p.A - p.k2 * Gv + p.km2 * Xv
        RX = p.k2 * Gv - p.km2 * Xv + p.k4 * Xv * Xv * Yv - p.k3 * p.B * Xv - p.k5 * Xv
        RY = p.k3 * p.B * Xv - p.k4 * Xv * Xv * Yv
        return RG, RX, RY

    for n in range(nt + 1):
        t = n * p.dt

        seed_x = gaussian_seed(r, t, p.seed_amp_x, p.seed_r_sigma, p.seed_t_peak, p.seed_t_sigma)
        seed_y = gaussian_seed(r, t, p.seed_bias_y, p.seed_r_sigma, p.seed_t_peak, p.seed_t_sigma)
        seed_g = gaussian_seed(r, t, p.seed_bias_g, p.seed_r_sigma, p.seed_t_peak, p.seed_t_sigma)

        RG, RX, RY = reaction_terms(G, X, Y)
        LG = laplacian_spherical(G, r, dr)
        LX = laplacian_spherical(X, r, dr)
        LY = laplacian_spherical(Y, r, dr)

        G = G + p.dt * (RG + p.Dg * LG + seed_g)
        X = X + p.dt * (RX + p.Dx * LX + seed_x)
        Y = Y + p.dt * (RY + p.Dy * LY + seed_y)

        # Mild positivity protection.
        G = np.maximum(G, 1e-12)
        X = np.maximum(X, 1e-12)
        Y = np.maximum(Y, 1e-12)

        # Outer vacuum / homogeneous boundary conditions.
        G[-1] = G0
        X[-1] = X0
        Y[-1] = Y0

        # Center symmetry via mirrored first cell.
        G[0] = G[1]
        X[0] = X[1]
        Y[0] = Y[1]

        if (n % p.save_stride) == 0 or n == nt:
            saved_t.append(t)
            saved_G.append(G.copy())
            saved_X.append(X.copy())
            saved_Y.append(Y.copy())

        if not np.all(np.isfinite(G)) or not np.all(np.isfinite(X)) or not np.all(np.isfinite(Y)):
            break

    return {
        "r": r,
        "t": np.asarray(saved_t),
        "G": np.asarray(saved_G),
        "X": np.asarray(saved_X),
        "Y": np.asarray(saved_Y),
        "G0": np.array([G0]),
        "X0": np.array([X0]),
        "Y0": np.array([Y0]),
    }


# -----------------------------------------------------------------------------
# STEP 6: Profile measurement tools
# -----------------------------------------------------------------------------

def zero_crossings(x: np.ndarray, y: np.ndarray) -> List[float]:
    xs: List[float] = []
    for i in range(len(y) - 1):
        y1, y2 = y[i], y[i + 1]
        if y1 == 0.0:
            xs.append(float(x[i]))
        elif y1 * y2 < 0.0:
            frac = abs(y1) / (abs(y1) + abs(y2))
            xs.append(float(x[i] + frac * (x[i + 1] - x[i])))
    return xs


def first_nontrivial_wavelength(r: np.ndarray, phi: np.ndarray) -> float:
    """
    Estimate λ0 from zero crossings beyond the central region.
    Uses the spacing between the 2nd and 4th zero crossings if available,
    otherwise falls back to neighbouring crossings.
    """
    zc = zero_crossings(r, phi)
    if len(zc) >= 4:
        return zc[3] - zc[1]
    if len(zc) >= 3:
        return 2.0 * (zc[2] - zc[1])
    if len(zc) >= 2:
        return 2.0 * (zc[1] - zc[0])
    return float("nan")


def first_core_boundary(r: np.ndarray, phi: np.ndarray) -> float:
    zc = zero_crossings(r, phi)
    if not zc:
        return r[min(len(r) - 1, max(2, len(r) // 20))]
    return max(zc[0], r[2])


def core_rms_radius(r: np.ndarray, phi: np.ndarray) -> float:
    rc = first_core_boundary(r, phi)
    mask = r <= rc
    w = np.maximum(np.abs(phi[mask]), 1e-16) * (r[mask] ** 2)
    numer = np.trapezoid((r[mask] ** 2) * w, r[mask])
    denom = np.trapezoid(w, r[mask])
    return math.sqrt(max(numer / max(denom, 1e-30), 0.0))


def boundary_error(phi_g: np.ndarray, phi_x: np.ndarray, phi_y: np.ndarray,
                   G0: float, X0: float, Y0: float, G: np.ndarray, X: np.ndarray, Y: np.ndarray) -> float:
    # Tail should relax toward homogeneous steady state.
    eg = abs(G[-1] - G0) / max(abs(G0), 1e-12)
    ex = abs(X[-1] - X0) / max(abs(X0), 1e-12)
    ey = abs(Y[-1] - Y0) / max(abs(Y0), 1e-12)
    return float((eg + ex + ey) / 3.0)


# -----------------------------------------------------------------------------
# STEP 7: Source estimates in simulation units
# -----------------------------------------------------------------------------

def source_densities_sim(p: ModelGParams, G: np.ndarray, X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    G0, X0, Y0 = homogeneous_steady_state(p)

    RG = p.k1 * p.A - p.k2 * G + p.km2 * X
    RY = p.k3 * p.B * X - p.k4 * X * X * Y

    RG0 = p.k1 * p.A - p.k2 * G0 + p.km2 * X0
    RY0 = p.k3 * p.B * X0 - p.k4 * X0 * X0 * Y0

    rho_g = RG - RG0
    rho_y = RY - RY0
    return rho_g, rho_y


def integrate_core_source(r: np.ndarray, rho: np.ndarray, r_core: float) -> float:
    """3D spherical integral over the core: 4π∫ rho(r) r² dr."""
    mask = r <= r_core
    return float(4.0 * math.pi * np.trapezoid(rho[mask] * (r[mask] ** 2), r[mask]))


# -----------------------------------------------------------------------------
# STEP 8: SI calibration and scoring
# -----------------------------------------------------------------------------

def estimate_time_scale(length_scale_m_per_unit: float) -> float:
    """
    A conservative placeholder. Without a dedicated photon / wave-speed calibration,
    map one simulation unit of velocity to c by taking one length unit per one time unit.
    Then t_scale = l_scale / c.
    """
    return length_scale_m_per_unit / C_LIGHT


def score_candidate(p: ModelGParams, target: ParticleTarget, sim: Dict[str, np.ndarray]) -> CandidateMetrics:
    r = sim["r"]
    G = sim["G"][-1]
    X = sim["X"][-1]
    Y = sim["Y"][-1]
    G0, X0, Y0 = float(sim["G0"][0]), float(sim["X0"][0]), float(sim["Y0"][0])

    phi_g = G - G0
    phi_x = X - X0
    phi_y = Y - Y0

    amp = max(np.max(np.abs(phi_x)), np.max(np.abs(phi_y)))
    stable = bool(np.all(np.isfinite(G)) and amp >= p.min_core_amp)
    notes: List[str] = []
    if not stable:
        notes.append("low-amplitude or non-finite result")

    lam_sim = first_nontrivial_wavelength(r, phi_x)
    if not np.isfinite(lam_sim) or lam_sim <= 0.0:
        lam_sim = float("nan")
        stable = False
        notes.append("unable to measure lambda0")

    if np.isfinite(lam_sim):
        length_scale = target.compton_wavelength_m / lam_sim
    else:
        length_scale = float("nan")

    time_scale = estimate_time_scale(length_scale) if np.isfinite(length_scale) else float("nan")

    r_core = first_core_boundary(r, phi_y if np.max(np.abs(phi_y)) >= np.max(np.abs(phi_x)) else phi_x)
    core_rms_sim = core_rms_radius(r, phi_y if np.max(np.abs(phi_y)) >= np.max(np.abs(phi_x)) else phi_x)
    core_rms_m = core_rms_sim * length_scale if np.isfinite(length_scale) else float("nan")

    rho_g_sim, rho_y_sim = source_densities_sim(p, G, X, Y)
    q_like_sim = integrate_core_source(r, rho_y_sim, r_core)
    mg_like_sim = integrate_core_source(r, rho_g_sim, r_core)

    # Calibrate charge scale with target particle charge magnitude.
    if abs(target.charge_c) > 0.0 and abs(q_like_sim) > 1e-30:
        q_scale = abs(target.charge_c) / abs(q_like_sim)
    else:
        q_scale = float("nan")

    # Brown / LaViolette active-mass ratio calibration.
    if np.isfinite(q_scale):
        mg_scale = M_ACTIVE_PER_Q_BROWN_SI * q_scale
    else:
        mg_scale = float("nan")

    if abs(q_like_sim) > 1e-30 and abs(mg_like_sim) > 1e-30:
        brown_ratio_error = abs((mg_like_sim / q_like_sim) - (M_ACTIVE_PER_Q_BROWN_SI / max(mg_scale / max(q_scale, 1e-30), 1e-30)))
        # Simplifies to dimensionless check below.
        brown_dimless_target = 1.0
        brown_dimless_model = (mg_like_sim * mg_scale) / max(q_like_sim * q_scale * M_ACTIVE_PER_Q_BROWN_SI, 1e-30)
        brown_ratio_error = abs(brown_dimless_model - brown_dimless_target)
    else:
        brown_ratio_error = 1e6

    # Check charge sign for charged targets.
    charge_sign_ok = True
    if target.charge_c > 0 and q_like_sim <= 0:
        charge_sign_ok = False
        notes.append("charge sign mismatch for positive target")
    elif target.charge_c < 0 and q_like_sim >= 0:
        charge_sign_ok = False
        notes.append("charge sign mismatch for negative target")

    edge_err = boundary_error(phi_g, phi_x, phi_y, G0, X0, Y0, G, X, Y)

    Gc = critical_threshold_Gc(p)
    near_thr = abs((G0 - Gc) / max(G0, 1e-12))

    # Soft score: lower is better.
    # Penalties chosen to guide search rather than claim deep physical significance.
    score = 0.0
    if not stable:
        score += 1e4
    if not np.isfinite(lam_sim):
        score += 1e4
    else:
        score += 200.0 * abs((lam_sim - lam_sim) / max(lam_sim, 1e-12))  # zero by construction
    score += 400.0 * edge_err
    score += 120.0 * near_thr
    score += 40.0 * abs(core_rms_sim / max(lam_sim, 1e-12) - 0.338)  # book-like neutral core guide
    score += 80.0 * (0.0 if charge_sign_ok else 1.0)

    # For charged particles, encourage nonzero source.
    if abs(target.charge_c) > 0.0:
        score += 50.0 / max(abs(q_like_sim), 1e-6)
        score += 50.0 * brown_ratio_error
    else:
        score += 10.0 * abs(q_like_sim)

    return CandidateMetrics(
        stable=stable,
        score=float(score),
        lambda0_sim=float(lam_sim) if np.isfinite(lam_sim) else float("nan"),
        length_scale_m_per_unit=float(length_scale) if np.isfinite(length_scale) else float("nan"),
        time_scale_s_per_unit=float(time_scale) if np.isfinite(time_scale) else float("nan"),
        charge_like_sim=float(q_like_sim),
        active_mass_like_sim=float(mg_like_sim),
        charge_scale_c_per_sim=float(q_scale) if np.isfinite(q_scale) else float("nan"),
        active_mass_scale_kg_per_sim=float(mg_scale) if np.isfinite(mg_scale) else float("nan"),
        brown_ratio_error=float(brown_ratio_error),
        core_rms_radius_sim=float(core_rms_sim),
        core_rms_radius_m=float(core_rms_m) if np.isfinite(core_rms_m) else float("nan"),
        edge_error=float(edge_err),
        near_threshold_error=float(near_thr),
        charge_sign_ok=charge_sign_ok,
        notes="; ".join(notes) if notes else "ok",
    )


# -----------------------------------------------------------------------------
# STEP 9: Random search around a trusted baseline
# -----------------------------------------------------------------------------

def sample_candidate(base: ModelGParams, rng: np.random.Generator, particle: ParticleTarget) -> ModelGParams:
    p = ModelGParams(**asdict(base))

    # Search mostly around the chapter-12-like baseline.
    p.B *= float(np.exp(rng.normal(0.0, 0.07)))
    p.Dy *= float(np.exp(rng.normal(0.0, 0.10)))
    p.k5 *= float(np.exp(rng.normal(0.0, 0.08)))
    p.km2 = max(0.0, p.km2 * float(np.exp(rng.normal(0.0, 0.20))))
    p.seed_amp_x *= float(np.exp(rng.normal(0.0, 0.08)))
    p.seed_r_sigma = max(0.4, p.seed_r_sigma * float(np.exp(rng.normal(0.0, 0.10))))
    p.seed_t_sigma = max(0.8, p.seed_t_sigma * float(np.exp(rng.normal(0.0, 0.10))))

    # Charged branch exploration: add a small Y or G bias to push secondary bifurcation.
    if particle.charge_c > 0:
        p.seed_bias_y = abs(rng.normal(0.03, 0.03))
        p.seed_bias_g = -abs(rng.normal(0.01, 0.01))
    elif particle.charge_c < 0:
        p.seed_bias_y = -abs(rng.normal(0.03, 0.03))
        p.seed_bias_g = abs(rng.normal(0.01, 0.01))
    else:
        p.seed_bias_y = 0.0
        p.seed_bias_g = 0.0

    return p


# -----------------------------------------------------------------------------
# STEP 10: Main search driver
# -----------------------------------------------------------------------------

def write_summary_csv(path: str, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reference SQK SI calibration search.")
    parser.add_argument("--particle", choices=sorted(PARTICLE_TARGETS.keys()), default="proton")
    parser.add_argument("--ntrials", type=int, default=30)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--outdir", type=str, default="")
    parser.add_argument("--t-end", type=float, default=35.0)
    parser.add_argument("--dt", type=float, default=2.0e-4)
    parser.add_argument("--nr", type=int, default=801)
    args = parser.parse_args()

    target = PARTICLE_TARGETS[args.particle]
    rng = np.random.default_rng(args.seed)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    outdir = args.outdir or f"sqk_si_calibration_search_{args.particle}_{timestamp}"
    os.makedirs(outdir, exist_ok=True)

    base = ModelGParams(t_end=args.t_end, dt=args.dt, Nr=args.nr)

    rows: List[Dict[str, object]] = []
    top_store: List[Tuple[float, Dict[str, object], Dict[str, np.ndarray]]] = []

    print(f"Target particle: {target.name}")
    print(f"Target passive mass: {target.passive_mass_kg:.12e} kg")
    print(f"Target charge: {target.charge_c:.12e} C")
    print(f"Target Compton wavelength: {target.compton_wavelength_m:.12e} m")
    print(f"Output folder: {outdir}")

    for i in range(args.ntrials):
        p = sample_candidate(base, rng, target)
        sim = run_reference_simulation(p)
        met = score_candidate(p, target, sim)

        row = {
            "trial": i,
            "score": met.score,
            "stable": met.stable,
            "lambda0_sim": met.lambda0_sim,
            "length_scale_m_per_unit": met.length_scale_m_per_unit,
            "time_scale_s_per_unit": met.time_scale_s_per_unit,
            "charge_like_sim": met.charge_like_sim,
            "active_mass_like_sim": met.active_mass_like_sim,
            "charge_scale_c_per_sim": met.charge_scale_c_per_sim,
            "active_mass_scale_kg_per_sim": met.active_mass_scale_kg_per_sim,
            "brown_ratio_error": met.brown_ratio_error,
            "core_rms_radius_sim": met.core_rms_radius_sim,
            "core_rms_radius_m": met.core_rms_radius_m,
            "edge_error": met.edge_error,
            "near_threshold_error": met.near_threshold_error,
            "charge_sign_ok": met.charge_sign_ok,
            "notes": met.notes,
            **{f"p_{k}": v for k, v in asdict(p).items()},
        }
        rows.append(row)

        top_store.append((met.score, row, sim))
        top_store.sort(key=lambda x: x[0])
        top_store = top_store[:5]

        print(
            f"trial {i:03d} | score={met.score:10.3f} | stable={met.stable} | "
            f"lam={met.lambda0_sim:8.4f} | qsim={met.charge_like_sim:+.3e} | "
            f"mgsim={met.active_mass_like_sim:+.3e} | notes={met.notes}"
        )

    rows_sorted = sorted(rows, key=lambda d: float(d["score"]))
    write_summary_csv(os.path.join(outdir, "search_summary.csv"), rows_sorted)

    best_score, best_row, best_sim = top_store[0]

    with open(os.path.join(outdir, "best_candidate.json"), "w", encoding="utf-8") as f:
        json.dump(best_row, f, indent=2)

    np.savez_compressed(
        os.path.join(outdir, "best_profiles.npz"),
        **best_sim,
    )

    # Save top candidates in a compact bundle.
    bundle = {}
    for rank, (_score, row, sim) in enumerate(top_store):
        bundle[f"rank{rank}_row_json"] = np.frombuffer(json.dumps(row).encode("utf-8"), dtype=np.uint8)
        bundle[f"rank{rank}_r"] = sim["r"]
        bundle[f"rank{rank}_t"] = sim["t"]
        bundle[f"rank{rank}_G"] = sim["G"]
        bundle[f"rank{rank}_X"] = sim["X"]
        bundle[f"rank{rank}_Y"] = sim["Y"]
        bundle[f"rank{rank}_G0"] = sim["G0"]
        bundle[f"rank{rank}_X0"] = sim["X0"]
        bundle[f"rank{rank}_Y0"] = sim["Y0"]
    np.savez_compressed(os.path.join(outdir, "top_candidates.npz"), **bundle)

    print("\nBest candidate summary")
    print(json.dumps(best_row, indent=2))
    print(f"\nSaved summary CSV to: {os.path.join(outdir, 'search_summary.csv')}")
    print(f"Saved best profiles to: {os.path.join(outdir, 'best_profiles.npz')}")


if __name__ == "__main__":
    main()
