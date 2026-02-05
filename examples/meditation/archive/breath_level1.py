#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Level-1 breath perception demo using pymdp and BreathEnv.

Hidden state (1 factor, 2 levels for inference):
    0: INHALE, 1: EXHALE

Environment GP now has 2 phases (INHALE, EXHALE).

Observation modality (1 modality, 2 outcomes):
    0: EXPANSION, 1: CONTRACTION

Runs a simple loop: environment emits observations; the agent infers the current
breath state using variational fixed-point iteration (VANILLA).

This module now includes two modes:
1. Fixed precision: zeta (ζ) is fixed throughout the simulation
2. Dynamic precision: zeta is updated based on prediction errors
   (implementing Parr et al. 2022 Equation B.45 for discrete models)

Note on naming convention:
    ζ (zeta) = likelihood/sensory precision (modulates A matrix)
    γ (gamma) = policy precision (modulates expected free energy)
    
This follows Parr et al. (2022) "Active Inference" Appendix B.
"""

import numpy as np

from pymdp.envs import BreathEnv
from pymdp.agent import Agent
from pymdp import utils
from pymdp.maths import (
    EPS_VAL, 
    scale_likelihood,
    update_likelihood_precision,
)


def run_session(T: int = 100, seed: int = 1, zeta: np.ndarray | float = 1.0):
    """Run a breath perception session with fixed likelihood precision.
    
    Parameters
    ----------
    T : int
        Number of timesteps
    seed : int
        Random seed
    zeta : float or np.ndarray
        Likelihood precision (ζ). Can be a scalar for constant precision
        or an array for time-varying precision.
    """
    # Build env
    env = BreathEnv(seed=seed)

    # Build agent manually
    p_correct = float(env.p_correct)
    A_base = np.array([[p_correct, 1.0 - p_correct], [1.0 - p_correct, p_correct]])

    stay_p_inhale = 1.0 - 1.0 / max(1, np.mean(env.inhale_range))
    stay_p_exhale = 1.0 - 1.0 / max(1, np.mean(env.exhale_range))

    B_base = np.zeros((2, 2, 1))
    B_base[:, 0, 0] = [stay_p_inhale, 1.0 - stay_p_inhale]
    B_base[:, 1, 0] = [1.0 - stay_p_exhale, stay_p_exhale]

    # Wrap B in object array to preserve 3D shape (avoids squeeze issue)
    B_obj = utils.obj_array(1)
    B_obj[0] = B_base

    agent = Agent(A=A_base, B=B_obj, save_belief_hist=True)

    # handle zeta schedule
    if np.isscalar(zeta):
        zeta_schedule = np.full(T, float(zeta))
    else:
        zeta_schedule = np.asarray(zeta).astype(float)
        if len(zeta_schedule) < T:
            raise ValueError(f"zeta array must have length >= T ({T}), but got {len(zeta_schedule)}")

    # Data logs
    true_states = np.zeros(T, dtype=int)
    observations = np.zeros(T, dtype=int)
    posteriors = np.zeros((T, agent.num_states[0]))

    # Reset env and get first observation
    obs = int(env.reset())
    print("A (base):\n", A_base)
    for t in range(T):
        # Store ground-truth state from env
        true_states[t] = env.state
        observations[t] = obs

        # Scale likelihood by current zeta and set on agent
        A_bar = scale_likelihood(A_base, zeta_schedule[t])
        agent.A = utils.to_obj_array(A_bar)

        # Agent inference requires list per modality: [obs_index]
        qs = agent.infer_states([obs])
        posteriors[t, :] = qs[0]

        # Advance environment
        obs = int(env.step(None))

    # Accuracy: argmax(qs) vs true state
    inferred = np.argmax(posteriors, axis=1)
    accuracy = (inferred == true_states).mean()

    print(f"Overall accuracy: {accuracy:.3f}")


    return {
        "true_states": true_states,
        "observations": observations,
        "posteriors": posteriors,
        "inferred": inferred,
        "accuracy": accuracy,     
    }


def run_session_with_precision_updating(
    T: int = 100, 
    seed: int = 1, 
    zeta_init: float = 1.0,
    log_zeta_prior_mean: float = 0.0,
    log_zeta_prior_var: float = 4.0,
    precision_learning_rate: float = 1.5,
    min_zeta: float = 0.1,
    max_zeta: float = 5.0
):
    """
    Run breath perception with dynamic precision (ζ) updating.
    
    This implements the precision belief updating from Parr et al. (2022)
    Appendix B Equation B.45, where sensory precision is updated based on 
    prediction errors.
    
    Parameters
    ----------
    T : int
        Number of timesteps
    seed : int
        Random seed
    zeta_init : float
        Initial likelihood precision (ζ) value
    log_zeta_prior_mean : float
        Prior mean for log-precision ln(ζ). Default 0.0 corresponds to ζ_prior ≈ 1.0
    log_zeta_prior_var : float
        Prior variance for log-precision. Larger = weaker regularization.
    precision_learning_rate : float
        Learning rate for precision updates. Default 1.5 for responsive dynamics.
    min_zeta : float
        Minimum allowed precision
    max_zeta : float
        Maximum allowed precision
        
    Returns
    -------
    dict : Results including precision history
    """
    # Build env
    env = BreathEnv(seed=seed)

    # Build agent manually
    p_correct = float(env.p_correct)
    A_base = np.array([[p_correct, 1.0 - p_correct], [1.0 - p_correct, p_correct]])

    stay_p_inhale = 1.0 - 1.0 / max(1, np.mean(env.inhale_range))
    stay_p_exhale = 1.0 - 1.0 / max(1, np.mean(env.exhale_range))

    B_base = np.zeros((2, 2, 1))
    B_base[:, 0, 0] = [stay_p_inhale, 1.0 - stay_p_inhale]
    B_base[:, 1, 0] = [1.0 - stay_p_exhale, stay_p_exhale]

    # Wrap B in object array to preserve 3D shape
    B_obj = utils.obj_array(1)
    B_obj[0] = B_base

    agent = Agent(A=A_base, B=B_obj, save_belief_hist=True)

    # Data logs
    true_states = np.zeros(T, dtype=int)
    observations = np.zeros(T, dtype=int)
    posteriors = np.zeros((T, agent.num_states[0]))
    zeta_history = np.zeros(T)
    prediction_errors = np.zeros(T)

    # Initialize precision and prior beliefs
    zeta = zeta_init
    prior = np.array([0.5, 0.5])  # Uniform prior at t=0
    
    # Reset env and get first observation
    obs = int(env.reset())
    
    print("Running with dynamic precision updating:")
    print(f"  zeta_init={zeta_init}, log_prior_mean={log_zeta_prior_mean}, lr={precision_learning_rate}")
    
    for t in range(T):
        # Store ground-truth state from env
        true_states[t] = env.state
        observations[t] = obs

        # 1. Update precision FIRST: compare observation to prior prediction
        #    This determines how much to trust the incoming sensory evidence
        zeta_new, pe, _ = update_likelihood_precision(
            zeta=zeta,
            A=A_base,  # Use base A for computing prediction error
            obs=obs,
            qs=prior,  # Use PRIOR beliefs for prediction error
            log_zeta_prior_mean=log_zeta_prior_mean,
            log_zeta_prior_var=log_zeta_prior_var,
            zeta_step=precision_learning_rate,
            min_zeta=min_zeta,
            max_zeta=max_zeta
        )
        prediction_errors[t] = pe
        zeta = zeta_new
        zeta_history[t] = zeta

        # 2. State inference: use updated precision to weight likelihood
        A_bar = scale_likelihood(A_base, zeta)
        agent.A = utils.to_obj_array(A_bar)
        qs = agent.infer_states([obs])
        posteriors[t, :] = qs[0]

        # 3. Prepare for next timestep: prior = B @ posterior
        prior = B_base[:, :, 0] @ qs[0]
        obs = int(env.step(None))

    # Accuracy: argmax(qs) vs true state
    inferred = np.argmax(posteriors, axis=1)
    accuracy = (inferred == true_states).mean()

    print(f"Overall accuracy: {accuracy:.3f}")
    print(f"Final precision (zeta): {zeta_history[-1]:.3f}")
    print(f"Mean prediction error: {prediction_errors.mean():.4f}")

    return {
        "true_states": true_states,
        "observations": observations,
        "posteriors": posteriors,
        "inferred": inferred,
        "accuracy": accuracy,
        "zeta_history": zeta_history,
        "prediction_errors": prediction_errors,
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Level-1 breath perception demo")
    parser.add_argument("--T", type=int, default=100, help="Number of timesteps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--mode", choices=["fixed", "dynamic"], default="fixed",
                        help="Precision mode: 'fixed' or 'dynamic'")
    parser.add_argument("--zeta", type=float, default=1.0, 
                        help="Precision value (fixed mode) or initial precision (dynamic mode)")
    parser.add_argument("--log-prior-mean", type=float, default=0.0,
                        help="Prior mean for log-precision (dynamic mode)")
    parser.add_argument("--log-prior-var", type=float, default=4.0,
                        help="Prior variance for log-precision (dynamic mode)")
    parser.add_argument("--lr", type=float, default=1.5,
                        help="Learning rate for precision updates (dynamic mode)")
    
    args = parser.parse_args()
    
    if args.mode == "fixed":
        run_session(T=args.T, seed=args.seed, zeta=args.zeta)
    else:
        run_session_with_precision_updating(
            T=args.T, 
            seed=args.seed, 
            zeta_init=args.zeta,
            log_zeta_prior_mean=args.log_prior_mean,
            log_zeta_prior_var=args.log_prior_var,
            precision_learning_rate=args.lr
        )

