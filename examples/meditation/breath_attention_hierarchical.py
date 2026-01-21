#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Hierarchical Breath Perception with Bidirectional Attention Coupling

Two-level model:
    Level 1: Breath perception with dynamic likelihood precision (ζ)
    Level 2: Attention state inference (focused vs. distracted)

Bidirectional coupling:
    DESCENDING: Attention beliefs → precision prior (Bayesian model average)
    ASCENDING:  Precision dynamics → discrete observation → attention update

Figure 3: Hierarchical breath perception with attention modulation
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pymdp.envs import BreathEnv
from pymdp.maths import (
    EPS_VAL,
    scale_likelihood,
    update_likelihood_precision,
)
from pymdp.control import update_posterior_policies
from pymdp import utils
from pymdp.learning import update_obs_likelihood_dirichlet, update_state_likelihood_dirichlet


# =============================================================================
# Constants
# =============================================================================

# Breath states (Level 1)
INHALE = 0
EXHALE = 1

# Attention states (Level 2)
FOCUSED = 0
DISTRACTED = 1

# Precision observations (ascending message to attention)
HIGH_PRECISION = 0
LOW_PRECISION = 1

# Awareness states (Level 2 - metacognitive)
AWARE = 0
NOT_AWARE = 1

# Awareness observations (ascending message from breath entropy)
LOW_ENTROPY = 0   # Confident breath beliefs → aware
HIGH_ENTROPY = 1  # Uncertain breath beliefs → not aware

# Attention actions (Level 2 - for Figure 5+)
STAY = 0
SWITCH = 1


# =============================================================================
# Helper Functions
# =============================================================================

def entropy_dist(p):
    """Compute entropy of a probability distribution (not a matrix)."""
    p = np.clip(p, EPS_VAL, 1 - EPS_VAL)
    return -np.sum(p * np.log(p))


def bayesian_update(A, obs, prior):
    """Simple Bayesian state inference."""
    likelihood = A[obs, :]
    posterior = likelihood * prior
    return posterior / (posterior.sum() + EPS_VAL)


# =============================================================================
# Main Simulation
# =============================================================================

def run_hierarchical_breath(
    T: int = 200,
    seed: int = 42,
    # Level 1 likelihood precision (baseline A1 matrix)
    A1_precision: float = 0.75,  # P(correct obs | state) - lower = noisier
    # Precision parameters for each attention state (m=2 multiplicative symmetry)
    zeta_focused: float = 2.0,
    zeta_distracted: float = 0.5,
    # Precision update parameters (B.45)
    zeta_step: float = 0.25,  # Step size for precision updates
    log_zeta_prior_var: float = 2.0,  # Variance of log-zeta prior
    # A2: Likelihood of precision observations given attention state
    A2_precision: float = 0.9,  # P(high_prec | focused) = P(low_prec | distracted)
    # Precision threshold for binning
    zeta_threshold: float = 1.0,
    # B2: Attention dynamics (how stable is attention?)
    attention_stay_prob: float = 0.95,
    # Initial priors
    initial_p_focused: float = 0.5,  # Initial prior P(focused)
    # Noise injection
    noise_start: int = None,
    noise_end: int = None,
    # Fixed attention mode (bypass attention inference)
    fixed_attention: str = None,  # None, "focused", or "distracted"
):
    """
    Run hierarchical breath perception with bidirectional attention coupling.
    
    Parameters
    ----------
    T : int
        Number of timesteps
    A1_precision : float
        Baseline precision of Level 1 likelihood (P(correct obs | state))
        Lower values = noisier observations, slower inference
    zeta_focused : float
        Precision associated with focused attention state
    zeta_distracted : float
        Precision associated with distracted attention state
    A2_precision : float
        How reliably precision observations map to attention states
    zeta_threshold : float
        Threshold for binning precision into high/low observations
    attention_stay_prob : float
        Probability of staying in current attention state (B2 diagonal)
    """
    np.random.seed(seed)
    
    # =========================================================================
    # Level 1: Breath Perception Setup
    # =========================================================================
    env = BreathEnv(seed=seed)
    
    # Use the provided A1 precision (can be lower than env default for noisier obs)
    p_correct = A1_precision
    A1 = np.array([
        [p_correct, 1 - p_correct],
        [1 - p_correct, p_correct]
    ])
    
    # Transition dynamics from breath environment
    stay_p_inhale = 1.0 - 1.0 / max(1, np.mean(env.inhale_range))
    stay_p_exhale = 1.0 - 1.0 / max(1, np.mean(env.exhale_range))
    B1 = np.array([
        [stay_p_inhale, 1 - stay_p_exhale],
        [1 - stay_p_inhale, stay_p_exhale]
    ])
    
    # =========================================================================
    # Level 2: Attention Setup
    # =========================================================================
    
    # A2: P(precision_obs | attention_state)
    #     Rows: HIGH_PRECISION, LOW_PRECISION
    #     Cols: FOCUSED, DISTRACTED
    A2 = np.array([
        [A2_precision, 1 - A2_precision],      # P(high_prec | state)
        [1 - A2_precision, A2_precision]       # P(low_prec | state)
    ])
    
    # B2: Attention transition dynamics
    B2 = np.array([
        [attention_stay_prob, 1 - attention_stay_prob],  # to FOCUSED
        [1 - attention_stay_prob, attention_stay_prob]   # to DISTRACTED
    ])
    
    # Precision parameters for each attention state
    zeta_by_state = np.array([zeta_focused, zeta_distracted])
    
    # =========================================================================
    # Data Logs
    # =========================================================================
    # Level 1
    true_breath = np.zeros(T, dtype=int)
    posterior_breath = np.zeros((T, 2))
    zeta_history = np.zeros(T)
    zeta_prior_history = np.zeros(T)  # Descending message
    is_noise = np.zeros(T, dtype=bool)  # Track noise periods
    
    # Level 2
    posterior_attention = np.zeros((T, 2))
    precision_obs_history = np.zeros(T, dtype=int)
    
    # =========================================================================
    # Initialize
    # =========================================================================
    # Level 1
    qs_breath = np.array([0.5, 0.5])  # Belief about breath state
    zeta = 1.0  # Current precision estimate
    obs_breath = int(env.reset())
    
    # Level 2
    qs_attention = np.array([initial_p_focused, 1 - initial_p_focused])  # Initial attention prior
    
    print(f"Hierarchical Breath Perception")
    print(f"  T={T}")
    print(f"  A1 precision={A1_precision}")
    print(f"  ζ_focused={zeta_focused}, ζ_distracted={zeta_distracted}")
    print(f"  A2 precision={A2_precision}")
    print(f"  Attention stay prob={attention_stay_prob}")
    if fixed_attention is not None:
        fixed_zeta = zeta_focused if fixed_attention == "focused" else zeta_distracted
        print(f"  FIXED ATTENTION: {fixed_attention} (ζ_prior = {fixed_zeta})")
    if noise_start is not None:
        print(f"  Noise injection: t={noise_start}-{noise_end}")
    print()
    
    # =========================================================================
    # Main Loop
    # =========================================================================
    for t in range(T):
        # Store true state
        true_breath[t] = env.state
        
        # =====================================================================
        # NOISE INJECTION: Random observations during noise period
        # =====================================================================
        if noise_start is not None and noise_start <= t < noise_end:
            obs_to_process = np.random.choice([0, 1])
            is_noise[t] = True
        else:
            obs_to_process = obs_breath
        
        # =====================================================================
        # DESCENDING MESSAGE: Attention → Precision Prior
        # =====================================================================
        if fixed_attention is not None:
            # Fixed attention mode: bypass attention inference
            zeta_prior = zeta_focused if fixed_attention == "focused" else zeta_distracted
        else:
            # Bayesian model average: expected precision given attention beliefs
            zeta_prior = np.sum(qs_attention * zeta_by_state)
        zeta_prior_history[t] = zeta_prior
        
        # =====================================================================
        # LEVEL 1: Precision Update (B.45) with attention-informed prior
        # =====================================================================
        # The key: B.45 updates FROM the attention-informed prior
        zeta, pe, _ = update_likelihood_precision(
            zeta=zeta_prior,  # ← Descending message sets starting point!
            A=A1,
            obs=obs_to_process,  # Use potentially noisy observation
            qs=qs_breath,
            log_zeta_prior_var=log_zeta_prior_var,
            zeta_step=zeta_step
        )
        zeta_history[t] = zeta
        
        # =====================================================================
        # LEVEL 1: Breath State Inference
        # =====================================================================
        A1_scaled = scale_likelihood(A1, zeta)
        qs_breath = bayesian_update(A1_scaled, obs_to_process, qs_breath)  # Use potentially noisy observation
        posterior_breath[t] = qs_breath
        
        # =====================================================================
        # ASCENDING MESSAGE: Precision → Discrete Observation
        # =====================================================================
        # Bin precision into discrete observation for Level 2
        if zeta > zeta_threshold:
            obs_precision = HIGH_PRECISION
        else:
            obs_precision = LOW_PRECISION
        precision_obs_history[t] = obs_precision
        
        # =====================================================================
        # LEVEL 2: Attention Inference
        # =====================================================================
        if fixed_attention is not None:
            # Fixed attention: set posterior to deterministic state
            if fixed_attention == "focused":
                qs_attention = np.array([1.0, 0.0])
            else:
                qs_attention = np.array([0.0, 1.0])
        else:
            # Update attention beliefs given precision observation
            qs_attention = bayesian_update(A2, obs_precision, qs_attention)
        posterior_attention[t] = qs_attention
        
        # =====================================================================
        # Advance to next timestep
        # =====================================================================
        # Predict next breath state
        qs_breath = B1 @ qs_breath
        
        # Predict next attention state (skip if fixed)
        if fixed_attention is None:
            qs_attention = B2 @ qs_attention
        
        # Get next observation from environment
        obs_breath = int(env.step(None))
    
    # =========================================================================
    # Summary
    # =========================================================================
    breath_acc = (np.argmax(posterior_breath, axis=1) == true_breath).mean()
    mean_p_focused = posterior_attention[:, FOCUSED].mean()
    
    print("Results:")
    print(f"  Breath inference accuracy: {breath_acc:.3f}")
    print(f"  Mean P(focused): {mean_p_focused:.3f}")
    print(f"  Mean ζ prior (descending): {zeta_prior_history.mean():.3f}")
    print(f"  Mean ζ posterior: {zeta_history.mean():.3f}")
    
    return {
        "true_breath": true_breath,
        "posterior_breath": posterior_breath,
        "zeta_history": zeta_history,
        "zeta_prior_history": zeta_prior_history,
        "posterior_attention": posterior_attention,
        "precision_obs_history": precision_obs_history,
        "zeta_focused": zeta_focused,
        "zeta_distracted": zeta_distracted,
        "zeta_step": zeta_step,
        "log_zeta_prior_var": log_zeta_prior_var,
        "zeta_threshold": zeta_threshold,
        "is_noise": is_noise,
        "noise_start": noise_start,
        "noise_end": noise_end,
        "fixed_attention": fixed_attention,
        "initial_p_focused": initial_p_focused,
    }


# =============================================================================
# Figure 4: Hierarchical Breath with Awareness
# =============================================================================

def run_hierarchical_with_awareness(
    T: int = 200,
    seed: int = 42,
    # Level 1 likelihood precision (baseline A1 matrix)
    A1_precision: float = 0.75,
    # Precision parameters for each attention state (m=2 multiplicative symmetry)
    zeta_focused: float = 2.0,
    zeta_distracted: float = 0.5,
    # Precision update parameters (B.45)
    zeta_step: float = 0.25,  # Step size for precision updates
    log_zeta_prior_var: float = 2.0,  # Variance of log-zeta prior (lower = stronger regularization)
    # A2: Likelihood of precision observations given attention state
    A2_precision: float = 0.9,  # Set to 0.5 for flat (no attention perception)
    # Precision threshold for binning
    zeta_threshold: float = 1.0,
    # B2: Attention dynamics
    attention_stay_prob: float = 0.95,  # Set to 0.5 for flat (no attention dynamics model)
    # Awareness parameters
    A3_precision: float = 0.9,  # P(low_entropy | aware) = P(high_entropy | not_aware)
    zeta_awareness: float = 1.0,  # Precision weighting for awareness likelihood (0.5 = not attending to it)
    # Initial priors (all default to 0.5 for uncertain start)
    initial_p_focused: float = 0.5,  # Initial prior P(focused)
    initial_p_aware: float = 0.5,  # Initial prior P(aware)
    entropy_threshold: float = 0.5,  # Below = low entropy (aware), above = high entropy
    awareness_stay_prob: float = 0.95,
    # Noise injection
    noise_start: int = None,
    noise_end: int = None,
    # Forced distraction (attention state set to distracted at this timestep)
    force_distract_at: int = None,
    # Fixed modes
    fixed_attention: str = None,
    fixed_awareness: str = None,
    # Novice mode: awareness observations inform attention inference
    awareness_informs_attention: bool = False,  # When True, entropy obs also update attention
):
    """
    Run hierarchical breath perception with attention AND awareness inference.
    
    Two metacognitive state factors at Level 2:
    1. Attention: Inferred from precision (ζ) - are we attending to breath?
    2. Awareness: Inferred from breath belief entropy - are we aware of breath?
    
    Parameters
    ----------
    entropy_threshold : float
        Threshold for binning breath entropy into aware/not-aware observations.
        Max entropy for binary = ln(2) ≈ 0.693. Threshold of 0.5 is moderate.
    """
    np.random.seed(seed)
    
    # =========================================================================
    # Level 1: Breath Perception Setup
    # =========================================================================
    env = BreathEnv(seed=seed)
    
    p_correct = A1_precision
    A1 = np.array([
        [p_correct, 1 - p_correct],
        [1 - p_correct, p_correct]
    ])
    
    stay_p_inhale = 1.0 - 1.0 / max(1, np.mean(env.inhale_range))
    stay_p_exhale = 1.0 - 1.0 / max(1, np.mean(env.exhale_range))
    B1 = np.array([
        [stay_p_inhale, 1 - stay_p_exhale],
        [1 - stay_p_inhale, stay_p_exhale]
    ])
    
    # =========================================================================
    # Level 2: Attention Setup
    # =========================================================================
    A2 = np.array([
        [A2_precision, 1 - A2_precision],
        [1 - A2_precision, A2_precision]
    ])
    
    B2 = np.array([
        [attention_stay_prob, 1 - attention_stay_prob],
        [1 - attention_stay_prob, attention_stay_prob]
    ])
    
    zeta_by_state = np.array([zeta_focused, zeta_distracted])
    
    # =========================================================================
    # Level 2: Awareness Setup
    # =========================================================================
    # A3: P(entropy_obs | awareness_state)
    #     Rows: LOW_ENTROPY, HIGH_ENTROPY
    #     Cols: AWARE, NOT_AWARE
    A3 = np.array([
        [A3_precision, 1 - A3_precision],      # P(low_entropy | state)
        [1 - A3_precision, A3_precision]       # P(high_entropy | state)
    ])
    
    B3 = np.array([
        [awareness_stay_prob, 1 - awareness_stay_prob],
        [1 - awareness_stay_prob, awareness_stay_prob]
    ])
    
    # =========================================================================
    # Data Logs
    # =========================================================================
    # Level 1
    true_breath = np.zeros(T, dtype=int)
    posterior_breath = np.zeros((T, 2))
    zeta_history = np.zeros(T)
    zeta_prior_history = np.zeros(T)
    breath_entropy_history = np.zeros(T)
    is_noise = np.zeros(T, dtype=bool)
    
    # Level 2 - Attention
    posterior_attention = np.zeros((T, 2))
    precision_obs_history = np.zeros(T, dtype=int)
    
    # Level 2 - Awareness
    posterior_awareness = np.zeros((T, 2))
    entropy_obs_history = np.zeros(T, dtype=int)
    
    # =========================================================================
    # Initialize
    # =========================================================================
    qs_breath = np.array([0.5, 0.5])
    zeta = 1.0
    obs_breath = int(env.reset())
    
    qs_attention = np.array([initial_p_focused, 1 - initial_p_focused])  # Initial attention prior
    qs_awareness = np.array([initial_p_aware, 1 - initial_p_aware])  # Initial awareness prior
    
    print(f"Hierarchical Breath with Awareness")
    print(f"  T={T}")
    print(f"  A1 precision={A1_precision}")
    print(f"  ζ_focused={zeta_focused}, ζ_distracted={zeta_distracted}")
    print(f"  ζ_step={zeta_step}, log_ζ_prior_var={log_zeta_prior_var}")
    print(f"  A2 precision (attention)={A2_precision}" + (" [FLAT]" if A2_precision == 0.5 else ""))
    print(f"  B2 stay prob (attention)={attention_stay_prob}" + (" [FLAT]" if attention_stay_prob == 0.5 else ""))
    print(f"  A3 precision (awareness)={A3_precision}")
    print(f"  ζ_awareness={zeta_awareness}" + (" [LOW - not attending]" if zeta_awareness < 1.0 else ""))
    print(f"  Initial P(focused)={initial_p_focused}, P(aware)={initial_p_aware}")
    print(f"  Entropy threshold={entropy_threshold} (max binary entropy = {np.log(2):.3f})")
    print(f"  Awareness informs attention: {awareness_informs_attention}")
    if fixed_attention is not None:
        print(f"  FIXED ATTENTION: {fixed_attention}")
    if fixed_awareness is not None:
        print(f"  FIXED AWARENESS: {fixed_awareness}")
    if noise_start is not None:
        print(f"  Noise injection: t={noise_start}-{noise_end}")
    if force_distract_at is not None:
        print(f"  FORCED DISTRACTION at t={force_distract_at}")
    print()
    
    # =========================================================================
    # Main Loop
    # =========================================================================
    for t in range(T):
        true_breath[t] = env.state
        
        # =====================================================================
        # NOISE INJECTION
        # =====================================================================
        if noise_start is not None and noise_start <= t < noise_end:
            obs_to_process = np.random.choice([0, 1])
            is_noise[t] = True
        else:
            obs_to_process = obs_breath
        
        # =====================================================================
        # DESCENDING MESSAGE: Attention → Precision Prior
        # =====================================================================
        if fixed_attention is not None:
            zeta_prior = zeta_focused if fixed_attention == "focused" else zeta_distracted
        else:
            zeta_prior = np.sum(qs_attention * zeta_by_state)
        zeta_prior_history[t] = zeta_prior
        
        # =====================================================================
        # LEVEL 1: Precision Update (B.45)
        # =====================================================================
        zeta, pe, _ = update_likelihood_precision(
            zeta=zeta_prior,  # Descending message sets starting point
            A=A1,
            obs=obs_to_process,
            qs=qs_breath,
            log_zeta_prior_var=log_zeta_prior_var,
            zeta_step=zeta_step
        )
        zeta_history[t] = zeta
        
        # =====================================================================
        # LEVEL 1: Breath State Inference
        # =====================================================================
        A1_scaled = scale_likelihood(A1, zeta)
        qs_breath = bayesian_update(A1_scaled, obs_to_process, qs_breath)
        posterior_breath[t] = qs_breath
        
        # Compute breath belief entropy
        breath_H = entropy_dist(qs_breath)
        breath_entropy_history[t] = breath_H
        
        # =====================================================================
        # ASCENDING MESSAGE 1: Precision → Attention
        # =====================================================================
        if zeta > zeta_threshold:
            obs_precision = HIGH_PRECISION
        else:
            obs_precision = LOW_PRECISION
        precision_obs_history[t] = obs_precision
        
        # =====================================================================
        # ASCENDING MESSAGE 2: Breath Entropy → Awareness
        # =====================================================================
        if breath_H < entropy_threshold:
            obs_entropy = LOW_ENTROPY  # Confident → aware
        else:
            obs_entropy = HIGH_ENTROPY  # Uncertain → not aware
        entropy_obs_history[t] = obs_entropy
        
        # =====================================================================
        # LEVEL 2: Attention Inference
        # =====================================================================
        if fixed_attention is not None:
            qs_attention = np.array([1.0, 0.0]) if fixed_attention == "focused" else np.array([0.0, 1.0])
        else:
            # Standard update from precision observation
            qs_attention = bayesian_update(A2, obs_precision, qs_attention)
            
            # If awareness informs attention, also update from entropy observation
            # This is the connection created by meditation practice:
            # "not aware of breath" → "probably not focused"
            if awareness_informs_attention:
                # A2_entropy: P(entropy_obs | attention_state)
                # LOW_ENTROPY → focused, HIGH_ENTROPY → distracted
                # Use same precision as A3 for this mapping
                A2_from_entropy = np.array([
                    [A3_precision, 1 - A3_precision],  # P(low_entropy | attention)
                    [1 - A3_precision, A3_precision]   # P(high_entropy | attention)
                ])
                qs_attention = bayesian_update(A2_from_entropy, obs_entropy, qs_attention)
        
        # FORCED DISTRACTION: Override attention at specified timestep
        if force_distract_at is not None and t == force_distract_at:
            qs_attention = np.array([0.0, 1.0])  # Force to distracted
        
        posterior_attention[t] = qs_attention
        
        # =====================================================================
        # LEVEL 2: Awareness Inference
        # =====================================================================
        if fixed_awareness is not None:
            qs_awareness = np.array([1.0, 0.0]) if fixed_awareness == "aware" else np.array([0.0, 1.0])
        else:
            # Apply precision scaling to awareness likelihood
            A3_scaled = scale_likelihood(A3, zeta_awareness)
            qs_awareness = bayesian_update(A3_scaled, obs_entropy, qs_awareness)
        posterior_awareness[t] = qs_awareness
        
        # =====================================================================
        # Advance to next timestep
        # =====================================================================
        qs_breath = B1 @ qs_breath
        
        if fixed_attention is None:
            qs_attention = B2 @ qs_attention
        
        if fixed_awareness is None:
            qs_awareness = B3 @ qs_awareness
        
        obs_breath = int(env.step(None))
    
    # =========================================================================
    # Summary
    # =========================================================================
    breath_acc = (np.argmax(posterior_breath, axis=1) == true_breath).mean()
    mean_p_focused = posterior_attention[:, FOCUSED].mean()
    mean_p_aware = posterior_awareness[:, AWARE].mean()
    mean_entropy = breath_entropy_history.mean()
    
    print(f"Results:")
    print(f"  Breath inference accuracy: {breath_acc:.3f}")
    print(f"  Mean P(focused): {mean_p_focused:.3f}")
    print(f"  Mean P(aware): {mean_p_aware:.3f}")
    print(f"  Mean breath entropy: {mean_entropy:.3f}")
    print(f"  Mean ζ prior: {zeta_prior_history.mean():.3f}")
    print(f"  Mean ζ posterior: {zeta_history.mean():.3f}")
    
    return {
        "true_breath": true_breath,
        "posterior_breath": posterior_breath,
        "zeta_history": zeta_history,
        "zeta_prior_history": zeta_prior_history,
        "breath_entropy_history": breath_entropy_history,
        "posterior_attention": posterior_attention,
        "precision_obs_history": precision_obs_history,
        "posterior_awareness": posterior_awareness,
        "entropy_obs_history": entropy_obs_history,
        "zeta_focused": zeta_focused,
        "zeta_distracted": zeta_distracted,
        "zeta_step": zeta_step,
        "log_zeta_prior_var": log_zeta_prior_var,
        "zeta_threshold": zeta_threshold,
        "entropy_threshold": entropy_threshold,
        "is_noise": is_noise,
        "noise_start": noise_start,
        "noise_end": noise_end,
        "force_distract_at": force_distract_at,
        "fixed_attention": fixed_attention,
        "fixed_awareness": fixed_awareness,
        "awareness_informs_attention": awareness_informs_attention,
        "A2_precision": A2_precision,
        "attention_stay_prob": attention_stay_prob,
        "A3_precision": A3_precision,
        "zeta_awareness": zeta_awareness,
        "initial_p_focused": initial_p_focused,
        "initial_p_aware": initial_p_aware,
    }


# =============================================================================
# Figure 5: Hierarchical Breath with Attention ACTION SELECTION
# =============================================================================

def run_hierarchical_with_action(
    T: int = 200,
    seed: int = 42,
    # Level 1 likelihood precision (baseline A1 matrix)
    A1_precision: float = 0.75,
    # Precision parameters for each attention state (m=2 multiplicative symmetry)
    zeta_focused: float = 2.0,
    zeta_distracted: float = 0.5,
    # Precision update parameters (B.45)
    zeta_step: float = 0.25,
    log_zeta_prior_var: float = 2.0,
    # A2: Likelihood of precision observations given attention state
    A2_precision: float = 0.9,
    # Precision threshold for binning
    zeta_threshold: float = 1.0,
    # B2: Attention dynamics (now action-dependent)
    B2_stay_prob: float = 0.95,  # P(stay|stay action)
    B2_switch_prob: float = 0.95,  # P(switch|switch action)
    # Initial beliefs
    initial_p_focused: float = 0.5,
    # Policy selection parameters
    gamma: float = 16.0,  # Policy precision
    # C2: Preferences over precision observations (log-probabilities)
    C2_high_prec_pref: float = 0.0,  # Preference for high precision obs
    C2_low_prec_pref: float = 0.0,   # Preference for low precision obs
    # E: Policy prior (habit)
    E_stay: float = 0.5,  # Prior probability of STAY action
    # Noise injection
    noise_start: int = None,
    noise_end: int = None,
    # Debug
    force_distract_at: int = None,
    # Terminal distraction mode: distraction is absorbing under STAY
    terminal_distraction: bool = False,
):
    """
    Figure 5: Hierarchical breath perception with attention ACTION SELECTION.
    
    Level 2 now has actions:
        - STAY: Maintain current attention state
        - SWITCH: Switch to other attention state
    
    Action selection is based on Expected Free Energy (EFE), computed using
    pymdp's update_posterior_policies function.
    
    Parameters
    ----------
    T : int
        Number of timesteps
    A1_precision : float
        Baseline precision of Level 1 breath likelihood
    zeta_focused, zeta_distracted : float
        Precision modulation for each attention state
    B2_stay_prob : float
        Probability of staying in current state when STAY action selected
    B2_switch_prob : float
        Probability of switching states when SWITCH action selected
    gamma : float
        Policy precision (inverse temperature for action selection)
    C2_high_prec_pref, C2_low_prec_pref : float
        Log-preferences for precision observations (0, 0 = flat preferences)
    """
    np.random.seed(seed)
    
    # =========================================================================
    # Setup Level 1: Breath perception
    # =========================================================================
    env = BreathEnv(p_correct=0.98, seed=seed)
    obs = int(env.reset())
    
    # Level 1 A matrix (breath observations)
    p_correct = A1_precision
    A1 = np.array([
        [p_correct, 1 - p_correct],  # P(expansion | inhale/exhale)
        [1 - p_correct, p_correct],  # P(contraction | inhale/exhale)
    ])
    
    # Level 1 B matrix (breath dynamics - probabilistic based on env durations)
    # Agent has correct beliefs about breath transition probabilities
    stay_p_inhale = 1.0 - 1.0 / max(1, np.mean(env.inhale_range))
    stay_p_exhale = 1.0 - 1.0 / max(1, np.mean(env.exhale_range))
    B1 = np.array([
        [stay_p_inhale, 1 - stay_p_exhale],  # P(inhale | inhale, exhale)
        [1 - stay_p_inhale, stay_p_exhale],  # P(exhale | inhale, exhale)
    ])
    
    # =========================================================================
    # Setup Level 2: Attention with action-dependent transitions
    # =========================================================================
    
    # A2: Observation model for attention (precision observations)
    # A2[o, s] = P(obs=o | state=s)
    A2_attention = np.array([
        [A2_precision, 1 - A2_precision],      # P(high_prec | focused, distracted)
        [1 - A2_precision, A2_precision],      # P(low_prec | focused, distracted)
    ])
    
    # B2: Action-dependent transition model for attention
    # B2[s', s, a] = P(s' | s, a)
    # Actions: STAY=0, SWITCH=1
    B2_attention = np.zeros((2, 2, 2))  # (next_state, current_state, action)
    
    if terminal_distraction:
        # Terminal distraction mode:
        # STAY: focus is unstable (B2_stay_prob), distraction is absorbing (100%)
        # SWITCH: deterministic transitions (100%)
        B2_attention[:, :, STAY] = np.array([
            [B2_stay_prob, 0.0],           # P(focused | focused/distracted, stay)
            [1 - B2_stay_prob, 1.0],       # P(distracted | focused/distracted, stay)
        ])
        B2_attention[:, :, SWITCH] = np.array([
            [0.0, 1.0],                    # P(focused | focused/distracted, switch)
            [1.0, 0.0],                    # P(distracted | focused/distracted, switch)
        ])
    else:
        # Standard mode: symmetric transitions
        # Action STAY (a=0): High probability of staying
        B2_attention[:, :, STAY] = np.array([
            [B2_stay_prob, 1 - B2_stay_prob],      # P(focused | focused/distracted, stay)
            [1 - B2_stay_prob, B2_stay_prob],      # P(distracted | focused/distracted, stay)
        ])
        
        # Action SWITCH (a=1): High probability of switching
        B2_attention[:, :, SWITCH] = np.array([
            [1 - B2_switch_prob, B2_switch_prob],  # P(focused | focused/distracted, switch)
            [B2_switch_prob, 1 - B2_switch_prob],  # P(distracted | focused/distracted, switch)
        ])
    
    # C2: Preferences over observations (log-probabilities)
    # Flat preferences for now: no preference for high vs low precision
    C2 = np.array([C2_high_prec_pref, C2_low_prec_pref])
    
    # Policies: Single-step policies for action selection
    # Each policy is a 2D array: (num_timesteps, num_factors)
    # We have 1 timestep and 1 factor (attention)
    policies = [
        np.array([[STAY]]),    # Policy 0: Stay
        np.array([[SWITCH]]),  # Policy 1: Switch
    ]
    
    # Convert to pymdp object arrays for compatibility
    A2_obj = utils.obj_array(1)
    A2_obj[0] = A2_attention
    
    B2_obj = utils.obj_array(1)
    B2_obj[0] = B2_attention
    
    C2_obj = utils.obj_array(1)
    C2_obj[0] = C2
    
    # E: Policy prior (habit) - prior probability over policies
    # E[0] = P(STAY), E[1] = P(SWITCH)
    E = np.array([E_stay, 1 - E_stay])
    
    # =========================================================================
    # Initialize state beliefs and histories
    # =========================================================================
    
    # Level 1: Breath
    qs_breath = np.array([0.5, 0.5])
    
    # Level 2: Attention
    qs_attention = np.array([initial_p_focused, 1 - initial_p_focused])
    
    # Precision
    zeta = 1.0
    
    # TRUE attention state (hidden from agent, exists in environment)
    # Actions affect this, and it generates observations
    true_attention_state = FOCUSED  # Start focused
    
    # Track previous action for belief prediction (start with STAY)
    prev_action = STAY
    
    # Storage
    true_breath = np.zeros(T, dtype=int)
    true_attention = np.zeros(T, dtype=int)  # Track true attention state
    posterior_breath = np.zeros((T, 2))
    zeta_history = np.zeros(T)
    zeta_prior_history = np.zeros(T)
    posterior_attention = np.zeros((T, 2))
    precision_obs_history = np.zeros(T, dtype=int)
    action_history = np.zeros(T, dtype=int)
    q_pi_history = np.zeros((T, 2))  # Policy posterior
    G_history = np.zeros((T, 2))      # EFE values
    is_noise = np.zeros(T, dtype=bool)
    
    # =========================================================================
    # Main loop
    # =========================================================================
    
    for t in range(T):
        # Record true states
        true_breath[t] = env.state
        true_attention[t] = true_attention_state
        
        # Check for noise injection (affects breath observations)
        if noise_start is not None and noise_end is not None:
            if noise_start <= t < noise_end:
                obs_to_process = np.random.randint(0, 2)
                is_noise[t] = True
            else:
                obs_to_process = obs
        else:
            obs_to_process = obs
        
        # =====================================================================
        # DESCENDING MESSAGE: TRUE Attention State → Precision Prior
        # The TRUE state (not beliefs) determines the precision environment
        # =====================================================================
        
        if true_attention_state == FOCUSED:
            zeta_prior = zeta_focused
        else:
            zeta_prior = zeta_distracted
        zeta_prior_history[t] = zeta_prior
        
        # =====================================================================
        # LEVEL 1: Precision Update (B.45)
        # =====================================================================
        
        zeta, pe, _ = update_likelihood_precision(
            zeta=zeta_prior,
            A=A1,
            obs=obs_to_process,
            qs=qs_breath,
            log_zeta_prior_var=log_zeta_prior_var,
            zeta_step=zeta_step
        )
        zeta_history[t] = zeta
        
        # =====================================================================
        # LEVEL 1: Breath State Inference
        # =====================================================================
        
        A1_scaled = scale_likelihood(A1, zeta)
        qs_breath = bayesian_update(A1_scaled, obs_to_process, qs_breath)
        posterior_breath[t] = qs_breath
        
        # =====================================================================
        # ASCENDING MESSAGE: TRUE Attention State → Observation
        # TRUE state generates DETERMINISTIC observations
        # Agent must infer the true state from these observations
        # =====================================================================
        
        if true_attention_state == FOCUSED:
            obs_precision = HIGH_PRECISION  # Deterministic: focused → high precision obs
        else:
            obs_precision = LOW_PRECISION   # Deterministic: distracted → low precision obs
        precision_obs_history[t] = obs_precision
        
        # Agent updates attention BELIEFS using proper active inference:
        # 1. PREDICTION: Compute prior from previous posterior + previous action + B model
        prior_attention = B2_attention[:, :, prev_action] @ qs_attention
        prior_attention = prior_attention / (prior_attention.sum() + EPS_VAL)
        
        # 2. UPDATE: Combine likelihood (from observation) with predicted prior
        likelihood_att = A2_attention[obs_precision, :]
        qs_attention = likelihood_att * prior_attention
        qs_attention = qs_attention / (qs_attention.sum() + EPS_VAL)
        
        posterior_attention[t] = qs_attention
        
        # =====================================================================
        # LEVEL 2: Action Selection via EFE (based on BELIEFS)
        # =====================================================================
        
        qs_attention_obj = utils.obj_array(1)
        qs_attention_obj[0] = qs_attention
        
        q_pi, G = update_posterior_policies(
            qs=qs_attention_obj,
            A=A2_obj,
            B=B2_obj,
            C=C2_obj,
            policies=policies,
            use_utility=True,
            use_states_info_gain=True,
            E=E,
            gamma=gamma,
        )
        
        q_pi_history[t] = q_pi.flatten()
        G_history[t] = G
        
        # Select action deterministically (argmax)
        action = np.argmax(q_pi.flatten())
        action_history[t] = action
        
        # Store action for next timestep's belief prediction
        prev_action = action
        
        # =====================================================================
        # TRUE Attention State Transition (action affects TRUE state)
        # =====================================================================
        
        # Forced distraction: sets TRUE state to distracted
        # Agent doesn't know this immediately - must infer from observations
        if force_distract_at is not None and t == force_distract_at:
            true_attention_state = DISTRACTED
            print(f"  FORCED DISTRACTION at t={t} (TRUE state → distracted)")
        else:
            # Stochastic transition of TRUE state based on action and B2
            transition_probs = B2_attention[:, true_attention_state, action]
            true_attention_state = np.random.choice([FOCUSED, DISTRACTED], p=transition_probs)
        
        # =====================================================================
        # Predict next states (for next timestep's prior)
        # =====================================================================
        qs_breath = B1 @ qs_breath  # Predict next breath state
        
        # =====================================================================
        # Environment step
        # =====================================================================
        obs = int(env.step(None))
    
    # =========================================================================
    # Results
    # =========================================================================
    
    accuracy = np.mean((posterior_breath[:, INHALE] > 0.5) == (true_breath == INHALE))
    
    print(f"Hierarchical Breath with Action Selection (Figure 5)")
    print(f"  T={T}")
    print(f"  A1 precision={A1_precision}")
    print(f"  ζ_focused={zeta_focused}, ζ_distracted={zeta_distracted}")
    print(f"  ζ_step={zeta_step}, log_ζ_prior_var={log_zeta_prior_var}")
    print(f"  B2 stay prob={B2_stay_prob}, switch prob={B2_switch_prob}")
    print(f"  Policy precision γ={gamma}")
    print(f"  Policy prior E: P(Stay)={E_stay}, P(Switch)={1-E_stay}")
    print(f"  C2 preferences: high={C2_high_prec_pref}, low={C2_low_prec_pref}")
    print(f"  Initial P(focused)={initial_p_focused}")
    print()
    print(f"Results:")
    print(f"  Breath inference accuracy: {accuracy:.3f}")
    print(f"  Mean P(focused): {posterior_attention[:, FOCUSED].mean():.3f}")
    print(f"  Mean ζ posterior: {zeta_history.mean():.3f}")
    print(f"  Action distribution: STAY={np.mean(action_history == STAY):.2f}, SWITCH={np.mean(action_history == SWITCH):.2f}")
    
    return {
        "T": T,
        "A1_precision": A1_precision,
        "A2_precision": A2_precision,
        "true_breath": true_breath,
        "true_attention": true_attention,  # NEW: true attention state history
        "posterior_breath": posterior_breath,
        "zeta_history": zeta_history,
        "zeta_prior_history": zeta_prior_history,
        "posterior_attention": posterior_attention,
        "precision_obs_history": precision_obs_history,
        "action_history": action_history,
        "q_pi_history": q_pi_history,
        "G_history": G_history,
        "zeta_focused": zeta_focused,
        "zeta_distracted": zeta_distracted,
        "zeta_step": zeta_step,
        "log_zeta_prior_var": log_zeta_prior_var,
        "zeta_threshold": zeta_threshold,
        "B2_stay_prob": B2_stay_prob,
        "B2_switch_prob": B2_switch_prob,
        "gamma": gamma,
        "E_stay": E_stay,
        "C2_high_prec_pref": C2_high_prec_pref,
        "C2_low_prec_pref": C2_low_prec_pref,
        "is_noise": is_noise,
        "noise_start": noise_start,
        "noise_end": noise_end,
        "force_distract_at": force_distract_at,
        "initial_p_focused": initial_p_focused,
        "terminal_distraction": terminal_distraction,
    }


# =============================================================================
# Figure 6: Hierarchical Breath with Action Selection AND Awareness
# =============================================================================

def run_hierarchical_with_action_and_awareness(
    T: int = 200,
    seed: int = 42,
    # Level 1 likelihood precision (baseline A1 matrix)
    A1_precision: float = 0.75,
    # Precision parameters for each attention state
    zeta_focused: float = 2.0,
    zeta_distracted: float = 0.5,
    # Precision update parameters (B.45)
    zeta_step: float = 0.25,
    log_zeta_prior_var: float = 2.0,
    # A2: Likelihood of precision observations given attention state
    A2_precision: float = 0.9,
    # Precision threshold for binning
    zeta_threshold: float = 1.0,
    # A3: Awareness likelihood
    A3_precision: float = 0.9,
    zeta_awareness: float = 1.0,
    entropy_threshold: float = 0.5,
    # B2: Attention dynamics (action-dependent)
    B2_stay_prob: float = 0.95,
    B2_switch_prob: float = 0.95,
    # B3: Awareness dynamics
    awareness_stay_prob: float = 0.95,
    # Initial beliefs
    initial_p_focused: float = 0.5,
    initial_p_aware: float = 0.5,
    # Policy selection parameters
    gamma: float = 16.0,
    # C2: Preferences over precision observations
    C2_high_prec_pref: float = 0.0,
    C2_low_prec_pref: float = 0.0,
    # E: Policy prior (habit)
    E_stay: float = 0.5,
    # Noise injection
    noise_start: int = None,
    noise_end: int = None,
    # Debug
    force_distract_at: int = None,
    terminal_distraction: bool = False,
):
    """
    Figure 6: Hierarchical breath perception with action selection AND awareness.
    
    Combines Figure 5's action selection with Figure 4's awareness inference.
    """
    np.random.seed(seed)
    
    # Environment
    env = BreathEnv(seed=seed)
    
    # Level 1: Breath perception matrices
    A1 = np.array([
        [A1_precision, 1 - A1_precision],
        [1 - A1_precision, A1_precision]
    ])
    
    inhale_duration = env.inhale_range[1] - env.inhale_range[0]
    exhale_duration = env.exhale_range[1] - env.exhale_range[0]
    p_stay_inhale = (inhale_duration - 1) / inhale_duration
    p_stay_exhale = (exhale_duration - 1) / exhale_duration
    
    B1 = np.array([
        [p_stay_inhale, 1 - p_stay_exhale],
        [1 - p_stay_inhale, p_stay_exhale]
    ])
    
    # Level 2: Attention observation model
    A2_attention = np.array([
        [A2_precision, 1 - A2_precision],
        [1 - A2_precision, A2_precision]
    ])
    
    # Level 2: Awareness observation model
    A3 = np.array([
        [A3_precision, 1 - A3_precision],
        [1 - A3_precision, A3_precision]
    ])
    
    # B3: Awareness transitions
    B3 = np.array([
        [awareness_stay_prob, 1 - awareness_stay_prob],
        [1 - awareness_stay_prob, awareness_stay_prob]
    ])
    
    # B2: Action-dependent transition model for attention
    B2_attention = np.zeros((2, 2, 2))
    
    if terminal_distraction:
        B2_attention[:, :, STAY] = np.array([
            [B2_stay_prob, 0.0],
            [1 - B2_stay_prob, 1.0],
        ])
        B2_attention[:, :, SWITCH] = np.array([
            [0.0, 1.0],
            [1.0, 0.0],
        ])
    else:
        B2_attention[:, :, STAY] = np.array([
            [B2_stay_prob, 1 - B2_stay_prob],
            [1 - B2_stay_prob, B2_stay_prob],
        ])
        B2_attention[:, :, SWITCH] = np.array([
            [1 - B2_switch_prob, B2_switch_prob],
            [B2_switch_prob, 1 - B2_switch_prob],
        ])
    
    # C2: Preferences
    C2 = np.array([C2_high_prec_pref, C2_low_prec_pref])
    
    # Policies
    policies = [
        np.array([[STAY]]),
        np.array([[SWITCH]]),
    ]
    
    # Convert to pymdp object arrays
    A2_obj = utils.obj_array(1)
    A2_obj[0] = A2_attention
    
    B2_obj = utils.obj_array(1)
    B2_obj[0] = B2_attention
    
    C2_obj = utils.obj_array(1)
    C2_obj[0] = C2
    
    # Policy prior E
    E = np.array([E_stay, 1 - E_stay])
    
    # Storage
    true_breath = np.zeros(T, dtype=int)
    obs_breath = np.zeros(T, dtype=int)
    posterior_breath = np.zeros((T, 2))
    zeta_history = np.zeros(T)
    zeta_prior_history = np.zeros(T)
    posterior_attention = np.zeros((T, 2))
    precision_obs_history = np.zeros(T, dtype=int)
    posterior_awareness = np.zeros((T, 2))
    entropy_obs_history = np.zeros(T, dtype=int)
    breath_entropy_history = np.zeros(T)
    true_attention_history = np.zeros(T, dtype=int)
    action_history = np.zeros(T, dtype=int)
    q_pi_history = np.zeros((T, 2))
    G_history = np.zeros((T, 2))
    is_noise = np.zeros(T, dtype=bool)
    
    # Initialize
    qs_breath = np.array([0.5, 0.5])
    qs_attention = np.array([initial_p_focused, 1 - initial_p_focused])
    qs_awareness = np.array([initial_p_aware, 1 - initial_p_aware])
    zeta = 1.0
    true_attention_state = FOCUSED
    prev_action = STAY
    
    obs_breath_val = int(env.step(None))
    
    print(f"Hierarchical Breath with Action & Awareness (Figure 6)")
    print(f"  T={T}")
    print(f"  A1 precision={A1_precision}")
    print(f"  ζ_focused={zeta_focused}, ζ_distracted={zeta_distracted}")
    print(f"  A2 precision={A2_precision}, A3 precision={A3_precision}")
    print(f"  ζ_awareness={zeta_awareness}")
    print(f"  B2 stay prob={B2_stay_prob}, switch prob={B2_switch_prob}")
    print(f"  γ={gamma}, E: P(Stay)={E_stay}")
    print(f"  C2: [{C2_high_prec_pref}, {C2_low_prec_pref}]")
    if terminal_distraction:
        print(f"  TERMINAL DISTRACTION MODE")
    print()
    
    # Main loop
    for t in range(T):
        true_breath[t] = env.state
        obs_breath[t] = obs_breath_val
        
        # Noise injection
        if noise_start is not None and noise_start <= t < noise_end:
            is_noise[t] = True
            obs_to_process = np.random.randint(0, 2)
        else:
            obs_to_process = obs_breath_val
        
        # Descending message: TRUE attention state → zeta prior
        # The TRUE state (not beliefs) determines the precision environment
        if true_attention_state == FOCUSED:
            zeta_prior = zeta_focused
        else:
            zeta_prior = zeta_distracted
        zeta_prior_history[t] = zeta_prior
        
        # Precision update (B.45) - start from prior each timestep
        zeta, _, _ = update_likelihood_precision(
            zeta=zeta_prior,
            A=A1,
            obs=obs_to_process,
            qs=qs_breath,
            log_zeta_prior_var=log_zeta_prior_var,
            zeta_step=zeta_step,
        )
        zeta_history[t] = zeta
        
        # State inference
        A1_scaled = scale_likelihood(A1, zeta)
        qs_breath = bayesian_update(A1_scaled, obs_to_process, qs_breath)
        posterior_breath[t] = qs_breath
        
        # Breath entropy
        breath_H = entropy_dist(qs_breath)
        breath_entropy_history[t] = breath_H
        
        # Ascending message 1: Precision → Attention observation
        if true_attention_state == FOCUSED:
            obs_precision = HIGH_PRECISION
        else:
            obs_precision = LOW_PRECISION
        precision_obs_history[t] = obs_precision
        
        # Ascending message 2: Entropy → Awareness observation
        if breath_H < entropy_threshold:
            obs_entropy = LOW_ENTROPY
        else:
            obs_entropy = HIGH_ENTROPY
        entropy_obs_history[t] = obs_entropy
        
        # Attention belief prediction
        prior_attention = B2_attention[:, :, prev_action] @ posterior_attention[t-1] if t > 0 else qs_attention
        
        # Attention belief update
        likelihood_att = A2_attention[obs_precision, :]
        qs_attention = likelihood_att * prior_attention
        qs_attention = qs_attention / (qs_attention.sum() + EPS_VAL)
        posterior_attention[t] = qs_attention
        
        # Awareness inference
        A3_scaled = scale_likelihood(A3, zeta_awareness)
        qs_awareness = bayesian_update(A3_scaled, obs_entropy, qs_awareness)
        posterior_awareness[t] = qs_awareness
        
        # Action selection via EFE
        qs_attention_obj = utils.obj_array(1)
        qs_attention_obj[0] = qs_attention
        
        q_pi, G = update_posterior_policies(
            qs=qs_attention_obj,
            A=A2_obj,
            B=B2_obj,
            C=C2_obj,
            policies=policies,
            use_utility=True,
            use_states_info_gain=True,
            E=E,
            gamma=gamma,
        )
        
        q_pi_history[t] = q_pi.flatten()
        G_history[t] = G
        
        # Select action deterministically (argmax)
        action = np.argmax(q_pi.flatten())
        action_history[t] = action
        prev_action = action
        
        # True attention state transition
        true_attention_history[t] = true_attention_state
        
        if force_distract_at is not None and t == force_distract_at:
            true_attention_state = DISTRACTED
            print(f"  FORCED DISTRACTION at t={t}")
        else:
            true_attention_one_hot = np.zeros(2)
            true_attention_one_hot[true_attention_state] = 1.0
            next_true_attention_one_hot = B2_attention[:, :, action] @ true_attention_one_hot
            true_attention_state = np.random.choice([FOCUSED, DISTRACTED], p=next_true_attention_one_hot)
        
        # Advance
        qs_breath = B1 @ qs_breath
        qs_breath = qs_breath / (qs_breath.sum() + EPS_VAL)
        qs_awareness = B3 @ qs_awareness
        qs_awareness = qs_awareness / (qs_awareness.sum() + EPS_VAL)
        
        obs_breath_val = int(env.step(None))
    
    # Summary
    accuracy = np.mean(np.argmax(posterior_breath, axis=1) == true_breath)
    mean_p_focused = np.mean(posterior_attention[:, FOCUSED])
    mean_zeta = np.mean(zeta_history)
    mean_p_aware = np.mean(posterior_awareness[:, AWARE])
    
    print(f"Results:")
    print(f"  Breath accuracy: {accuracy:.3f}")
    print(f"  Mean P(focused): {mean_p_focused:.3f}")
    print(f"  Mean P(aware): {mean_p_aware:.3f}")
    print(f"  Mean ζ: {mean_zeta:.3f}")
    stay_pct = np.mean(action_history == STAY)
    switch_pct = np.mean(action_history == SWITCH)
    print(f"  Actions: STAY={stay_pct:.2f}, SWITCH={switch_pct:.2f}")
    
    return {
        "true_breath": true_breath,
        "obs_breath": obs_breath,
        "posterior_breath": posterior_breath,
        "zeta_history": zeta_history,
        "zeta_prior_history": zeta_prior_history,
        "posterior_attention": posterior_attention,
        "precision_obs_history": precision_obs_history,
        "posterior_awareness": posterior_awareness,
        "entropy_obs_history": entropy_obs_history,
        "breath_entropy_history": breath_entropy_history,
        "true_attention_history": true_attention_history,
        "action_history": action_history,
        "q_pi_history": q_pi_history,
        "G_history": G_history,
        "T": T,
        "A1_precision": A1_precision,
        "A2_precision": A2_precision,
        "A3_precision": A3_precision,
        "zeta_focused": zeta_focused,
        "zeta_distracted": zeta_distracted,
        "zeta_step": zeta_step,
        "log_zeta_prior_var": log_zeta_prior_var,
        "zeta_awareness": zeta_awareness,
        "B2_stay_prob": B2_stay_prob,
        "B2_switch_prob": B2_switch_prob,
        "gamma": gamma,
        "E_stay": E_stay,
        "C2_high_prec_pref": C2_high_prec_pref,
        "C2_low_prec_pref": C2_low_prec_pref,
        "is_noise": is_noise,
        "noise_start": noise_start,
        "noise_end": noise_end,
        "force_distract_at": force_distract_at,
        "terminal_distraction": terminal_distraction,
    }


# =============================================================================
# Figure 6.1: Baseline with B2_true/B2 separation (Non-meditator within-sit)
# Uses same structure as run_hierarchical_with_action_and_awareness but with
# separate B2_true (distraction attractor) and B2_agent (flat)
# =============================================================================

def run_figure6_1(
    T: int = 100,
    seed: int = 42,
    # Level 1 parameters
    A1_precision: float = 0.75,
    zeta_focused: float = 2.0,
    zeta_distracted: float = 0.5,
    zeta_step: float = 0.25,
    log_zeta_prior_var: float = 2.0,
    # Level 2: Attention - SEPARATE true vs agent
    A2_precision: float = 0.52,  # Agent's A2 (flat/uncertain)
    A2_true_precision: float = 0.9,  # True A2 for generating observations
    B2_agent_stay_prob: float = 0.52,  # Agent's uncertain B2
    # B2_true uses distraction attractor (hardcoded)
    # Level 2: Awareness
    A3_precision: float = 0.75,
    entropy_threshold: float = 0.5,
    zeta_awareness: float = 1.0,
    # B3: Awareness dynamics
    awareness_stay_prob: float = 0.95,
    # Initial beliefs
    initial_p_focused: float = 0.5,
    initial_p_aware: float = 0.5,
    # Policy selection
    gamma: float = 16.0,
    E_stay: float = 0.9,
    # C2: Preferences
    C2_high_prec_pref: float = 0.0,
    C2_low_prec_pref: float = 0.0,
):
    """
    Figure 6.1: Non-meditator baseline within-sit dynamics.

    Based on run_hierarchical_with_action_and_awareness but with:
    - B2_true: distraction attractor (80% → distracted regardless of state)
    - B2_agent: flat (0.52) - agent doesn't know attention dynamics
    - A2_true vs A2_agent separation
    """
    np.random.seed(seed)
    env = BreathEnv(seed=seed)

    # Level 1: Breath perception matrices (same as original)
    A1 = np.array([
        [A1_precision, 1 - A1_precision],
        [1 - A1_precision, A1_precision]
    ])

    inhale_duration = env.inhale_range[1] - env.inhale_range[0]
    exhale_duration = env.exhale_range[1] - env.exhale_range[0]
    p_stay_inhale = (inhale_duration - 1) / inhale_duration
    p_stay_exhale = (exhale_duration - 1) / exhale_duration
    B1 = np.array([
        [p_stay_inhale, 1 - p_stay_exhale],
        [1 - p_stay_inhale, p_stay_exhale]
    ])

    # Level 2: Attention observation model - SEPARATE true vs agent
    A2_true = np.array([
        [A2_true_precision, 1 - A2_true_precision],
        [1 - A2_true_precision, A2_true_precision]
    ])
    A2_attention = np.array([
        [A2_precision, 1 - A2_precision],
        [1 - A2_precision, A2_precision]
    ])

    # Level 2: Awareness observation model (same as original)
    A3 = np.array([
        [A3_precision, 1 - A3_precision],
        [1 - A3_precision, A3_precision]
    ])

    # B3: Awareness transitions (same as original)
    B3 = np.array([
        [awareness_stay_prob, 1 - awareness_stay_prob],
        [1 - awareness_stay_prob, awareness_stay_prob]
    ])

    # B2_true: Distraction is absorbing under STAY, toggle for SWITCH
    B2_true = np.zeros((2, 2, 2))
    B2_true[:, :, STAY] = np.array([
        [0.8, 0.0],   # P(->foc | stay): can maintain focus, can't return from distracted
        [0.2, 1.0],   # P(->dist | stay): distraction is absorbing
    ])
    B2_true[:, :, SWITCH] = np.array([
        [0.1, 0.9],   # P(->foc | switch): foc->dist, dist->foc
        [0.9, 0.1],   # P(->dist | switch): toggles state
    ])

    # B2_agent: Agent's model (flat/uncertain)
    B2_attention = np.zeros((2, 2, 2))
    B2_attention[:, :, STAY] = np.array([
        [B2_agent_stay_prob, 1 - B2_agent_stay_prob],
        [1 - B2_agent_stay_prob, B2_agent_stay_prob],
    ])
    B2_attention[:, :, SWITCH] = np.array([
        [1 - B2_agent_stay_prob, B2_agent_stay_prob],
        [B2_agent_stay_prob, 1 - B2_agent_stay_prob],
    ])

    # Policy setup (same as original)
    E = np.array([E_stay, 1 - E_stay])
    C2 = np.array([C2_high_prec_pref, C2_low_prec_pref])
    policies = [np.array([[STAY]]), np.array([[SWITCH]])]

    # pymdp object arrays
    A2_obj = utils.obj_array(1)
    A2_obj[0] = A2_attention
    B2_obj = utils.obj_array(1)
    B2_obj[0] = B2_attention
    C2_obj = utils.obj_array(1)
    C2_obj[0] = C2

    # Storage (same as original)
    true_breath = np.zeros(T, dtype=int)
    obs_breath = np.zeros(T, dtype=int)
    posterior_breath = np.zeros((T, 2))
    zeta_history = np.zeros(T)
    zeta_prior_history = np.zeros(T)
    posterior_attention = np.zeros((T, 2))
    precision_obs_history = np.zeros(T, dtype=int)
    posterior_awareness = np.zeros((T, 2))
    entropy_obs_history = np.zeros(T, dtype=int)
    breath_entropy_history = np.zeros(T)
    true_attention_history = np.zeros(T, dtype=int)
    action_history = np.zeros(T, dtype=int)
    q_pi_history = np.zeros((T, 2))
    G_history = np.zeros((T, 2))

    # Initialize (same as original)
    qs_breath = np.array([0.5, 0.5])
    qs_attention = np.array([initial_p_focused, 1 - initial_p_focused])
    qs_awareness = np.array([initial_p_aware, 1 - initial_p_aware])
    zeta = 1.0
    true_attention_state = FOCUSED
    prev_action = STAY
    obs_breath_val = int(env.step(None))

    print(f"Figure 6.1: Non-meditator Baseline (within-sit)")
    print(f"  T={T}")
    print(f"  A1 precision={A1_precision}")
    print(f"  ζ_focused={zeta_focused}, ζ_distracted={zeta_distracted}")
    print(f"  A2 agent={A2_precision}, A2 true={A2_true_precision}")
    print(f"  B2 agent stay={B2_agent_stay_prob}, B2 true=distraction attractor")
    print(f"  γ={gamma}, E: P(Stay)={E_stay}")
    print()

    # Main loop - canonical inference order: prior = B @ qs_prev, then likelihood update
    for t in range(T):
        true_breath[t] = env.state
        obs_breath[t] = obs_breath_val
        true_attention_history[t] = true_attention_state

        # Descending: TRUE attention → precision prior
        zeta_prior = zeta_focused if true_attention_state == FOCUSED else zeta_distracted
        zeta_prior_history[t] = zeta_prior

        # Precision update
        zeta, _, _ = update_likelihood_precision(
            zeta=zeta_prior, A=A1, obs=obs_breath_val, qs=qs_breath,
            log_zeta_prior_var=log_zeta_prior_var, zeta_step=zeta_step,
        )
        zeta_history[t] = zeta

        # BREATH INFERENCE: prior = B1 @ qs_prev, then likelihood update
        prior_breath = B1 @ qs_breath if t > 0 else qs_breath
        A1_scaled = scale_likelihood(A1, zeta)
        likelihood_breath = A1_scaled[obs_breath_val, :]
        qs_breath = likelihood_breath * prior_breath
        qs_breath = qs_breath / (qs_breath.sum() + EPS_VAL)
        posterior_breath[t] = qs_breath

        # Breath entropy
        breath_H = entropy_dist(qs_breath)
        breath_entropy_history[t] = breath_H

        # Ascending observations
        obs_precision = np.random.choice([HIGH_PRECISION, LOW_PRECISION], p=A2_true[:, true_attention_state])
        precision_obs_history[t] = obs_precision

        obs_entropy = LOW_ENTROPY if breath_H < entropy_threshold else HIGH_ENTROPY
        entropy_obs_history[t] = obs_entropy

        # AWARENESS INFERENCE: prior = B3 @ qs_prev, then likelihood update
        prior_awareness = B3 @ qs_awareness if t > 0 else qs_awareness
        A3_scaled = scale_likelihood(A3, zeta_awareness)
        likelihood_awareness_state = A3_scaled[obs_entropy, :]
        qs_awareness = likelihood_awareness_state * prior_awareness
        qs_awareness = qs_awareness / (qs_awareness.sum() + EPS_VAL)
        posterior_awareness[t] = qs_awareness

        # ATTENTION INFERENCE: prior = B2 @ qs_prev, then likelihood update
        prior_attention = B2_attention[:, :, prev_action] @ qs_attention if t > 0 else qs_attention
        likelihood_att = A2_attention[obs_precision, :]
        qs_attention = likelihood_att * prior_attention
        qs_attention = qs_attention / (qs_attention.sum() + EPS_VAL)
        posterior_attention[t] = qs_attention

        # Action selection
        qs_attention_obj = utils.obj_array(1)
        qs_attention_obj[0] = qs_attention

        q_pi, G = update_posterior_policies(
            qs=qs_attention_obj, A=A2_obj, B=B2_obj, C=C2_obj,
            policies=policies, use_utility=True, use_states_info_gain=True,
            E=E, gamma=gamma,
        )

        q_pi_history[t] = q_pi.flatten()
        G_history[t] = G

        action = np.argmax(q_pi.flatten())
        action_history[t] = action
        prev_action = action

        # TRUE attention transition - USES B2_TRUE (distraction attractor)
        true_att_onehot = np.zeros(2)
        true_att_onehot[true_attention_state] = 1.0
        next_att_probs = B2_true[:, :, action] @ true_att_onehot
        true_attention_state = np.random.choice([FOCUSED, DISTRACTED], p=next_att_probs)

        # Get next observation
        obs_breath_val = int(env.step(None))

    # Compute metrics
    raw_att_acc = np.mean([posterior_attention[t, true_attention_history[t]] for t in range(T)])
    att_accuracy = (raw_att_acc - 0.5) / 0.5
    mean_p_focused = np.mean(posterior_attention[:, FOCUSED])
    mean_p_aware = np.mean(posterior_awareness[:, AWARE])
    mean_zeta = np.mean(zeta_history)

    print(f"Results:")
    print(f"  Attention accuracy: {att_accuracy:.1%}")
    print(f"  Mean P(focused): {mean_p_focused:.3f}")
    print(f"  Mean P(aware): {mean_p_aware:.3f}")
    print(f"  Mean ζ: {mean_zeta:.3f}")
    print(f"  Time in distracted: {np.mean(true_attention_history == DISTRACTED):.2%}")
    print(f"  Actions: STAY={np.mean(action_history == STAY):.2%}")

    return {
        "true_breath": true_breath,
        "obs_breath": obs_breath,
        "posterior_breath": posterior_breath,
        "zeta_history": zeta_history,
        "zeta_prior_history": zeta_prior_history,
        "posterior_attention": posterior_attention,
        "precision_obs_history": precision_obs_history,
        "posterior_awareness": posterior_awareness,
        "entropy_obs_history": entropy_obs_history,
        "breath_entropy_history": breath_entropy_history,
        "true_attention_history": true_attention_history,
        "action_history": action_history,
        "q_pi_history": q_pi_history,
        "G_history": G_history,
        "T": T,
        "A1_precision": A1_precision,
        "A2_precision": A2_precision,
        "A2_true_precision": A2_true_precision,
        "A3_precision": A3_precision,
        "zeta_focused": zeta_focused,
        "zeta_distracted": zeta_distracted,
        "zeta_awareness": zeta_awareness,
        "B2_agent_stay_prob": B2_agent_stay_prob,
        "gamma": gamma,
        "E_stay": E_stay,
        "C2_high_prec_pref": C2_high_prec_pref,
        "C2_low_prec_pref": C2_low_prec_pref,
    }


# =============================================================================
# Figure 6.2: With Meditation Instruction (awareness informs attention)
# =============================================================================

def run_figure6_2(
    T: int = 100,
    seed: int = 42,
    # Level 1 parameters
    A1_precision: float = 0.75,
    zeta_focused: float = 2.0,
    zeta_distracted: float = 0.5,
    zeta_step: float = 0.25,
    log_zeta_prior_var: float = 2.0,
    # Level 2: Attention - SEPARATE true vs agent
    A2_precision: float = 0.52,
    A2_true_precision: float = 0.9,
    B2_agent_stay_prob: float = 0.52,
    # Meditation instruction: awareness → attention (second modality)
    A2_awareness_precision: float = 0.75,
    # Level 2: Awareness
    A3_precision: float = 0.75,
    entropy_threshold: float = 0.5,
    zeta_awareness: float = 1.0,
    awareness_stay_prob: float = 0.95,
    # Preferences (meditation instruction)
    C_awareness_aware: float = 2.0,
    C_awareness_unaware: float = 0.0,
    # Initial beliefs
    initial_p_focused: float = 0.5,
    initial_p_aware: float = 0.5,
    # Policy selection
    gamma: float = 16.0,
    E_stay: float = 0.9,
    C2_high_prec_pref: float = 0.0,
    C2_low_prec_pref: float = 0.0,
):
    """
    Figure 6.2: With meditation instruction.

    Same as Figure 6.1, plus meditation instruction implemented as:
    - A2 extended to 2 modalities: precision obs AND awareness obs both → attention
    - C extended to 2 modalities: includes awareness preference [2, 0]
    - EFE naturally incorporates both observation types
    """
    np.random.seed(seed)
    env = BreathEnv(seed=seed)

    # Level 1 matrices (same as 6.1)
    A1 = np.array([
        [A1_precision, 1 - A1_precision],
        [1 - A1_precision, A1_precision]
    ])

    inhale_duration = env.inhale_range[1] - env.inhale_range[0]
    exhale_duration = env.exhale_range[1] - env.exhale_range[0]
    p_stay_inhale = (inhale_duration - 1) / inhale_duration
    p_stay_exhale = (exhale_duration - 1) / exhale_duration
    B1 = np.array([
        [p_stay_inhale, 1 - p_stay_exhale],
        [1 - p_stay_inhale, p_stay_exhale]
    ])

    # Level 2: A2_true for generating precision observations
    A2_true = np.array([
        [A2_true_precision, 1 - A2_true_precision],
        [1 - A2_true_precision, A2_true_precision]
    ])

    # Level 2: Agent's A2 with TWO MODALITIES (meditation instruction)
    # Modality 0: precision obs → attention (agent's weak model)
    A2_precision_mod = np.array([
        [A2_precision, 1 - A2_precision],
        [1 - A2_precision, A2_precision]
    ])
    # Modality 1: awareness obs → attention (meditation instruction)
    # "If I lose awareness of breath, my attention has wandered"
    A2_awareness_mod = np.array([
        [A2_awareness_precision, 1 - A2_awareness_precision],
        [1 - A2_awareness_precision, A2_awareness_precision],
    ])

    # Awareness model (for separate awareness state inference)
    A3 = np.array([
        [A3_precision, 1 - A3_precision],
        [1 - A3_precision, A3_precision]
    ])

    # B3: Awareness transitions
    B3 = np.array([
        [awareness_stay_prob, 1 - awareness_stay_prob],
        [1 - awareness_stay_prob, awareness_stay_prob]
    ])

    # B2_true: Distraction is absorbing under STAY, toggle for SWITCH
    B2_true = np.zeros((2, 2, 2))
    B2_true[:, :, STAY] = np.array([
        [0.8, 0.0],   # P(->foc | stay): can maintain focus, can't return from distracted
        [0.2, 1.0],   # P(->dist | stay): distraction is absorbing
    ])
    B2_true[:, :, SWITCH] = np.array([
        [0.1, 0.9],   # P(->foc | switch): foc->dist, dist->foc
        [0.9, 0.1],   # P(->dist | switch): toggles state
    ])

    # B2_agent: flat for STAY, knows SWITCH toggles state
    B2_attention = np.zeros((2, 2, 2))
    B2_attention[:, :, STAY] = np.array([
        [B2_agent_stay_prob, 1 - B2_agent_stay_prob],
        [1 - B2_agent_stay_prob, B2_agent_stay_prob],
    ])
    # Agent knows SWITCH toggles attention state (same as B2_true)
    B2_attention[:, :, SWITCH] = np.array([
        [0.1, 0.9],   # P(->foc | switch): foc->dist, dist->foc
        [0.9, 0.1],   # P(->dist | switch): toggles state
    ])

    # Policy setup
    E = np.array([E_stay, 1 - E_stay])
    policies = [np.array([[STAY]]), np.array([[SWITCH]])]

    # pymdp object arrays - TWO MODALITIES for attention
    # A2_obj[0]: precision obs → attention (weak)
    # A2_obj[1]: awareness obs → attention (meditation instruction)
    A2_obj = utils.obj_array(2)
    A2_obj[0] = A2_precision_mod
    A2_obj[1] = A2_awareness_mod

    B2_obj = utils.obj_array(1)
    B2_obj[0] = B2_attention

    # C2_obj with TWO MODALITIES
    # C2_obj[0]: preferences over precision obs (neutral)
    # C2_obj[1]: preferences over awareness obs (prefer aware)
    C2_obj = utils.obj_array(2)
    C2_obj[0] = np.array([C2_high_prec_pref, C2_low_prec_pref])
    C2_obj[1] = np.array([C_awareness_aware, C_awareness_unaware])

    # Storage
    true_breath = np.zeros(T, dtype=int)
    obs_breath = np.zeros(T, dtype=int)
    posterior_breath = np.zeros((T, 2))
    zeta_history = np.zeros(T)
    zeta_prior_history = np.zeros(T)
    posterior_attention = np.zeros((T, 2))
    precision_obs_history = np.zeros(T, dtype=int)
    posterior_awareness = np.zeros((T, 2))
    entropy_obs_history = np.zeros(T, dtype=int)
    breath_entropy_history = np.zeros(T)
    true_attention_history = np.zeros(T, dtype=int)
    action_history = np.zeros(T, dtype=int)
    q_pi_history = np.zeros((T, 2))
    G_history = np.zeros((T, 2))

    qs_breath = np.array([0.5, 0.5])
    qs_attention = np.array([initial_p_focused, 1 - initial_p_focused])
    qs_awareness = np.array([initial_p_aware, 1 - initial_p_aware])
    zeta = 1.0
    true_attention_state = FOCUSED
    prev_action = STAY
    obs_breath_val = int(env.step(None))

    print(f"Figure 6.2: With Meditation Instruction (within-sit)")
    print(f"  T={T}")
    print(f"  A2 has 2 modalities:")
    print(f"    Modality 0 (precision→attention): precision={A2_precision}")
    print(f"    Modality 1 (awareness→attention): precision={A2_awareness_precision}")
    print(f"  C has 2 modalities:")
    print(f"    Modality 0 (precision prefs): [{C2_high_prec_pref}, {C2_low_prec_pref}]")
    print(f"    Modality 1 (awareness prefs): [{C_awareness_aware}, {C_awareness_unaware}]")
    print(f"  γ={gamma}, E: P(Stay)={E_stay}")
    print()

    for t in range(T):
        true_breath[t] = env.state
        obs_breath[t] = obs_breath_val
        true_attention_history[t] = true_attention_state

        # Descending: TRUE attention → precision prior
        zeta_prior = zeta_focused if true_attention_state == FOCUSED else zeta_distracted
        zeta_prior_history[t] = zeta_prior

        # Precision update
        zeta, _, _ = update_likelihood_precision(
            zeta=zeta_prior, A=A1, obs=obs_breath_val, qs=qs_breath,
            log_zeta_prior_var=log_zeta_prior_var, zeta_step=zeta_step,
        )
        zeta_history[t] = zeta

        # BREATH INFERENCE: prior = B1 @ qs_prev, then likelihood update
        prior_breath = B1 @ qs_breath if t > 0 else qs_breath
        A1_scaled = scale_likelihood(A1, zeta)
        likelihood_breath = A1_scaled[obs_breath_val, :]
        qs_breath = likelihood_breath * prior_breath
        qs_breath = qs_breath / (qs_breath.sum() + EPS_VAL)
        posterior_breath[t] = qs_breath

        # Breath entropy
        breath_H = entropy_dist(qs_breath)
        breath_entropy_history[t] = breath_H

        # Ascending observations
        obs_precision = np.random.choice([HIGH_PRECISION, LOW_PRECISION], p=A2_true[:, true_attention_state])
        precision_obs_history[t] = obs_precision

        obs_entropy = LOW_ENTROPY if breath_H < entropy_threshold else HIGH_ENTROPY
        entropy_obs_history[t] = obs_entropy

        # AWARENESS INFERENCE: prior = B3 @ qs_prev, then likelihood update
        prior_awareness = B3 @ qs_awareness if t > 0 else qs_awareness
        A3_scaled = scale_likelihood(A3, zeta_awareness)
        likelihood_awareness_state = A3_scaled[obs_entropy, :]
        qs_awareness = likelihood_awareness_state * prior_awareness
        qs_awareness = qs_awareness / (qs_awareness.sum() + EPS_VAL)
        posterior_awareness[t] = qs_awareness

        # ATTENTION INFERENCE: prior = B2 @ qs_prev, then likelihood update
        prior_attention = B2_attention[:, :, prev_action] @ qs_attention if t > 0 else qs_attention

        # Combined likelihood from both modalities
        likelihood_precision = A2_precision_mod[obs_precision, :]  # Modality 0
        likelihood_awareness = A2_awareness_mod[obs_entropy, :]    # Modality 1
        likelihood_combined = likelihood_precision * likelihood_awareness

        qs_attention = likelihood_combined * prior_attention
        qs_attention = qs_attention / (qs_attention.sum() + EPS_VAL)
        posterior_attention[t] = qs_attention

        # Action selection via EFE (now uses both modalities automatically)
        qs_attention_obj = utils.obj_array(1)
        qs_attention_obj[0] = qs_attention

        q_pi, G = update_posterior_policies(
            qs=qs_attention_obj, A=A2_obj, B=B2_obj, C=C2_obj,
            policies=policies, use_utility=True, use_states_info_gain=True,
            E=E, gamma=gamma,
        )

        q_pi_history[t] = q_pi.flatten()
        G_history[t] = G

        action = np.argmax(q_pi)
        action_history[t] = action
        prev_action = action

        # TRUE attention transition
        true_att_onehot = np.zeros(2)
        true_att_onehot[true_attention_state] = 1.0
        next_att_probs = B2_true[:, :, action] @ true_att_onehot
        true_attention_state = np.random.choice([FOCUSED, DISTRACTED], p=next_att_probs)

        # Get next observation
        obs_breath_val = int(env.step(None))

    raw_att_acc = np.mean([posterior_attention[t, true_attention_history[t]] for t in range(T)])
    att_accuracy = (raw_att_acc - 0.5) / 0.5
    mean_p_aware = np.mean(posterior_awareness[:, AWARE])

    print(f"Results:")
    print(f"  Attention accuracy: {att_accuracy:.1%}")
    print(f"  Mean P(aware): {mean_p_aware:.3f}")
    print(f"  Time in distracted: {np.mean(true_attention_history == DISTRACTED):.2%}")
    print(f"  Actions: STAY={np.mean(action_history == STAY):.2%}, SWITCH={np.mean(action_history == SWITCH):.2%}")

    return {
        "true_breath": true_breath,
        "obs_breath": obs_breath,
        "posterior_breath": posterior_breath,
        "zeta_history": zeta_history,
        "zeta_prior_history": zeta_prior_history,
        "posterior_attention": posterior_attention,
        "precision_obs_history": precision_obs_history,
        "posterior_awareness": posterior_awareness,
        "entropy_obs_history": entropy_obs_history,
        "breath_entropy_history": breath_entropy_history,
        "true_attention_history": true_attention_history,
        "action_history": action_history,
        "q_pi_history": q_pi_history,
        "G_history": G_history,
        "T": T,
        "A2_precision": A2_precision,
        "A2_awareness_precision": A2_awareness_precision,
        "C_awareness_aware": C_awareness_aware,
        "C_awareness_unaware": C_awareness_unaware,
        "zeta_focused": zeta_focused,
        "zeta_distracted": zeta_distracted,
        "zeta_awareness": zeta_awareness,
        "gamma": gamma,
        "E_stay": E_stay,
    }


def plot_figure6_1_2(results: dict, title: str, save_path: str = None):
    """Plot Figure 6.1 or 6.2 using same style as plot_figure6."""
    plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 10})

    T = results["T"]
    time = np.arange(T)

    fig, axes = plt.subplots(5, 1, figsize=(12, 10), sharex=True)

    BLUE = '#2563eb'
    ORANGE = '#ea580c'
    TEAL = '#0d9488'
    GRAY = '#6b7280'

    # Panel 1: Awareness (same as original figure 6)
    ax = axes[0]
    ax.plot(time, results["posterior_awareness"][:, AWARE], color=TEAL, linewidth=1.5, label='P(Aware)')
    entropy_obs = results["entropy_obs_history"]
    ax.scatter(time[entropy_obs == LOW_ENTROPY],
               np.ones(np.sum(entropy_obs == LOW_ENTROPY)) * 0.95,
               c=GRAY, s=15, alpha=0.7, marker='s', label='Low entropy')
    ax.scatter(time[entropy_obs == HIGH_ENTROPY],
               np.zeros(np.sum(entropy_obs == HIGH_ENTROPY)) + 0.05,
               c=ORANGE, s=15, alpha=0.7, marker='s', label='High entropy')
    ax.set_ylabel("Probability")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='upper right', fontsize=8)
    ax.set_title("Level 2: Awareness State (from breath entropy)", fontsize=11)

    # Panel 2: Action Selection
    ax = axes[1]
    actions = results["action_history"]
    q_pi = results["q_pi_history"]
    ax.plot(time, q_pi[:, STAY], color=TEAL, linewidth=1.5, label='P(Stay)')
    ax.scatter(time[actions == STAY], np.ones(np.sum(actions == STAY)) * 1.0,
               c=TEAL, s=20, marker='.', label='Selected: Stay')
    ax.scatter(time[actions == SWITCH], np.ones(np.sum(actions == SWITCH)) * 0.0,
               c=ORANGE, s=20, marker='.', label='Selected: Switch')
    ax.set_ylabel("Probability")
    ax.set_ylim(-0.1, 1.1)
    ax.legend(loc='upper right', fontsize=8)
    ax.set_title("Level 2: Action Selection (EFE-based)", fontsize=11)

    # Panel 3: Attention State
    ax = axes[2]
    true_att = results["true_attention_history"]
    ax.scatter(time[true_att == FOCUSED], np.ones(np.sum(true_att == FOCUSED)) * 0.95,
               c=GRAY, s=15, alpha=0.7, marker='*')
    ax.scatter(time[true_att == DISTRACTED], np.zeros(np.sum(true_att == DISTRACTED)) + 0.05,
               c=ORANGE, s=15, alpha=0.7, marker='*')
    ax.plot(time, results["posterior_attention"][:, FOCUSED], color=BLUE,
            linewidth=1.5, label='P(Focused)')
    ax.axhline(y=0.5, color=GRAY, linestyle=':', alpha=0.5)
    ax.set_ylabel("Probability")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='upper right', fontsize=8)
    ax.set_title("Level 2: Attention State (true vs inferred)", fontsize=11)

    # Panel 4: Precision
    ax = axes[3]
    ax.plot(time, results["zeta_history"], color=ORANGE, linewidth=1.5, label='ζ posterior')
    ax.plot(time, results["zeta_prior_history"], color=BLUE, linewidth=1,
            alpha=0.5, linestyle='--', label='ζ prior (↓)')
    ax.axhline(y=results["zeta_focused"], color=GRAY, linestyle=':', alpha=0.5)
    ax.axhline(y=results["zeta_distracted"], color=GRAY, linestyle=':', alpha=0.5)
    ax.text(T + 1, results["zeta_focused"], f'ζ_foc={results["zeta_focused"]}', fontsize=8, va='center')
    ax.text(T + 1, results["zeta_distracted"], f'ζ_dist={results["zeta_distracted"]}', fontsize=8, va='center')
    ax.set_ylabel("Precision (ζ)")
    ax.legend(loc='upper right', fontsize=8)
    ax.set_title("Likelihood Precision", fontsize=11)

    # Panel 5: Breath Perception
    ax = axes[4]
    ax.plot(time, results["posterior_breath"][:, INHALE], color=BLUE, linewidth=1.5, label='P(Inhaling)')
    true_breath = results["true_breath"]
    ax.scatter(time[true_breath == INHALE], np.ones(np.sum(true_breath == INHALE)) * 0.95,
               c=BLUE, s=10, alpha=0.5, marker='*')
    ax.scatter(time[true_breath == EXHALE], np.zeros(np.sum(true_breath == EXHALE)) + 0.05,
               c=ORANGE, s=10, alpha=0.5, marker='*')
    ax.axhline(y=0.5, color=GRAY, linestyle=':', alpha=0.5)
    ax.set_ylabel("Probability")
    ax.set_xlabel("Time step")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='upper right', fontsize=8)
    ax.set_title("Level 1: Breath Perception", fontsize=11)

    plt.suptitle(title, fontsize=13, fontweight='bold')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor='white')
        print(f"Saved: {save_path}")

    return fig


# =============================================================================
# Figure 7: A2 Matrix Learning Across Sits
# =============================================================================

def normalize_A(pA):
    """Normalize Dirichlet parameters to get A matrix."""
    return pA / pA.sum(axis=0, keepdims=True)


def run_A2_learning(
    num_sits: int = 200,
    T_per_sit: int = 200,
    seed: int = 42,
    lr: float = 1.0,
    fr: float = 0.9,
    initial_pA2_strength: float = 2.0,
    initial_A2_precision: float = 0.6,  # Agent's initial A2 diagonal
    # Level 1 parameters
    A1_precision: float = 0.75,
    zeta_focused: float = 2.0,
    zeta_distracted: float = 0.5,
    zeta_step: float = 0.25,
    log_zeta_prior_var: float = 2.0,
    # A2 true precision (what we're trying to learn)
    A2_true_precision: float = 0.9,
    # B2 parameters
    B2_stay_prob: float = 0.95,
    B2_switch_prob: float = 0.95,
    terminal_distraction: bool = False,
    # Policy parameters (no preferences for learning)
    gamma: float = 16.0,
    E_stay: float = 0.9,
):
    """
    Figure 7: A2 matrix learning across meditation sits.
    
    The agent learns the attention observation model A2 from experience:
    A2[o, s] = P(precision_obs=o | attention_state=s)
    
    True A2: P(high_prec | focused) = A2_true_precision
    """
    np.random.seed(seed)
    
    # True A2 matrix
    A2_true = np.array([
        [A2_true_precision, 1 - A2_true_precision],
        [1 - A2_true_precision, A2_true_precision]
    ])
    
    # Initialize pA2 with specified initial precision (as object array for pymdp)
    pA2_obj = utils.obj_array(1)
    pA2_obj[0] = np.array([
        [initial_A2_precision, 1 - initial_A2_precision],
        [1 - initial_A2_precision, initial_A2_precision]
    ]) * initial_pA2_strength
    A2 = normalize_A(pA2_obj[0])
    
    # Level 1 matrices (fixed, not learned)
    A1 = np.array([
        [A1_precision, 1 - A1_precision],
        [1 - A1_precision, A1_precision]
    ])
    
    # B2 action-dependent transitions
    B2_attention = np.zeros((2, 2, 2))
    if terminal_distraction:
        B2_attention[:, :, STAY] = np.array([
            [B2_stay_prob, 0.0],
            [1 - B2_stay_prob, 1.0],
        ])
        B2_attention[:, :, SWITCH] = np.array([
            [0.0, 1.0],
            [1.0, 0.0],
        ])
    else:
        B2_attention[:, :, STAY] = np.array([
            [B2_stay_prob, 1 - B2_stay_prob],
            [1 - B2_stay_prob, B2_stay_prob],
        ])
        B2_attention[:, :, SWITCH] = np.array([
            [1 - B2_switch_prob, B2_switch_prob],
            [B2_switch_prob, 1 - B2_switch_prob],
        ])
    
    # Policy prior and preferences (flat preferences for learning)
    E = np.array([E_stay, 1 - E_stay])
    C2 = np.array([0.0, 0.0])  # No preferences
    
    print(f"A2 Matrix Learning (Figure 7)")
    print(f"  Sits: {num_sits}, T/sit: {T_per_sit}")
    print(f"  True A2 diagonal: {A2_true_precision:.3f}")
    print(f"  Initial A2 diagonal: [{A2[0,0]:.3f}, {A2[1,1]:.3f}]")
    print(f"  Learning rate: {lr}, Forgetting rate: {fr}")
    print(f"  Terminal distraction: {terminal_distraction}")
    print()
    
    # Logging
    A2_diagonal_history = np.zeros((num_sits + 1, 2))
    A2_diagonal_history[0] = [A2[0, 0], A2[1, 1]]
    accuracy_history = np.zeros(num_sits)
    mean_p_focused_history = np.zeros(num_sits)
    switch_rate_history = np.zeros(num_sits)
    
    # pymdp object arrays for policy computation
    policies = [np.array([[STAY]]), np.array([[SWITCH]])]
    
    for sit in range(num_sits):
        env = BreathEnv(seed=seed + sit)
        
        # Compute B1 from env
        inhale_duration = env.inhale_range[1] - env.inhale_range[0]
        exhale_duration = env.exhale_range[1] - env.exhale_range[0]
        p_stay_inhale = (inhale_duration - 1) / inhale_duration
        p_stay_exhale = (exhale_duration - 1) / exhale_duration
        B1 = np.array([
            [p_stay_inhale, 1 - p_stay_exhale],
            [1 - p_stay_inhale, p_stay_exhale]
        ])
        
        # Initialize for this sit
        qs_breath = np.array([0.5, 0.5])
        qs_attention = np.array([0.5, 0.5])
        true_attention_state = FOCUSED
        prev_action = STAY
        obs_breath_val = int(env.step(None))
        
        # pymdp object arrays
        A2_obj = utils.obj_array(1)
        A2_obj[0] = A2
        B2_obj = utils.obj_array(1)
        B2_obj[0] = B2_attention
        C2_obj = utils.obj_array(1)
        C2_obj[0] = C2
        
        # Storage for this sit
        precision_obs_list = []
        posterior_attention_list = []
        true_breath_list = []
        posterior_breath_list = []
        action_list = []
        true_attention_list = []
        
        for t in range(T_per_sit):
            true_breath_list.append(env.state)
            true_attention_list.append(true_attention_state)
            
            # Precision from true attention state
            if true_attention_state == FOCUSED:
                zeta_prior = zeta_focused
            else:
                zeta_prior = zeta_distracted
            
            # Precision update
            zeta, _, _ = update_likelihood_precision(
                zeta=zeta_prior,
                A=A1,
                obs=obs_breath_val,
                qs=qs_breath,
                log_zeta_prior_var=log_zeta_prior_var,
                zeta_step=zeta_step,
            )
            
            # Breath inference
            A1_scaled = scale_likelihood(A1, zeta)
            qs_breath = bayesian_update(A1_scaled, obs_breath_val, qs_breath)
            posterior_breath_list.append(qs_breath.copy())
            
            # Ascending message: TRUE state → precision observation (probabilistic)
            # Sample from true A2 likelihood
            obs_precision = np.random.choice(
                [HIGH_PRECISION, LOW_PRECISION],
                p=A2_true[:, true_attention_state]
            )
            precision_obs_list.append(obs_precision)
            
            # Attention belief update
            prior_attention = B2_attention[:, :, prev_action] @ qs_attention if t > 0 else qs_attention
            likelihood_att = A2[obs_precision, :]
            qs_attention = likelihood_att * prior_attention
            qs_attention = qs_attention / (qs_attention.sum() + EPS_VAL)
            posterior_attention_list.append(qs_attention.copy())
            
            # Action selection
            qs_attention_obj = utils.obj_array(1)
            qs_attention_obj[0] = qs_attention
            
            q_pi, G = update_posterior_policies(
                qs=qs_attention_obj,
                A=A2_obj,
                B=B2_obj,
                C=C2_obj,
                policies=policies,
                use_utility=True,
                use_states_info_gain=True,
                E=E,
                gamma=gamma,
            )
            
            action = np.argmax(q_pi.flatten())
            action_list.append(action)
            prev_action = action
            
            # True attention transition
            true_attention_one_hot = np.zeros(2)
            true_attention_one_hot[true_attention_state] = 1.0
            next_true_attention = B2_attention[:, :, action] @ true_attention_one_hot
            true_attention_state = np.random.choice([FOCUSED, DISTRACTED], p=next_true_attention)
            
            # Advance
            qs_breath = B1 @ qs_breath
            qs_breath = qs_breath / (qs_breath.sum() + EPS_VAL)
            obs_breath_val = int(env.step(None))
        
        # Compute breath accuracy for this sit (normalized: 0%=chance, 100%=perfect)
        posterior_breath_arr = np.array(posterior_breath_list)
        true_breath_arr = np.array(true_breath_list)
        raw_breath_acc = np.mean(posterior_breath_arr[np.arange(len(true_breath_arr)), true_breath_arr])
        accuracy = (raw_breath_acc - 0.5) / 0.5  # Normalize to [0, 1] where 0=chance
        accuracy_history[sit] = accuracy

        # Compute attention accuracy for this sit (normalized: 0%=chance, 100%=perfect)
        posterior_attention_arr = np.array(posterior_attention_list)
        true_attention_arr = np.array(true_attention_list)
        raw_attention_acc = np.mean(posterior_attention_arr[np.arange(len(true_attention_arr)), true_attention_arr])
        attention_accuracy = (raw_attention_acc - 0.5) / 0.5  # Normalize to [0, 1] where 0=chance
        mean_p_focused_history[sit] = attention_accuracy
        
        # Switch rate
        action_arr = np.array(action_list)
        switch_rate_history[sit] = np.mean(action_arr == SWITCH)
        
        # Update A2 after sit (Dirichlet learning) using pymdp learning function
        # Apply forgetting once per sit (before accumulating new evidence)
        pA2_obj[0] = fr * pA2_obj[0]

        # Accumulate learning from all timesteps (fr=1.0 within sit)
        for t_idx in range(len(precision_obs_list)):
            obs = precision_obs_list[t_idx]
            qs_obj = utils.obj_array(1)
            qs_obj[0] = posterior_attention_list[t_idx]
            pA2_obj = update_obs_likelihood_dirichlet(
                pA=pA2_obj, A=A2_obj, obs=obs, qs=qs_obj, lr=lr, fr=1.0
            )
        A2 = normalize_A(pA2_obj[0])
        
        # Log
        A2_diagonal_history[sit + 1] = [A2[0, 0], A2[1, 1]]
        
        if (sit + 1) % 20 == 0 or sit == 0:
            print(f"  Sit {sit+1:3d}: BreathAcc={accuracy:.3f}, AttAcc={attention_accuracy:.3f}, "
                  f"A2_diag=[{A2[0,0]:.3f}, {A2[1,1]:.3f}], Switch={switch_rate_history[sit]:.2f}")
    
    print()
    print(f"Final A2 diagonal: [{A2[0,0]:.4f}, {A2[1,1]:.4f}]")
    print(f"True A2 diagonal:  [{A2_true_precision:.4f}, {A2_true_precision:.4f}]")
    
    return {
        "A2_diagonal_history": A2_diagonal_history,
        "accuracy_history": accuracy_history,
        "attention_accuracy_history": mean_p_focused_history,  # Renamed
        "switch_rate_history": switch_rate_history,
        "A2_true_precision": A2_true_precision,
        "initial_A2_precision": initial_A2_precision,
        "A2_final": A2,
        "num_sits": num_sits,
        "lr": lr,
        "fr": fr,
        "terminal_distraction": terminal_distraction,
    }


# =============================================================================
# Figure 7.1: Joint A2 + B2 Learning (Non-meditator baseline)
# =============================================================================

def run_A2_B2_learning(
    num_sits: int = 200,
    T_per_sit: int = 100,
    seed: int = 42,
    lr: float = 1.0,
    fr: float = 0.9,
    # A2 parameters
    initial_pA2_strength: float = 2.0,
    initial_A2_precision: float = 0.52,  # Agent's initial A2 diagonal
    A2_true_precision: float = 0.9,  # True A2 (generates observations)
    # B2 parameters - NOW SEPARATE for true vs agent
    initial_pB2_strength: float = 2.0,
    initial_B2_stay_prob: float = 0.5,  # Agent starts with flat/uncertain B2
    B2_true_stay_prob: float = 0.8,  # True attention dynamics
    # Level 1 parameters
    A1_precision: float = 0.75,
    zeta_focused: float = 2.0,
    zeta_distracted: float = 0.5,
    zeta_step: float = 0.25,
    log_zeta_prior_var: float = 2.0,
    # Policy parameters
    gamma: float = 16.0,
    E_stay: float = 0.9,
):
    """
    Figure 7.1: Joint A2 + B2 learning across meditation sits.

    Non-meditator baseline: Agent must learn BOTH:
    - A2: observation model (precision → attention state)
    - B2: dynamics model (attention transitions)

    This is harder because:
    - Accurate posteriors need both good A2 and good B2
    - Learning each requires accurate posteriors
    - Chicken-and-egg problem leads to slower convergence
    """
    np.random.seed(seed)

    # =========================================================================
    # TRUE models (generate observations and transitions)
    # =========================================================================
    A2_true = np.array([
        [A2_true_precision, 1 - A2_true_precision],
        [1 - A2_true_precision, A2_true_precision]
    ])

    # B2_true: Distraction is absorbing under STAY, toggle for SWITCH
    B2_true = np.zeros((2, 2, 2))
    B2_true[:, :, STAY] = np.array([
        [0.8, 0.0],   # P(->foc | stay): can maintain focus, can't return from distracted
        [0.2, 1.0],   # P(->dist | stay): distraction is absorbing
    ])
    B2_true[:, :, SWITCH] = np.array([
        [0.1, 0.9],   # P(->foc | switch): foc->dist, dist->foc (toggle)
        [0.9, 0.1],   # P(->dist | switch): toggles state
    ])

    # =========================================================================
    # AGENT's models (learned from experience)
    # =========================================================================

    # Initialize pA2 (Dirichlet parameters for A2 learning)
    pA2_obj = utils.obj_array(1)
    pA2_obj[0] = np.array([
        [initial_A2_precision, 1 - initial_A2_precision],
        [1 - initial_A2_precision, initial_A2_precision]
    ]) * initial_pA2_strength
    A2 = normalize_A(pA2_obj[0])

    # Initialize pB2 (Dirichlet parameters for B2 learning)
    # Agent starts with uncertain/flat beliefs about attention dynamics
    pB2_obj = utils.obj_array(1)
    pB2_obj[0] = np.zeros((2, 2, 2))
    pB2_obj[0][:, :, STAY] = np.array([
        [initial_B2_stay_prob, 1 - initial_B2_stay_prob],
        [1 - initial_B2_stay_prob, initial_B2_stay_prob],
    ]) * initial_pB2_strength
    pB2_obj[0][:, :, SWITCH] = np.array([
        [1 - initial_B2_stay_prob, initial_B2_stay_prob],
        [initial_B2_stay_prob, 1 - initial_B2_stay_prob],
    ]) * initial_pB2_strength

    # Normalize to get B2
    B2 = np.zeros_like(pB2_obj[0])
    for a in range(2):
        for s in range(2):
            col_sum = pB2_obj[0][:, s, a].sum()
            B2[:, s, a] = pB2_obj[0][:, s, a] / col_sum if col_sum > 0 else 0.5

    # Level 1 matrices (fixed, not learned)
    A1 = np.array([
        [A1_precision, 1 - A1_precision],
        [1 - A1_precision, A1_precision]
    ])

    # Policy prior and preferences (flat preferences)
    E = np.array([E_stay, 1 - E_stay])
    C2 = np.array([0.0, 0.0])

    print(f"Joint A2 + B2 Learning (Figure 7.1)")
    print(f"  Sits: {num_sits}, T/sit: {T_per_sit}")
    print(f"  A2 - True precision: {A2_true_precision:.3f}, Initial: {initial_A2_precision:.3f}")
    print(f"  B2 - True stay prob: {B2_true_stay_prob:.3f}, Initial: {initial_B2_stay_prob:.3f}")
    print(f"  Learning rate: {lr}, Forgetting rate: {fr}")
    print()

    # =========================================================================
    # Logging
    # =========================================================================
    A2_diagonal_history = np.zeros((num_sits + 1, 2))
    A2_diagonal_history[0] = [A2[0, 0], A2[1, 1]]

    B2_stay_history = np.zeros((num_sits + 1, 2))  # B2 stay probs for STAY action
    B2_stay_history[0] = [B2[0, 0, STAY], B2[1, 1, STAY]]  # P(foc|foc,stay), P(dist|dist,stay)

    accuracy_history = np.zeros(num_sits)
    attention_accuracy_history = np.zeros(num_sits)
    switch_rate_history = np.zeros(num_sits)

    # pymdp policies for action selection
    policies = [np.array([[STAY]]), np.array([[SWITCH]])]

    # =========================================================================
    # Main learning loop
    # =========================================================================
    for sit in range(num_sits):
        env = BreathEnv(seed=seed + sit)

        # Compute B1 from env
        inhale_duration = env.inhale_range[1] - env.inhale_range[0]
        exhale_duration = env.exhale_range[1] - env.exhale_range[0]
        p_stay_inhale = (inhale_duration - 1) / inhale_duration
        p_stay_exhale = (exhale_duration - 1) / exhale_duration
        B1 = np.array([
            [p_stay_inhale, 1 - p_stay_exhale],
            [1 - p_stay_inhale, p_stay_exhale]
        ])

        # Initialize for this sit
        qs_breath = np.array([0.5, 0.5])
        qs_attention = np.array([0.5, 0.5])
        qs_attention_prev = np.array([0.5, 0.5])  # For B2 learning
        true_attention_state = FOCUSED
        prev_action = STAY
        obs_breath_val = int(env.step(None))

        # pymdp object arrays for policy computation (use agent's learned models)
        A2_obj = utils.obj_array(1)
        A2_obj[0] = A2
        B2_obj = utils.obj_array(1)
        B2_obj[0] = B2  # Agent's learned B2
        C2_obj = utils.obj_array(1)
        C2_obj[0] = C2

        # Storage for this sit
        precision_obs_list = []
        posterior_attention_list = []
        posterior_attention_prev_list = []  # For B2 learning
        action_list = []
        true_breath_list = []
        posterior_breath_list = []
        true_attention_list = []

        for t in range(T_per_sit):
            true_breath_list.append(env.state)
            true_attention_list.append(true_attention_state)

            # Store previous posterior for B2 learning
            qs_attention_prev = qs_attention.copy()
            if t > 0:
                posterior_attention_prev_list.append(qs_attention_prev)

            # Precision from TRUE attention state
            if true_attention_state == FOCUSED:
                zeta_prior = zeta_focused
            else:
                zeta_prior = zeta_distracted

            # Precision update
            zeta, _, _ = update_likelihood_precision(
                zeta=zeta_prior,
                A=A1,
                obs=obs_breath_val,
                qs=qs_breath,
                log_zeta_prior_var=log_zeta_prior_var,
                zeta_step=zeta_step,
            )

            # Breath inference
            A1_scaled = scale_likelihood(A1, zeta)
            qs_breath = bayesian_update(A1_scaled, obs_breath_val, qs_breath)
            posterior_breath_list.append(qs_breath.copy())

            # Ascending message: TRUE state → precision observation (sampled from A2_true)
            obs_precision = np.random.choice(
                [HIGH_PRECISION, LOW_PRECISION],
                p=A2_true[:, true_attention_state]
            )
            precision_obs_list.append(obs_precision)

            # Attention belief update using AGENT's A2 and B2
            prior_attention = B2[:, :, prev_action] @ qs_attention if t > 0 else qs_attention
            likelihood_att = A2[obs_precision, :]
            qs_attention = likelihood_att * prior_attention
            qs_attention = qs_attention / (qs_attention.sum() + EPS_VAL)
            posterior_attention_list.append(qs_attention.copy())

            # Action selection using agent's models
            qs_attention_obj = utils.obj_array(1)
            qs_attention_obj[0] = qs_attention

            q_pi, G = update_posterior_policies(
                qs=qs_attention_obj,
                A=A2_obj,
                B=B2_obj,
                C=C2_obj,
                policies=policies,
                use_utility=True,
                use_states_info_gain=True,
                E=E,
                gamma=gamma,
            )

            action = np.argmax(q_pi.flatten())
            action_list.append(action)
            prev_action = action

            # TRUE attention transition (using B2_true, not agent's B2)
            true_attention_one_hot = np.zeros(2)
            true_attention_one_hot[true_attention_state] = 1.0
            next_true_attention = B2_true[:, :, action] @ true_attention_one_hot
            true_attention_state = np.random.choice([FOCUSED, DISTRACTED], p=next_true_attention)

            # Advance breath
            qs_breath = B1 @ qs_breath
            qs_breath = qs_breath / (qs_breath.sum() + EPS_VAL)
            obs_breath_val = int(env.step(None))

        # =====================================================================
        # Compute accuracies for this sit
        # =====================================================================

        # Breath accuracy (normalized: 0%=chance, 100%=perfect)
        posterior_breath_arr = np.array(posterior_breath_list)
        true_breath_arr = np.array(true_breath_list)
        raw_breath_acc = np.mean(posterior_breath_arr[np.arange(len(true_breath_arr)), true_breath_arr])
        accuracy_history[sit] = (raw_breath_acc - 0.5) / 0.5

        # Attention accuracy (normalized: 0%=chance, 100%=perfect)
        posterior_attention_arr = np.array(posterior_attention_list)
        true_attention_arr = np.array(true_attention_list)
        raw_attention_acc = np.mean(posterior_attention_arr[np.arange(len(true_attention_arr)), true_attention_arr])
        attention_accuracy_history[sit] = (raw_attention_acc - 0.5) / 0.5

        # Switch rate
        action_arr = np.array(action_list)
        switch_rate_history[sit] = np.mean(action_arr == SWITCH)

        # =====================================================================
        # Learning updates
        # =====================================================================

        # A2 learning: apply forgetting, then accumulate
        pA2_obj[0] = fr * pA2_obj[0]
        for t_idx in range(len(precision_obs_list)):
            obs = precision_obs_list[t_idx]
            qs_obj = utils.obj_array(1)
            qs_obj[0] = posterior_attention_list[t_idx]
            pA2_obj = update_obs_likelihood_dirichlet(
                pA=pA2_obj, A=A2_obj, obs=obs, qs=qs_obj, lr=lr, fr=1.0
            )
        A2 = normalize_A(pA2_obj[0])
        A2_obj[0] = A2  # Update for next sit

        # B2 learning: apply forgetting, then accumulate
        pB2_obj[0] = fr * pB2_obj[0]
        for t_idx in range(len(posterior_attention_prev_list)):
            # qs_prev is posterior at t, qs is posterior at t+1
            qs_prev_obj = utils.obj_array(1)
            qs_prev_obj[0] = posterior_attention_prev_list[t_idx]
            qs_curr_obj = utils.obj_array(1)
            qs_curr_obj[0] = posterior_attention_list[t_idx + 1]  # t+1
            action_t = action_list[t_idx]

            pB2_obj = update_state_likelihood_dirichlet(
                pB=pB2_obj,
                B=B2_obj,
                actions=np.array([action_t]),
                qs=qs_curr_obj,
                qs_prev=qs_prev_obj,
                lr=lr,
            )

        # Normalize B2
        for a in range(2):
            for s in range(2):
                col_sum = pB2_obj[0][:, s, a].sum()
                B2[:, s, a] = pB2_obj[0][:, s, a] / col_sum if col_sum > 0 else 0.5
        B2_obj[0] = B2  # Update for next sit

        # Log
        A2_diagonal_history[sit + 1] = [A2[0, 0], A2[1, 1]]
        B2_stay_history[sit + 1] = [B2[0, 0, STAY], B2[1, 1, STAY]]

        if (sit + 1) % 20 == 0 or sit == 0:
            print(f"  Sit {sit+1:3d}: BreathAcc={accuracy_history[sit]:.3f}, "
                  f"AttAcc={attention_accuracy_history[sit]:.3f}, "
                  f"A2=[{A2[0,0]:.3f}, {A2[1,1]:.3f}], "
                  f"B2_stay=[{B2[0,0,STAY]:.3f}, {B2[1,1,STAY]:.3f}]")

    print()
    print(f"Final A2 diagonal: [{A2[0,0]:.4f}, {A2[1,1]:.4f}] (True: {A2_true_precision})")
    print(f"Final B2 stay:     [{B2[0,0,STAY]:.4f}, {B2[1,1,STAY]:.4f}] (True: {B2_true_stay_prob})")

    return {
        "A2_diagonal_history": A2_diagonal_history,
        "B2_stay_history": B2_stay_history,
        "accuracy_history": accuracy_history,
        "attention_accuracy_history": attention_accuracy_history,
        "switch_rate_history": switch_rate_history,
        "A2_true_precision": A2_true_precision,
        "B2_true_stay_prob": B2_true_stay_prob,
        "initial_A2_precision": initial_A2_precision,
        "initial_B2_stay_prob": initial_B2_stay_prob,
        "A2_final": A2,
        "B2_final": B2,
        "num_sits": num_sits,
        "lr": lr,
        "fr": fr,
    }


def plot_figure7_1(results: dict, save_path: str = None):
    """
    Plot Figure 7.1: Joint A2 + B2 learning trajectory.

    Three panels:
    A) A2 diagonal entries over sits
    B) B2 stay probabilities over sits
    C) Inference accuracy over sits
    """
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 11,
    })

    num_sits = results["num_sits"]
    sit_range = np.arange(num_sits + 1)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    BLUE = '#2563eb'
    ORANGE = '#ea580c'
    PURPLE = '#7c3aed'
    TEAL = '#0d9488'
    GRAY = '#6b7280'

    # Panel A: A2 diagonal learning
    ax = axes[0]
    A2_diag = results["A2_diagonal_history"]
    ax.plot(sit_range, A2_diag[:, 0], color=BLUE, linewidth=1.5, label='A2[0,0] - P(high|foc)')
    ax.plot(sit_range, A2_diag[:, 1], color=ORANGE, linewidth=1.5, label='A2[1,1] - P(low|dist)')
    ax.axhline(y=results["A2_true_precision"], color=GRAY, linestyle='--',
               linewidth=1.5, label=f'True = {results["A2_true_precision"]}')
    ax.axhline(y=0.5, color=GRAY, linestyle=':', alpha=0.5)
    ax.set_ylabel("A2 diagonal entry")
    ax.set_ylim(0.4, 1.0)
    ax.legend(loc='lower right', fontsize=9)
    ax.set_title(f"A2 Learning (Observation Model)", fontsize=12)

    # Panel B: B2 stay probability learning
    ax = axes[1]
    B2_stay = results["B2_stay_history"]
    ax.plot(sit_range, B2_stay[:, 0], color=BLUE, linewidth=1.5, label='B2[foc,foc,stay]')
    ax.plot(sit_range, B2_stay[:, 1], color=ORANGE, linewidth=1.5, label='B2[dist,dist,stay]')
    ax.axhline(y=results["B2_true_stay_prob"], color=GRAY, linestyle='--',
               linewidth=1.5, label=f'True = {results["B2_true_stay_prob"]}')
    ax.axhline(y=0.5, color=GRAY, linestyle=':', alpha=0.5)
    ax.set_ylabel("B2 stay probability")
    ax.set_ylim(0.4, 1.0)
    ax.legend(loc='lower right', fontsize=9)
    ax.set_title(f"B2 Learning (Dynamics Model)", fontsize=12)

    # Panel C: Inference accuracy
    ax = axes[2]
    sit_range_acc = np.arange(num_sits)
    ax.plot(sit_range_acc, results["accuracy_history"], color=PURPLE,
            linewidth=1.5, label='Breath')
    ax.plot(sit_range_acc, results["attention_accuracy_history"], color=TEAL,
            linewidth=1.5, label='Attention')
    ax.axhline(y=1.0, color=GRAY, linestyle=':', alpha=0.5, label='Perfect')
    ax.axhline(y=0.0, color=GRAY, linestyle=':', alpha=0.5, label='Chance')
    ax.set_xlabel("Sit")
    ax.set_ylabel("Inference Accuracy\n(0%=chance, 100%=perfect)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='lower right', fontsize=9)
    ax.set_title("Inference Accuracy", fontsize=12)

    plt.suptitle(f"Joint A2 + B2 Learning (lr={results['lr']}, fr={results['fr']})",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()

    if save_path:
        suffix = f"_A2init{results['initial_A2_precision']}_B2init{results['initial_B2_stay_prob']}"
        suffix += f"_lr{results['lr']}_fr{results['fr']}"
        png_path = save_path.replace(".png", "") + suffix + ".png"
        fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor='white')
        print(f"Saved: {png_path}")

    return fig


# =============================================================================
# Figure 7.2: Learning with Meditation Instruction introduced at sit N
# =============================================================================

def run_figure7_2(
    num_sits: int = 200,
    T_per_sit: int = 100,
    meditation_start_sit: int = 100,
    seed: int = 42,
    lr: float = 1.0,
    fr: float = 0.9,
    # A2 parameters
    initial_pA2_strength: float = 2.0,
    initial_A2_precision: float = 0.52,
    A2_true_precision: float = 0.9,
    # Meditation instruction parameters
    A2_awareness_precision: float = 0.75,
    A3_precision: float = 0.75,
    C_awareness_aware: float = 2.0,
    C_awareness_unaware: float = 0.0,
    entropy_threshold: float = 0.5,
    # B2 parameters
    initial_pB2_strength: float = 2.0,
    initial_B2_stay_prob: float = 0.5,
    # Level 1 parameters
    A1_precision: float = 0.75,
    zeta_focused: float = 2.0,
    zeta_distracted: float = 0.5,
    zeta_step: float = 0.25,
    log_zeta_prior_var: float = 2.0,
    # Policy parameters
    gamma: float = 16.0,
    E_stay: float = 0.9,
):
    """
    Figure 7.2: A2 + B2 learning with meditation instruction introduced at sit N.

    Before meditation_start_sit: Non-meditator baseline (same as Figure 7.1)
    After meditation_start_sit: Meditation instruction added:
        - A2 gets second modality: awareness obs → attention
        - C_awareness preference for being aware
        - Agent knows SWITCH toggles attention state
    """
    np.random.seed(seed)

    # =========================================================================
    # TRUE models (same as Figure 7.1)
    # =========================================================================
    A2_true = np.array([
        [A2_true_precision, 1 - A2_true_precision],
        [1 - A2_true_precision, A2_true_precision]
    ])

    # B2_true: Distraction is absorbing under STAY, toggle for SWITCH
    B2_true = np.zeros((2, 2, 2))
    B2_true[:, :, STAY] = np.array([
        [0.8, 0.0],   # P(->foc | stay): can maintain focus, can't return from distracted
        [0.2, 1.0],   # P(->dist | stay): distraction is absorbing
    ])
    B2_true[:, :, SWITCH] = np.array([
        [0.1, 0.9],   # P(->foc | switch): foc->dist, dist->foc (toggle)
        [0.9, 0.1],   # P(->dist | switch): toggles state
    ])

    # A3: awareness observation model (entropy → awareness)
    A3 = np.array([
        [A3_precision, 1 - A3_precision],
        [1 - A3_precision, A3_precision]
    ])

    # A2_awareness: meditation instruction (awareness obs → attention)
    A2_awareness = np.array([
        [A2_awareness_precision, 1 - A2_awareness_precision],
        [1 - A2_awareness_precision, A2_awareness_precision]
    ])

    # =========================================================================
    # AGENT's models (learned from experience)
    # =========================================================================

    # Initialize pA2 (Dirichlet parameters for A2 learning)
    pA2_obj = utils.obj_array(1)
    pA2_obj[0] = np.array([
        [initial_A2_precision, 1 - initial_A2_precision],
        [1 - initial_A2_precision, initial_A2_precision]
    ]) * initial_pA2_strength
    A2 = normalize_A(pA2_obj[0])

    # Initialize pB2 (Dirichlet parameters for B2 learning)
    pB2_obj = utils.obj_array(1)
    pB2_obj[0] = np.zeros((2, 2, 2))
    pB2_obj[0][:, :, STAY] = np.array([
        [initial_B2_stay_prob, 1 - initial_B2_stay_prob],
        [1 - initial_B2_stay_prob, initial_B2_stay_prob],
    ]) * initial_pB2_strength
    pB2_obj[0][:, :, SWITCH] = np.array([
        [1 - initial_B2_stay_prob, initial_B2_stay_prob],
        [initial_B2_stay_prob, 1 - initial_B2_stay_prob],
    ]) * initial_pB2_strength

    # Normalize to get B2
    B2 = np.zeros_like(pB2_obj[0])
    for a in range(2):
        for s in range(2):
            col_sum = pB2_obj[0][:, s, a].sum()
            B2[:, s, a] = pB2_obj[0][:, s, a] / col_sum if col_sum > 0 else 0.5

    # Level 1 matrices (fixed, not learned)
    A1 = np.array([
        [A1_precision, 1 - A1_precision],
        [1 - A1_precision, A1_precision]
    ])

    # Policy prior
    E = np.array([E_stay, 1 - E_stay])

    print(f"Figure 7.2: Learning with Meditation Instruction at Sit {meditation_start_sit}")
    print(f"  Sits: {num_sits}, T/sit: {T_per_sit}")
    print(f"  A2 - True precision: {A2_true_precision:.3f}, Initial: {initial_A2_precision:.3f}")
    print(f"  A2_awareness precision: {A2_awareness_precision:.3f}")
    print(f"  C_awareness: [{C_awareness_aware}, {C_awareness_unaware}]")
    print(f"  Learning rate: {lr}, Forgetting rate: {fr}")
    print()

    # =========================================================================
    # Logging
    # =========================================================================
    A2_diagonal_history = np.zeros((num_sits + 1, 2))
    A2_diagonal_history[0] = [A2[0, 0], A2[1, 1]]

    B2_stay_history = np.zeros((num_sits + 1, 2))
    B2_stay_history[0] = [B2[0, 0, STAY], B2[1, 1, STAY]]

    B2_switch_history = np.zeros((num_sits + 1, 2))
    B2_switch_history[0] = [B2[0, 0, SWITCH], B2[1, 1, SWITCH]]

    accuracy_history = np.zeros(num_sits)
    attention_accuracy_history = np.zeros(num_sits)
    switch_rate_history = np.zeros(num_sits)
    time_distracted_history = np.zeros(num_sits)

    policies = [np.array([[STAY]]), np.array([[SWITCH]])]

    # =========================================================================
    # Main learning loop
    # =========================================================================
    for sit in range(num_sits):
        env = BreathEnv(seed=seed + sit)

        # Check if meditation instruction is active
        meditation_active = sit >= meditation_start_sit

        # Compute B1 from env
        inhale_duration = env.inhale_range[1] - env.inhale_range[0]
        exhale_duration = env.exhale_range[1] - env.exhale_range[0]
        p_stay_inhale = (inhale_duration - 1) / inhale_duration
        p_stay_exhale = (exhale_duration - 1) / exhale_duration
        B1 = np.array([
            [p_stay_inhale, 1 - p_stay_exhale],
            [1 - p_stay_inhale, p_stay_exhale]
        ])

        # When meditation starts, agent learns SWITCH toggles
        if sit == meditation_start_sit:
            print(f"  *** Meditation instruction introduced at sit {sit} ***")
            # Update B2 SWITCH to reflect toggle knowledge
            B2[:, :, SWITCH] = np.array([
                [0.1, 0.9],
                [0.9, 0.1],
            ])
            # Also update pB2 to reflect this knowledge (high confidence)
            pB2_obj[0][:, :, SWITCH] = np.array([
                [0.1, 0.9],
                [0.9, 0.1],
            ]) * 10.0  # High confidence

        # Initialize for this sit
        qs_breath = np.array([0.5, 0.5])
        qs_attention = np.array([0.5, 0.5])
        qs_awareness = np.array([0.5, 0.5])
        true_attention_state = FOCUSED
        prev_action = STAY
        obs_breath_val = int(env.step(None))

        # Setup pymdp objects based on whether meditation is active
        if meditation_active:
            # Two modalities for A2
            A2_obj = utils.obj_array(2)
            A2_obj[0] = A2
            A2_obj[1] = A2_awareness
            C2_obj = utils.obj_array(2)
            C2_obj[0] = np.array([0.0, 0.0])
            C2_obj[1] = np.array([C_awareness_aware, C_awareness_unaware])
        else:
            # Single modality
            A2_obj = utils.obj_array(1)
            A2_obj[0] = A2
            C2_obj = utils.obj_array(1)
            C2_obj[0] = np.array([0.0, 0.0])

        B2_obj = utils.obj_array(1)
        B2_obj[0] = B2

        # Storage for this sit
        precision_obs_list = []
        entropy_obs_list = []
        posterior_attention_list = []
        posterior_attention_prev_list = []
        action_list = []
        true_breath_list = []
        posterior_breath_list = []
        true_attention_list = []

        for t in range(T_per_sit):
            true_breath_list.append(env.state)
            true_attention_list.append(true_attention_state)

            qs_attention_prev = qs_attention.copy()
            if t > 0:
                posterior_attention_prev_list.append(qs_attention_prev)

            # Precision from TRUE attention state
            zeta_prior = zeta_focused if true_attention_state == FOCUSED else zeta_distracted

            # Precision update
            zeta, _, _ = update_likelihood_precision(
                zeta=zeta_prior, A=A1, obs=obs_breath_val, qs=qs_breath,
                log_zeta_prior_var=log_zeta_prior_var, zeta_step=zeta_step,
            )

            # Breath inference
            prior_breath = B1 @ qs_breath if t > 0 else qs_breath
            A1_scaled = scale_likelihood(A1, zeta)
            likelihood_breath = A1_scaled[obs_breath_val, :]
            qs_breath = likelihood_breath * prior_breath
            qs_breath = qs_breath / (qs_breath.sum() + EPS_VAL)
            posterior_breath_list.append(qs_breath.copy())

            # Breath entropy for awareness observation
            breath_H = entropy_dist(qs_breath)
            obs_entropy = LOW_ENTROPY if breath_H < entropy_threshold else HIGH_ENTROPY
            entropy_obs_list.append(obs_entropy)

            # Ascending: precision observation (sampled from A2_true)
            obs_precision = np.random.choice([HIGH_PRECISION, LOW_PRECISION], p=A2_true[:, true_attention_state])
            precision_obs_list.append(obs_precision)

            # Attention inference
            prior_attention = B2[:, :, prev_action] @ qs_attention if t > 0 else qs_attention

            if meditation_active:
                # Combined likelihood from both modalities
                likelihood_precision = A2[obs_precision, :]
                likelihood_awareness = A2_awareness[obs_entropy, :]
                likelihood_combined = likelihood_precision * likelihood_awareness
            else:
                # Single modality
                likelihood_combined = A2[obs_precision, :]

            qs_attention = likelihood_combined * prior_attention
            qs_attention = qs_attention / (qs_attention.sum() + EPS_VAL)
            posterior_attention_list.append(qs_attention.copy())

            # Action selection
            qs_attention_obj = utils.obj_array(1)
            qs_attention_obj[0] = qs_attention

            q_pi, G = update_posterior_policies(
                qs=qs_attention_obj, A=A2_obj, B=B2_obj, C=C2_obj,
                policies=policies, use_utility=True, use_states_info_gain=True,
                E=E, gamma=gamma,
            )

            action = np.argmax(q_pi.flatten())
            action_list.append(action)
            prev_action = action

            # TRUE attention transition
            true_attention_one_hot = np.zeros(2)
            true_attention_one_hot[true_attention_state] = 1.0
            next_true_attention = B2_true[:, :, action] @ true_attention_one_hot
            true_attention_state = np.random.choice([FOCUSED, DISTRACTED], p=next_true_attention)

            # Next observation
            obs_breath_val = int(env.step(None))

        # =====================================================================
        # Compute metrics for this sit
        # =====================================================================
        posterior_breath_arr = np.array(posterior_breath_list)
        true_breath_arr = np.array(true_breath_list)
        raw_breath_acc = np.mean(posterior_breath_arr[np.arange(len(true_breath_arr)), true_breath_arr])
        accuracy_history[sit] = (raw_breath_acc - 0.5) / 0.5

        posterior_attention_arr = np.array(posterior_attention_list)
        true_attention_arr = np.array(true_attention_list)
        raw_attention_acc = np.mean(posterior_attention_arr[np.arange(len(true_attention_arr)), true_attention_arr])
        attention_accuracy_history[sit] = (raw_attention_acc - 0.5) / 0.5

        action_arr = np.array(action_list)
        switch_rate_history[sit] = np.mean(action_arr == SWITCH)
        time_distracted_history[sit] = np.mean(np.array(true_attention_list) == DISTRACTED)

        # =====================================================================
        # Learning updates (A2 and B2)
        # =====================================================================
        pA2_obj[0] = fr * pA2_obj[0]
        for t_idx in range(len(precision_obs_list)):
            obs = precision_obs_list[t_idx]
            qs_obj = utils.obj_array(1)
            qs_obj[0] = posterior_attention_list[t_idx]
            A2_obj_single = utils.obj_array(1)
            A2_obj_single[0] = A2
            pA2_obj = update_obs_likelihood_dirichlet(
                pA=pA2_obj, A=A2_obj_single, obs=obs, qs=qs_obj, lr=lr, fr=1.0
            )
        A2 = normalize_A(pA2_obj[0])

        # B2 learning
        pB2_obj[0][:, :, STAY] = fr * pB2_obj[0][:, :, STAY]
        # Only learn STAY dynamics (SWITCH is fixed after meditation instruction)
        for t_idx in range(len(posterior_attention_prev_list)):
            action_t = action_list[t_idx]
            if action_t == STAY:  # Only learn STAY dynamics
                qs_prev_obj = utils.obj_array(1)
                qs_prev_obj[0] = posterior_attention_prev_list[t_idx]
                qs_curr_obj = utils.obj_array(1)
                qs_curr_obj[0] = posterior_attention_list[t_idx + 1]

                pB2_obj = update_state_likelihood_dirichlet(
                    pB=pB2_obj, B=B2_obj,
                    actions=np.array([action_t]),
                    qs=qs_curr_obj, qs_prev=qs_prev_obj, lr=lr,
                )

        # Normalize B2
        for a in range(2):
            for s in range(2):
                col_sum = pB2_obj[0][:, s, a].sum()
                B2[:, s, a] = pB2_obj[0][:, s, a] / col_sum if col_sum > 0 else 0.5

        # Log
        A2_diagonal_history[sit + 1] = [A2[0, 0], A2[1, 1]]
        B2_stay_history[sit + 1] = [B2[0, 0, STAY], B2[1, 1, STAY]]
        B2_switch_history[sit + 1] = [B2[0, 0, SWITCH], B2[1, 1, SWITCH]]

        if (sit + 1) % 20 == 0 or sit == 0 or sit == meditation_start_sit:
            print(f"  Sit {sit+1:3d}: AttAcc={attention_accuracy_history[sit]:.3f}, "
                  f"Dist={time_distracted_history[sit]:.2f}, Switch={switch_rate_history[sit]:.2f}, "
                  f"A2=[{A2[0,0]:.3f}, {A2[1,1]:.3f}]")

    print()
    print(f"Final A2 diagonal: [{A2[0,0]:.4f}, {A2[1,1]:.4f}] (True: {A2_true_precision})")
    print(f"Final B2 stay:     [{B2[0,0,STAY]:.4f}, {B2[1,1,STAY]:.4f}]")

    return {
        "A2_diagonal_history": A2_diagonal_history,
        "B2_stay_history": B2_stay_history,
        "B2_switch_history": B2_switch_history,
        "accuracy_history": accuracy_history,
        "attention_accuracy_history": attention_accuracy_history,
        "switch_rate_history": switch_rate_history,
        "time_distracted_history": time_distracted_history,
        "A2_true_precision": A2_true_precision,
        "initial_A2_precision": initial_A2_precision,
        "initial_B2_stay_prob": initial_B2_stay_prob,
        "meditation_start_sit": meditation_start_sit,
        "num_sits": num_sits,
        "lr": lr,
        "fr": fr,
    }


def plot_figure7_2(results: dict, save_path: str = None):
    """Plot Figure 7.2: Learning with meditation instruction."""
    plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 11})

    num_sits = results["num_sits"]
    meditation_start = results["meditation_start_sit"]
    sit_range = np.arange(num_sits + 1)
    sit_range_acc = np.arange(num_sits)

    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

    BLUE = '#2563eb'
    ORANGE = '#ea580c'
    TEAL = '#0d9488'
    GRAY = '#6b7280'
    RED = '#dc2626'

    # Panel A: A2 learning
    ax = axes[0]
    A2_diag = results["A2_diagonal_history"]
    ax.plot(sit_range, A2_diag[:, 0], color=BLUE, linewidth=1.5, label='A2[0,0] - P(high|foc)')
    ax.plot(sit_range, A2_diag[:, 1], color=ORANGE, linewidth=1.5, label='A2[1,1] - P(low|dist)')
    ax.axhline(y=results["A2_true_precision"], color=GRAY, linestyle='--', linewidth=1.5, label=f'True = {results["A2_true_precision"]}')
    ax.axvline(x=meditation_start, color=RED, linestyle='--', alpha=0.7, label='Meditation starts')
    ax.axhline(y=0.5, color=GRAY, linestyle=':', alpha=0.5)
    ax.set_ylabel("A2 diagonal")
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc='right', fontsize=8)
    ax.set_title("A2 Learning (Observation Model)")

    # Panel B: Attention accuracy
    ax = axes[1]
    ax.plot(sit_range_acc, results["attention_accuracy_history"], color=TEAL, linewidth=1.5)
    ax.axvline(x=meditation_start, color=RED, linestyle='--', alpha=0.7)
    ax.axhline(y=0.0, color=GRAY, linestyle=':', alpha=0.5)
    ax.set_ylabel("Attention Accuracy\n(0%=chance)")
    ax.set_ylim(-0.1, 1.0)
    ax.set_title("Attention Inference Accuracy")

    # Panel C: Time distracted
    ax = axes[2]
    ax.plot(sit_range_acc, results["time_distracted_history"], color=ORANGE, linewidth=1.5)
    ax.axvline(x=meditation_start, color=RED, linestyle='--', alpha=0.7)
    ax.set_ylabel("Fraction Distracted")
    ax.set_ylim(0, 1.0)
    ax.set_title("Time Spent Distracted")

    # Panel D: Switch rate
    ax = axes[3]
    ax.plot(sit_range_acc, results["switch_rate_history"], color=BLUE, linewidth=1.5)
    ax.axvline(x=meditation_start, color=RED, linestyle='--', alpha=0.7)
    ax.set_xlabel("Sit")
    ax.set_ylabel("Switch Rate")
    ax.set_ylim(0, 1.0)
    ax.set_title("Action Selection (Switch Rate)")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor='white')
        print(f"Saved: {save_path}")

    return fig


def plot_figure7(results: dict, save_path: str = None):
    """
    Plot Figure 7: A2 learning trajectory.

    Two panels:
    A) A2 diagonal entries over sits (learning trajectory)
    B) Breath accuracy and mean P(focused) over sits
    """
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 11,
    })
    
    num_sits = results["num_sits"]
    sit_range = np.arange(num_sits + 1)
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    
    BLUE = '#2563eb'
    ORANGE = '#ea580c'
    PURPLE = '#7c3aed'
    TEAL = '#0d9488'
    GRAY = '#6b7280'
    
    # Panel A: A2 diagonal learning
    ax = axes[0]
    A2_diag = results["A2_diagonal_history"]
    ax.plot(sit_range, A2_diag[:, 0], color=BLUE, linewidth=1.5, label='A2[0,0] - P(high|foc)')
    ax.plot(sit_range, A2_diag[:, 1], color=ORANGE, linewidth=1.5, label='A2[1,1] - P(low|dist)')
    ax.axhline(y=results["A2_true_precision"], color=GRAY, linestyle='--', 
               linewidth=1.5, label=f'True = {results["A2_true_precision"]}')
    ax.axhline(y=0.5, color=GRAY, linestyle=':', alpha=0.5)
    ax.set_ylabel("A2 diagonal entry")
    ax.set_ylim(0.4, 1.0)
    ax.legend(loc='lower right', fontsize=9)
    ax.set_title(f"A2 Matrix Learning (lr={results['lr']}, fr={results['fr']})", fontsize=12)
    
    # Panel B: Breath and Attention Inference Accuracy (normalized: 0%=chance, 100%=perfect)
    ax = axes[1]
    sit_range_acc = np.arange(num_sits)
    ax.plot(sit_range_acc, results["accuracy_history"], color=PURPLE,
            linewidth=1.5, label='Breath')
    ax.plot(sit_range_acc, results["attention_accuracy_history"], color=TEAL,
            linewidth=1.5, label='Attention')
    ax.axhline(y=1.0, color=GRAY, linestyle=':', alpha=0.5, label='Perfect')
    ax.axhline(y=0.0, color=GRAY, linestyle=':', alpha=0.5, label='Chance')
    ax.set_xlabel("Sit")
    ax.set_ylabel("Inference Accuracy\n(0%=chance, 100%=perfect)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='lower right', fontsize=9)
    ax.set_title("Breath and Attention Inference Accuracy", fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        suffix = f"_A2true{results['A2_true_precision']}_A2init{results['initial_A2_precision']}_lr{results['lr']}_fr{results['fr']}"
        if results.get("terminal_distraction"):
            suffix += "_termDist"
        
        png_path = save_path.replace(".png", "") + suffix + ".png"
        fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor='white')
        print(f"Saved: {png_path}")
    
    return fig


def plot_figure6(results: dict, save_path: str = None, precision_y_max: float = 3.0):
    """
    Generate Figure 6: Hierarchical Breath with Action Selection AND Awareness.
    
    Five-panel figure (top to bottom):
    A) Awareness state inference
    B) Action selection (P(Stay) vs P(Switch)) + selected actions
    C) Attention state inference + true state
    D) Precision dynamics
    E) Breath state inference
    """
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
    })
    
    T = results["T"]
    t_range = np.arange(T)
    
    # Colors
    BLUE = '#2563eb'
    ORANGE = '#ea580c'
    GREEN = '#16a34a'
    PURPLE = '#7c3aed'
    TEAL = '#0d9488'
    GRAY = '#6b7280'
    RED_LIGHT = '#fecaca'
    MAGENTA = '#c026d3'
    
    fig, axes = plt.subplots(5, 1, figsize=(12, 11), sharex=True,
                              gridspec_kw={'height_ratios': [1, 1, 1, 1, 1], 'hspace': 0.35})
    
    force_distract_at = results.get("force_distract_at")
    
    # =========================================================================
    # Panel A: Awareness State
    # =========================================================================
    ax = axes[0]
    
    if force_distract_at is not None:
        ax.axvline(x=force_distract_at, color=MAGENTA, linestyle='--', linewidth=1.5, alpha=0.7)
    
    p_aware = results["posterior_awareness"][:, AWARE]
    ax.plot(t_range, p_aware, color=TEAL, linewidth=1.5, label='P(Aware)')
    ax.axhline(y=0.5, color=GRAY, linestyle='--', linewidth=1, alpha=0.5)
    
    # Entropy observations as scatter
    obs_entropy = results["entropy_obs_history"]
    obs_entropy_binary = 1 - obs_entropy
    ax.scatter(t_range, obs_entropy_binary, s=25, c=TEAL, alpha=0.3, edgecolors='none',
               label='Entropy obs', zorder=5, marker='s')
    
    ax.set_ylabel("Probability")
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks([0, 0.5, 1])
    ax.legend(loc='lower right', framealpha=0.9, fontsize=9)
    ax.set_title("Level 2: Awareness State (from breath entropy)", fontsize=12)
    
    # =========================================================================
    # Panel B: Action Selection
    # =========================================================================
    ax = axes[1]
    
    if force_distract_at is not None:
        ax.axvline(x=force_distract_at, color=MAGENTA, linestyle='--', linewidth=1.5, alpha=0.7)
        ax.text(force_distract_at + 2, 0.95, 'Forced distraction',
                fontsize=9, color=MAGENTA, alpha=0.8)
    
    ax.plot(t_range, results["q_pi_history"][:, STAY], 
            color=TEAL, linewidth=1.5, label='P(Stay)')
    
    stay_mask = results["action_history"] == STAY
    switch_mask = results["action_history"] == SWITCH
    
    ax.scatter(t_range[stay_mask], np.ones(np.sum(stay_mask)) * 1.02,
               s=15, c=TEAL, alpha=0.6, marker='|', label='Selected: Stay')
    ax.scatter(t_range[switch_mask], np.zeros(np.sum(switch_mask)) - 0.02,
               s=15, c=PURPLE, alpha=0.8, marker='|', label='Selected: Switch')
    
    ax.axhline(y=0.5, color=GRAY, linestyle='--', linewidth=1, alpha=0.5)
    ax.set_ylabel("Probability")
    ax.set_ylim(-0.1, 1.1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.legend(loc='upper right', framealpha=0.9, fontsize=9)
    ax.set_title("Level 2: Action Selection (EFE-based)", fontsize=12)
    
    # =========================================================================
    # Panel C: Attention State
    # =========================================================================
    ax = axes[2]
    
    if force_distract_at is not None:
        ax.axvline(x=force_distract_at, color=MAGENTA, linestyle='--', linewidth=1.5, alpha=0.7)
    
    ax.plot(t_range, results["posterior_attention"][:, FOCUSED],
            color=PURPLE, linewidth=1.5, label='P(Focused)')
    
    # True attention state as dots
    true_att = results["true_attention_history"]
    focused_mask = true_att == FOCUSED
    distracted_mask = true_att == DISTRACTED
    ax.scatter(t_range[focused_mask], np.ones(np.sum(focused_mask)) * 1.02,
               s=15, c=GRAY, alpha=0.5, marker='o', label='True state')
    ax.scatter(t_range[distracted_mask], np.zeros(np.sum(distracted_mask)) - 0.02,
               s=15, c='red', alpha=0.5, marker='o')
    
    ax.axhline(y=0.5, color=GRAY, linestyle='--', linewidth=1, alpha=0.5)
    ax.set_ylabel("Probability")
    ax.set_ylim(-0.1, 1.1)
    ax.set_yticks([0, 0.5, 1])
    ax.legend(loc='lower right', framealpha=0.9, fontsize=9)
    ax.set_title("Level 2: Attention State (true vs inferred)", fontsize=12)
    
    # =========================================================================
    # Panel D: Precision
    # =========================================================================
    ax = axes[3]
    
    if force_distract_at is not None:
        ax.axvline(x=force_distract_at, color=MAGENTA, linestyle='--', linewidth=1.5, alpha=0.7)
    
    ax.plot(t_range, results["zeta_history"], color=ORANGE, linewidth=1.5, 
            label='ζ posterior')
    ax.plot(t_range, results["zeta_prior_history"], color=ORANGE, linewidth=1,
            linestyle='--', alpha=0.6, label='ζ prior (↓)')
    
    zeta_foc = results["zeta_focused"]
    zeta_dist = results["zeta_distracted"]
    ax.axhline(y=zeta_foc, color=GRAY, linestyle=':', alpha=0.4)
    ax.axhline(y=zeta_dist, color=GRAY, linestyle=':', alpha=0.4)
    ax.text(T - 5, zeta_foc + 0.05, f'ζ_foc={zeta_foc}', fontsize=8, alpha=0.6, ha='right')
    ax.text(T - 5, zeta_dist + 0.05, f'ζ_dist={zeta_dist}', fontsize=8, alpha=0.6, ha='right')
    
    ax.set_ylabel("Precision (ζ)")
    ax.set_ylim(0, precision_y_max)
    ax.legend(loc='upper right', framealpha=0.9, fontsize=9)
    ax.set_title("Likelihood Precision", fontsize=12)
    
    # =========================================================================
    # Panel E: Breath Perception
    # =========================================================================
    ax = axes[4]
    
    if force_distract_at is not None:
        ax.axvline(x=force_distract_at, color=MAGENTA, linestyle='--', linewidth=1.5, alpha=0.7)
    
    p_inhale = results["posterior_breath"][:, INHALE]
    ax.plot(t_range, p_inhale, color=BLUE, linewidth=1.5, label='P(Inhaling)')
    
    true_state = results["true_breath"]
    inhale_mask = true_state == INHALE
    exhale_mask = true_state == EXHALE
    ax.scatter(t_range[inhale_mask], np.ones(np.sum(inhale_mask)) * 1.02,
               s=12, c=GRAY, alpha=0.4, marker='o', label='True state')
    ax.scatter(t_range[exhale_mask], np.zeros(np.sum(exhale_mask)) - 0.02,
               s=12, c=GRAY, alpha=0.4, marker='o')
    
    ax.axhline(y=0.5, color=GRAY, linestyle='--', linewidth=1, alpha=0.5)
    ax.set_ylabel("Probability")
    ax.set_ylim(-0.1, 1.1)
    ax.set_yticks([0, 0.5, 1])
    ax.legend(loc='lower right', framealpha=0.9, fontsize=9)
    ax.set_xlabel("Time step")
    ax.set_title("Level 1: Breath Perception", fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        suffix = f"_A1{results['A1_precision']}_A2{results['A2_precision']}_A3{results['A3_precision']}"
        suffix += f"_zfoc{results['zeta_focused']}_zdist{results['zeta_distracted']}"
        suffix += f"_gamma{results['gamma']}_Estay{results['E_stay']}"
        suffix += f"_C2_{results['C2_high_prec_pref']}_{results['C2_low_prec_pref']}"
        if results.get("terminal_distraction"):
            suffix += "_termDist"
        if results.get("force_distract_at"):
            suffix += f"_distract{results['force_distract_at']}"
        
        base_path = save_path.replace(".png", "")
        png_path = f"{base_path}{suffix}.png"
        
        fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor='white')
        print(f"Saved: {png_path}")
    
    return fig


def plot_figure5(results: dict, save_path: str = None, precision_y_max: float = 3.0):
    """
    Generate Figure 5: Hierarchical Breath with Attention Action Selection.
    
    Five-panel figure (top to bottom):
    A) Action selection (P(Stay) vs P(Switch)) + selected actions
    B) Attention state inference + precision observations
    C) Precision dynamics with descending prior
    D) Breath state inference (Level 1)
    """
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })
    
    T = results["T"]
    t_range = np.arange(T)
    
    # Colors
    blue = '#2563eb'
    orange = '#ea580c'
    gray = '#6b7280'
    green = '#2E7D32'
    purple = '#7c3aed'
    teal = '#0d9488'
    
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True,
                             gridspec_kw={'height_ratios': [1, 1, 1, 1]})
    
    # =========================================================================
    # Panel A: Action Selection
    # =========================================================================
    ax = axes[0]
    
    # Plot P(Stay) as line
    ax.plot(t_range, results["q_pi_history"][:, STAY], 
            color=teal, linewidth=1.5, label='P(Stay)')
    
    # Plot selected actions as scatter
    stay_mask = results["action_history"] == STAY
    switch_mask = results["action_history"] == SWITCH
    ax.scatter(t_range[stay_mask], np.ones(stay_mask.sum()) * 1.05, 
               c=teal, s=15, alpha=0.5, marker='|', label='Selected: Stay')
    ax.scatter(t_range[switch_mask], np.ones(switch_mask.sum()) * -0.05, 
               c=purple, s=15, alpha=0.5, marker='|', label='Selected: Switch')
    
    ax.axhline(y=0.5, color=gray, linestyle='--', linewidth=1, alpha=0.5)
    ax.set_ylim(-0.15, 1.15)
    ax.set_ylabel('Probability')
    ax.set_title('Level 2: Action Selection (EFE-based)')
    ax.legend(loc='right', fontsize=8)
    
    # Forced distraction marker
    if results.get("force_distract_at"):
        for a in axes:
            a.axvline(x=results["force_distract_at"], color='magenta', 
                     linestyle='--', linewidth=1.5, alpha=0.7)
        axes[0].text(results["force_distract_at"] + 2, 0.9, 'Forced distraction',
                    fontsize=8, color='magenta')
    
    # =========================================================================
    # Panel B: Attention State Inference
    # =========================================================================
    ax = axes[1]
    
    # Agent's beliefs about attention state
    ax.plot(t_range, results["posterior_attention"][:, FOCUSED], 
            color=purple, linewidth=1.5, label='P(Focused)')
    
    # True attention state as scatter dots
    true_attention = results.get("true_attention")
    if true_attention is not None:
        # Plot true state: 1.0 for focused, 0.0 for distracted
        true_state_binary = 1.0 - true_attention  # FOCUSED=0→1.0, DISTRACTED=1→0.0
        ax.scatter(t_range, true_state_binary, c=gray, s=12, alpha=0.5, 
                   marker='o', label='True state')
    
    ax.axhline(y=0.5, color=gray, linestyle='--', linewidth=1, alpha=0.5)
    ax.set_ylim(-0.1, 1.1)
    ax.set_ylabel('Probability')
    ax.set_title('Level 2: Attention State (true vs inferred)')
    ax.legend(loc='lower right', fontsize=8)
    
    # =========================================================================
    # Panel C: Likelihood Precision
    # =========================================================================
    ax = axes[2]
    
    ax.plot(t_range, results["zeta_history"], color=orange, linewidth=1.5,
            label='ζ posterior')
    ax.plot(t_range, results["zeta_prior_history"], color=orange, linewidth=1,
            linestyle='--', alpha=0.7, label='ζ prior (↓)')
    
    ax.axhline(y=results["zeta_focused"], color=gray, linestyle=':', 
               linewidth=1, alpha=0.5, label=f'ζ_foc={results["zeta_focused"]}')
    ax.axhline(y=results["zeta_distracted"], color=gray, linestyle=':', 
               linewidth=1, alpha=0.5, label=f'ζ_dist={results["zeta_distracted"]}')
    ax.axhline(y=1.0, color=gray, linestyle=':', linewidth=1, alpha=0.3)
    
    ax.set_ylim(0, precision_y_max)
    ax.set_ylabel('Precision (ζ)')
    ax.set_title('Likelihood Precision')
    ax.legend(loc='upper right', fontsize=8)
    
    # =========================================================================
    # Panel D: Breath State Inference
    # =========================================================================
    ax = axes[3]
    
    ax.plot(t_range, results["posterior_breath"][:, INHALE], 
            color=blue, linewidth=1.5, label='P(Inhaling)')
    
    # True states as scatter
    inhale_mask = results["true_breath"] == INHALE
    exhale_mask = results["true_breath"] == EXHALE
    ax.scatter(t_range[inhale_mask], np.ones(inhale_mask.sum()) * 1.02,
               c=gray, s=8, alpha=0.5, marker='o')
    ax.scatter(t_range[exhale_mask], np.zeros(exhale_mask.sum()) - 0.02,
               c=gray, s=8, alpha=0.5, marker='o', label='True state')
    
    # Noise shading
    if results.get("noise_start") and results.get("noise_end"):
        ax.axvspan(results["noise_start"], results["noise_end"], 
                  alpha=0.2, color='red', label='Noise')
    
    ax.axhline(y=0.5, color=gray, linestyle='--', linewidth=1, alpha=0.5)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlabel('Time step')
    ax.set_ylabel('Probability')
    ax.set_title('Level 1: Breath Perception')
    ax.legend(loc='lower right', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        # Build filename with parameters
        suffix = f"_A1{results['A1_precision']}_A2{results['A2_precision']}"
        suffix += f"_zfoc{results['zeta_focused']}_zdist{results['zeta_distracted']}"
        suffix += f"_gamma{results['gamma']}_Estay{results['E_stay']}"
        suffix += f"_C2_{results['C2_high_prec_pref']}_{results['C2_low_prec_pref']}"
        if results.get("terminal_distraction"):
            suffix += "_termDist"
        if results.get("force_distract_at"):
            suffix += f"_distract{results['force_distract_at']}"
        
        base_path = save_path.replace(".png", "")
        png_path = f"{base_path}{suffix}.png"
        
        fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor='white')
        print(f"Saved: {png_path}")
    
    return fig


def plot_figure4(results: dict, save_path: str = None, precision_y_max: float = 2.0):
    """
    Generate Figure 4: Hierarchical Breath with Attention AND Awareness.
    
    Four-panel figure (top to bottom):
    A) Awareness state inference + entropy observations
    B) Attention state inference + precision observations  
    C) Precision dynamics with descending prior
    D) Breath state inference (Level 1)
    """
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })
    
    T = len(results["true_breath"])
    t_range = np.arange(T)
    
    noise_start = results.get("noise_start")
    noise_end = results.get("noise_end")
    has_noise = noise_start is not None and noise_end is not None
    force_distract_at = results.get("force_distract_at")
    
    # Colors
    BLUE = '#2563eb'
    ORANGE = '#ea580c'
    GREEN = '#16a34a'
    PURPLE = '#7c3aed'
    TEAL = '#0d9488'
    GRAY = '#6b7280'
    RED_LIGHT = '#fecaca'
    MAGENTA = '#c026d3'
    
    fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True,
                              gridspec_kw={'height_ratios': [1, 1, 1, 1], 'hspace': 0.35})
    
    # =========================================================================
    # Panel A: Awareness State - TOP
    # =========================================================================
    ax = axes[0]
    
    if has_noise:
        ax.axvspan(noise_start, noise_end, alpha=0.3, color=RED_LIGHT, label='Noise period')
    if force_distract_at is not None:
        ax.axvline(x=force_distract_at, color=MAGENTA, linestyle='--', linewidth=2, label='Forced distraction')
    
    p_aware = results["posterior_awareness"][:, AWARE]
    ax.plot(t_range, p_aware, color=TEAL, linewidth=1.5, label='P(Aware)')
    ax.axhline(y=0.5, color=GRAY, linestyle='--', linewidth=1, alpha=0.5)
    
    # Overlay entropy observations as scatter
    obs_entropy = results["entropy_obs_history"]
    obs_entropy_binary = 1 - obs_entropy  # LOW_ENTROPY=0 → 1, HIGH_ENTROPY=1 → 0
    ax.scatter(t_range, obs_entropy_binary, s=35, c=TEAL, alpha=0.4, edgecolors='none',
               label='Entropy obs (↑)', zorder=5, marker='s')
    
    ax.set_ylabel("Probability")
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks([0, 0.5, 1])
    ax.legend(loc='lower right', framealpha=0.9)
    ax.text(-0.06, 1.05, 'A', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')
    
    fixed_awareness = results.get("fixed_awareness")
    zeta_awareness = results.get("zeta_awareness", 1.0)
    if fixed_awareness:
        ax.set_title(f"Level 2: Awareness State (FIXED: {fixed_awareness})", fontsize=12)
    elif zeta_awareness < 1.0:
        ax.set_title(f"Level 2: Awareness State (ζ_aware={zeta_awareness} - not attending)", fontsize=12)
    else:
        ax.set_title("Level 2: Awareness State (from breath entropy)", fontsize=12)
    
    # =========================================================================
    # Panel B: Attention State
    # =========================================================================
    ax = axes[1]
    
    if has_noise:
        ax.axvspan(noise_start, noise_end, alpha=0.3, color=RED_LIGHT)
    if force_distract_at is not None:
        ax.axvline(x=force_distract_at, color=MAGENTA, linestyle='--', linewidth=2)
    
    p_focused = results["posterior_attention"][:, FOCUSED]
    ax.plot(t_range, p_focused, color=PURPLE, linewidth=1.5, label='P(Focused)')
    ax.axhline(y=0.5, color=GRAY, linestyle='--', linewidth=1, alpha=0.5)
    
    obs_prec = results["precision_obs_history"]
    obs_prec_binary = 1 - obs_prec
    ax.scatter(t_range, obs_prec_binary, s=35, c=GREEN, alpha=0.6, edgecolors='none',
               label='Precision obs (↑)', zorder=5)
    
    ax.set_ylabel("Probability")
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks([0, 0.5, 1])
    ax.legend(loc='lower right', framealpha=0.9)
    ax.text(-0.06, 1.05, 'B', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')
    
    fixed_attention = results.get("fixed_attention")
    A2_prec = results.get("A2_precision", 0.9)
    awareness_informs = results.get("awareness_informs_attention", False)
    
    if fixed_attention:
        ax.set_title(f"Level 2: Attention State (FIXED: {fixed_attention})", fontsize=12)
    elif A2_prec == 0.5:
        if awareness_informs:
            ax.set_title("Level 2: Attention State (NOVICE + awareness→attention)", fontsize=12)
        else:
            ax.set_title("Level 2: Attention State (NOVICE: flat A2/B2)", fontsize=12)
    else:
        ax.set_title("Level 2: Attention State (from precision)", fontsize=12)
    
    # =========================================================================
    # Panel C: Precision Dynamics
    # =========================================================================
    ax = axes[2]
    
    if has_noise:
        ax.axvspan(noise_start, noise_end, alpha=0.3, color=RED_LIGHT)
    if force_distract_at is not None:
        ax.axvline(x=force_distract_at, color=MAGENTA, linestyle='--', linewidth=2)
    
    ax.plot(t_range, results["zeta_history"], color=ORANGE, linewidth=1.5, 
            label='ζ posterior')
    ax.plot(t_range, results["zeta_prior_history"], color=ORANGE, linewidth=1.5,
            linestyle='--', alpha=0.6, label='ζ prior (↓)')
    ax.axhline(y=results["zeta_threshold"], color=GRAY, linestyle=':', 
               linewidth=1, alpha=0.6, label='Threshold')
    ax.axhline(y=results["zeta_focused"], color=PURPLE, linestyle=':', 
               linewidth=1, alpha=0.4)
    ax.axhline(y=results["zeta_distracted"], color=PURPLE, linestyle=':', 
               linewidth=1, alpha=0.4)
    ax.set_ylabel("Precision (ζ)")
    ax.set_ylim(0, precision_y_max)
    ax.legend(loc='lower right', framealpha=0.9, ncol=3)
    ax.text(-0.06, 1.05, 'C', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')
    ax.set_title("Likelihood Precision", fontsize=12)
    
    if results["zeta_focused"] < precision_y_max:
        ax.text(T + 1, results["zeta_focused"], 'ζ_foc', fontsize=9, 
                color=PURPLE, alpha=0.6, va='center')
    if results["zeta_distracted"] < precision_y_max:
        ax.text(T + 1, results["zeta_distracted"], 'ζ_dist', fontsize=9, 
                color=PURPLE, alpha=0.6, va='center')
    
    # =========================================================================
    # Panel D: Breath State Inference - BOTTOM
    # =========================================================================
    ax = axes[3]
    
    if has_noise:
        ax.axvspan(noise_start, noise_end, alpha=0.3, color=RED_LIGHT)
    if force_distract_at is not None:
        ax.axvline(x=force_distract_at, color=MAGENTA, linestyle='--', linewidth=2)
    
    p_inhale = results["posterior_breath"][:, INHALE]
    ax.plot(t_range, p_inhale, color=BLUE, linewidth=1.5, label='P(Inhaling)')
    true_breath_binary = 1.0 - results["true_breath"]
    ax.scatter(t_range, true_breath_binary, s=15, color=GRAY, alpha=0.5)
    ax.set_ylabel("Probability")
    ax.set_xlabel("Time step")
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks([0, 0.5, 1])
    legend = [Line2D([0], [0], color=BLUE, linewidth=1.5, label='P(Inhaling)'),
              Line2D([0], [0], marker='o', color='w', markerfacecolor=GRAY, 
                     markersize=5, label='True state', alpha=0.5)]
    ax.legend(handles=legend, loc='lower right', framealpha=0.9)
    ax.text(-0.06, 1.05, 'D', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')
    ax.set_title("Level 1: Breath Perception", fontsize=12)
    
    fig.align_ylabels(axes)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor='white')
        print(f"Saved: {save_path}")
    
    return fig


# =============================================================================
# Plotting - Figure 3
# =============================================================================

def plot_figure3(results: dict, save_path: str = None, precision_y_max: float = 2.0):
    """
    Generate Figure 3: Hierarchical Breath Perception with Attention.
    
    Three-panel figure (top to bottom):
    A) Attention state inference + precision observations overlay
    B) Precision dynamics with descending prior
    C) Breath state inference (Level 1)
    """
    # Publication style
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })
    
    T = len(results["true_breath"])
    t_range = np.arange(T)
    
    # Check for noise period
    noise_start = results.get("noise_start")
    noise_end = results.get("noise_end")
    has_noise = noise_start is not None and noise_end is not None
    
    # Colors
    BLUE = '#2563eb'
    ORANGE = '#ea580c'
    GREEN = '#16a34a'
    PURPLE = '#7c3aed'
    GRAY = '#6b7280'
    RED_LIGHT = '#fecaca'  # Light red for noise shading
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True,
                              gridspec_kw={'height_ratios': [1, 1, 1], 'hspace': 0.35})
    
    # =========================================================================
    # Panel A: Attention State + Precision Observations - TOP
    # =========================================================================
    ax = axes[0]
    
    # Add noise shading
    if has_noise:
        ax.axvspan(noise_start, noise_end, alpha=0.3, color=RED_LIGHT, label='Noise period')
    
    p_focused = results["posterior_attention"][:, FOCUSED]
    ax.plot(t_range, p_focused, color=PURPLE, linewidth=1.5, label='P(Focused)')
    ax.axhline(y=0.5, color=GRAY, linestyle='--', linewidth=1, alpha=0.5)
    
    # Overlay precision observations as scatter
    obs = results["precision_obs_history"]
    obs_binary = 1 - obs  # HIGH_PRECISION=0 → 1, LOW_PRECISION=1 → 0
    ax.scatter(t_range, obs_binary, s=35, c=GREEN, alpha=0.6, edgecolors='none',
               label='Precision obs (↑)', zorder=5)
    
    ax.set_ylabel("Probability")
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks([0, 0.5, 1])
    ax.legend(loc='lower right', framealpha=0.9)
    ax.text(-0.06, 1.05, 'A', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')
    fixed_attention = results.get("fixed_attention")
    if fixed_attention:
        ax.set_title(f"Level 2: Attention State (FIXED: {fixed_attention})", fontsize=12)
    else:
        ax.set_title("Level 2: Attention State", fontsize=12)
    
    # =========================================================================
    # Panel B: Precision Dynamics - MIDDLE
    # =========================================================================
    ax = axes[1]
    
    # Add noise shading
    if has_noise:
        ax.axvspan(noise_start, noise_end, alpha=0.3, color=RED_LIGHT)
    
    ax.plot(t_range, results["zeta_history"], color=ORANGE, linewidth=1.5, 
            label='ζ posterior')
    ax.plot(t_range, results["zeta_prior_history"], color=ORANGE, linewidth=1.5,
            linestyle='--', alpha=0.6, label='ζ prior (↓)')
    ax.axhline(y=results["zeta_threshold"], color=GRAY, linestyle=':', 
               linewidth=1, alpha=0.6, label='Threshold')
    ax.axhline(y=results["zeta_focused"], color=PURPLE, linestyle=':', 
               linewidth=1, alpha=0.4)
    ax.axhline(y=results["zeta_distracted"], color=PURPLE, linestyle=':', 
               linewidth=1, alpha=0.4)
    ax.set_ylabel("Precision (ζ)")
    ax.set_ylim(0, precision_y_max)
    ax.legend(loc='lower right', framealpha=0.9, ncol=3)
    ax.text(-0.06, 1.05, 'B', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')
    ax.set_title("Likelihood Precision", fontsize=12)
    
    # Annotate precision levels (only if they fit in the plot)
    if results["zeta_focused"] < precision_y_max:
        ax.text(T + 1, results["zeta_focused"], 'ζ_focused', fontsize=9, 
                color=PURPLE, alpha=0.6, va='center')
    if results["zeta_distracted"] < precision_y_max:
        ax.text(T + 1, results["zeta_distracted"], 'ζ_distracted', fontsize=9, 
                color=PURPLE, alpha=0.6, va='center')
    
    # =========================================================================
    # Panel C: Breath State Inference (Level 1) - BOTTOM
    # =========================================================================
    ax = axes[2]
    
    # Add noise shading
    if has_noise:
        ax.axvspan(noise_start, noise_end, alpha=0.3, color=RED_LIGHT)
    
    p_inhale = results["posterior_breath"][:, INHALE]
    ax.plot(t_range, p_inhale, color=BLUE, linewidth=1.5, label='P(Inhaling)')
    true_breath_binary = 1.0 - results["true_breath"]  # INHALE=0 → 1.0
    ax.scatter(t_range, true_breath_binary, s=15, color=GRAY, alpha=0.5)
    ax.set_ylabel("Probability")
    ax.set_xlabel("Time step")
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks([0, 0.5, 1])
    legend = [Line2D([0], [0], color=BLUE, linewidth=1.5, label='P(Inhaling)'),
              Line2D([0], [0], marker='o', color='w', markerfacecolor=GRAY, 
                     markersize=5, label='True state', alpha=0.5)]
    ax.legend(handles=legend, loc='lower right', framealpha=0.9)
    ax.text(-0.06, 1.05, 'C', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')
    ax.set_title("Level 1: Breath Perception", fontsize=12)
    
    fig.align_ylabels(axes)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor='white')
        print(f"Saved: {save_path}")
    
    return fig


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="Hierarchical breath perception with attention")
    parser.add_argument("--T", type=int, default=100, help="Number of timesteps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--A1-precision", type=float, default=0.75,
                        help="Level 1 likelihood precision (lower = noisier)")
    parser.add_argument("--zeta-focused", type=float, default=2.0, 
                        help="Precision for focused state (m=2 default)")
    parser.add_argument("--zeta-distracted", type=float, default=0.5, 
                        help="Precision for distracted state (1/m=0.5 default)")
    parser.add_argument("--zeta-step", type=float, default=0.25,
                        help="Step size for precision updates (B.45)")
    parser.add_argument("--log-zeta-prior-var", type=float, default=2.0,
                        help="Variance of log-zeta prior (lower = stronger regularization)")
    parser.add_argument("--A2-precision", type=float, default=0.9,
                        help="A2 likelihood precision")
    parser.add_argument("--initial-p-focused", type=float, default=0.5,
                        help="Initial prior P(focused)")
    parser.add_argument("--attention-stay", type=float, default=0.95,
                        help="Probability of staying in current attention state")
    parser.add_argument("--attention-switch", type=float, default=None,
                        help="Probability of switching attention state (default: same as stay)")
    parser.add_argument("--noise-start", type=int, default=None,
                        help="Start timestep for noise injection")
    parser.add_argument("--noise-end", type=int, default=None,
                        help="End timestep for noise injection")
    parser.add_argument("--force-distract-at", type=int, default=None,
                        help="Force attention to distracted at this timestep")
    parser.add_argument("--terminal-distraction", action="store_true",
                        help="Make distraction absorbing under STAY, focus unstable, SWITCH deterministic")
    parser.add_argument("--fixed-attention", type=str, default=None, choices=["focused", "distracted"],
                        help="Fix attention state (bypass attention inference)")
    parser.add_argument("--fixed-awareness", type=str, default=None, choices=["aware", "not_aware"],
                        help="Fix awareness state (bypass awareness inference)")
    parser.add_argument("--entropy-threshold", type=float, default=0.5,
                        help="Threshold for binning breath entropy into aware/not-aware")
    parser.add_argument("--precision-y-max", type=float, default=2.0,
                        help="Y-axis max for precision panel")
    parser.add_argument("--figure4", action="store_true", 
                        help="Run Figure 4 (with awareness inference)")
    parser.add_argument("--figure5", action="store_true", 
                        help="Run Figure 5 (with attention action selection)")
    parser.add_argument("--figure6", action="store_true",
                        help="Run Figure 6 (action selection + awareness)")
    parser.add_argument("--figure7", action="store_true",
                        help="Run Figure 7 (A2 learning across sits)")
    parser.add_argument("--figure7-1", action="store_true",
                        help="Run Figure 7.1 (joint A2 + B2 learning - non-meditator baseline)")
    parser.add_argument("--num-sits", type=int, default=200,
                        help="Number of sits for learning (figure 7/7.1)")
    parser.add_argument("--B2-init", type=float, default=0.52,
                        help="Initial B2 stay probability for agent's model (figure 7.1)")
    parser.add_argument("--B2-true", type=float, default=0.8,
                        help="True B2 stay probability (figure 7.1)")
    parser.add_argument("--A2-lr", type=float, default=1.0,
                        help="Learning rate for A2 learning")
    parser.add_argument("--A2-fr", type=float, default=0.9,
                        help="Forgetting rate for A2 learning")
    parser.add_argument("--A2-init", type=float, default=0.52,
                        help="Initial A2 precision for agent's model")
    parser.add_argument("--novice", action="store_true",
                        help="Novice mode: flat A2/B2 for attention (no perception/prediction of attention)")
    parser.add_argument("--gamma", type=float, default=16.0,
                        help="Policy precision for action selection")
    parser.add_argument("--E-stay", type=float, default=0.9,
                        help="Policy prior P(Stay) - habit towards staying")
    parser.add_argument("--C2-high", type=float, default=0.0,
                        help="Preference for high precision observation (focused)")
    parser.add_argument("--C2-low", type=float, default=0.0,
                        help="Preference for low precision observation (distracted)")
    parser.add_argument("--zeta-awareness", type=float, default=1.0,
                        help="Precision for awareness likelihood (0.5 = not habitually attending to it)")
    parser.add_argument("--initial-p-aware", type=float, default=0.5,
                        help="Initial prior P(aware) (0.5 = uncertain)")
    parser.add_argument("--awareness-informs-attention", action="store_true",
                        help="Awareness observations also inform attention inference (meditation connection)")
    parser.add_argument("--save", action="store_true", help="Save plot")
    args = parser.parse_args()
    
    here = os.path.dirname(__file__)
    outdir = os.path.join(here, "outputs")
    os.makedirs(outdir, exist_ok=True)
    
    if args.figure4:
        # Set novice parameters if requested
        A2_prec = 0.5 if args.novice else args.A2_precision
        B2_stay = 0.5 if args.novice else args.attention_stay
        
        # Run Figure 4 with awareness inference
        results = run_hierarchical_with_awareness(
            T=args.T,
            seed=args.seed,
            A1_precision=args.A1_precision,
            zeta_focused=args.zeta_focused,
            zeta_distracted=args.zeta_distracted,
            zeta_step=args.zeta_step,
            log_zeta_prior_var=args.log_zeta_prior_var,
            A2_precision=A2_prec,
            attention_stay_prob=B2_stay,
            zeta_awareness=args.zeta_awareness,
            initial_p_focused=args.initial_p_focused,
            initial_p_aware=args.initial_p_aware,
            entropy_threshold=args.entropy_threshold,
            noise_start=args.noise_start,
            noise_end=args.noise_end,
            force_distract_at=args.force_distract_at,
            fixed_attention=args.fixed_attention,
            fixed_awareness=args.fixed_awareness,
            awareness_informs_attention=args.awareness_informs_attention,
        )
        
        # Determine filename
        mode_suffix = ""
        if args.novice:
            mode_suffix = "_novice"
        if args.zeta_awareness < 1.0:
            mode_suffix += f"_zetaAware{args.zeta_awareness}"
        if args.awareness_informs_attention:
            mode_suffix += "_aware2attn"
        # Add precision params to suffix
        mode_suffix += f"_A1{args.A1_precision}"
        mode_suffix += f"_zfoc{args.zeta_focused}_zdist{args.zeta_distracted}"
        mode_suffix += f"_step{args.zeta_step}_var{args.log_zeta_prior_var}"
        if args.force_distract_at is not None:
            mode_suffix += f"_distract{args.force_distract_at}"
        if args.noise_start is not None:
            filename = f"figure4{mode_suffix}_noise{args.noise_start}-{args.noise_end}.png"
        else:
            filename = f"figure4{mode_suffix}.png"
        save_path = os.path.join(outdir, filename) if args.save else None
        
        fig = plot_figure4(results, save_path=save_path, precision_y_max=args.precision_y_max)
    
    elif args.figure5:
        # Run Figure 5 with action selection
        results = run_hierarchical_with_action(
            T=args.T,
            seed=args.seed,
            A1_precision=args.A1_precision,
            zeta_focused=args.zeta_focused,
            zeta_distracted=args.zeta_distracted,
            zeta_step=args.zeta_step,
            log_zeta_prior_var=args.log_zeta_prior_var,
            A2_precision=args.A2_precision,
            B2_stay_prob=args.attention_stay,
            B2_switch_prob=args.attention_switch if args.attention_switch is not None else args.attention_stay,
            initial_p_focused=args.initial_p_focused,
            gamma=args.gamma,
            E_stay=args.E_stay,
            C2_high_prec_pref=args.C2_high,
            C2_low_prec_pref=args.C2_low,
            noise_start=args.noise_start,
            noise_end=args.noise_end,
            force_distract_at=args.force_distract_at,
            terminal_distraction=args.terminal_distraction,
        )
        
        save_path = os.path.join(outdir, "figure5") if args.save else None
        fig = plot_figure5(results, save_path=save_path, precision_y_max=args.precision_y_max)
    
    elif args.figure6:
        # Run Figure 6 with action selection AND awareness
        results = run_hierarchical_with_action_and_awareness(
            T=args.T,
            seed=args.seed,
            A1_precision=args.A1_precision,
            zeta_focused=args.zeta_focused,
            zeta_distracted=args.zeta_distracted,
            zeta_step=args.zeta_step,
            log_zeta_prior_var=args.log_zeta_prior_var,
            A2_precision=args.A2_precision,
            A3_precision=args.A3_precision if hasattr(args, 'A3_precision') else 0.9,
            zeta_awareness=args.zeta_awareness if hasattr(args, 'zeta_awareness') else 1.0,
            B2_stay_prob=args.attention_stay,
            B2_switch_prob=args.attention_switch if args.attention_switch is not None else args.attention_stay,
            initial_p_focused=args.initial_p_focused,
            gamma=args.gamma,
            E_stay=args.E_stay,
            C2_high_prec_pref=args.C2_high,
            C2_low_prec_pref=args.C2_low,
            noise_start=args.noise_start,
            noise_end=args.noise_end,
            force_distract_at=args.force_distract_at,
            terminal_distraction=args.terminal_distraction,
        )
        
        save_path = os.path.join(outdir, "figure6") if args.save else None
        fig = plot_figure6(results, save_path=save_path, precision_y_max=args.precision_y_max)
    
    elif args.figure7:
        # Run Figure 7: A2 learning across sits
        results = run_A2_learning(
            num_sits=args.num_sits,
            T_per_sit=args.T,
            seed=args.seed,
            lr=args.A2_lr,
            fr=args.A2_fr,
            initial_A2_precision=args.A2_init,
            A1_precision=args.A1_precision,
            zeta_focused=args.zeta_focused,
            zeta_distracted=args.zeta_distracted,
            zeta_step=args.zeta_step,
            log_zeta_prior_var=args.log_zeta_prior_var,
            A2_true_precision=args.A2_precision,
            B2_stay_prob=args.attention_stay,
            B2_switch_prob=args.attention_switch if args.attention_switch is not None else args.attention_stay,
            terminal_distraction=args.terminal_distraction,
            gamma=args.gamma,
            E_stay=args.E_stay,
        )
        
        save_path = os.path.join(outdir, "figure7") if args.save else None
        fig = plot_figure7(results, save_path=save_path)

    elif args.figure7_1:
        # Run Figure 7.1: Joint A2 + B2 learning (non-meditator baseline)
        results = run_A2_B2_learning(
            num_sits=args.num_sits,
            T_per_sit=args.T,
            seed=args.seed,
            lr=args.A2_lr,
            fr=args.A2_fr,
            initial_A2_precision=args.A2_init,
            A2_true_precision=args.A2_precision,
            initial_B2_stay_prob=args.B2_init,
            B2_true_stay_prob=args.B2_true,
            A1_precision=args.A1_precision,
            zeta_focused=args.zeta_focused,
            zeta_distracted=args.zeta_distracted,
            zeta_step=args.zeta_step,
            log_zeta_prior_var=args.log_zeta_prior_var,
            gamma=args.gamma,
            E_stay=args.E_stay,
        )

        save_path = os.path.join(outdir, "figure7_1") if args.save else None
        fig = plot_figure7_1(results, save_path=save_path)

    else:
        # Run Figure 3 (original)
        results = run_hierarchical_breath(
            T=args.T,
            seed=args.seed,
            A1_precision=args.A1_precision,
            zeta_focused=args.zeta_focused,
            zeta_distracted=args.zeta_distracted,
            zeta_step=args.zeta_step,
            log_zeta_prior_var=args.log_zeta_prior_var,
            A2_precision=args.A2_precision,
            attention_stay_prob=args.attention_stay,
            initial_p_focused=args.initial_p_focused,
            noise_start=args.noise_start,
            noise_end=args.noise_end,
            fixed_attention=args.fixed_attention,
        )
        
        # Determine filename based on mode
        if args.fixed_attention is not None:
            filename = f"figure3_fixed_{args.fixed_attention}_A1{args.A1_precision}.png"
        elif args.noise_start is not None:
            filename = f"figure3.1_hierarchical_breath_noise{args.noise_start}-{args.noise_end}.png"
        else:
            filename = "figure3_hierarchical_breath.png"
        save_path = os.path.join(outdir, filename) if args.save else None
        
        fig = plot_figure3(results, save_path=save_path, precision_y_max=args.precision_y_max)
    
    if not args.save:
        plt.show()

