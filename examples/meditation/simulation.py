#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simulation functions for focused attention meditation model.

This module contains simulation loops for each figure:
    Figure 1: Breath perception with dynamic precision
    Figure 2: Attention modulates precision (comparison)
    Figure 3: Precision dynamics improve A1 learning
    Figure 4: The attention trap (non-meditator)
    Figure 5: Meditation instruction breaks the cycle
    Figure 6: Learning across sits

Each run_figure_X function returns a results dict for plotting.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Optional

from pymdp import utils, control, learning
from pymdp.maths import softmax, spm_log_single as log_stable
from pymdp.maths import scale_likelihood, update_likelihood_precision
from pymdp.envs import BreathEnv

try:
    from .models import (
        ModelParams,
        build_breath_model, build_attention_model, build_awareness_model,
        build_environment, build_dirichlet_prior,
        initialize_A2_for_learning, initialize_B2_for_learning,
        INHALE, EXHALE, FOCUSED, DISTRACTED, AWARE, UNAWARE,
        STAY, SWITCH, PRECISE, IMPRECISE, OBS_AWARE, OBS_UNAWARE,
    )
except ImportError:
    from examples.meditation.models import (
        ModelParams,
        build_breath_model, build_attention_model, build_awareness_model,
        build_environment, build_dirichlet_prior,
        initialize_A2_for_learning, initialize_B2_for_learning,
        INHALE, EXHALE, FOCUSED, DISTRACTED, AWARE, UNAWARE,
        STAY, SWITCH, PRECISE, IMPRECISE, OBS_AWARE, OBS_UNAWARE,
    )


EPS_VAL = 1e-16


# =============================================================================
# Helper Functions
# =============================================================================

def bayesian_update(A: np.ndarray, obs: int, prior: np.ndarray) -> np.ndarray:
    """Simple Bayesian state inference: posterior = likelihood * prior."""
    likelihood = A[obs, :]
    posterior = likelihood * prior
    return posterior / (posterior.sum() + EPS_VAL)


def compute_entropy(p: np.ndarray) -> float:
    """Compute entropy H(p) = -sum(p * log(p))."""
    p_safe = np.clip(p, EPS_VAL, 1.0)
    return -np.sum(p_safe * np.log(p_safe))


def sample_action(Q_pi: np.ndarray, rng: np.random.Generator) -> int:
    """Sample action from policy distribution."""
    return int(rng.choice(len(Q_pi), p=Q_pi))


def compute_expected_free_energy(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    qs: np.ndarray,
    gamma: float = 16.0
) -> np.ndarray:
    """
    Compute expected free energy G for each action.

    G(a) = ambiguity + risk
         = E[H[P(o|s)]] - E[log P(o|C)]
    """
    num_actions = B.shape[2]
    G = np.zeros(num_actions)

    for a in range(num_actions):
        # Predicted state after action
        qs_next = B[:, :, a] @ qs

        # Expected observation
        qo = A @ qs_next

        # Ambiguity: expected entropy of likelihood
        H_A = -np.sum(A * np.log(A + EPS_VAL), axis=0)  # Entropy per state
        ambiguity = np.dot(qs_next, H_A)

        # Risk: divergence from preferences
        log_C = log_stable(softmax(C))
        risk = -np.dot(qo, log_C)

        G[a] = ambiguity + risk

    return G


def select_action(
    A2,
    B2: np.ndarray,
    C2,
    qs_attention: np.ndarray,
    gamma: float,
    rng: np.random.Generator
) -> int:
    """
    Select action using expected free energy.

    Handles both single modality (np.ndarray) and multi-modality (obj_array) A2/C2.
    """
    num_actions = B2.shape[2]
    G = np.zeros(num_actions)

    # Handle multi-modality A2
    if isinstance(A2, np.ndarray) and A2.dtype == object:
        num_modalities = len(A2)
    else:
        num_modalities = 1
        A2 = [A2]
        C2 = [C2]

    for a in range(num_actions):
        qs_next = B2[:, :, a] @ qs_attention

        for m in range(num_modalities):
            A_m = A2[m]
            C_m = C2[m]

            qo = A_m @ qs_next
            H_A = -np.sum(A_m * np.log(A_m + EPS_VAL), axis=0)
            ambiguity = np.dot(qs_next, H_A)

            log_C = log_stable(softmax(C_m))
            risk = -np.dot(qo, log_C)

            G[a] += ambiguity + risk

    # Policy as softmax of negative EFE
    Q_pi = softmax(-gamma * G)
    return sample_action(Q_pi, rng), Q_pi


# =============================================================================
# Figure 1: Breath Perception with Dynamic Precision
# =============================================================================

def run_figure1(
    T: int = 200,
    seed: int = 42,
    params: ModelParams = None,
) -> Dict[str, Any]:
    """
    Run breath perception with dynamic precision (B.45).

    Demonstrates how precision (zeta) tracks predictability:
    - Increases when predictions are accurate
    - Decreases when predictions are poor

    Returns
    -------
    results : dict
        Contains time series for plotting
    """
    if params is None:
        params = ModelParams()

    rng = np.random.default_rng(seed)

    # Build models
    A1, B1 = build_breath_model(params)
    env = BreathEnv(seed=seed)

    # Initialize
    qs_breath = np.array([0.5, 0.5])
    zeta = 1.0  # Start at prior mean

    # Logs
    true_states = np.zeros(T, dtype=int)
    posteriors = np.zeros((T, 2))
    zeta_history = np.zeros(T)
    prediction_errors = np.zeros(T)
    observations = np.zeros(T, dtype=int)

    obs = int(env.reset())

    for t in range(T):
        true_states[t] = env.state
        observations[t] = obs

        # Prior from transition
        prior = B1 @ qs_breath if t > 0 else qs_breath

        # Update precision BEFORE inference
        zeta_new, pe, _ = update_likelihood_precision(
            zeta=zeta,
            A=A1,
            obs=obs,
            qs=prior,
            log_zeta_prior_mean=0.0,
            log_zeta_prior_var=params.zeta_prior_var,
            zeta_step=params.zeta_step,
            min_zeta=params.zeta_min,
            max_zeta=params.zeta_max
        )
        zeta = zeta_new
        zeta_history[t] = zeta
        prediction_errors[t] = pe

        # State inference with precision-scaled likelihood
        A1_scaled = scale_likelihood(A1, zeta)
        qs_breath = bayesian_update(A1_scaled, obs, prior)
        posteriors[t] = qs_breath

        # Step environment
        obs = int(env.step(None))

    # Metrics
    inferred = np.argmax(posteriors, axis=1)
    accuracy = (inferred == true_states).mean()

    return {
        "true_states": true_states,
        "observations": observations,
        "posteriors": posteriors,
        "zeta_history": zeta_history,
        "prediction_errors": prediction_errors,
        "accuracy": accuracy,
        "params": params,
        "T": T,
    }


# =============================================================================
# Figure 2: Attention Modulates Precision
# =============================================================================

def run_figure2(
    T: int = 200,
    seed: int = 42,
    params: ModelParams = None,
    distraction_onset: int = 100,
) -> Dict[str, Any]:
    """
    Compare breath perception when focused vs distracted.

    Runs a single simulation where attention switches from
    focused to distracted at distraction_onset.

    Demonstrates:
    - Focused: high precision prior -> accurate inference
    - Distracted: low precision prior -> poor inference
    """
    if params is None:
        params = ModelParams()

    rng = np.random.default_rng(seed)

    # Build models
    A1, B1 = build_breath_model(params)
    env = BreathEnv(seed=seed)

    # Initialize
    qs_breath = np.array([0.5, 0.5])

    # Logs
    true_states = np.zeros(T, dtype=int)
    posteriors = np.zeros((T, 2))
    zeta_history = np.zeros(T)
    attention_states = np.zeros(T, dtype=int)  # 0=focused, 1=distracted

    obs = int(env.reset())

    for t in range(T):
        true_states[t] = env.state

        # Attention state (externally controlled for this demo)
        attention = DISTRACTED if t >= distraction_onset else FOCUSED
        attention_states[t] = attention

        # Set precision based on attention
        zeta = params.zeta_focused if attention == FOCUSED else params.zeta_distracted
        zeta_history[t] = zeta

        # Prior from transition
        prior = B1 @ qs_breath if t > 0 else qs_breath

        # State inference
        A1_scaled = scale_likelihood(A1, zeta)
        qs_breath = bayesian_update(A1_scaled, obs, prior)
        posteriors[t] = qs_breath

        obs = int(env.step(None))

    # Metrics by attention state
    focused_mask = attention_states == FOCUSED
    distracted_mask = attention_states == DISTRACTED

    inferred = np.argmax(posteriors, axis=1)
    acc_focused = (inferred[focused_mask] == true_states[focused_mask]).mean()
    acc_distracted = (inferred[distracted_mask] == true_states[distracted_mask]).mean()

    return {
        "true_states": true_states,
        "posteriors": posteriors,
        "zeta_history": zeta_history,
        "attention_states": attention_states,
        "accuracy_focused": acc_focused,
        "accuracy_distracted": acc_distracted,
        "distraction_onset": distraction_onset,
        "params": params,
        "T": T,
    }


# =============================================================================
# Figure 3: Precision Dynamics Improve Learning
# =============================================================================

def run_figure3(
    T: int = 500,
    seed: int = 42,
    params: ModelParams = None,
    mode: str = "fixed",  # "fixed", "dynamic", "attention"
    fixed_attention: int = FOCUSED,  # For "attention" mode
) -> Dict[str, Any]:
    """
    Compare A1 learning under different precision regimes.

    Modes:
    - "fixed": Fixed zeta=1.0, standard Dirichlet update
    - "dynamic": B.45 precision updates, precision-weighted learning
    - "attention": Known attention state sets precision

    Demonstrates that precision-weighted learning converges
    faster and closer to truth.
    """
    if params is None:
        params = ModelParams()

    rng = np.random.default_rng(seed)

    # Build true model
    A1_true, B1 = build_breath_model(params)
    env = BreathEnv(seed=seed)

    # Initialize learnable A1 (start near uniform)
    pA1 = np.ones((2, 2)) * 0.5 + rng.random((2, 2)) * 0.1
    pA1 = pA1 / pA1.sum(axis=0, keepdims=True)  # Normalize
    pA1_concentration = pA1 * 1.0  # Weak prior

    # Initialize
    qs_breath = np.array([0.5, 0.5])
    zeta = 1.0

    # Logs
    true_states = np.zeros(T, dtype=int)
    A1_history = np.zeros((T, 2, 2))  # Track A1 learning
    zeta_history = np.zeros(T)

    obs = int(env.reset())

    for t in range(T):
        true_states[t] = env.state

        # Get current A1 estimate
        A1_est = pA1_concentration / pA1_concentration.sum(axis=0, keepdims=True)
        A1_history[t] = A1_est

        # Prior
        prior = B1 @ qs_breath if t > 0 else qs_breath

        # Determine precision based on mode
        if mode == "fixed":
            zeta = 1.0
        elif mode == "dynamic":
            zeta_new, pe, _ = update_likelihood_precision(
                zeta=zeta, A=A1_est, obs=obs, qs=prior,
                log_zeta_prior_mean=0.0,
                log_zeta_prior_var=params.zeta_prior_var,
                zeta_step=params.zeta_step,
                min_zeta=params.zeta_min,
                max_zeta=params.zeta_max
            )
            zeta = zeta_new
        elif mode == "attention":
            zeta = params.zeta_focused if fixed_attention == FOCUSED else params.zeta_distracted

        zeta_history[t] = zeta

        # State inference
        A1_scaled = scale_likelihood(A1_est, zeta)
        qs_breath = bayesian_update(A1_scaled, obs, prior)

        # Learning: update A1 concentration parameters
        # Precision-weighted update: higher zeta = faster learning
        obs_onehot = np.zeros(2)
        obs_onehot[obs] = 1.0

        lr = params.A_learning_rate * (zeta if mode != "fixed" else 1.0)
        pA1_concentration[:, true_states[t]] += lr * obs_onehot

        obs = int(env.step(None))

    # Compute distance to true A1 over time
    A1_error = np.zeros(T)
    for t in range(T):
        A1_error[t] = np.mean(np.abs(A1_history[t] - A1_true))

    return {
        "true_states": true_states,
        "A1_history": A1_history,
        "A1_true": A1_true,
        "A1_error": A1_error,
        "zeta_history": zeta_history,
        "mode": mode,
        "params": params,
        "T": T,
    }


# =============================================================================
# Figure 4: The Attention Trap (Non-Meditator)
# =============================================================================

def run_figure4(
    T: int = 300,
    seed: int = 42,
    params: ModelParams = None,
    distraction_onset: int = 50,
) -> Dict[str, Any]:
    """
    Full hierarchical model without meditation instruction.

    Demonstrates the attention trap:
    - Agent starts focused
    - Environment forces distraction at distraction_onset
    - Agent can't accurately infer attention state (weak A2)
    - Can't select appropriate actions
    - Remains trapped in distraction
    """
    if params is None:
        params = ModelParams()

    rng = np.random.default_rng(seed)

    # Build models
    A1, B1 = build_breath_model(params)
    A2, B2, C2 = build_attention_model(params, include_awareness_modality=False)
    A3, B3 = build_awareness_model(params)
    B2_true = build_environment(params)

    env = BreathEnv(seed=seed)

    # Initialize beliefs
    qs_breath = np.array([0.5, 0.5])
    qs_attention = np.array([0.9, 0.1])  # Start believing focused
    qs_awareness = np.array([0.5, 0.5])

    # True states
    true_attention = FOCUSED
    true_awareness = AWARE
    zeta = params.zeta_focused

    # Logs
    T_logs = T
    true_breath_states = np.zeros(T_logs, dtype=int)
    true_attention_states = np.zeros(T_logs, dtype=int)
    true_awareness_states = np.zeros(T_logs, dtype=int)

    posterior_breath = np.zeros((T_logs, 2))
    posterior_attention = np.zeros((T_logs, 2))
    posterior_awareness = np.zeros((T_logs, 2))

    zeta_history = np.zeros(T_logs)
    actions = np.zeros(T_logs, dtype=int)

    obs_breath = int(env.reset())
    prev_action = STAY

    for t in range(T):
        # Force distraction at onset
        if t == distraction_onset:
            true_attention = DISTRACTED

        # Store true states
        true_breath_states[t] = env.state
        true_attention_states[t] = true_attention
        true_awareness_states[t] = true_awareness

        # Set precision based on true attention
        zeta = params.zeta_focused if true_attention == FOCUSED else params.zeta_distracted
        zeta_history[t] = zeta

        # === BREATH INFERENCE ===
        prior_breath = B1 @ qs_breath if t > 0 else qs_breath
        A1_scaled = scale_likelihood(A1, zeta)
        qs_breath = bayesian_update(A1_scaled, obs_breath, prior_breath)
        posterior_breath[t] = qs_breath

        # === AWARENESS INFERENCE ===
        # Observe entropy of breath posterior as proxy for precision
        entropy = compute_entropy(qs_breath)
        obs_entropy = OBS_AWARE if entropy < 0.5 else OBS_UNAWARE

        prior_awareness = B3 @ qs_awareness if t > 0 else qs_awareness
        qs_awareness = bayesian_update(A3, obs_entropy, prior_awareness)
        posterior_awareness[t] = qs_awareness

        # === ATTENTION INFERENCE ===
        # Observe precision level
        obs_precision = PRECISE if zeta > 1.0 else IMPRECISE

        prior_attention = B2[:, :, prev_action] @ qs_attention if t > 0 else qs_attention
        qs_attention = bayesian_update(A2, obs_precision, prior_attention)
        posterior_attention[t] = qs_attention

        # === ACTION SELECTION ===
        action, Q_pi = select_action(A2, B2, C2, qs_attention, params.gamma, rng)
        actions[t] = action

        # === ENVIRONMENT TRANSITION ===
        # True attention transitions based on action
        p_transition = B2_true[:, true_attention, action]
        true_attention = rng.choice(2, p=p_transition)

        # Awareness transitions
        p_aware = B3[:, true_awareness]
        true_awareness = rng.choice(2, p=p_aware)

        prev_action = action
        obs_breath = int(env.step(None))

    # Metrics
    inferred_attention = np.argmax(posterior_attention, axis=1)
    attention_accuracy = (inferred_attention == true_attention_states).mean()
    time_distracted = (true_attention_states == DISTRACTED).mean()
    switch_rate = (actions == SWITCH).mean()

    return {
        "true_breath": true_breath_states,
        "true_attention": true_attention_states,
        "true_awareness": true_awareness_states,
        "posterior_breath": posterior_breath,
        "posterior_attention": posterior_attention,
        "posterior_awareness": posterior_awareness,
        "zeta_history": zeta_history,
        "actions": actions,
        "attention_accuracy": attention_accuracy,
        "time_distracted": time_distracted,
        "switch_rate": switch_rate,
        "distraction_onset": distraction_onset,
        "params": params,
        "T": T,
    }


# =============================================================================
# Figure 5: Meditation Instruction Breaks the Cycle
# =============================================================================

def run_figure5(
    T: int = 300,
    seed: int = 42,
    params: ModelParams = None,
    distraction_onset: int = 50,
) -> Dict[str, Any]:
    """
    Full hierarchical model WITH meditation instruction.

    Meditation instruction provides:
    1. Two-modality A2: precision obs + awareness obs -> attention
    2. Preference for awareness (C2)

    Demonstrates escape from the attention trap:
    - Agent can infer attention via awareness observations
    - Selects SWITCH action when distracted
    - Returns to focused state
    """
    if params is None:
        params = ModelParams()

    rng = np.random.default_rng(seed)

    # Build models - WITH meditation instruction
    A1, B1 = build_breath_model(params)
    A2, B2, C2 = build_attention_model(params, include_awareness_modality=True)
    A3, B3 = build_awareness_model(params)
    B2_true = build_environment(params)

    env = BreathEnv(seed=seed)

    # Initialize beliefs
    qs_breath = np.array([0.5, 0.5])
    qs_attention = np.array([0.9, 0.1])
    qs_awareness = np.array([0.5, 0.5])

    # True states
    true_attention = FOCUSED
    true_awareness = AWARE
    zeta = params.zeta_focused

    # Logs
    true_breath_states = np.zeros(T, dtype=int)
    true_attention_states = np.zeros(T, dtype=int)
    true_awareness_states = np.zeros(T, dtype=int)

    posterior_breath = np.zeros((T, 2))
    posterior_attention = np.zeros((T, 2))
    posterior_awareness = np.zeros((T, 2))

    zeta_history = np.zeros(T)
    actions = np.zeros(T, dtype=int)

    obs_breath = int(env.reset())
    prev_action = STAY

    for t in range(T):
        # Force distraction at onset
        if t == distraction_onset:
            true_attention = DISTRACTED

        # Store true states
        true_breath_states[t] = env.state
        true_attention_states[t] = true_attention
        true_awareness_states[t] = true_awareness

        # Set precision
        zeta = params.zeta_focused if true_attention == FOCUSED else params.zeta_distracted
        zeta_history[t] = zeta

        # === BREATH INFERENCE ===
        prior_breath = B1 @ qs_breath if t > 0 else qs_breath
        A1_scaled = scale_likelihood(A1, zeta)
        qs_breath = bayesian_update(A1_scaled, obs_breath, prior_breath)
        posterior_breath[t] = qs_breath

        # === AWARENESS INFERENCE ===
        entropy = compute_entropy(qs_breath)
        obs_entropy = OBS_AWARE if entropy < 0.5 else OBS_UNAWARE

        prior_awareness = B3 @ qs_awareness if t > 0 else qs_awareness
        qs_awareness = bayesian_update(A3, obs_entropy, prior_awareness)
        posterior_awareness[t] = qs_awareness

        # === ATTENTION INFERENCE (with meditation instruction) ===
        obs_precision = PRECISE if zeta > 1.0 else IMPRECISE

        prior_attention = B2[:, :, prev_action] @ qs_attention if t > 0 else qs_attention

        # Combine both modalities
        likelihood_precision = A2[0][obs_precision, :]
        likelihood_awareness = A2[1][obs_entropy, :]
        likelihood_combined = likelihood_precision * likelihood_awareness

        qs_attention = likelihood_combined * prior_attention
        qs_attention = qs_attention / (qs_attention.sum() + EPS_VAL)
        posterior_attention[t] = qs_attention

        # === ACTION SELECTION ===
        action, Q_pi = select_action(A2, B2, C2, qs_attention, params.gamma, rng)
        actions[t] = action

        # === ENVIRONMENT TRANSITION ===
        p_transition = B2_true[:, true_attention, action]
        true_attention = rng.choice(2, p=p_transition)

        p_aware = B3[:, true_awareness]
        true_awareness = rng.choice(2, p=p_aware)

        prev_action = action
        obs_breath = int(env.step(None))

    # Metrics
    inferred_attention = np.argmax(posterior_attention, axis=1)
    attention_accuracy = (inferred_attention == true_attention_states).mean()
    time_distracted = (true_attention_states == DISTRACTED).mean()
    switch_rate = (actions == SWITCH).mean()

    return {
        "true_breath": true_breath_states,
        "true_attention": true_attention_states,
        "true_awareness": true_awareness_states,
        "posterior_breath": posterior_breath,
        "posterior_attention": posterior_attention,
        "posterior_awareness": posterior_awareness,
        "zeta_history": zeta_history,
        "actions": actions,
        "attention_accuracy": attention_accuracy,
        "time_distracted": time_distracted,
        "switch_rate": switch_rate,
        "distraction_onset": distraction_onset,
        "params": params,
        "T": T,
    }


# =============================================================================
# Figure 6: Learning Across Sits
# =============================================================================

def run_figure6(
    num_sits: int = 200,
    T_per_sit: int = 100,
    seed: int = 42,
    params: ModelParams = None,
    meditation_start_sit: Optional[int] = None,  # None = no meditation instruction
) -> Dict[str, Any]:
    """
    Learning A2 and B2 across multiple meditation sits.

    Parameters
    ----------
    meditation_start_sit : int or None
        If None, no meditation instruction (non-meditator baseline).
        If int, meditation instruction introduced at this sit number.

    Demonstrates:
    - Non-meditator: trapped in distraction, can't learn A2/B2
    - Meditator: instruction breaks cycle, learns correct models
    """
    if params is None:
        params = ModelParams()

    rng = np.random.default_rng(seed)

    # Build models
    A1, B1 = build_breath_model(params)
    A3, B3 = build_awareness_model(params)
    B2_true = build_environment(params)

    # Initialize learnable parameters
    pA2 = initialize_A2_for_learning(params)
    pB2 = initialize_B2_for_learning(params)

    # Logs (per sit)
    attention_accuracy_per_sit = np.zeros(num_sits)
    time_distracted_per_sit = np.zeros(num_sits)
    switch_rate_per_sit = np.zeros(num_sits)
    A2_diagonal_per_sit = np.zeros(num_sits)  # Track learning
    B2_stay_foc_per_sit = np.zeros(num_sits)  # B2[FOCUSED, FOCUSED, STAY]
    B2_stay_dist_per_sit = np.zeros(num_sits)  # B2[DISTRACTED, DISTRACTED, STAY]

    for sit in range(num_sits):
        # Check if meditation instruction is active
        has_instruction = (meditation_start_sit is not None and sit >= meditation_start_sit)

        # Build A2/C2 based on instruction status
        if has_instruction:
            A2, B2, C2 = build_attention_model(params, include_awareness_modality=True)
        else:
            A2, B2, C2 = build_attention_model(params, include_awareness_modality=False)

        # Use learned A2 (single modality only for learning tracking)
        A2_learned = pA2 / pA2.sum(axis=0, keepdims=True)
        B2_learned = pB2 / pB2.sum(axis=0, keepdims=True)

        # Initialize for this sit
        env = BreathEnv(seed=seed + sit)
        qs_breath = np.array([0.5, 0.5])
        qs_attention = np.array([0.9, 0.1])
        qs_awareness = np.array([0.5, 0.5])

        true_attention = FOCUSED
        true_awareness = AWARE
        prev_action = STAY

        # Sit logs
        true_attention_sit = np.zeros(T_per_sit, dtype=int)
        actions_sit = np.zeros(T_per_sit, dtype=int)
        inferred_attention_sit = np.zeros(T_per_sit, dtype=int)

        obs_breath = int(env.reset())

        for t in range(T_per_sit):
            true_attention_sit[t] = true_attention

            # Precision based on attention
            zeta = params.zeta_focused if true_attention == FOCUSED else params.zeta_distracted

            # Breath inference
            prior_breath = B1 @ qs_breath if t > 0 else qs_breath
            A1_scaled = scale_likelihood(A1, zeta)
            qs_breath = bayesian_update(A1_scaled, obs_breath, prior_breath)

            # Awareness inference
            entropy = compute_entropy(qs_breath)
            obs_entropy = OBS_AWARE if entropy < 0.5 else OBS_UNAWARE
            prior_awareness = B3 @ qs_awareness if t > 0 else qs_awareness
            qs_awareness = bayesian_update(A3, obs_entropy, prior_awareness)

            # Attention inference
            obs_precision = PRECISE if zeta > 1.0 else IMPRECISE
            prior_attention = B2_learned[:, :, prev_action] @ qs_attention if t > 0 else qs_attention

            if has_instruction:
                # Two modalities
                likelihood_precision = A2[0][obs_precision, :]
                likelihood_awareness = A2[1][obs_entropy, :]
                likelihood_combined = likelihood_precision * likelihood_awareness
                qs_attention = likelihood_combined * prior_attention
            else:
                # Single modality (learned A2)
                qs_attention = A2_learned[obs_precision, :] * prior_attention

            qs_attention = qs_attention / (qs_attention.sum() + EPS_VAL)
            inferred_attention_sit[t] = np.argmax(qs_attention)

            # Action selection
            if has_instruction:
                action, _ = select_action(A2, B2_learned, C2, qs_attention, params.gamma, rng)
            else:
                action, _ = select_action(A2_learned, B2_learned, C2, qs_attention, params.gamma, rng)
            actions_sit[t] = action

            # Environment transition
            p_transition = B2_true[:, true_attention, action]
            new_attention = rng.choice(2, p=p_transition)

            # Learning updates (only if we have some experience)
            if t > 0:
                # A2 learning: obs_precision -> attention
                lr_A = params.A_learning_rate
                pA2[obs_precision, true_attention] += lr_A

                # B2 learning: (prev_attention, prev_action) -> attention
                lr_B = params.B_learning_rate
                pB2[new_attention, true_attention, prev_action] += lr_B

            # Forgetting (decay toward prior)
            pA2 *= params.forgetting_rate
            pA2 += (1 - params.forgetting_rate) * 0.5
            pB2 *= params.forgetting_rate
            pB2 += (1 - params.forgetting_rate) * 0.5

            true_attention = new_attention
            prev_action = action
            obs_breath = int(env.step(None))

        # Record metrics for this sit
        attention_accuracy_per_sit[sit] = (inferred_attention_sit == true_attention_sit).mean()
        time_distracted_per_sit[sit] = (true_attention_sit == DISTRACTED).mean()
        switch_rate_per_sit[sit] = (actions_sit == SWITCH).mean()

        A2_current = pA2 / pA2.sum(axis=0, keepdims=True)
        A2_diagonal_per_sit[sit] = (A2_current[0, 0] + A2_current[1, 1]) / 2

        B2_current = pB2 / pB2.sum(axis=0, keepdims=True)
        B2_stay_foc_per_sit[sit] = B2_current[FOCUSED, FOCUSED, STAY]
        B2_stay_dist_per_sit[sit] = B2_current[DISTRACTED, DISTRACTED, STAY]

    return {
        "attention_accuracy": attention_accuracy_per_sit,
        "time_distracted": time_distracted_per_sit,
        "switch_rate": switch_rate_per_sit,
        "A2_diagonal": A2_diagonal_per_sit,
        "B2_stay_focused": B2_stay_foc_per_sit,
        "B2_stay_distracted": B2_stay_dist_per_sit,
        "meditation_start_sit": meditation_start_sit,
        "num_sits": num_sits,
        "T_per_sit": T_per_sit,
        "params": params,
    }
