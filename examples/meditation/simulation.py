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
from pymdp.control import update_posterior_policies
from pymdp.envs import BreathEnv

try:
    from .models import (
        ModelParams,
        build_breath_model, build_attention_model, build_awareness_model,
        build_environment, build_A2_true, build_dirichlet_prior,
        initialize_A2_for_learning, initialize_B2_for_learning,
        INHALE, EXHALE, FOCUSED, DISTRACTED, AWARE, UNAWARE,
        STAY, SWITCH, PRECISE, IMPRECISE, OBS_AWARE, OBS_UNAWARE,
    )
except ImportError:
    from examples.meditation.models import (
        ModelParams,
        build_breath_model, build_attention_model, build_awareness_model,
        build_environment, build_A2_true, build_dirichlet_prior,
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
    rng: np.random.Generator,
    E: np.ndarray = None,
) -> int:
    """
    Select action using expected free energy.

    Handles both single modality (np.ndarray) and multi-modality (obj_array) A2/C2.

    Parameters
    ----------
    E : np.ndarray, optional
        Policy prior. If None, uniform prior is used.
        E[0] = P(STAY), E[1] = P(SWITCH)
    """
    num_actions = B2.shape[2]
    G = np.zeros(num_actions)

    # Default uniform policy prior
    if E is None:
        E = np.ones(num_actions) / num_actions

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

    # Policy as softmax of negative EFE + log policy prior
    log_E = np.log(E + EPS_VAL)
    Q_pi = softmax(-gamma * G + log_E)
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
    Attention inference modulates precision with dynamic updating.

    Runs a simulation where:
    1. True attention switches from focused to distracted at distraction_onset
    2. Agent infers attention state from precision observations
    3. Inferred attention provides descending prior for likelihood precision
    4. Precision evolves dynamically using B.45 update

    Demonstrates the hierarchical message passing:
    - Attention inference -> precision prior (descending message)
    - Precision dynamics -> breath inference quality
    """
    if params is None:
        params = ModelParams()

    rng = np.random.default_rng(seed)

    # Build models
    A1, B1 = build_breath_model(params)
    A2, B2, C2 = build_attention_model(params, include_awareness_modality=False)
    A2_true = build_A2_true(params)  # For generating precision observations
    env = BreathEnv(seed=seed)

    # Initialize beliefs
    qs_breath = np.array([0.5, 0.5])
    qs_attention = np.array([0.5, 0.5])  # Start uncertain about attention
    zeta = 1.0  # Start at prior mean

    # Logs
    true_breath_states = np.zeros(T, dtype=int)
    true_attention_states = np.zeros(T, dtype=int)
    breath_posteriors = np.zeros((T, 2))
    attention_posteriors = np.zeros((T, 2))
    zeta_history = np.zeros(T)
    zeta_prior_history = np.zeros(T)  # Descending prior from attention
    prediction_errors = np.zeros(T)

    obs_breath = int(env.reset())

    for t in range(T):
        # True states
        true_breath_states[t] = env.state
        true_attention = DISTRACTED if t >= distraction_onset else FOCUSED
        true_attention_states[t] = true_attention

        # === ATTENTION INFERENCE ===
        # Sample precision observation from TRUE attention state
        obs_precision = rng.choice([PRECISE, IMPRECISE], p=A2_true[:, true_attention])

        # Attention transition prior (allows tracking of state changes)
        # Use a transition matrix that assumes attention can change
        # B_att[next, current] - some probability of switching
        p_stay = 0.9  # Attention tends to persist
        B_att = np.array([
            [p_stay, 1 - p_stay],      # P(focused | prev)
            [1 - p_stay, p_stay]       # P(distracted | prev)
        ])
        prior_attention = B_att @ qs_attention

        # Update attention beliefs with observation
        qs_attention = bayesian_update(A2, obs_precision, prior_attention)
        attention_posteriors[t] = qs_attention

        # === DESCENDING MESSAGE: Attention -> Precision Prior ===
        # Inferred attention sets the STARTING POINT for precision
        # Expected precision = P(focused) * zeta_focused + P(distracted) * zeta_distracted
        zeta_prior = (qs_attention[FOCUSED] * params.zeta_focused +
                      qs_attention[DISTRACTED] * params.zeta_distracted)
        zeta_prior_history[t] = zeta_prior

        # === DYNAMIC PRECISION UPDATE (B.45) ===
        # Key: descending message sets starting point, B.45 adjusts with pull towards ζ=1
        prior_breath = B1 @ qs_breath if t > 0 else qs_breath

        zeta, pe, _ = update_likelihood_precision(
            zeta=zeta_prior,  # Descending message sets starting point
            A=A1,
            obs=obs_breath,
            qs=prior_breath,
            # log_zeta_prior_mean defaults to 0.0 (pulls towards ζ=1)
            log_zeta_prior_var=params.zeta_prior_var,
            zeta_step=params.zeta_step,
            min_zeta=params.zeta_min,
            max_zeta=params.zeta_max
        )
        zeta_history[t] = zeta
        prediction_errors[t] = pe

        # === BREATH INFERENCE ===
        A1_scaled = scale_likelihood(A1, zeta)
        qs_breath = bayesian_update(A1_scaled, obs_breath, prior_breath)
        breath_posteriors[t] = qs_breath

        # Step environment
        obs_breath = int(env.step(None))

    # Metrics
    focused_mask = true_attention_states == FOCUSED
    distracted_mask = true_attention_states == DISTRACTED

    inferred_breath = np.argmax(breath_posteriors, axis=1)
    acc_focused = (inferred_breath[focused_mask] == true_breath_states[focused_mask]).mean()
    acc_distracted = (inferred_breath[distracted_mask] == true_breath_states[distracted_mask]).mean()

    inferred_attention = np.argmax(attention_posteriors, axis=1)
    attention_accuracy = (inferred_attention == true_attention_states).mean()

    return {
        "true_states": true_breath_states,
        "true_attention_states": true_attention_states,
        "posteriors": breath_posteriors,
        "attention_posteriors": attention_posteriors,
        "zeta_history": zeta_history,
        "zeta_prior_history": zeta_prior_history,  # Descending prior from attention
        "prediction_errors": prediction_errors,
        "attention_states": true_attention_states,  # For backward compatibility
        "accuracy_focused": acc_focused,
        "accuracy_distracted": acc_distracted,
        "attention_accuracy": attention_accuracy,
        "distraction_onset": distraction_onset,
        "params": params,
        "T": T,
    }


# =============================================================================
# Figure 2.1: Attention with Mental Action (Natural Transitions)
# =============================================================================

def run_figure2_1(
    T: int = 200,
    seed: int = 42,
    params: ModelParams = None,
) -> Dict[str, Any]:
    """
    Attention inference with mental action and natural attention transitions.

    Like Figure 2 but with:
    - Mental action selection (STAY/SWITCH)
    - Natural attention transitions via B2_true (no forced distraction)
    - Distraction is absorbing under STAY

    Demonstrates:
    - How action selection depends on attention inference
    - How precision observations inform attention beliefs
    - The challenge of maintaining focus when distraction is absorbing
    """
    if params is None:
        params = ModelParams()

    rng = np.random.default_rng(seed)

    # Build models
    A1, B1 = build_breath_model(params)
    A2, B2, C2 = build_attention_model(params, include_awareness_modality=False)
    A2_true = build_A2_true(params)  # For generating precision observations
    B2_true = build_environment(params)  # True attention dynamics (absorbing distraction)
    env = BreathEnv(seed=seed)

    # Convert to pymdp object arrays for update_posterior_policies
    A2_obj = utils.obj_array(1)
    A2_obj[0] = A2
    B2_obj = utils.obj_array(1)
    B2_obj[0] = B2
    C2_obj = utils.obj_array(1)
    C2_obj[0] = C2

    # Define policies: STAY and SWITCH
    policies = [
        np.array([[STAY]]),    # Policy 0: Stay
        np.array([[SWITCH]]),  # Policy 1: Switch
    ]

    # Policy prior: favors STAY
    E = np.array([params.E_stay, 1 - params.E_stay])

    # Initialize beliefs
    qs_breath = np.array([0.5, 0.5])
    qs_attention = np.array([0.5, 0.5])  # Start uncertain about attention
    zeta = 1.0  # Start at prior mean

    # True states - start focused, will transition naturally
    true_attention = FOCUSED
    prev_action = STAY

    # Logs
    true_breath_states = np.zeros(T, dtype=int)
    true_attention_states = np.zeros(T, dtype=int)
    breath_posteriors = np.zeros((T, 2))
    attention_posteriors = np.zeros((T, 2))
    zeta_history = np.zeros(T)
    zeta_prior_history = np.zeros(T)
    actions = np.zeros(T, dtype=int)
    q_pi_history = np.zeros((T, 2))  # Policy posterior [P(STAY), P(SWITCH)]
    precision_obs_history = np.zeros(T, dtype=int)

    obs_breath = int(env.reset())

    for t in range(T):
        # Store true states
        true_breath_states[t] = env.state
        true_attention_states[t] = true_attention

        # === ATTENTION INFERENCE ===
        # Sample precision observation from TRUE attention state (probabilistic)
        obs_precision = rng.choice([PRECISE, IMPRECISE], p=A2_true[:, true_attention])
        precision_obs_history[t] = obs_precision

        # Attention prior using agent's B2 model and previous action
        prior_attention = B2[:, :, prev_action] @ qs_attention if t > 0 else qs_attention

        # Update attention beliefs with observation
        qs_attention = bayesian_update(A2, obs_precision, prior_attention)
        attention_posteriors[t] = qs_attention

        # === DESCENDING MESSAGE: Attention -> Precision Prior ===
        # Based on TRUE attention state (not agent's posterior)
        zeta_prior = params.zeta_focused if true_attention == FOCUSED else params.zeta_distracted
        zeta_prior_history[t] = zeta_prior

        # === DYNAMIC PRECISION UPDATE (B.45) ===
        prior_breath = B1 @ qs_breath if t > 0 else qs_breath

        zeta, pe, _ = update_likelihood_precision(
            zeta=zeta_prior,  # Descending message sets starting point
            A=A1,
            obs=obs_breath,
            qs=prior_breath,
            log_zeta_prior_var=params.zeta_prior_var,
            zeta_step=params.zeta_step,
            min_zeta=params.zeta_min,
            max_zeta=params.zeta_max
        )
        zeta_history[t] = zeta

        # === BREATH INFERENCE ===
        A1_scaled = scale_likelihood(A1, zeta)
        qs_breath = bayesian_update(A1_scaled, obs_breath, prior_breath)
        breath_posteriors[t] = qs_breath

        # === ACTION SELECTION (using pymdp) ===
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
            gamma=params.gamma,
        )

        q_pi_history[t] = q_pi.flatten()

        # Select action via argmax (deterministic)
        action = int(np.argmax(q_pi.flatten()))
        actions[t] = action

        # === ENVIRONMENT TRANSITION ===
        # True attention transitions based on action using B2_true
        p_transition = B2_true[:, true_attention, action]
        true_attention = int(rng.choice(2, p=p_transition))

        prev_action = action
        obs_breath = int(env.step(None))

    # Metrics
    inferred_attention = np.argmax(attention_posteriors, axis=1)
    attention_accuracy = (inferred_attention == true_attention_states).mean()
    time_distracted = (true_attention_states == DISTRACTED).mean()
    switch_rate = (actions == SWITCH).mean()

    return {
        "true_states": true_breath_states,
        "true_attention_states": true_attention_states,
        "posteriors": breath_posteriors,
        "attention_posteriors": attention_posteriors,
        "zeta_history": zeta_history,
        "zeta_prior_history": zeta_prior_history,
        "actions": actions,
        "q_pi_history": q_pi_history,
        "precision_obs_history": precision_obs_history,
        "attention_accuracy": attention_accuracy,
        "time_distracted": time_distracted,
        "switch_rate": switch_rate,
        "params": params,
        "T": T,
    }


def run_figure2_1_with_A2_precision(
    T: int = 200,
    seed: int = 42,
    params: ModelParams = None,
    A2_precision_override: float = None,
) -> Dict[str, Any]:
    """
    Run Figure 2.1 with configurable A2 precision.

    Same as run_figure2_1 but allows overriding the agent's A2 precision
    to test how meta-awareness precision affects ability to maintain focus.

    Parameters
    ----------
    A2_precision_override : float
        Override for A2_precision_obs. If None, uses params.A2_precision_obs.
        0.5 = no meta-awareness (flat)
        1.0 = perfect meta-awareness
    """
    if params is None:
        params = ModelParams()

    rng = np.random.default_rng(seed)

    # Build models
    A1, B1 = build_breath_model(params)
    A2_true = build_A2_true(params)
    B2_true = build_environment(params)
    env = BreathEnv(seed=seed)

    # Build agent's A2 with override precision
    A2_prec = A2_precision_override if A2_precision_override is not None else params.A2_precision_obs
    A2 = np.array([
        [A2_prec, 1 - A2_prec],
        [1 - A2_prec, A2_prec],
    ])

    # Agent's B2 beliefs
    p_stay = params.B2_stay_prob
    B2 = np.zeros((2, 2, 2))
    B2[:, :, STAY] = np.array([
        [p_stay, 1 - p_stay],
        [1 - p_stay, p_stay],
    ])
    p_switch = params.B2_switch_prob
    B2[:, :, SWITCH] = np.array([
        [1 - p_switch, p_switch],
        [p_switch, 1 - p_switch],
    ])

    C2 = np.array([params.C_precision_precise, params.C_precision_imprecise])

    # Convert to pymdp object arrays
    A2_obj = utils.obj_array(1)
    A2_obj[0] = A2
    B2_obj = utils.obj_array(1)
    B2_obj[0] = B2
    C2_obj = utils.obj_array(1)
    C2_obj[0] = C2

    # Policies and prior
    policies = [np.array([[STAY]]), np.array([[SWITCH]])]
    E = np.array([params.E_stay, 1 - params.E_stay])

    # Initialize
    qs_breath = np.array([0.5, 0.5])
    qs_attention = np.array([0.5, 0.5])
    zeta = 1.0
    true_attention = FOCUSED
    prev_action = STAY

    # Logs
    true_breath_states = np.zeros(T, dtype=int)
    true_attention_states = np.zeros(T, dtype=int)
    breath_posteriors = np.zeros((T, 2))
    attention_posteriors = np.zeros((T, 2))
    zeta_history = np.zeros(T)
    zeta_prior_history = np.zeros(T)
    actions = np.zeros(T, dtype=int)
    q_pi_history = np.zeros((T, 2))
    precision_obs_history = np.zeros(T, dtype=int)

    obs_breath = int(env.reset())

    for t in range(T):
        true_breath_states[t] = env.state
        true_attention_states[t] = true_attention

        # Sample precision observation from TRUE attention state
        obs_precision = rng.choice([PRECISE, IMPRECISE], p=A2_true[:, true_attention])
        precision_obs_history[t] = obs_precision

        # Attention inference
        prior_attention = B2[:, :, prev_action] @ qs_attention if t > 0 else qs_attention
        qs_attention = bayesian_update(A2, obs_precision, prior_attention)
        attention_posteriors[t] = qs_attention

        # Descending message: TRUE attention sets zeta prior
        zeta_prior = params.zeta_focused if true_attention == FOCUSED else params.zeta_distracted
        zeta_prior_history[t] = zeta_prior

        # Dynamic precision update
        prior_breath = B1 @ qs_breath if t > 0 else qs_breath
        zeta, pe, _ = update_likelihood_precision(
            zeta=zeta_prior,
            A=A1,
            obs=obs_breath,
            qs=prior_breath,
            log_zeta_prior_var=params.zeta_prior_var,
            zeta_step=params.zeta_step,
            min_zeta=params.zeta_min,
            max_zeta=params.zeta_max
        )
        zeta_history[t] = zeta

        # Breath inference
        A1_scaled = scale_likelihood(A1, zeta)
        qs_breath = bayesian_update(A1_scaled, obs_breath, prior_breath)
        breath_posteriors[t] = qs_breath

        # Action selection
        qs_attention_obj = utils.obj_array(1)
        qs_attention_obj[0] = qs_attention
        q_pi, G = update_posterior_policies(
            qs=qs_attention_obj, A=A2_obj, B=B2_obj, C=C2_obj,
            policies=policies, use_utility=True, use_states_info_gain=True,
            E=E, gamma=params.gamma,
        )
        q_pi_history[t] = q_pi.flatten()
        action = int(np.argmax(q_pi.flatten()))
        actions[t] = action

        # True attention transition
        p_transition = B2_true[:, true_attention, action]
        true_attention = int(rng.choice(2, p=p_transition))
        prev_action = action
        obs_breath = int(env.step(None))

    # Metrics
    inferred_attention = np.argmax(attention_posteriors, axis=1)
    attention_accuracy = (inferred_attention == true_attention_states).mean()
    time_distracted = (true_attention_states == DISTRACTED).mean()
    switch_rate = (actions == SWITCH).mean()

    return {
        "true_states": true_breath_states,
        "true_attention_states": true_attention_states,
        "posteriors": breath_posteriors,
        "attention_posteriors": attention_posteriors,
        "zeta_history": zeta_history,
        "zeta_prior_history": zeta_prior_history,
        "actions": actions,
        "q_pi_history": q_pi_history,
        "precision_obs_history": precision_obs_history,
        "attention_accuracy": attention_accuracy,
        "time_distracted": time_distracted,
        "switch_rate": switch_rate,
        "A2_precision": A2_prec,
        "params": params,
        "T": T,
    }


# =============================================================================
# Figure 3: Precision Dynamics Improve Learning
# =============================================================================

def compute_kl_divergence(P: np.ndarray, Q: np.ndarray) -> float:
    """Compute KL(P || Q) for two matrices, averaged over columns."""
    kl = 0.0
    for col in range(P.shape[1]):
        p = np.clip(P[:, col], EPS_VAL, 1.0)
        q = np.clip(Q[:, col], EPS_VAL, 1.0)
        kl += np.sum(p * np.log(p / q))
    return kl / P.shape[1]


def run_figure3(
    num_sits: int = 200,
    T_per_sit: int = 100,
    seed: int = 42,
    params: ModelParams = None,
    mode: str = "fixed",  # "fixed", "dynamic", "attention"
    A1_init_precision: float = 0.55,  # Agent's initial A1 belief
    initial_pA_strength: float = 2.0,  # Initial pseudo-count strength
) -> Dict[str, Any]:
    """
    Compare A1 learning across sits under different precision regimes.

    Modes:
    - "fixed": Fixed zeta=1.0, standard Dirichlet update
    - "dynamic": B.45 precision updates, precision-weighted learning
    - "attention": Hierarchical - zeta determines true attention, attention inference
                   sets zeta prior, B.45 updates from there

    Learning is batched at the end of each sit, with forgetting applied BEFORE update.
    Uses beliefs (posterior) for learning, not true states.
    """
    if params is None:
        params = ModelParams()

    rng = np.random.default_rng(seed)

    # Build true model (uses params.A1_precision)
    A1_true, B1 = build_breath_model(params)

    # For attention mode: build attention model
    if mode == "attention":
        A2, _, _ = build_attention_model(params, include_awareness_modality=False)
        A2_true = build_A2_true(params)

    # Initialize agent's A1 beliefs (starts at A1_init_precision)
    p = A1_init_precision
    pA1 = np.array([
        [p, 1 - p],
        [1 - p, p]
    ]) * initial_pA_strength

    # Logs per sit
    A1_kl = np.zeros(num_sits)
    A1_error = np.zeros(num_sits)
    A1_diagonal = np.zeros(num_sits)  # Mean of diagonal elements

    for sit in range(num_sits):
        # New environment for each sit (use same p_correct as A1_true)
        env = BreathEnv(seed=seed + sit, p_correct=params.A1_precision)
        qs_breath = np.array([0.5, 0.5])
        qs_attention = np.array([0.5, 0.5])  # For attention mode
        zeta = 1.0
        obs = int(env.reset())

        # Collect observations and beliefs during sit
        observations = []
        beliefs = []

        for t in range(T_per_sit):
            # Get current A1 estimate
            A1_est = pA1 / pA1.sum(axis=0, keepdims=True)

            # Prior from transition
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
                # 1. Current zeta determines true attention state
                true_attention = FOCUSED if zeta > 1.0 else DISTRACTED

                # 2. True attention generates observation probabilistically (ascending)
                obs_precision = rng.choice([PRECISE, IMPRECISE], p=A2_true[:, true_attention])

                # 3. Update attention beliefs from ascending observation
                p_stay = 0.9
                B_att = np.array([[p_stay, 1 - p_stay], [1 - p_stay, p_stay]])
                prior_attention = B_att @ qs_attention if t > 0 else qs_attention
                qs_attention = bayesian_update(A2, obs_precision, prior_attention)

                # 4. Attention beliefs set zeta prior (descending message)
                zeta_prior = (qs_attention[FOCUSED] * params.zeta_focused +
                              qs_attention[DISTRACTED] * params.zeta_distracted)

                # 5. B.45 updates zeta from prior
                zeta, pe, _ = update_likelihood_precision(
                    zeta=zeta_prior,  # Descending message sets starting point
                    A=A1_est, obs=obs, qs=prior,
                    log_zeta_prior_mean=0.0,  # Pull towards 1
                    log_zeta_prior_var=params.zeta_prior_var,
                    zeta_step=params.zeta_step,
                    min_zeta=params.zeta_min,
                    max_zeta=params.zeta_max
                )

            # State inference with precision-scaled likelihood
            A1_scaled = scale_likelihood(A1_est, zeta)
            qs_breath = bayesian_update(A1_scaled, obs, prior)

            # Store for batch learning
            observations.append(obs)
            beliefs.append(qs_breath.copy())

            obs = int(env.step(None))

        # End of sit: apply forgetting FIRST
        pA1 = pA1 * params.forgetting_rate

        # Then batch learning update using beliefs (not true states)
        for obs_t, qs_t in zip(observations, beliefs):
            obs_onehot = np.zeros(2)
            obs_onehot[obs_t] = 1.0
            # Outer product: obs ⊗ qs (as in original)
            dfda = np.outer(obs_onehot, qs_t)
            lr = params.A_learning_rate * (zeta if mode != "fixed" else 1.0)
            pA1 = pA1 + lr * dfda

        # Record metrics at end of sit
        A1_est = pA1 / pA1.sum(axis=0, keepdims=True)
        A1_kl[sit] = compute_kl_divergence(A1_true, A1_est)
        A1_error[sit] = np.mean(np.abs(A1_est - A1_true))
        A1_diagonal[sit] = (A1_est[0, 0] + A1_est[1, 1]) / 2  # Mean diagonal

    # True A1 diagonal
    A1_true_diagonal = (A1_true[0, 0] + A1_true[1, 1]) / 2

    return {
        "A1_true": A1_true,
        "A1_true_diagonal": A1_true_diagonal,
        "A1_kl": A1_kl,
        "A1_error": A1_error,
        "A1_diagonal": A1_diagonal,
        "mode": mode,
        "params": params,
        "num_sits": num_sits,
        "T_per_sit": T_per_sit,
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

    Key: Observations generated from A2_true (0.9 precision),
         but agent infers using weak A2 (0.52 precision).
    """
    if params is None:
        params = ModelParams()

    rng = np.random.default_rng(seed)

    # Build models
    A1, B1 = build_breath_model(params)
    A2, B2, C2 = build_attention_model(params, include_awareness_modality=False)
    A2_true = build_A2_true(params)  # For generating observations
    A3, B3 = build_awareness_model(params)
    B2_true = build_environment(params)

    # Convert to pymdp object arrays for update_posterior_policies
    A2_obj = utils.obj_array(1)
    A2_obj[0] = A2
    B2_obj = utils.obj_array(1)
    B2_obj[0] = B2
    C2_obj = utils.obj_array(1)
    C2_obj[0] = C2

    # Define policies: STAY and SWITCH
    policies = [
        np.array([[STAY]]),    # Policy 0: Stay
        np.array([[SWITCH]]),  # Policy 1: Switch
    ]

    # Policy prior: favors STAY
    E = np.array([params.E_stay, 1 - params.E_stay])

    env = BreathEnv(seed=seed)

    # Initialize beliefs
    qs_breath = np.array([0.5, 0.5])
    qs_attention = np.array([0.5, 0.5])  # Start uncertain
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
    precision_obs_history = np.zeros(T_logs, dtype=int)

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
        obs_entropy = OBS_AWARE if entropy < params.entropy_threshold else OBS_UNAWARE

        prior_awareness = B3 @ qs_awareness if t > 0 else qs_awareness
        qs_awareness = bayesian_update(A3, obs_entropy, prior_awareness)
        posterior_awareness[t] = qs_awareness

        # === ATTENTION INFERENCE ===
        # Sample precision observation from TRUE A2 (probabilistic)
        obs_precision = rng.choice([PRECISE, IMPRECISE], p=A2_true[:, true_attention])
        precision_obs_history[t] = obs_precision

        prior_attention = B2[:, :, prev_action] @ qs_attention if t > 0 else qs_attention
        qs_attention = bayesian_update(A2, obs_precision, prior_attention)
        posterior_attention[t] = qs_attention

        # === ACTION SELECTION (using pymdp) ===
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
            gamma=params.gamma,
        )

        # Sample action from policy posterior
        action = int(rng.choice(len(q_pi), p=q_pi.flatten()))
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
    3. Knowledge that SWITCH action toggles attention state

    Demonstrates escape from the attention trap:
    - Agent can infer attention via awareness observations
    - Selects SWITCH action when distracted
    - Returns to focused state

    Key: Uses same A2_true as figure 4 for observation generation.
    """
    if params is None:
        params = ModelParams()

    rng = np.random.default_rng(seed)

    # Build models - WITH meditation instruction
    A1, B1 = build_breath_model(params)
    A2, B2, C2 = build_attention_model(params, include_awareness_modality=True)
    A2_true = build_A2_true(params)  # For generating observations
    A3, B3 = build_awareness_model(params)
    B2_true = build_environment(params)

    # Convert to pymdp object arrays for update_posterior_policies
    # A2 is already an object array (2 modalities) from build_attention_model
    B2_obj = utils.obj_array(1)
    B2_obj[0] = B2
    # C2 is already an object array (2 modalities)

    # Define policies: STAY and SWITCH
    policies = [
        np.array([[STAY]]),    # Policy 0: Stay
        np.array([[SWITCH]]),  # Policy 1: Switch
    ]

    # Policy prior: favors STAY
    E = np.array([params.E_stay, 1 - params.E_stay])

    env = BreathEnv(seed=seed)

    # Initialize beliefs
    qs_breath = np.array([0.5, 0.5])
    qs_attention = np.array([0.5, 0.5])  # Start uncertain
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
    precision_obs_history = np.zeros(T, dtype=int)

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
        entropy = compute_entropy(qs_breath)
        obs_entropy = OBS_AWARE if entropy < params.entropy_threshold else OBS_UNAWARE

        prior_awareness = B3 @ qs_awareness if t > 0 else qs_awareness
        qs_awareness = bayesian_update(A3, obs_entropy, prior_awareness)
        posterior_awareness[t] = qs_awareness

        # === ATTENTION INFERENCE (with meditation instruction) ===
        # Sample precision observation from TRUE A2 (probabilistic)
        obs_precision = rng.choice([PRECISE, IMPRECISE], p=A2_true[:, true_attention])
        precision_obs_history[t] = obs_precision

        prior_attention = B2[:, :, prev_action] @ qs_attention if t > 0 else qs_attention

        # Combine both modalities for inference
        likelihood_precision = A2[0][obs_precision, :]
        likelihood_awareness = A2[1][obs_entropy, :]
        likelihood_combined = likelihood_precision * likelihood_awareness

        qs_attention = likelihood_combined * prior_attention
        qs_attention = qs_attention / (qs_attention.sum() + EPS_VAL)
        posterior_attention[t] = qs_attention

        # === ACTION SELECTION (using pymdp) ===
        qs_attention_obj = utils.obj_array(1)
        qs_attention_obj[0] = qs_attention

        q_pi, G = update_posterior_policies(
            qs=qs_attention_obj,
            A=A2,  # Already object array with 2 modalities
            B=B2_obj,
            C=C2,  # Already object array with 2 modalities
            policies=policies,
            use_utility=True,
            use_states_info_gain=True,
            E=E,
            gamma=params.gamma,
        )

        # Sample action from policy posterior
        action = int(rng.choice(len(q_pi), p=q_pi.flatten()))
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

    Key: Observations generated from A2_true (0.9 precision).
    When meditation starts, agent learns SWITCH toggles attention.
    """
    if params is None:
        params = ModelParams()

    rng = np.random.default_rng(seed)

    # Build models
    A1, B1 = build_breath_model(params)
    A2_true = build_A2_true(params)  # For generating observations
    A3, B3 = build_awareness_model(params)
    B2_true = build_environment(params)

    # Policy prior: favors STAY
    E = np.array([params.E_stay, 1 - params.E_stay])

    # Define policies: STAY and SWITCH
    policies = [
        np.array([[STAY]]),    # Policy 0: Stay
        np.array([[SWITCH]]),  # Policy 1: Switch
    ]

    # Initialize learnable parameters (Dirichlet concentrations)
    pA2 = initialize_A2_for_learning(params)
    pB2 = initialize_B2_for_learning(params)

    # Track whether meditation has started (to update B2 SWITCH once)
    meditation_started = False

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

        # When meditation starts, agent learns that SWITCH toggles attention
        if has_instruction and not meditation_started:
            meditation_started = True
            # Update pB2 SWITCH to reflect toggle knowledge (high confidence)
            pB2[:, :, SWITCH] = np.array([
                [0.1, 0.9],
                [0.9, 0.1],
            ]) * 10.0  # High confidence from instruction

        # Build A2/C2 based on instruction status
        if has_instruction:
            A2, B2, C2 = build_attention_model(params, include_awareness_modality=True)
        else:
            A2, B2, C2 = build_attention_model(params, include_awareness_modality=False)

        # Normalize learned parameters to get current A2/B2
        A2_learned = pA2 / pA2.sum(axis=0, keepdims=True)
        B2_learned = pB2 / pB2.sum(axis=0, keepdims=True)

        # Initialize for this sit
        env = BreathEnv(seed=seed + sit)
        qs_breath = np.array([0.5, 0.5])
        qs_attention = np.array([0.5, 0.5])  # Start uncertain
        qs_awareness = np.array([0.5, 0.5])

        true_attention = FOCUSED
        true_awareness = AWARE
        prev_attention = FOCUSED  # Track for B2 learning
        prev_action = STAY

        # Sit logs
        true_attention_sit = np.zeros(T_per_sit, dtype=int)
        actions_sit = np.zeros(T_per_sit, dtype=int)
        inferred_attention_sit = np.zeros(T_per_sit, dtype=int)

        obs_breath = int(env.reset())

        for t in range(T_per_sit):
            true_attention_sit[t] = true_attention

            # Precision based on true attention
            zeta = params.zeta_focused if true_attention == FOCUSED else params.zeta_distracted

            # Breath inference
            prior_breath = B1 @ qs_breath if t > 0 else qs_breath
            A1_scaled = scale_likelihood(A1, zeta)
            qs_breath = bayesian_update(A1_scaled, obs_breath, prior_breath)

            # Awareness inference
            entropy = compute_entropy(qs_breath)
            obs_entropy = OBS_AWARE if entropy < params.entropy_threshold else OBS_UNAWARE
            prior_awareness = B3 @ qs_awareness if t > 0 else qs_awareness
            qs_awareness = bayesian_update(A3, obs_entropy, prior_awareness)

            # Sample precision observation from TRUE A2 (probabilistic)
            obs_precision = rng.choice([PRECISE, IMPRECISE], p=A2_true[:, true_attention])

            # Attention inference
            prior_attention = B2_learned[:, :, prev_action] @ qs_attention if t > 0 else qs_attention

            if has_instruction:
                # Two modalities for meditator
                likelihood_precision = A2[0][obs_precision, :]
                likelihood_awareness = A2[1][obs_entropy, :]
                likelihood_combined = likelihood_precision * likelihood_awareness
                qs_attention = likelihood_combined * prior_attention
            else:
                # Single modality using learned A2
                qs_attention = A2_learned[obs_precision, :] * prior_attention

            qs_attention = qs_attention / (qs_attention.sum() + EPS_VAL)
            inferred_attention_sit[t] = np.argmax(qs_attention)

            # Action selection (using pymdp)
            qs_attention_obj = utils.obj_array(1)
            qs_attention_obj[0] = qs_attention

            # Prepare B2 as object array
            B2_learned_obj = utils.obj_array(1)
            B2_learned_obj[0] = B2_learned

            if has_instruction:
                # A2 and C2 are already object arrays (2 modalities)
                q_pi, G = update_posterior_policies(
                    qs=qs_attention_obj,
                    A=A2,
                    B=B2_learned_obj,
                    C=C2,
                    policies=policies,
                    use_utility=True,
                    use_states_info_gain=True,
                    E=E,
                    gamma=params.gamma,
                )
            else:
                # Single modality - wrap in object arrays
                A2_learned_obj = utils.obj_array(1)
                A2_learned_obj[0] = A2_learned
                C2_obj = utils.obj_array(1)
                C2_obj[0] = C2

                q_pi, G = update_posterior_policies(
                    qs=qs_attention_obj,
                    A=A2_learned_obj,
                    B=B2_learned_obj,
                    C=C2_obj,
                    policies=policies,
                    use_utility=True,
                    use_states_info_gain=True,
                    E=E,
                    gamma=params.gamma,
                )

            # Sample action from policy posterior
            action = int(rng.choice(len(q_pi), p=q_pi.flatten()))
            actions_sit[t] = action

            # Environment transition
            p_transition = B2_true[:, true_attention, action]
            new_attention = rng.choice(2, p=p_transition)

            # Learning updates
            if t > 0:
                # A2 learning: obs_precision -> true attention state
                lr_A = params.A_learning_rate
                pA2[obs_precision, prev_attention] += lr_A

                # B2 learning: (prev_attention, prev_action) -> new attention
                lr_B = params.B_learning_rate
                pB2[true_attention, prev_attention, prev_action] += lr_B

            # Forgetting (decay toward uniform)
            pA2 *= params.forgetting_rate
            pA2 += (1 - params.forgetting_rate) * 0.5
            pB2 *= params.forgetting_rate
            pB2 += (1 - params.forgetting_rate) * 0.5

            prev_attention = true_attention
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


# =============================================================================
# Figure 4: The Distraction Trap - Learning Sweep
# =============================================================================

from pymdp.learning import update_obs_likelihood_dirichlet, update_state_likelihood_dirichlet


def scale_B_matrix(B, omega):
    """
    Scale a B (transition) matrix by precision parameter omega.

    Uses the same approach as scale_likelihood: B^omega then renormalize.
    - omega=0: uniform (flat, no knowledge of dynamics)
    - omega=1: original B matrix (true dynamics)

    Parameters
    ----------
    B : np.ndarray, shape (n_states, n_states, n_actions)
        Transition matrix
    omega : float
        Precision parameter in [0, 1]

    Returns
    -------
    B_scaled : np.ndarray
        Scaled and renormalized transition matrix
    """
    B_scaled = np.zeros_like(B)
    for a in range(B.shape[2]):
        # Scale each action's transition matrix like a likelihood
        B_action = B[:, :, a]
        B_action_scaled = B_action ** omega
        # Renormalize columns
        col_sums = B_action_scaled.sum(axis=0, keepdims=True)
        B_scaled[:, :, a] = B_action_scaled / (col_sums + EPS_VAL)
    return B_scaled


def run_single_learning_trajectory(
    zeta_A2: float,
    omega_B2: float,
    num_sits: int,
    T_per_sit: int,
    seed: int,
    params: ModelParams,
) -> Dict[str, Any]:
    """
    Run a single learning trajectory with full Figure 2.1 hierarchical stack.

    Each sit runs the full hierarchical inference:
    - Level 1: Breath perception with dynamic precision
    - Level 2: Attention inference from precision observations
    - Action selection based on attention beliefs
    - True attention transitions

    At end of each sit, A2 and B2 are updated via Dirichlet learning.

    Initial A2/B2 are precision-scaled versions of true models:
    - A2_agent = scale_likelihood(A2_true, zeta_A2)
    - B2_agent = scale_B_matrix(B2_true, omega_B2)

    Where:
    - zeta/omega = 0: completely flat (no knowledge)
    - zeta/omega = 1: matches true model exactly

    Returns average % time focused across all sits.
    """
    rng = np.random.default_rng(seed)

    # === BUILD TRUE MODELS ===
    A1, B1 = build_breath_model(params)  # Breath model
    A2_true = build_A2_true(params)  # True precision observation model
    B2_true = build_environment(params)  # True attention dynamics

    # === INITIALIZE AGENT'S MODELS ===
    # A2: precision-scaled from true
    A2_init = scale_likelihood(A2_true, zeta_A2)

    # B2: precision-scaled from true
    B2_init = scale_B_matrix(B2_true, omega_B2)

    # Dirichlet parameters for learning
    initial_strength = 2.0
    pA2_obj = utils.obj_array(1)
    pA2_obj[0] = A2_init * initial_strength + EPS_VAL

    pB2_obj = utils.obj_array(1)
    pB2_obj[0] = B2_init * initial_strength + EPS_VAL

    # Current beliefs (normalized)
    A2 = A2_init.copy()
    B2 = B2_init.copy()

    # Preferences
    C2 = np.array([params.C_precision_precise, params.C_precision_imprecise])

    # Wrap for pymdp
    A2_obj = utils.obj_array(1)
    A2_obj[0] = A2
    B2_obj = utils.obj_array(1)
    B2_obj[0] = B2
    C2_obj = utils.obj_array(1)
    C2_obj[0] = C2

    # Policy setup
    E = np.array([params.E_stay, 1 - params.E_stay])
    policies = [np.array([[STAY]]), np.array([[SWITCH]])]

    # Logs
    time_focused_per_sit = np.zeros(num_sits)
    A2_diagonal_per_sit = np.zeros(num_sits)
    B2_diagonal_per_sit = np.zeros(num_sits)

    for sit in range(num_sits):
        # === FORGETTING: decay toward uniform ===
        pA2_obj[0] = params.forgetting_rate * pA2_obj[0] + (1 - params.forgetting_rate) * 0.5
        pB2_obj[0] = params.forgetting_rate * pB2_obj[0] + (1 - params.forgetting_rate) * 0.5

        # Normalize to get current beliefs
        A2 = pA2_obj[0] / (pA2_obj[0].sum(axis=0, keepdims=True) + EPS_VAL)
        for a in range(2):
            for s in range(2):
                col_sum = pB2_obj[0][:, s, a].sum()
                B2[:, s, a] = pB2_obj[0][:, s, a] / col_sum if col_sum > EPS_VAL else 0.5
        A2_obj[0] = A2
        B2_obj[0] = B2

        # Track learning progress
        A2_diagonal_per_sit[sit] = (A2[0, 0] + A2[1, 1]) / 2
        B2_diagonal_per_sit[sit] = (B2[FOCUSED, FOCUSED, STAY] + B2[DISTRACTED, DISTRACTED, STAY] +
                                    B2[FOCUSED, DISTRACTED, SWITCH] + B2[DISTRACTED, FOCUSED, SWITCH]) / 4

        # === INITIALIZE FOR THIS SIT ===
        sit_seed = rng.integers(0, 2**31)
        sit_rng = np.random.default_rng(sit_seed)
        env = BreathEnv(seed=sit_seed)

        # Initialize beliefs
        qs_breath = np.array([0.5, 0.5])
        qs_attention = np.array([0.5, 0.5])
        zeta = 1.0  # Start at prior mean

        # True states - 50/50 chance of starting focused or distracted
        true_attention = sit_rng.choice([FOCUSED, DISTRACTED])
        prev_action = STAY
        time_focused = 0

        # Storage for batch learning
        precision_obs_list = []
        posterior_attention_list = []
        posterior_attention_prev_list = []
        action_list = []

        obs_breath = int(env.reset())

        # === WITHIN-SIT LOOP (Full Figure 2.1 stack) ===
        for t in range(T_per_sit):
            if true_attention == FOCUSED:
                time_focused += 1

            # Store previous posterior for B2 learning
            qs_attention_prev = qs_attention.copy()
            if t > 0:
                posterior_attention_prev_list.append(qs_attention_prev)

            # --- ASCENDING: Precision observation from TRUE attention ---
            obs_precision = sit_rng.choice([PRECISE, IMPRECISE], p=A2_true[:, true_attention])
            precision_obs_list.append(obs_precision)

            # --- ATTENTION INFERENCE using agent's learned A2 and B2 ---
            prior_attention = B2[:, :, prev_action] @ qs_attention if t > 0 else qs_attention
            qs_attention = bayesian_update(A2, obs_precision, prior_attention)
            posterior_attention_list.append(qs_attention.copy())

            # --- DESCENDING: TRUE attention sets zeta prior ---
            zeta_prior = params.zeta_focused if true_attention == FOCUSED else params.zeta_distracted

            # --- DYNAMIC PRECISION UPDATE (B.45) ---
            prior_breath = B1 @ qs_breath if t > 0 else qs_breath
            zeta, pe, _ = update_likelihood_precision(
                zeta=zeta_prior,  # Descending message
                A=A1,
                obs=obs_breath,
                qs=prior_breath,
                log_zeta_prior_var=params.zeta_prior_var,
                zeta_step=params.zeta_step,
                min_zeta=params.zeta_min,
                max_zeta=params.zeta_max
            )

            # --- BREATH INFERENCE with precision-scaled A1 ---
            A1_scaled = scale_likelihood(A1, zeta)
            qs_breath = bayesian_update(A1_scaled, obs_breath, prior_breath)

            # --- ACTION SELECTION ---
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
                gamma=params.gamma,
            )

            action = int(np.argmax(q_pi.flatten()))
            action_list.append(action)
            prev_action = action

            # --- TRUE ATTENTION TRANSITION ---
            p_transition = B2_true[:, true_attention, action]
            true_attention = sit_rng.choice(2, p=p_transition)

            # --- BREATH STEP ---
            obs_breath = int(env.step(None))

        time_focused_per_sit[sit] = time_focused / T_per_sit

        # === BATCH LEARNING at end of sit ===
        # A2 learning: (obs, inferred posterior) pairs
        for t_idx in range(len(precision_obs_list)):
            obs = precision_obs_list[t_idx]
            qs_obj = utils.obj_array(1)
            qs_obj[0] = posterior_attention_list[t_idx]
            pA2_obj = update_obs_likelihood_dirichlet(
                pA=pA2_obj, A=A2_obj, obs=obs, qs=qs_obj,
                lr=params.A_learning_rate, modalities="all"
            )

        # B2 learning: (qs_prev, qs_curr, action) tuples
        for t_idx in range(len(posterior_attention_prev_list)):
            qs_prev_obj = utils.obj_array(1)
            qs_prev_obj[0] = posterior_attention_prev_list[t_idx]
            qs_curr_obj = utils.obj_array(1)
            qs_curr_obj[0] = posterior_attention_list[t_idx + 1]
            action_t = action_list[t_idx]

            pB2_obj = update_state_likelihood_dirichlet(
                pB=pB2_obj, B=B2_obj,
                actions=np.array([action_t]),
                qs=qs_curr_obj, qs_prev=qs_prev_obj,
                lr=params.B_learning_rate, factors="all"
            )

    # Final outcome: average across ALL sits
    final_time_focused = time_focused_per_sit.mean()

    return {
        "final_time_focused": final_time_focused,
        "time_focused_per_sit": time_focused_per_sit,
        "A2_diagonal_per_sit": A2_diagonal_per_sit,
        "B2_diagonal_per_sit": B2_diagonal_per_sit,
    }


def run_figure4_diagnostic(
    zeta_A2: float,
    omega_B2: float,
    num_sits: int,
    T_per_sit: int,
    seed: int,
    params: ModelParams,
    capture_sits: list = None,
) -> Dict[str, Any]:
    """
    Run a single learning trajectory with detailed within-sit dynamics capture.

    Similar to run_single_learning_trajectory but captures timestep-by-timestep
    dynamics for specified sits (for debugging/visualization).

    Parameters
    ----------
    capture_sits : list
        List of sit indices to capture detailed dynamics for (e.g., [0, 49, 99])
    """
    if capture_sits is None:
        capture_sits = [0, num_sits // 2, num_sits - 1]

    rng = np.random.default_rng(seed)

    # Build TRUE models
    A2_true = build_A2_true(params)
    B2_true = build_environment(params)

    # Policy prior and policies
    E = np.array([params.E_stay, 1 - params.E_stay])
    policies = [np.array([[STAY]]), np.array([[SWITCH]])]

    # Initialize agent's A2/B2 by scaling true models
    A2_init = scale_likelihood(A2_true, zeta_A2)
    B2_init = scale_B_matrix(B2_true, omega_B2)

    # Initialize Dirichlet parameters
    initial_strength = 2.0
    pA2_obj = utils.obj_array(1)
    pA2_obj[0] = A2_init * initial_strength + EPS_VAL

    pB2_obj = utils.obj_array(1)
    pB2_obj[0] = B2_init * initial_strength + EPS_VAL

    # Current beliefs
    A2 = A2_init.copy()
    B2 = B2_init.copy()

    # Preferences
    C2 = np.array([params.C_precision_precise, params.C_precision_imprecise])

    # Wrap for pymdp
    A2_obj = utils.obj_array(1)
    A2_obj[0] = A2
    B2_obj = utils.obj_array(1)
    B2_obj[0] = B2
    C2_obj = utils.obj_array(1)
    C2_obj[0] = C2

    # Storage for captured sits
    captured_dynamics = {}

    for sit in range(num_sits):
        # Apply forgetting
        pA2_obj[0] = params.forgetting_rate * pA2_obj[0] + (1 - params.forgetting_rate) * 0.5
        pB2_obj[0] = params.forgetting_rate * pB2_obj[0] + (1 - params.forgetting_rate) * 0.5

        # Normalize
        A2 = pA2_obj[0] / (pA2_obj[0].sum(axis=0, keepdims=True) + EPS_VAL)
        for a in range(2):
            for s in range(2):
                col_sum = pB2_obj[0][:, s, a].sum()
                B2[:, s, a] = pB2_obj[0][:, s, a] / col_sum if col_sum > EPS_VAL else 0.5
        A2_obj[0] = A2
        B2_obj[0] = B2

        # Initialize for this sit with fresh RNG
        sit_seed = rng.integers(0, 2**31)
        sit_rng = np.random.default_rng(sit_seed)
        env = BreathEnv(seed=sit_seed)
        qs_attention = np.array([0.5, 0.5])
        true_attention = FOCUSED
        prev_action = STAY

        # Per-timestep storage if capturing this sit
        capturing = sit in capture_sits
        if capturing:
            sit_data = {
                "true_attention": [],
                "qs_focused": [],
                "obs_precision": [],
                "action": [],
                "A2_diagonal": (A2[0, 0] + A2[1, 1]) / 2,
                "B2_stay_diag": (B2[FOCUSED, FOCUSED, STAY] + B2[DISTRACTED, DISTRACTED, STAY]) / 2,
            }

        # Storage for batch learning
        precision_obs_list = []
        posterior_attention_list = []
        posterior_attention_prev_list = []
        action_list = []

        obs_breath = int(env.reset())

        for t in range(T_per_sit):
            qs_attention_prev = qs_attention.copy()
            if t > 0:
                posterior_attention_prev_list.append(qs_attention_prev)

            # Generate precision observation from TRUE attention state
            obs_precision = sit_rng.choice([PRECISE, IMPRECISE], p=A2_true[:, true_attention])
            precision_obs_list.append(obs_precision)

            # Attention inference
            prior_attention = B2[:, :, prev_action] @ qs_attention if t > 0 else qs_attention
            likelihood = A2[obs_precision, :]
            qs_attention = likelihood * prior_attention
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
                gamma=params.gamma,
            )

            action = int(np.argmax(q_pi.flatten()))
            action_list.append(action)
            prev_action = action

            # Capture if needed
            if capturing:
                sit_data["true_attention"].append(true_attention)
                sit_data["qs_focused"].append(qs_attention[FOCUSED])
                sit_data["obs_precision"].append(obs_precision)
                sit_data["action"].append(action)

            # TRUE attention transition
            p_transition = B2_true[:, true_attention, action]
            true_attention = sit_rng.choice(2, p=p_transition)

            obs_breath = int(env.step(None))

        if capturing:
            sit_data["true_attention"] = np.array(sit_data["true_attention"])
            sit_data["qs_focused"] = np.array(sit_data["qs_focused"])
            sit_data["obs_precision"] = np.array(sit_data["obs_precision"])
            sit_data["action"] = np.array(sit_data["action"])
            captured_dynamics[sit] = sit_data

        # Batch learning
        for t_idx in range(len(precision_obs_list)):
            obs = precision_obs_list[t_idx]
            qs_obj = utils.obj_array(1)
            qs_obj[0] = posterior_attention_list[t_idx]
            pA2_obj = update_obs_likelihood_dirichlet(
                pA=pA2_obj, A=A2_obj, obs=obs, qs=qs_obj,
                lr=params.A_learning_rate, modalities="all"
            )

        for t_idx in range(len(posterior_attention_prev_list)):
            qs_prev_obj = utils.obj_array(1)
            qs_prev_obj[0] = posterior_attention_prev_list[t_idx]
            qs_curr_obj = utils.obj_array(1)
            qs_curr_obj[0] = posterior_attention_list[t_idx + 1]
            action_t = action_list[t_idx]

            pB2_obj = update_state_likelihood_dirichlet(
                pB=pB2_obj, B=B2_obj,
                actions=np.array([action_t]),
                qs=qs_curr_obj, qs_prev=qs_prev_obj,
                lr=params.B_learning_rate, factors="all"
            )

    return {
        "captured_dynamics": captured_dynamics,
        "capture_sits": capture_sits,
        "zeta_A2": zeta_A2,
        "omega_B2": omega_B2,
    }


def run_figure4_learning_sweep(
    zeta_range: np.ndarray = None,
    omega_range: np.ndarray = None,
    num_sits: int = 100,
    T_per_sit: int = 50,
    seed: int = 42,
    params: ModelParams = None,
    n_seeds: int = 1,
) -> Dict[str, Any]:
    """
    Run a parameter sweep over initial A2 (zeta) and B2 (omega) precision.

    Demonstrates the "distraction trap": without sufficient initial structure
    in A2 (likelihood) or B2 (transition), the agent cannot learn their way
    out of distraction.

    Uses scale_likelihood to create agent's initial beliefs:
    - A2_agent = scale_likelihood(A2_true, zeta)
    - B2_agent = scale_B_matrix(B2_true, omega)

    Parameters
    ----------
    zeta_range : array-like
        A2 precision values to sweep (0 = flat, 1 = true model)
    omega_range : array-like
        B2 precision values to sweep (0 = flat, 1 = true model)
    num_sits : int
        Number of meditation sits per simulation
    T_per_sit : int
        Timesteps per sit
    seed : int
        Random seed
    params : ModelParams
        Model parameters
    n_seeds : int
        Number of random seeds to average over per parameterization

    Returns
    -------
    results : dict
        Contains 2D arrays for heatmap plotting
    """
    if params is None:
        params = ModelParams()

    if zeta_range is None:
        zeta_range = np.linspace(0.0, 1.0, 11)  # 0 = flat, 1 = true
    if omega_range is None:
        omega_range = np.linspace(0.0, 1.0, 11)

    n_zeta = len(zeta_range)
    n_omega = len(omega_range)

    # Output arrays
    final_time_focused = np.zeros((n_zeta, n_omega))

    print(f"Running {n_zeta * n_omega} simulations (averaging over {n_seeds} seeds)...")

    for i, zeta in enumerate(zeta_range):
        for j, omega in enumerate(omega_range):
            # Average over multiple seeds
            results_seeds = []
            for s in range(n_seeds):
                result = run_single_learning_trajectory(
                    zeta_A2=zeta,
                    omega_B2=omega,
                    num_sits=num_sits,
                    T_per_sit=T_per_sit,
                    seed=seed + i * 1000 + j * 10 + s,
                    params=params,
                )
                results_seeds.append(result["final_time_focused"])
            final_time_focused[i, j] = np.mean(results_seeds)

        print(f"  Completed ζ={zeta:.2f} ({i+1}/{n_zeta})")

    return {
        "final_time_focused": final_time_focused,
        "zeta_range": zeta_range,
        "omega_range": omega_range,
        "num_sits": num_sits,
        "T_per_sit": T_per_sit,
        "params": params,
    }
