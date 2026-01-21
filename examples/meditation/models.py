#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generative model builders for focused attention meditation simulation.

This module contains:
- State/observation constants
- Model parameter defaults
- Functions to build A, B, C matrices for each level of the hierarchy

Hierarchy:
    Level 1: Breath perception (interoceptive inference)
    Level 2: Attention state (focused/distracted) with action selection
    Level 3: Awareness (meta-cognitive monitoring of attention)

References:
    Parr, T., Pezzulo, G., & Friston, K. J. (2022). Active Inference.
    Appendix B, Equation B.45 for precision updating.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple

from pymdp import utils


# =============================================================================
# Constants
# =============================================================================

# Breath states (Level 1)
INHALE = 0
EXHALE = 1
BREATH_STATES = ["INHALE", "EXHALE"]

# Breath observations
EXPANSION = 0
CONTRACTION = 1
BREATH_OBS = ["EXPANSION", "CONTRACTION"]

# Attention states (Level 2)
FOCUSED = 0
DISTRACTED = 1
ATTENTION_STATES = ["FOCUSED", "DISTRACTED"]

# Attention actions
STAY = 0
SWITCH = 1
ATTENTION_ACTIONS = ["STAY", "SWITCH"]

# Precision observations (Level 2 observation modality 0)
PRECISE = 0
IMPRECISE = 1
PRECISION_OBS = ["PRECISE", "IMPRECISE"]

# Awareness states (Level 3)
AWARE = 0
UNAWARE = 1
AWARENESS_STATES = ["AWARE", "UNAWARE"]

# Awareness observations (Level 2 observation modality 1, Level 3 observation)
OBS_AWARE = 0
OBS_UNAWARE = 1
AWARENESS_OBS = ["AWARE", "UNAWARE"]


# =============================================================================
# Default Parameters
# =============================================================================

@dataclass
class ModelParams:
    """Default parameters for the meditation model."""

    # Breath model (Level 1)
    A1_precision: float = 0.75  # P(correct obs | state)
    inhale_duration: float = 4.0  # Mean timesteps in inhale
    exhale_duration: float = 6.0  # Mean timesteps in exhale

    # Attention model (Level 2)
    A2_precision_obs: float = 0.52  # Weak: precision obs -> attention
    A2_awareness_obs: float = 0.75  # Meditation instruction: awareness obs -> attention

    # Awareness model (Level 3)
    A3_precision: float = 0.75  # P(correct awareness obs | awareness state)

    # Precision dynamics
    zeta_focused: float = 2.0  # Precision when focused
    zeta_distracted: float = 0.5  # Precision when distracted
    zeta_step: float = 0.25  # Learning rate for precision updates
    zeta_prior_var: float = 1.0  # Prior variance on log-precision
    zeta_min: float = 0.1
    zeta_max: float = 5.0

    # Environment (B2_true)
    p_stay_focused: float = 0.8  # P(focused -> focused | STAY)
    p_stay_distracted: float = 1.0  # P(distracted -> distracted | STAY) - absorbing!
    p_switch_success: float = 0.9  # P(state flips | SWITCH)

    # Agent's B2 beliefs (may differ from truth)
    B2_stay_prob: float = 0.9  # Agent's belief about STAY
    B2_switch_prob: float = 0.9  # Agent's belief about SWITCH toggling

    # Awareness dynamics (B3)
    p_aware_to_aware: float = 0.7
    p_unaware_to_unaware: float = 0.9

    # Policy selection
    gamma: float = 16.0  # Policy precision

    # Preferences
    C_awareness_aware: float = 2.0  # Preference for being aware
    C_awareness_unaware: float = 0.0

    # Learning
    A_learning_rate: float = 1.0
    B_learning_rate: float = 1.0
    forgetting_rate: float = 0.9


# =============================================================================
# Model Builders
# =============================================================================

def build_breath_model(params: ModelParams = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build Level 1 breath perception model.

    Returns
    -------
    A1 : np.ndarray, shape (2, 2)
        Likelihood mapping: P(observation | breath_state)
        Rows: observations (EXPANSION, CONTRACTION)
        Cols: states (INHALE, EXHALE)
    B1 : np.ndarray, shape (2, 2)
        Transition matrix: P(breath_state' | breath_state)
        Based on mean durations of inhale/exhale phases.
    """
    if params is None:
        params = ModelParams()

    p = params.A1_precision
    A1 = np.array([
        [p, 1 - p],      # P(EXPANSION | INHALE), P(EXPANSION | EXHALE)
        [1 - p, p],      # P(CONTRACTION | INHALE), P(CONTRACTION | EXHALE)
    ])

    # Transition probabilities based on mean phase durations
    p_stay_inhale = 1.0 - 1.0 / max(1, params.inhale_duration)
    p_stay_exhale = 1.0 - 1.0 / max(1, params.exhale_duration)

    B1 = np.array([
        [p_stay_inhale, 1 - p_stay_exhale],  # P(INHALE | prev_state)
        [1 - p_stay_inhale, p_stay_exhale],  # P(EXHALE | prev_state)
    ])

    return A1, B1


def build_attention_model(
    params: ModelParams = None,
    include_awareness_modality: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build Level 2 attention model.

    Parameters
    ----------
    params : ModelParams
        Model parameters
    include_awareness_modality : bool
        If True, A2 has two modalities (precision obs + awareness obs).
        This represents the meditation instruction.
        If False, A2 has only precision observations.

    Returns
    -------
    A2 : np.ndarray or object array
        Likelihood mapping(s) for attention state
    B2 : np.ndarray, shape (2, 2, 2)
        Transition matrix: P(attention' | attention, action)
        Third dimension: actions (STAY, SWITCH)
    C2 : np.ndarray or object array
        Preferences over observations
    """
    if params is None:
        params = ModelParams()

    # Precision observation modality: weak mapping
    p_prec = params.A2_precision_obs
    A2_precision = np.array([
        [p_prec, 1 - p_prec],      # P(PRECISE | FOCUSED), P(PRECISE | DISTRACTED)
        [1 - p_prec, p_prec],      # P(IMPRECISE | FOCUSED), P(IMPRECISE | DISTRACTED)
    ])

    if include_awareness_modality:
        # Awareness observation modality: meditation instruction
        p_aware = params.A2_awareness_obs
        A2_awareness = np.array([
            [p_aware, 1 - p_aware],      # P(AWARE | FOCUSED), P(AWARE | DISTRACTED)
            [1 - p_aware, p_aware],      # P(UNAWARE | FOCUSED), P(UNAWARE | DISTRACTED)
        ])

        A2 = utils.obj_array(2)
        A2[0] = A2_precision
        A2[1] = A2_awareness

        C2 = utils.obj_array(2)
        C2[0] = np.array([0.0, 0.0])  # No preference on precision obs
        C2[1] = np.array([params.C_awareness_aware, params.C_awareness_unaware])
    else:
        A2 = A2_precision
        C2 = np.array([0.0, 0.0])  # No preference without awareness

    # Agent's beliefs about attention transitions
    # B2[:, :, action] = P(next_state | current_state, action)
    B2 = np.zeros((2, 2, 2))

    # STAY action: agent believes state mostly persists
    p_stay = params.B2_stay_prob
    B2[:, :, STAY] = np.array([
        [p_stay, 1 - p_stay],      # P(FOCUSED | prev, STAY)
        [1 - p_stay, p_stay],      # P(DISTRACTED | prev, STAY)
    ])

    # SWITCH action: agent believes state toggles
    p_switch = params.B2_switch_prob
    B2[:, :, SWITCH] = np.array([
        [1 - p_switch, p_switch],  # P(FOCUSED | prev, SWITCH)
        [p_switch, 1 - p_switch],  # P(DISTRACTED | prev, SWITCH)
    ])

    return A2, B2, C2


def build_awareness_model(params: ModelParams = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build Level 3 awareness model.

    Returns
    -------
    A3 : np.ndarray, shape (2, 2)
        Likelihood: P(awareness_obs | awareness_state)
    B3 : np.ndarray, shape (2, 2)
        Transition: P(awareness' | awareness)
    """
    if params is None:
        params = ModelParams()

    p = params.A3_precision
    A3 = np.array([
        [p, 1 - p],      # P(OBS_AWARE | AWARE), P(OBS_AWARE | UNAWARE)
        [1 - p, p],      # P(OBS_UNAWARE | AWARE), P(OBS_UNAWARE | UNAWARE)
    ])

    B3 = np.array([
        [params.p_aware_to_aware, 1 - params.p_unaware_to_unaware],
        [1 - params.p_aware_to_aware, params.p_unaware_to_unaware],
    ])

    return A3, B3


def build_environment(params: ModelParams = None) -> np.ndarray:
    """
    Build the TRUE environment dynamics for attention (B2_true).

    This represents the actual transition probabilities that the agent
    experiences, which may differ from the agent's beliefs (B2).

    Key feature: distraction is ABSORBING under STAY action.
    This creates the "attention trap" - once distracted, can't passively return.

    Returns
    -------
    B2_true : np.ndarray, shape (2, 2, 2)
        True transition matrix: P(attention' | attention, action)
    """
    if params is None:
        params = ModelParams()

    B2_true = np.zeros((2, 2, 2))

    # STAY action: distraction is absorbing
    B2_true[:, :, STAY] = np.array([
        [params.p_stay_focused, 0.0],                    # P(FOCUSED | prev, STAY)
        [1 - params.p_stay_focused, params.p_stay_distracted],  # P(DISTRACTED | prev, STAY)
    ])

    # SWITCH action: toggles state
    p = params.p_switch_success
    B2_true[:, :, SWITCH] = np.array([
        [1 - p, p],    # P(FOCUSED | prev, SWITCH): foc->dist, dist->foc
        [p, 1 - p],    # P(DISTRACTED | prev, SWITCH)
    ])

    return B2_true


def build_dirichlet_prior(
    matrix: np.ndarray,
    scale: float = 1.0
) -> np.ndarray:
    """
    Build Dirichlet prior from a probability matrix.

    Parameters
    ----------
    matrix : np.ndarray
        Probability matrix (columns sum to 1)
    scale : float
        Concentration parameter. Higher = stronger prior.

    Returns
    -------
    prior : np.ndarray
        Dirichlet concentration parameters (same shape as matrix)
    """
    return matrix * scale + 1e-6  # Small offset for numerical stability


def initialize_A2_for_learning(
    params: ModelParams = None,
    initial_precision: float = 0.52
) -> np.ndarray:
    """
    Initialize A2 concentration parameters for learning.

    Starts near uniform (weak prior) to allow learning.
    """
    if params is None:
        params = ModelParams()

    p = initial_precision
    A2_init = np.array([
        [p, 1 - p],
        [1 - p, p],
    ])

    return build_dirichlet_prior(A2_init, scale=1.0)


def initialize_B2_for_learning(
    params: ModelParams = None,
    initial_stay_prob: float = 0.5
) -> np.ndarray:
    """
    Initialize B2 concentration parameters for learning.

    Starts near uniform for STAY action to allow learning the
    asymmetric dynamics.
    """
    if params is None:
        params = ModelParams()

    pB2 = np.zeros((2, 2, 2))

    # STAY: start near uniform
    p = initial_stay_prob
    pB2[:, :, STAY] = np.array([
        [p, 1 - p],
        [1 - p, p],
    ])

    # SWITCH: agent knows this toggles (from instruction)
    p_switch = params.B2_switch_prob
    pB2[:, :, SWITCH] = np.array([
        [1 - p_switch, p_switch],
        [p_switch, 1 - p_switch],
    ])

    return build_dirichlet_prior(pB2, scale=1.0)
