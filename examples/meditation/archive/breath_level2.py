#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Level-2 hierarchical breath perception with attentional state inference.

Hierarchy:
    Level 1: Breath perception (INHALE/EXHALE from noisy sensory observations)
             - Likelihood precision ζ (zeta) is dynamically updated (B.45)
             
    Level 2: Attentional state inference (FOCUSED/NOT_FOCUSED)
             - Observations are generated from Level 1's precision:
               ζ > 1 → "focused" observation
               ζ ≤ 1 → "not focused" observation

This implements a computational model of meta-awareness in meditation,
where the meditator infers their own attentional state from the precision
of their breath perception.

Note on naming convention:
    ζ (zeta) = likelihood/sensory precision (modulates A matrix)
    γ (gamma) = policy precision (modulates expected free energy)
    
This follows Parr et al. (2022) "Active Inference" Appendix B.
"""

import numpy as np
from pymdp.envs import BreathEnv
from pymdp.agent import Agent
from pymdp import utils
from pymdp.maths import EPS_VAL


# =============================================================================
# Constants
# =============================================================================

# Level 1: Breath states
INHALE = 0
EXHALE = 1

# Level 2: Attentional states  
FOCUSED = 0
NOT_FOCUSED = 1

# Level 2: Observations (generated from precision)
OBS_FOCUSED = 0      # ζ > 1: "focused on breath"
OBS_NOT_FOCUSED = 1  # ζ ≤ 1: "not focused on breath"


# =============================================================================
# Helper Functions
# =============================================================================

def scale_likelihood(A_base: np.ndarray, zeta: float) -> np.ndarray:
    """Scale likelihood matrix with precision parameter ζ."""
    log_A = np.log(A_base + EPS_VAL) * zeta
    A_scaled = np.exp(log_A)
    A_scaled = A_scaled / A_scaled.sum(axis=0, keepdims=True)
    return A_scaled


def discretize_precision(zeta: float) -> int:
    """
    Convert continuous precision to discrete observation for Level 2.
    
    Simple binary mapping:
        ζ > 1  → OBS_FOCUSED (0)     : "focused on breath"
        ζ ≤ 1  → OBS_NOT_FOCUSED (1) : "not focused on breath"
    """
    if zeta > 1.0:
        return OBS_FOCUSED
    else:
        return OBS_NOT_FOCUSED


def compute_prediction_error_with_prior(A, obs, qs_prior):
    """Compute prediction error using prior beliefs."""
    expected_obs = A @ qs_prior
    actual_obs = np.zeros(A.shape[0])
    actual_obs[int(obs)] = 1.0
    pe = np.sum((actual_obs - expected_obs) ** 2)
    return pe


def update_precision(zeta, A, obs, qs_prior, 
                     log_zeta_prior_mean=0.0, log_zeta_prior_var=4.0,
                     lr=1.5, min_zeta=0.1, max_zeta=5.0):
    """
    Update likelihood precision (ζ) based on prediction error (B.45).
    
    Implements Parr et al. (2022) Equation B.45 adapted for discrete 
    state-space models.
    """
    log_zeta = np.log(zeta + EPS_VAL)
    
    pe = compute_prediction_error_with_prior(A, obs, qs_prior)
    
    uniform = np.ones(len(qs_prior)) / len(qs_prior)
    baseline_pe = compute_prediction_error_with_prior(A, obs, uniform)
    
    precision_weighted_error = zeta * pe
    expected_error = np.exp(log_zeta_prior_mean) * baseline_pe
    
    error_drive = 0.5 * (expected_error - precision_weighted_error)
    prior_term = (log_zeta - log_zeta_prior_mean) / log_zeta_prior_var
    
    log_zeta_new = log_zeta + lr * (error_drive - prior_term)
    zeta_new = np.exp(log_zeta_new)
    zeta_new = np.clip(zeta_new, min_zeta, max_zeta)
    
    return zeta_new, pe


def build_level2_agent():
    """
    Build the Level-2 agent for attentional state inference.
    
    Hidden states: [FOCUSED, NOT_FOCUSED]
    Observations: [OBS_FOCUSED, OBS_NOT_FOCUSED]
    
    A matrix encodes: P(observation | attentional_state)
        - When FOCUSED: high probability of OBS_FOCUSED
        - When NOT_FOCUSED: high probability of OBS_NOT_FOCUSED
    """
    # A matrix: P(observation | hidden_state)
    # Shape: (num_obs, num_states) = (2, 2)
    A_level2 = np.array([
        #  FOCUSED  NOT_FOCUSED
        [0.9,      0.2],   # P(OBS_FOCUSED | state)
        [0.1,      0.8],   # P(OBS_NOT_FOCUSED | state)
    ])
    
    # B matrix: Transition dynamics for attention
    # Attention is somewhat sticky but can drift
    # Shape: (num_states, num_states, num_actions) = (2, 2, 1)
    B_level2 = np.zeros((2, 2, 1))
    B_level2[:, :, 0] = np.array([
        #  from_FOCUSED  from_NOT_FOCUSED
        [0.9,           0.3],   # to_FOCUSED
        [0.1,           0.7],   # to_NOT_FOCUSED
    ])
    
    # D vector: Prior over initial states (slightly favor focused)
    D_level2 = np.array([0.6, 0.4])
    
    B_obj = utils.obj_array(1)
    B_obj[0] = B_level2
    
    agent = Agent(A=A_level2, B=B_obj, D=D_level2, save_belief_hist=True)
    
    return agent


# =============================================================================
# Main Simulation
# =============================================================================

def run_hierarchical_session(
    T: int = 150,
    seed: int = 42,
    # Level 1 precision parameters
    zeta_init: float = 1.0,
    log_zeta_prior_mean: float = 0.0,
    log_zeta_prior_var: float = 4.0,
    precision_lr: float = 1.5,
    # Noise injection for testing
    noise_start: int = None,
    noise_end: int = None,
):
    """
    Run the hierarchical simulation with both Level 1 (breath) and Level 2 (attention).
    
    Parameters
    ----------
    T : int
        Number of timesteps
    seed : int
        Random seed
    zeta_init : float
        Initial precision (ζ) for Level 1
    log_zeta_prior_mean : float
        Prior mean for log-precision ln(ζ). Default 0.0 → ζ ≈ 1.0
    log_zeta_prior_var : float
        Prior variance for log-precision
    precision_lr : float
        Learning rate for precision updates
    noise_start, noise_end : int or None
        If provided, inject random observations during this period
        
    Returns
    -------
    dict : Results containing beliefs and observations from both levels
    """
    np.random.seed(seed)
    
    # =========================================================================
    # Build Level 1 agent (breath perception)
    # =========================================================================
    env = BreathEnv(seed=seed)
    
    p_correct = float(env.p_correct)
    A_level1 = np.array([
        [p_correct, 1.0 - p_correct],
        [1.0 - p_correct, p_correct]
    ])
    
    stay_p_inhale = 1.0 - 1.0 / max(1, np.mean(env.inhale_range))
    stay_p_exhale = 1.0 - 1.0 / max(1, np.mean(env.exhale_range))
    
    B_level1 = np.zeros((2, 2, 1))
    B_level1[:, 0, 0] = [stay_p_inhale, 1.0 - stay_p_inhale]
    B_level1[:, 1, 0] = [1.0 - stay_p_exhale, stay_p_exhale]
    
    B_obj_l1 = utils.obj_array(1)
    B_obj_l1[0] = B_level1
    
    agent_level1 = Agent(A=A_level1, B=B_obj_l1, save_belief_hist=True)
    
    # =========================================================================
    # Build Level 2 agent (attentional state inference)
    # =========================================================================
    agent_level2 = build_level2_agent()
    
    # =========================================================================
    # Data logs
    # =========================================================================
    # Level 1
    true_breath_states = np.zeros(T, dtype=int)
    breath_observations = np.zeros(T, dtype=int)
    breath_posteriors = np.zeros((T, 2))
    breath_priors = np.zeros((T, 2))
    zeta_history = np.zeros(T)
    prediction_errors = np.zeros(T)
    
    # Level 2
    precision_observations = np.zeros(T, dtype=int)  # Binary: focused/not focused
    attention_posteriors = np.zeros((T, 2))
    attention_priors = np.zeros((T, 2))
    
    # =========================================================================
    # Initialize
    # =========================================================================
    zeta = zeta_init
    obs_breath = int(env.reset())
    qs_prior_l1 = np.array([0.5, 0.5])  # Initial prior for Level 1
    qs_prior_l2 = np.array([0.6, 0.4])  # Initial prior for Level 2 (match D)
    
    print(f"Running hierarchical simulation:")
    print(f"  T={T}, zeta_init={zeta_init}, precision_lr={precision_lr}")
    print(f"  Observation threshold: ζ > 1 → 'focused', ζ ≤ 1 → 'not focused'")
    if noise_start is not None:
        print(f"  Noise period: t={noise_start} to t={noise_end}")
    print()
    
    # =========================================================================
    # Main loop
    # =========================================================================
    for t in range(T):
        # ---------------------------------------------------------------------
        # Level 1: Breath perception
        # ---------------------------------------------------------------------
        true_breath_states[t] = env.state
        breath_priors[t] = qs_prior_l1
        
        # Optionally inject noise
        if noise_start is not None and noise_start <= t < noise_end:
            obs_received = np.random.choice([0, 1])
        else:
            obs_received = obs_breath
        
        breath_observations[t] = obs_received
        
        # 1. Update precision FIRST using prior beliefs (B.45)
        zeta_new, pe = update_precision(
            zeta, A_level1, obs_received, qs_prior_l1,
            log_zeta_prior_mean=log_zeta_prior_mean,
            log_zeta_prior_var=log_zeta_prior_var,
            lr=precision_lr
        )
        prediction_errors[t] = pe
        zeta = zeta_new
        zeta_history[t] = zeta
        
        # 2. Infer breath state with updated precision
        A_scaled = scale_likelihood(A_level1, zeta)
        agent_level1.A = utils.to_obj_array(A_scaled)
        qs_breath = agent_level1.infer_states([obs_received])
        breath_posteriors[t] = qs_breath[0]
        
        # 3. Compute prior for next timestep
        qs_prior_l1 = B_level1[:, :, 0] @ qs_breath[0]
        
        # ---------------------------------------------------------------------
        # Level 2: Attentional state inference
        # ---------------------------------------------------------------------
        attention_priors[t] = qs_prior_l2
        
        # Generate observation from precision (the "ascending message")
        # Simple binary: ζ > 1 → focused, ζ ≤ 1 → not focused
        obs_precision = discretize_precision(zeta)
        precision_observations[t] = obs_precision
        
        # Infer attentional state from precision observation
        qs_attention = agent_level2.infer_states([obs_precision])
        attention_posteriors[t] = qs_attention[0]
        
        # Compute attention prior for next timestep
        B_l2 = agent_level2.B[0]
        qs_prior_l2 = B_l2[:, :, 0] @ qs_attention[0]
        
        # ---------------------------------------------------------------------
        # Advance environment
        # ---------------------------------------------------------------------
        obs_breath = int(env.step(None))
    
    # =========================================================================
    # Print summary
    # =========================================================================
    breath_accuracy = (np.argmax(breath_posteriors, axis=1) == true_breath_states).mean()
    mean_zeta = zeta_history.mean()
    pct_focused_obs = (precision_observations == OBS_FOCUSED).mean() * 100
    mean_p_focused = attention_posteriors[:, FOCUSED].mean()
    
    print("Results:")
    print(f"  Level 1 breath accuracy: {breath_accuracy:.3f}")
    print(f"  Mean precision (ζ): {mean_zeta:.3f}")
    print(f"  Focused observations: {pct_focused_obs:.1f}% of timesteps")
    print(f"  Mean P(Focused): {mean_p_focused:.3f}")
    
    return {
        # Level 1
        "true_breath_states": true_breath_states,
        "breath_observations": breath_observations,
        "breath_posteriors": breath_posteriors,
        "breath_priors": breath_priors,
        "zeta_history": zeta_history,
        "prediction_errors": prediction_errors,
        # Level 2
        "precision_observations": precision_observations,
        "attention_posteriors": attention_posteriors,
        "attention_priors": attention_priors,
        # Metadata
        "noise_start": noise_start,
        "noise_end": noise_end,
    }


# =============================================================================
# Plotting
# =============================================================================

def plot_hierarchical_results(results: dict, save_path: str = None):
    """Plot results from the hierarchical simulation."""
    import matplotlib.pyplot as plt
    
    # Publication style
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })
    
    T = len(results["true_breath_states"])
    t_range = np.arange(T)
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True,
                              gridspec_kw={'height_ratios': [1, 1, 1], 'hspace': 0.25})
    
    noise_start = results.get("noise_start")
    noise_end = results.get("noise_end")
    
    # Colors
    BLUE = '#2563eb'
    ORANGE = '#ea580c'
    PURPLE = '#7c3aed'
    GRAY = '#6b7280'
    
    # =========================================================================
    # Panel A: Level 1 - Breath inference
    # =========================================================================
    ax = axes[0]
    p_inhale = results["breath_posteriors"][:, INHALE]
    ax.plot(t_range, p_inhale, color=BLUE, linewidth=1.5, label='P(Inhaling)')
    true_binary = 1.0 - results["true_breath_states"]  # INHALE=0 -> 1
    ax.scatter(t_range, true_binary, s=12, color=GRAY, alpha=0.6, label='True state')
    if noise_start is not None:
        ax.axvspan(noise_start, noise_end, alpha=0.15, color='red', label='Noise')
    ax.set_ylabel("Probability")
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks([0, 0.5, 1])
    ax.legend(loc='upper right', framealpha=0.9)
    ax.text(-0.08, 1.05, 'A', transform=ax.transAxes, fontsize=14, 
            fontweight='bold', va='top')
    ax.set_title("Level 1: Breath State Inference", fontsize=12, fontweight='normal')
    
    # =========================================================================
    # Panel B: Precision (ζ) with observation threshold
    # =========================================================================
    ax = axes[1]
    ax.plot(t_range, results["zeta_history"], color=ORANGE, linewidth=1.5)
    ax.axhline(y=1.0, color=GRAY, linestyle='--', linewidth=1, alpha=0.6,
               label='Threshold (ζ=1)')
    ax.fill_between(t_range, 1.0, results["zeta_history"], 
                     where=results["zeta_history"] > 1.0,
                     alpha=0.2, color='green', label='Focused obs')
    ax.fill_between(t_range, results["zeta_history"], 1.0,
                     where=results["zeta_history"] <= 1.0,
                     alpha=0.2, color='red', label='Not focused obs')
    if noise_start is not None:
        ax.axvspan(noise_start, noise_end, alpha=0.15, color='red')
    ax.set_ylabel("Precision (ζ)")
    ax.set_ylim(0, 2.5)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.text(-0.08, 1.05, 'B', transform=ax.transAxes, fontsize=14, 
            fontweight='bold', va='top')
    ax.set_title("Likelihood Precision → Level 2 Observations", fontsize=12, fontweight='normal')
    
    # =========================================================================
    # Panel C: Level 2 - Attentional state inference
    # =========================================================================
    ax = axes[2]
    p_focused = results["attention_posteriors"][:, FOCUSED]
    ax.plot(t_range, p_focused, color=PURPLE, linewidth=1.5, label='P(Focused)')
    ax.fill_between(t_range, 0, p_focused, alpha=0.2, color=PURPLE)
    ax.axhline(y=0.5, color=GRAY, linestyle='--', linewidth=1, alpha=0.6)
    if noise_start is not None:
        ax.axvspan(noise_start, noise_end, alpha=0.15, color='red')
    ax.set_ylabel("Probability")
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel("Time step")
    ax.legend(loc='upper right', framealpha=0.9)
    ax.text(-0.08, 1.05, 'C', transform=ax.transAxes, fontsize=14, 
            fontweight='bold', va='top')
    ax.set_title("Level 2: Attentional State Inference", fontsize=12, fontweight='normal')
    
    fig.align_ylabels(axes)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor='white')
        # Also save PDF
        pdf_path = save_path.rsplit('.', 1)[0] + '.pdf'
        plt.savefig(pdf_path, bbox_inches="tight", facecolor='white')
        print(f"Saved: {save_path}")
        print(f"Saved: {pdf_path}")
    
    return fig


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="Hierarchical breath perception with attentional state inference")
    parser.add_argument("--T", type=int, default=150, help="Number of timesteps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--zeta-init", type=float, default=1.0, help="Initial precision (ζ)")
    parser.add_argument("--lr", type=float, default=1.5, help="Precision learning rate")
    parser.add_argument("--log-prior-var", type=float, default=4.0, help="Prior variance for log-precision")
    parser.add_argument("--noise-start", type=int, default=None, help="Start of noise period")
    parser.add_argument("--noise-end", type=int, default=None, help="End of noise period")
    parser.add_argument("--save", action="store_true", help="Save plot")
    args = parser.parse_args()
    
    results = run_hierarchical_session(
        T=args.T,
        seed=args.seed,
        zeta_init=args.zeta_init,
        precision_lr=args.lr,
        log_zeta_prior_var=args.log_prior_var,
        noise_start=args.noise_start,
        noise_end=args.noise_end,
    )
    
    here = os.path.dirname(__file__)
    outdir = os.path.join(here, "outputs")
    os.makedirs(outdir, exist_ok=True)
    
    if args.noise_start is not None:
        filename = f"hierarchical_breath_noise_{args.noise_start}_{args.noise_end}.png"
    else:
        filename = "hierarchical_breath_baseline.png"
    
    save_path = os.path.join(outdir, filename) if args.save else None
    fig = plot_hierarchical_results(results, save_path=save_path)
    
    if not args.save:
        import matplotlib.pyplot as plt
        plt.show()
