#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Figure 0: A Matrix Learning Across Meditation Sits

Demonstrates how the observation likelihood (A matrix) is learned over 
multiple meditation sessions through experience with breath observations.

Each "sit" is a trial of breath perception. Between sits, the A matrix is 
updated using Dirichlet learning. We track:
    - Evolution of A matrix entries (diagonal = correct observation probs)
    - Breath inference accuracy improving as A converges to truth

No likelihood precision inference in this figure - just basic A learning.
"""

import numpy as np
import matplotlib.pyplot as plt
from pymdp.envs import BreathEnv
from pymdp import learning
from pymdp.maths import (
    softmax, 
    spm_norm as normalize_A,  # Alias for clarity
    scale_likelihood,
    update_likelihood_precision,
    EPS_VAL,
)


# =============================================================================
# Constants
# =============================================================================

INHALE = 0
EXHALE = 1


def run_single_sit(env, A, B, T=200, zeta=1.0, noise_start=None, noise_end=None):
    """
    Run a single meditation sit (trial) with fixed A matrix and fixed zeta.
    
    Parameters
    ----------
    env : BreathEnv
        Environment
    A : np.ndarray
        Observation likelihood matrix
    B : np.ndarray
        Transition matrix
    T : int
        Timesteps per sit
    zeta : float
        Likelihood precision (attention). zeta=1 is standard,
        zeta>1 sharpens likelihood (more attention),
        zeta<1 blurs likelihood (less attention).
    noise_start, noise_end : int or None
        If set, inject random observations during [noise_start, noise_end)
    
    Returns observations, beliefs, true states, and accuracy.
    """
    observations = []
    beliefs = []
    true_states = []
    
    # Initialize
    qs = np.array([0.5, 0.5])
    obs = int(env.reset())
    
    for t in range(T):
        # Store true state
        true_states.append(env.state)
        
        # Inject noise if in noise period
        if noise_start is not None and noise_start <= t < noise_end:
            obs_received = np.random.choice([0, 1])
        else:
            obs_received = obs
        
        # Store observation
        observations.append(obs_received)
        
        # State inference with precision-weighted likelihood
        log_likelihood = np.log(A[obs_received, :] + 1e-16)
        likelihood = softmax(zeta * log_likelihood)
        qs_posterior = likelihood * qs
        qs_posterior = qs_posterior / qs_posterior.sum()
        
        # Store belief
        beliefs.append(qs_posterior.copy())
        
        # Predict next state
        qs = B @ qs_posterior
        
        # Get next observation
        obs = int(env.step(None))
    
    # Compute accuracy
    inferred_states = [np.argmax(b) for b in beliefs]
    accuracy = np.mean([inf == true for inf, true in zip(inferred_states, true_states)])
    
    return observations, beliefs, true_states, accuracy


def run_single_sit_with_precision_inference(
    env, A, B, T=200, 
    zeta_prior: float = 1.0,
    log_zeta_prior_var: float = 2.0,
    zeta_step: float = 0.25,
    noise_start: int = None,
    noise_end: int = None,
):
    """
    Run a single meditation sit with dynamic precision inference.
    
    Parameters
    ----------
    env : BreathEnv
        Environment
    A : np.ndarray
        Observation likelihood matrix
    B : np.ndarray
        Transition matrix
    T : int
        Timesteps per sit
    noise_start, noise_end : int or None
        If set, inject random observations during [noise_start, noise_end)
    zeta_prior : float
        Prior mean for likelihood precision
    log_zeta_prior_var : float
        Prior variance for log(zeta)
    zeta_step : float
        Learning rate for precision updates
    
    Returns observations, beliefs, true states, accuracy, mean zeta, and full zeta history.
    """
    observations = []
    beliefs = []
    true_states = []
    zeta_history = []
    is_noise = []  # Track which timesteps had noise
    
    # Initialize
    qs = np.array([0.5, 0.5])
    zeta = zeta_prior  # Start at prior
    obs = int(env.reset())
    
    log_zeta_prior_mean = np.log(zeta_prior)
    
    for t in range(T):
        # Store true state
        true_states.append(env.state)
        
        # Inject noise if in noise period
        if noise_start is not None and noise_start <= t < noise_end:
            obs_received = np.random.choice([0, 1])
            is_noise.append(True)
        else:
            obs_received = obs
            is_noise.append(False)
        
        # Store observation
        observations.append(obs_received)
        
        # Update precision first (before state inference)
        # Using the function from pymdp.maths
        zeta, _, _ = update_likelihood_precision(
            zeta, A, obs_received, qs,
            log_zeta_prior_mean=log_zeta_prior_mean,
            log_zeta_prior_var=log_zeta_prior_var,
            zeta_step=zeta_step,
        )
        zeta_history.append(zeta)
        
        # State inference with current precision
        log_likelihood = np.log(A[obs_received, :] + 1e-16)
        likelihood = softmax(zeta * log_likelihood)
        qs_posterior = likelihood * qs
        qs_posterior = qs_posterior / qs_posterior.sum()
        
        # Store belief
        beliefs.append(qs_posterior.copy())
        
        # Predict next state
        qs = B @ qs_posterior
        
        # Get next observation
        obs = int(env.step(None))
    
    # Compute accuracy
    inferred_states = [np.argmax(b) for b in beliefs]
    accuracy = np.mean([inf == true for inf, true in zip(inferred_states, true_states)])
    mean_zeta = np.mean(zeta_history)
    
    return observations, beliefs, true_states, accuracy, mean_zeta, zeta_history, is_noise


def update_A_after_sit(pA, A, observations, beliefs, lr=1.0, fr=1.0):
    """
    Update Dirichlet parameters after a sit.
    
    For each timestep, increment pA based on observation and belief.
    Uses forgetting rate (fr) to decay existing counts before update.
    
    Parameters
    ----------
    pA : np.ndarray
        Current Dirichlet parameters
    A : np.ndarray
        Current observation likelihood
    observations : list
        Observations from the sit
    beliefs : list
        State beliefs from the sit
    lr : float
        Learning rate
    fr : float
        Forgetting rate in [0, 1]. Decays existing counts.
        fr=1.0 means no forgetting.
    """
    # Apply forgetting to existing counts
    pA_new = fr * pA.copy()
    
    for obs, qs in zip(observations, beliefs):
        # One-hot observation
        obs_onehot = np.zeros(2)
        obs_onehot[obs] = 1.0
        
        # Outer product: obs ⊗ qs
        dfda = np.outer(obs_onehot, qs)
        
        # Only update where A > 0 (in our case, everywhere)
        pA_new = pA_new + lr * dfda
    
    # Derive A from pA
    A_new = normalize_A(pA_new)
    
    return pA_new, A_new


# =============================================================================
# Main Simulation
# =============================================================================

def run_A_learning(
    num_sits: int = 30,
    T_per_sit: int = 200,
    seed: int = 42,
    lr: float = 1.0,
    fr: float = 1.0,
    zeta: float = 1.0,
    initial_pA_strength: float = 2.0,  # Initial pseudo-counts (higher = stronger prior)
    p_correct: float = None,  # Override environment's p_correct if set
):
    """
    Run A matrix learning across multiple meditation sits.
    
    Parameters
    ----------
    num_sits : int
        Number of meditation sessions
    T_per_sit : int
        Timesteps per session
    lr : float
        Learning rate for Dirichlet update
    fr : float
        Forgetting rate in [0, 1]. Decays existing counts each sit.
        fr=1.0 means no forgetting.
    zeta : float
        Likelihood precision (attention level). zeta=1 is standard,
        zeta>1 means more attention, zeta<1 means less attention.
    initial_pA_strength : float
        Initial pseudo-count sum per column (higher = slower learning)
    p_correct : float or None
        Override the environment's p_correct. If None, uses env default (0.98).
    """
    np.random.seed(seed)
    
    # =========================================================================
    # Setup Environment
    # =========================================================================
    if p_correct is not None:
        env = BreathEnv(seed=seed, p_correct=p_correct)
    else:
        env = BreathEnv(seed=seed)
    
    # True A matrix (what we're trying to learn)
    p_correct = float(env.p_correct)
    A_true = np.array([
        [p_correct, 1 - p_correct],
        [1 - p_correct, p_correct]
    ])
    
    # Transition dynamics (known, not learned)
    stay_p_inhale = 1.0 - 1.0 / max(1, np.mean(env.inhale_range))
    stay_p_exhale = 1.0 - 1.0 / max(1, np.mean(env.exhale_range))
    B = np.array([
        [stay_p_inhale, 1 - stay_p_exhale],
        [1 - stay_p_inhale, stay_p_exhale]
    ])
    
    # =========================================================================
    # Initialize A with weakly informative prior
    # =========================================================================
    # Start with slight bias toward correct (diagonal higher than off-diagonal)
    # This gives the agent a "head start" to bootstrap learning
    initial_diagonal_bias = 0.6  # Initial P(correct) = 0.6
    pA = np.array([
        [initial_diagonal_bias, 1 - initial_diagonal_bias],
        [1 - initial_diagonal_bias, initial_diagonal_bias]
    ]) * initial_pA_strength
    A = normalize_A(pA)
    
    print(f"A Matrix Learning Across Meditation Sits")
    print(f"  Sits: {num_sits}, Timesteps/sit: {T_per_sit}")
    print(f"  True A diagonal: [{p_correct:.3f}, {p_correct:.3f}]")
    print(f"  Initial A diagonal: [{A[0,0]:.3f}, {A[1,1]:.3f}]")
    print(f"  Learning rate: {lr}, Forgetting rate: {fr}, Precision (zeta): {zeta}")
    print()
    
    # =========================================================================
    # Data Logs
    # =========================================================================
    A_diagonal_history = np.zeros((num_sits + 1, 2))
    A_diagonal_history[0] = [A[0, 0], A[1, 1]]  # Initial
    
    accuracy_history = np.zeros(num_sits)
    pA_total_counts = np.zeros(num_sits + 1)
    pA_total_counts[0] = pA.sum()
    
    # =========================================================================
    # Run Sits
    # =========================================================================
    for sit in range(num_sits):
        # Reset environment for new sit (use same p_correct as initial env)
        env = BreathEnv(seed=seed + sit, p_correct=p_correct)
        
        # Run sit with current A
        observations, beliefs, true_states, accuracy = run_single_sit(env, A, B, T_per_sit, zeta=zeta)
        accuracy_history[sit] = accuracy
        
        # Update A after sit
        pA, A = update_A_after_sit(pA, A, observations, beliefs, lr=lr, fr=fr)
        
        # Log
        A_diagonal_history[sit + 1] = [A[0, 0], A[1, 1]]
        pA_total_counts[sit + 1] = pA.sum()
        
        if (sit + 1) % 10 == 0 or sit == 0:
            print(f"  Sit {sit+1:3d}: Accuracy={accuracy:.3f}, "
                  f"A_diag=[{A[0,0]:.3f}, {A[1,1]:.3f}]")
    
    print()
    print(f"Final A matrix:")
    print(f"  [[{A[0,0]:.4f}, {A[0,1]:.4f}],")
    print(f"   [{A[1,0]:.4f}, {A[1,1]:.4f}]]")
    print(f"True A matrix:")
    print(f"  [[{A_true[0,0]:.4f}, {A_true[0,1]:.4f}],")
    print(f"   [{A_true[1,0]:.4f}, {A_true[1,1]:.4f}]]")
    
    return {
        "A_diagonal_history": A_diagonal_history,
        "accuracy_history": accuracy_history,
        "pA_total_counts": pA_total_counts,
        "A_true": A_true,
        "A_final": A,
        "num_sits": num_sits,
        "lr": lr,
        "fr": fr,
        "zeta": zeta,
    }


def run_A_learning_with_precision(
    num_sits: int = 30, T_per_sit: int = 200, seed: int = 42,
    lr: float = 1.0, fr: float = 0.9, zeta_prior: float = 1.0,
    log_zeta_prior_var: float = 2.0, zeta_step: float = 0.25,
    initial_pA_strength: float = 2.0,
    p_correct: float = None,
):
    """Run A matrix learning with dynamic precision inference during sits."""
    np.random.seed(seed)
    if p_correct is not None:
        env = BreathEnv(seed=seed, p_correct=p_correct)
    else:
        env = BreathEnv(seed=seed)
    p_correct = float(env.p_correct)
    A_true = np.array([[p_correct, 1-p_correct], [1-p_correct, p_correct]])
    stay_p = 1.0 - 1.0 / max(1, np.mean(env.inhale_range))
    B = np.array([[stay_p, 1-stay_p], [1-stay_p, stay_p]])
    pA = np.array([[0.6, 0.4], [0.4, 0.6]]) * initial_pA_strength
    A = normalize_A(pA)
    
    print(f"A Learning WITH Precision: zeta_prior={zeta_prior}")
    A_diagonal_history = np.zeros((num_sits + 1, 2))
    A_diagonal_history[0] = [A[0,0], A[1,1]]
    accuracy_history = np.zeros(num_sits)
    mean_zeta_history = np.zeros(num_sits)
    
    for sit in range(num_sits):
        env = BreathEnv(seed=seed + sit, p_correct=p_correct)
        obs, beliefs, _, acc, mz, _, _ = run_single_sit_with_precision_inference(
            env, A, B, T_per_sit, zeta_prior, log_zeta_prior_var, zeta_step)
        accuracy_history[sit], mean_zeta_history[sit] = acc, mz
        pA, A = update_A_after_sit(pA, A, obs, beliefs, lr=lr, fr=fr)
        A_diagonal_history[sit + 1] = [A[0,0], A[1,1]]
        if (sit + 1) % 10 == 0 or sit == 0:
            print(f"  Sit {sit+1:3d}: A=[{A[0,0]:.3f},{A[1,1]:.3f}], zeta={mz:.3f}")
    
    print(f"Final: A=[{A[0,0]:.3f},{A[1,1]:.3f}]")
    return {"A_diagonal_history": A_diagonal_history, "accuracy_history": accuracy_history,
            "mean_zeta_history": mean_zeta_history, "A_true": A_true, "num_sits": num_sits,
            "zeta_prior": zeta_prior}


# =============================================================================
# Non-stationary Environment Learning
# =============================================================================

def run_A_learning_nonstationary(
    num_sits: int = 200, T_per_sit: int = 200, seed: int = 42,
    lr: float = 1.0, fr: float = 0.9, zeta_prior: float = 1.0,
    log_zeta_prior_var: float = 2.0, zeta_step: float = 0.25,
    initial_pA_strength: float = 2.0,
    p_correct_phase1: float = 0.98,
    p_correct_phase2: float = 0.90,
    switch_sit: int = 100,
):
    """
    Run A matrix learning in a non-stationary environment.
    
    Environment starts with p_correct_phase1, then switches to p_correct_phase2
    at switch_sit.
    """
    np.random.seed(seed)
    
    # Initial setup with phase 1 parameters
    env = BreathEnv(seed=seed, p_correct=p_correct_phase1)
    stay_p = 1.0 - 1.0 / max(1, np.mean(env.inhale_range))
    B = np.array([[stay_p, 1-stay_p], [1-stay_p, stay_p]])
    pA = np.array([[0.6, 0.4], [0.4, 0.6]]) * initial_pA_strength
    A = normalize_A(pA)
    
    print(f"Non-stationary Learning: zeta_prior={zeta_prior}")
    print(f"  Phase 1 (sits 1-{switch_sit}): p_correct={p_correct_phase1}")
    print(f"  Phase 2 (sits {switch_sit+1}-{num_sits}): p_correct={p_correct_phase2}")
    
    A_diagonal_history = np.zeros((num_sits + 1, 2))
    A_diagonal_history[0] = [A[0,0], A[1,1]]
    accuracy_history = np.zeros(num_sits)
    mean_zeta_history = np.zeros(num_sits)
    p_correct_history = np.zeros(num_sits)
    
    for sit in range(num_sits):
        # Determine which phase we're in
        if sit < switch_sit:
            current_p_correct = p_correct_phase1
        else:
            current_p_correct = p_correct_phase2
        
        p_correct_history[sit] = current_p_correct
        
        # Create environment with current p_correct
        env = BreathEnv(seed=seed + sit, p_correct=current_p_correct)
        
        obs, beliefs, _, acc, mz, _, _ = run_single_sit_with_precision_inference(
            env, A, B, T_per_sit, zeta_prior, log_zeta_prior_var, zeta_step)
        accuracy_history[sit], mean_zeta_history[sit] = acc, mz
        pA, A = update_A_after_sit(pA, A, obs, beliefs, lr=lr, fr=fr)
        A_diagonal_history[sit + 1] = [A[0,0], A[1,1]]
        
        if (sit + 1) % 20 == 0 or sit == 0 or sit == switch_sit:
            phase = "Phase 1" if sit < switch_sit else "Phase 2"
            print(f"  Sit {sit+1:3d} ({phase}): A=[{A[0,0]:.3f},{A[1,1]:.3f}], acc={acc:.3f}, ζ={mz:.3f}")
    
    print(f"Final: A=[{A[0,0]:.3f},{A[1,1]:.3f}]")
    
    return {
        "A_diagonal_history": A_diagonal_history,
        "accuracy_history": accuracy_history,
        "mean_zeta_history": mean_zeta_history,
        "p_correct_history": p_correct_history,
        "num_sits": num_sits,
        "zeta_prior": zeta_prior,
        "switch_sit": switch_sit,
        "p_correct_phase1": p_correct_phase1,
        "p_correct_phase2": p_correct_phase2,
    }


def run_nonstationary_comparison(
    zeta_prior_values: list = [0.95, 1.0, 1.05, 1.1, 1.15],
    num_sits: int = 200,
    T_per_sit: int = 200,
    lr: float = 1.0,
    fr: float = 0.9,
    seed: int = 42,
    p_correct_phase1: float = 0.98,
    p_correct_phase2: float = 0.90,
    switch_sit: int = 100,
    save_dir: str = None,
):
    """
    Compare A matrix learning with different ζ priors in a non-stationary environment.
    
    Tests how well agents adapt when environment reliability drops mid-learning.
    """
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.size': 11, 'axes.labelsize': 12,
        'axes.titlesize': 13, 'legend.fontsize': 9, 'figure.dpi': 150,
        'axes.spines.top': False, 'axes.spines.right': False,
    })
    
    cmap = plt.get_cmap('RdYlGn')
    zeta_min, zeta_max = min(zeta_prior_values), max(zeta_prior_values)
    colors, labels = {}, {}
    for zp in zeta_prior_values:
        norm = (zp - zeta_min) / (zeta_max - zeta_min) if zeta_max > zeta_min else 0.5
        colors[zp] = cmap(norm)
        labels[zp] = f'ζ prior={zp}'
    
    all_results = {}
    for zp in zeta_prior_values:
        print(f"\n=== zeta_prior={zp} ===")
        results = run_A_learning_nonstationary(
            num_sits=num_sits, T_per_sit=T_per_sit, seed=seed,
            lr=lr, fr=fr, zeta_prior=zp,
            p_correct_phase1=p_correct_phase1,
            p_correct_phase2=p_correct_phase2,
            switch_sit=switch_sit)
        all_results[zp] = results
    
    # Create 3-panel figure
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True,
                              gridspec_kw={'height_ratios': [1, 1, 1], 'hspace': 0.25})
    GRAY = '#6b7280'
    
    # Panel A: A matrix learning
    ax = axes[0]
    for zp in zeta_prior_values:
        A_diag = all_results[zp]["A_diagonal_history"]
        sit_range = np.arange(all_results[zp]["num_sits"] + 1)
        A_avg = (A_diag[:, 0] + A_diag[:, 1]) / 2
        ax.plot(sit_range, A_avg, color=colors[zp], linewidth=2, label=labels[zp])
    
    # True A lines for both phases
    ax.axhline(y=p_correct_phase1, color=GRAY, linestyle='--', linewidth=1.5, alpha=0.7)
    ax.axhline(y=p_correct_phase2, color=GRAY, linestyle=':', linewidth=1.5, alpha=0.7)
    ax.axvline(x=switch_sit, color='red', linestyle='-', linewidth=2, alpha=0.5, label='Switch')
    
    # Add text annotations for true values
    ax.text(switch_sit/2, p_correct_phase1 + 0.015, f'True ({p_correct_phase1:.2f})', 
            ha='center', fontsize=9, color=GRAY)
    ax.text((switch_sit + num_sits)/2, p_correct_phase2 - 0.025, f'True ({p_correct_phase2:.2f})', 
            ha='center', fontsize=9, color=GRAY)
    
    ax.set_ylabel("P(correct)")
    ax.set_ylim(0.55, 1.02)
    ax.legend(loc='lower right', framealpha=0.9, ncol=2)
    ax.text(-0.06, 1.05, 'A', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')
    ax.set_title("Observation Likelihood Learning (Non-stationary Environment)", fontsize=12)
    
    # Panel B: Mean zeta during sits
    ax = axes[1]
    for zp in zeta_prior_values:
        mz = all_results[zp]["mean_zeta_history"]
        sit_range = np.arange(1, all_results[zp]["num_sits"] + 1)
        ax.plot(sit_range, mz, color=colors[zp], linewidth=2, label=labels[zp])
        ax.axhline(y=zp, color=colors[zp], linestyle=':', linewidth=1, alpha=0.5)
    ax.axvline(x=switch_sit, color='red', linestyle='-', linewidth=2, alpha=0.5)
    ax.set_ylabel("Mean ζ")
    ax.set_ylim(0.5, 2.0)
    ax.legend(loc='upper right', framealpha=0.9, ncol=2)
    ax.text(-0.06, 1.05, 'B', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')
    ax.set_title("Dynamic Precision (mean per sit)", fontsize=12)
    
    # Panel C: Accuracy
    ax = axes[2]
    for zp in zeta_prior_values:
        acc = all_results[zp]["accuracy_history"]
        sit_range = np.arange(1, all_results[zp]["num_sits"] + 1)
        ax.plot(sit_range, acc, color=colors[zp], linewidth=2, label=labels[zp])
    ax.axvline(x=switch_sit, color='red', linestyle='-', linewidth=2, alpha=0.5, label='Switch')
    ax.axhline(y=p_correct_phase1, color=GRAY, linestyle='--', linewidth=1, alpha=0.5)
    ax.axhline(y=p_correct_phase2, color=GRAY, linestyle=':', linewidth=1, alpha=0.5)
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Meditation Sit")
    ax.set_ylim(0.5, 1.05)
    ax.legend(loc='lower right', framealpha=0.9, ncol=2)
    ax.text(-0.06, 1.05, 'C', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')
    ax.set_title("Inference Accuracy", fontsize=12)
    
    fig.suptitle(f"Non-stationary Environment: p_correct {p_correct_phase1}→{p_correct_phase2} at sit {switch_sit}", 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_dir:
        save_path = os.path.join(save_dir, f"figure_nonstationary_{p_correct_phase1}to{p_correct_phase2}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor='white')
        plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches="tight", facecolor='white')
        print(f"\nSaved: {save_path}")
    
    return fig, all_results


# =============================================================================
# Figure 0.3: A Learning with Hierarchical Attention
# =============================================================================

# Attention states
FOCUSED = 0
DISTRACTED = 1

# Precision observations (ascending message)
HIGH_PRECISION = 0
LOW_PRECISION = 1


def bayesian_update_simple(A, obs, prior):
    """Simple Bayesian state inference."""
    likelihood = A[obs, :]
    posterior = likelihood * prior
    return posterior / (posterior.sum() + EPS_VAL)


def run_single_sit_with_attention(
    env, A, B, T=200,
    # Attention parameters
    zeta_focused: float = 2.0,
    zeta_distracted: float = 0.5,
    A2_precision: float = 0.9,
    zeta_threshold: float = 1.0,
    attention_stay_prob: float = 0.95,
    initial_p_focused: float = 0.7,
    # Precision B.45 parameters (optimized from parameter sweep)
    zeta_step: float = 0.25,
    log_zeta_prior_var: float = 2.0,
):
    """
    Run a single meditation sit with hierarchical attention-perception loop.
    
    DESCENDING: Attention beliefs → precision prior (Bayesian model average)
    ASCENDING:  Precision dynamics → discrete observation → attention update
    
    Parameters
    ----------
    env : BreathEnv
        Environment
    A : np.ndarray
        Observation likelihood matrix (Level 1)
    B : np.ndarray
        Transition matrix (Level 1)
    T : int
        Timesteps per sit
    zeta_focused : float
        Precision associated with FOCUSED attention state
    zeta_distracted : float  
        Precision associated with DISTRACTED attention state
    A2_precision : float
        P(high_prec | focused) = P(low_prec | distracted)
    zeta_threshold : float
        Threshold for binning precision into discrete observations
    attention_stay_prob : float
        Probability of staying in current attention state
    initial_p_focused : float
        Initial belief P(focused)
    zeta_step : float
        Learning rate for B.45 precision update
    log_zeta_prior_var : float
        Prior variance for log(zeta)
        
    Returns
    -------
    dict with observations, beliefs, true_states, accuracy, and attention metrics
    """
    # Level 2 setup
    A2 = np.array([
        [A2_precision, 1 - A2_precision],      # P(high_prec | state)
        [1 - A2_precision, A2_precision]       # P(low_prec | state)
    ])
    B2 = np.array([
        [attention_stay_prob, 1 - attention_stay_prob],
        [1 - attention_stay_prob, attention_stay_prob]
    ])
    zeta_by_state = np.array([zeta_focused, zeta_distracted])
    
    # Data logs
    observations = []
    beliefs = []
    true_states = []
    zeta_history = []
    zeta_prior_history = []
    p_focused_history = []
    
    # Initialize
    qs_breath = np.array([0.5, 0.5])
    qs_attention = np.array([initial_p_focused, 1 - initial_p_focused])
    obs = int(env.reset())
    
    # Initialize zeta at the expected value given initial attention beliefs
    zeta = np.sum(qs_attention * zeta_by_state)
    
    for t in range(T):
        # Store true state
        true_states.append(env.state)
        observations.append(obs)
        
        # =====================================================================
        # DESCENDING MESSAGE: Attention → Precision Prior
        # =====================================================================
        zeta_prior = np.sum(qs_attention * zeta_by_state)
        zeta_prior_history.append(zeta_prior)
        
        # =====================================================================
        # LEVEL 1: Precision Update (B.45) with attention-informed prior
        # =====================================================================
        # Use update_likelihood_precision from maths.py with attention-derived prior
        log_zeta_prior_mean = np.log(zeta_prior + EPS_VAL)
        zeta, _, _ = update_likelihood_precision(
            zeta=zeta,
            A=A,
            obs=obs,
            qs=qs_breath,
            log_zeta_prior_mean=log_zeta_prior_mean,
            log_zeta_prior_var=log_zeta_prior_var,
            learning_rate=zeta_step,
        )
        zeta_history.append(zeta)
        
        # =====================================================================
        # LEVEL 1: Breath State Inference
        # =====================================================================
        A_scaled = scale_likelihood(A, zeta)  # Use imported function
        qs_breath = bayesian_update_simple(A_scaled, obs, qs_breath)
        beliefs.append(qs_breath.copy())
        
        # =====================================================================
        # ASCENDING MESSAGE: Precision → Discrete Observation
        # =====================================================================
        obs_precision = HIGH_PRECISION if zeta > zeta_threshold else LOW_PRECISION
        
        # =====================================================================
        # LEVEL 2: Attention Inference
        # =====================================================================
        qs_attention = bayesian_update_simple(A2, obs_precision, qs_attention)
        p_focused_history.append(qs_attention[FOCUSED])
        
        # =====================================================================
        # Advance to next timestep
        # =====================================================================
        qs_breath = B @ qs_breath
        qs_attention = B2 @ qs_attention
        obs = int(env.step(None))
    
    # Compute metrics
    inferred_states = [np.argmax(b) for b in beliefs]
    accuracy = np.mean([inf == true for inf, true in zip(inferred_states, true_states)])
    mean_zeta = np.mean(zeta_history)
    mean_zeta_prior = np.mean(zeta_prior_history)
    mean_p_focused = np.mean(p_focused_history)
    
    return {
        "observations": observations,
        "beliefs": beliefs,
        "true_states": true_states,
        "accuracy": accuracy,
        "mean_zeta": mean_zeta,
        "mean_zeta_prior": mean_zeta_prior,
        "mean_p_focused": mean_p_focused,
        "zeta_history": zeta_history,
        "zeta_prior_history": zeta_prior_history,
        "p_focused_history": p_focused_history,
    }


def run_A_learning_with_attention(
    num_sits: int = 200,
    T_per_sit: int = 200,
    seed: int = 42,
    lr: float = 1.0,
    fr: float = 0.9,
    # Attention parameters
    zeta_focused: float = 2.0,
    zeta_distracted: float = 0.5,
    A2_precision: float = 0.9,
    zeta_threshold: float = 1.0,
    attention_stay_prob: float = 0.95,
    initial_p_focused: float = 0.7,
    # Precision B.45 parameters (optimized from parameter sweep)
    zeta_step: float = 0.25,
    log_zeta_prior_var: float = 2.0,
    # A matrix initialization
    initial_pA_strength: float = 2.0,
    # Environment
    p_correct: float = None,
):
    """
    Run A matrix learning with hierarchical attention-perception loop.
    
    Figure 0.3: How does A matrix learning change when the agent has beliefs
    about their attentional state, which in turn modulates precision?
    """
    np.random.seed(seed)
    
    # Setup environment - use provided p_correct or default from env
    env = BreathEnv(seed=seed, p_correct=p_correct if p_correct is not None else 0.98)
    actual_p_correct = float(env.p_correct)
    A_true = np.array([[actual_p_correct, 1-actual_p_correct], [1-actual_p_correct, actual_p_correct]])
    
    # Transition dynamics (known)
    stay_p = 1.0 - 1.0 / max(1, np.mean(env.inhale_range))
    B = np.array([[stay_p, 1-stay_p], [1-stay_p, stay_p]])
    
    # Initialize A with weakly informative prior
    pA = np.array([[0.6, 0.4], [0.4, 0.6]]) * initial_pA_strength
    A = normalize_A(pA)
    
    # Expected zeta under attention prior
    expected_zeta_prior = initial_p_focused * zeta_focused + (1 - initial_p_focused) * zeta_distracted
    
    print(f"A Learning WITH Attention: ζ_focused={zeta_focused}, ζ_distracted={zeta_distracted}")
    print(f"  Initial P(focused)={initial_p_focused} → Expected ζ prior ≈ {expected_zeta_prior:.3f}")
    print(f"  Environment p_correct={actual_p_correct:.2f}")
    
    # History arrays
    A_diagonal_history = np.zeros((num_sits + 1, 2))
    A_diagonal_history[0] = [A[0,0], A[1,1]]
    accuracy_history = np.zeros(num_sits)
    mean_zeta_history = np.zeros(num_sits)
    mean_zeta_prior_history = np.zeros(num_sits)
    mean_p_focused_history = np.zeros(num_sits)
    
    for sit in range(num_sits):
        env = BreathEnv(seed=seed + sit, p_correct=actual_p_correct)
        
        sit_result = run_single_sit_with_attention(
            env, A, B, T_per_sit,
            zeta_focused=zeta_focused,
            zeta_distracted=zeta_distracted,
            A2_precision=A2_precision,
            zeta_threshold=zeta_threshold,
            attention_stay_prob=attention_stay_prob,
            initial_p_focused=initial_p_focused,
            zeta_step=zeta_step,
            log_zeta_prior_var=log_zeta_prior_var,
        )
        
        accuracy_history[sit] = sit_result["accuracy"]
        mean_zeta_history[sit] = sit_result["mean_zeta"]
        mean_zeta_prior_history[sit] = sit_result["mean_zeta_prior"]
        mean_p_focused_history[sit] = sit_result["mean_p_focused"]
        
        # Update A matrix
        pA, A = update_A_after_sit(
            pA, A, sit_result["observations"], sit_result["beliefs"], lr=lr, fr=fr
        )
        A_diagonal_history[sit + 1] = [A[0,0], A[1,1]]
        
        if (sit + 1) % 20 == 0 or sit == 0:
            print(f"  Sit {sit+1:3d}: A=[{A[0,0]:.3f},{A[1,1]:.3f}], "
                  f"ζ={sit_result['mean_zeta']:.3f}, P(foc)={sit_result['mean_p_focused']:.3f}")
    
    print(f"Final: A=[{A[0,0]:.3f},{A[1,1]:.3f}]")
    
    return {
        "A_diagonal_history": A_diagonal_history,
        "accuracy_history": accuracy_history,
        "mean_zeta_history": mean_zeta_history,
        "mean_zeta_prior_history": mean_zeta_prior_history,
        "mean_p_focused_history": mean_p_focused_history,
        "A_true": A_true,
        "num_sits": num_sits,
        "zeta_focused": zeta_focused,
        "zeta_distracted": zeta_distracted,
        "initial_p_focused": initial_p_focused,
        "lr": lr,
        "fr": fr,
    }


def run_attention_comparison(
    zeta_focused_values: list = [1.0, 1.25, 1.5, 2.0],
    zeta_distracted: float = 0.8,
    num_sits: int = 200,
    T_per_sit: int = 200,
    lr: float = 1.0,
    fr: float = 0.9,
    seed: int = 42,
    save_dir: str = None,
):
    """
    Compare A learning across different attention configurations.
    
    Sweeps zeta_focused while keeping zeta_distracted fixed.
    Also includes baseline comparison from Figure 0.2 (fixed prior).
    """
    all_results = {}
    
    # First: baselines from fixed prior (Figure 0.2 style)
    # Run fixed prior that matches expected attention prior
    print("\n=== Baseline: Fixed ζ prior ===")
    for zf in zeta_focused_values:
        initial_p_focused = 0.7
        expected_zeta = initial_p_focused * zf + (1 - initial_p_focused) * zeta_distracted
        print(f"\nFixed ζ prior = {expected_zeta:.3f} (matching ζ_focused={zf})")
        
        results = run_A_learning_with_precision(
            num_sits=num_sits,
            T_per_sit=T_per_sit,
            seed=seed,
            lr=lr,
            fr=fr,
            zeta_prior=expected_zeta,
        )
        all_results[f"fixed_{zf}"] = results
    
    # Then: attention-based runs
    print("\n=== Attention-based runs ===")
    for zf in zeta_focused_values:
        print(f"\n=== zeta_focused={zf} ===")
        results = run_A_learning_with_attention(
            num_sits=num_sits,
            T_per_sit=T_per_sit,
            seed=seed,
            lr=lr,
            fr=fr,
            zeta_focused=zf,
            zeta_distracted=zeta_distracted,
        )
        all_results[f"attention_{zf}"] = results
    
    # Plot comparison
    fig = plot_figure0_3(all_results, zeta_focused_values, zeta_distracted, save_dir)
    
    return fig, all_results


def plot_figure0_3(
    all_results: dict,
    zeta_focused_values: list,
    zeta_distracted: float,
    save_dir: str = None,
):
    """
    Generate Figure 0.3: A Learning with Attention vs Fixed Prior.
    
    Four-panel figure:
    A) A matrix learning curves (attention-based)
    B) A matrix learning curves (fixed prior baselines)
    C) Mean P(focused) over sits
    D) Mean ζ dynamics over sits
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
    
    # Colors
    colors = ['#b91c1c', '#ea580c', '#65a30d', '#059669', '#0284c7']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Figure 0.3: A Learning with Hierarchical Attention\n"
                 f"(ζ_distracted={zeta_distracted})", 
                 fontsize=14, fontweight='bold', y=1.02)
    
    # Get first result to determine num_sits
    first_key = list(all_results.keys())[0]
    num_sits = all_results[first_key]["num_sits"]
    sit_range = np.arange(num_sits + 1)
    A_true = all_results[first_key]["A_true"][0, 0]
    
    # =========================================================================
    # Panel A: Attention-based A learning
    # =========================================================================
    ax = axes[0, 0]
    for i, zf in enumerate(zeta_focused_values):
        key = f"attention_{zf}"
        if key in all_results:
            A_diag = all_results[key]["A_diagonal_history"]
            # Plot mean of both diagonal entries
            A_mean = (A_diag[:, 0] + A_diag[:, 1]) / 2
            ax.plot(sit_range, A_mean, color=colors[i], linewidth=2, 
                   label=f'ζ_foc={zf}')
    
    ax.axhline(y=A_true, color='gray', linestyle='--', linewidth=1.5, 
               alpha=0.7, label=f'True ({A_true:.2f})')
    ax.set_ylabel("P(correct)")
    ax.set_ylim(0.55, 1.0)
    ax.legend(loc='lower right', fontsize=8)
    ax.text(-0.08, 1.05, 'A', transform=ax.transAxes, fontsize=14, fontweight='bold')
    ax.set_title("A Learning (with Attention)", fontsize=11)
    
    # =========================================================================
    # Panel B: Fixed prior baselines
    # =========================================================================
    ax = axes[0, 1]
    for i, zf in enumerate(zeta_focused_values):
        key = f"fixed_{zf}"
        if key in all_results:
            A_diag = all_results[key]["A_diagonal_history"]
            A_mean = (A_diag[:, 0] + A_diag[:, 1]) / 2
            expected_zeta = 0.7 * zf + 0.3 * zeta_distracted
            ax.plot(sit_range, A_mean, color=colors[i], linewidth=2, 
                   linestyle='--', label=f'ζ prior={expected_zeta:.2f}')
    
    ax.axhline(y=A_true, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.set_ylabel("P(correct)")
    ax.set_ylim(0.55, 1.0)
    ax.legend(loc='lower right', fontsize=8)
    ax.text(-0.08, 1.05, 'B', transform=ax.transAxes, fontsize=14, fontweight='bold')
    ax.set_title("A Learning (Fixed Prior - Baseline)", fontsize=11)
    
    # =========================================================================
    # Panel C: Mean P(focused) over sits
    # =========================================================================
    ax = axes[1, 0]
    sit_range_acc = np.arange(1, num_sits + 1)
    for i, zf in enumerate(zeta_focused_values):
        key = f"attention_{zf}"
        if key in all_results:
            p_focused = all_results[key]["mean_p_focused_history"]
            ax.plot(sit_range_acc, p_focused, color=colors[i], linewidth=2, 
                   label=f'ζ_foc={zf}')
    
    ax.axhline(y=0.7, color='gray', linestyle=':', linewidth=1, alpha=0.5, 
               label='Initial prior')
    ax.set_ylabel("Mean P(focused)")
    ax.set_xlabel("Meditation Sit")
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc='lower right', fontsize=8)
    ax.text(-0.08, 1.05, 'C', transform=ax.transAxes, fontsize=14, fontweight='bold')
    ax.set_title("Attention Beliefs Over Sits", fontsize=11)
    
    # =========================================================================
    # Panel D: Mean ζ dynamics
    # =========================================================================
    ax = axes[1, 1]
    for i, zf in enumerate(zeta_focused_values):
        key = f"attention_{zf}"
        if key in all_results:
            mean_zeta = all_results[key]["mean_zeta_history"]
            mean_zeta_prior = all_results[key]["mean_zeta_prior_history"]
            ax.plot(sit_range_acc, mean_zeta, color=colors[i], linewidth=2, 
                   label=f'ζ post (ζ_foc={zf})')
            ax.plot(sit_range_acc, mean_zeta_prior, color=colors[i], linewidth=1, 
                   linestyle=':', alpha=0.7)
    
    ax.axhline(y=1.0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.set_ylabel("Mean ζ")
    ax.set_xlabel("Meditation Sit")
    ax.set_ylim(0.4, 2.2)
    ax.legend(loc='upper right', fontsize=8)
    ax.text(-0.08, 1.05, 'D', transform=ax.transAxes, fontsize=14, fontweight='bold')
    ax.set_title("Precision Dynamics (solid=post, dotted=prior)", fontsize=11)
    
    plt.tight_layout()
    
    if save_dir:
        save_path = f"{save_dir}/figure0.3_attention_comparison.png"
        plt.savefig(save_path, bbox_inches="tight", facecolor='white')
        plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches="tight", facecolor='white')
        print(f"\nSaved: {save_path}")
    
    return fig


def run_attention_gap_sweep(
    multiplier_values: list = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0],
    num_sits: int = 200,
    T_per_sit: int = 200,
    lr: float = 1.0,
    fr: float = 0.9,
    seed: int = 42,
    save_dir: str = None,
    p_correct: float = 0.98,
):
    """
    Sweep across attention gaps using MULTIPLICATIVE symmetry.
    
    The precision ζ scales log-likelihood, so multiplicative symmetry is correct:
    - multiplier=1.0 → focused=1.0, distracted=1.0 (no gap)
    - multiplier=1.5 → focused=1.5, distracted=0.667 (1/1.5)
    - multiplier=2.0 → focused=2.0, distracted=0.5 (1/2)
    - multiplier=3.0 → focused=3.0, distracted=0.333 (1/3)
    
    This ensures symmetric effects: ζ=2 doubles log-lik differences,
    ζ=0.5 halves them (equal and opposite).
    """
    all_results = {}
    
    for mult in multiplier_values:
        zeta_focused = mult
        zeta_distracted = 1.0 / mult
        
        print(f"\n=== Multiplier={mult:.2f}: ζ_foc={zeta_focused:.2f}, ζ_dist={zeta_distracted:.3f} ===")
        
        results = run_A_learning_with_attention(
            num_sits=num_sits,
            T_per_sit=T_per_sit,
            seed=seed,
            lr=lr,
            fr=fr,
            zeta_focused=zeta_focused,
            zeta_distracted=zeta_distracted,
            p_correct=p_correct,
        )
        all_results[f"mult_{mult}"] = results
    
    # Plot
    fig = plot_attention_gap_sweep(all_results, multiplier_values, save_dir, p_correct=p_correct)
    
    return fig, all_results


def plot_attention_gap_sweep(
    all_results: dict,
    multiplier_values: list,
    save_dir: str = None,
    p_correct: float = 0.98,
):
    """
    Plot A learning across different attention multipliers.
    
    Three-panel figure:
    A) A matrix learning curves for each multiplier
    B) Final A value vs multiplier
    C) Mean P(focused) vs multiplier
    """
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 8,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })
    
    # Colormap from red (small multiplier) to green (large multiplier)
    import matplotlib.cm as cm
    cmap = plt.colormaps['RdYlGn']
    n_vals = len(multiplier_values)
    colors = [cmap(i / (n_vals - 1)) if n_vals > 1 else cmap(0.5) for i in range(n_vals)]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Figure 0.4: A Learning vs Attention Multiplier (p_correct={p_correct})\n"
                 "(ζ_focused = m, ζ_distracted = 1/m — multiplicative symmetry)", 
                 fontsize=14, fontweight='bold', y=1.02)
    
    # Get num_sits from first result
    first_key = list(all_results.keys())[0]
    num_sits = all_results[first_key]["num_sits"]
    sit_range = np.arange(num_sits + 1)
    A_true = all_results[first_key]["A_true"][0, 0]
    
    # Arrays for summary plots
    final_A_values = []
    mean_p_focused_values = []
    mean_zeta_values = []
    
    # =========================================================================
    # Panel A: A learning curves
    # =========================================================================
    ax = axes[0]
    for i, mult in enumerate(multiplier_values):
        key = f"mult_{mult}"
        if key in all_results:
            A_diag = all_results[key]["A_diagonal_history"]
            A_mean = (A_diag[:, 0] + A_diag[:, 1]) / 2
            zeta_dist = 1.0 / mult
            ax.plot(sit_range, A_mean, color=colors[i], linewidth=1.5, 
                   label=f'm={mult:.2f} (ζ:{mult:.1f}/{zeta_dist:.2f})', alpha=0.8)
            
            final_A_values.append(A_mean[-1])
            mean_p_focused_values.append(np.mean(all_results[key]["mean_p_focused_history"]))
            mean_zeta_values.append(np.mean(all_results[key]["mean_zeta_history"]))
    
    ax.axhline(y=A_true, color='gray', linestyle='--', linewidth=1.5, 
               alpha=0.7, label=f'True ({A_true:.2f})')
    ax.set_ylabel("P(correct)")
    ax.set_xlabel("Meditation Sit")
    ax.set_ylim(0.45, 1.02)
    ax.legend(loc='lower right', fontsize=6, ncol=2)
    ax.text(-0.08, 1.05, 'A', transform=ax.transAxes, fontsize=14, fontweight='bold')
    ax.set_title("A Learning Trajectories", fontsize=11)
    
    # =========================================================================
    # Panel B: Final A vs Multiplier
    # =========================================================================
    ax = axes[1]
    ax.plot(multiplier_values, final_A_values, 'o-', color='#2563eb', linewidth=2, markersize=8)
    ax.axhline(y=A_true, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.set_xlabel("Attention Multiplier (m)")
    ax.set_ylabel("Final P(correct)")
    ax.set_ylim(0.45, 1.02)
    ax.set_xlim(0.9, max(multiplier_values) + 0.1)
    ax.text(-0.08, 1.05, 'B', transform=ax.transAxes, fontsize=14, fontweight='bold')
    ax.set_title("Final A vs Multiplier", fontsize=11)
    
    # Add secondary x-axis showing distracted zeta
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    tick_positions = multiplier_values
    tick_labels = [f'{1/m:.2f}' for m in multiplier_values]
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels(tick_labels, fontsize=8)
    ax2.set_xlabel("ζ_distracted (= 1/m)", fontsize=10)
    
    # =========================================================================
    # Panel C: Mean P(focused) vs Multiplier
    # =========================================================================
    ax = axes[2]
    ax.plot(multiplier_values, mean_p_focused_values, 's-', color='#16a34a', linewidth=2, markersize=8)
    ax.axhline(y=0.7, color='gray', linestyle=':', linewidth=1, alpha=0.5, label='Initial prior')
    ax.set_xlabel("Attention Multiplier (m)")
    ax.set_ylabel("Mean P(focused)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0.9, max(multiplier_values) + 0.1)
    ax.legend(loc='lower right', fontsize=8)
    ax.text(-0.08, 1.05, 'C', transform=ax.transAxes, fontsize=14, fontweight='bold')
    ax.set_title("Attention Stability vs Multiplier", fontsize=11)
    
    plt.tight_layout()
    
    if save_dir:
        save_path = f"{save_dir}/figure0.4_attention_multiplier_sweep_pcorrect{p_correct}.png"
        plt.savefig(save_path, bbox_inches="tight", facecolor='white')
        plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches="tight", facecolor='white')
        print(f"\nSaved: {save_path}")
    
    return fig


# =============================================================================
# Plotting - Figure 0
# =============================================================================

def plot_figure0(results: dict, save_path: str = None):
    """
    Generate Figure 0: A Matrix Learning Over Sits.
    
    Two-panel figure:
    A) A matrix diagonal entries converging to true values
    B) Breath inference accuracy improving over sits
    """
    # Publication style
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })
    
    num_sits = results["num_sits"]
    sit_range = np.arange(num_sits + 1)
    sit_range_acc = np.arange(1, num_sits + 1)
    
    A_diag = results["A_diagonal_history"]
    accuracy = results["accuracy_history"]
    A_true = results["A_true"]
    lr = results.get("lr", 1.0)
    fr = results.get("fr", 1.0)
    
    # Colors
    BLUE = '#2563eb'
    ORANGE = '#ea580c'
    GREEN = '#16a34a'
    GRAY = '#6b7280'
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                              gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.25})
    
    # Add suptitle with parameters
    fig.suptitle(f"A Matrix Learning (lr={lr}, fr={fr})", fontsize=14, fontweight='bold', y=1.02)
    
    # =========================================================================
    # Panel A: A Matrix Diagonal Entries
    # =========================================================================
    ax = axes[0]
    ax.plot(sit_range, A_diag[:, 0], color=BLUE, linewidth=2, 
            label='P(obs=inhale | state=inhale)')
    ax.plot(sit_range, A_diag[:, 1], color=ORANGE, linewidth=2, 
            label='P(obs=exhale | state=exhale)')
    ax.axhline(y=A_true[0, 0], color=GRAY, linestyle='--', linewidth=1.5, 
               alpha=0.7, label=f'True value ({A_true[0,0]:.2f})')
    ax.axhline(y=0.5, color=GRAY, linestyle=':', linewidth=1, alpha=0.5)
    ax.set_ylabel("Probability")
    ax.set_ylim(0.45, 1.0)
    ax.legend(loc='lower right', framealpha=0.9)
    ax.text(-0.06, 1.05, 'A', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')
    ax.set_title("Observation Likelihood Learning", fontsize=12)
    
    # =========================================================================
    # Panel B: Inference Accuracy
    # =========================================================================
    ax = axes[1]
    ax.plot(sit_range_acc, accuracy, color=GREEN, linewidth=2, label='Accuracy')
    ax.axhline(y=accuracy[-1], color=GRAY, linestyle='--', linewidth=1, alpha=0.5)
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Meditation Sit")
    ax.set_ylim(0.45, 1.0)
    ax.legend(loc='lower right', framealpha=0.9)
    ax.text(-0.06, 1.05, 'B', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')
    ax.set_title("Breath Inference Accuracy", fontsize=12)
    
    fig.align_ylabels(axes)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor='white')
        pdf_path = save_path.rsplit('.', 1)[0] + '.pdf'
        plt.savefig(pdf_path, bbox_inches="tight", facecolor='white')
        print(f"Saved: {save_path}")
        print(f"Saved: {pdf_path}")
    
    return fig


def run_parameter_sweep(
    lr_values: list = [0.5, 1.0, 2.0],
    fr_values: list = [1.0, 0.99, 0.95, 0.9],
    num_sits: int = 100,
    T_per_sit: int = 200,
    seed: int = 42,
    save_dir: str = None,
):
    """
    Run a parameter sweep over learning rate and forgetting rate.
    
    Creates a grid plot showing A matrix convergence for each combination.
    """
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 9,
        'axes.labelsize': 10,
        'axes.titlesize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 7,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })
    
    n_lr = len(lr_values)
    n_fr = len(fr_values)
    
    # Colors
    BLUE = '#2563eb'
    ORANGE = '#ea580c'
    GRAY = '#6b7280'
    
    fig, axes = plt.subplots(n_lr, n_fr, figsize=(3.5 * n_fr, 3 * n_lr), 
                              sharex=True, sharey=True)
    
    # Ensure axes is 2D
    if n_lr == 1 and n_fr == 1:
        axes = np.array([[axes]])
    elif n_lr == 1:
        axes = axes.reshape(1, -1)
    elif n_fr == 1:
        axes = axes.reshape(-1, 1)
    
    all_results = {}
    
    for i, lr in enumerate(lr_values):
        for j, fr in enumerate(fr_values):
            print(f"\n=== lr={lr}, fr={fr} ===")
            results = run_A_learning(
                num_sits=num_sits,
                T_per_sit=T_per_sit,
                seed=seed,
                lr=lr,
                fr=fr,
            )
            all_results[(lr, fr)] = results
            
            ax = axes[i, j]
            A_diag = results["A_diagonal_history"]
            A_true = results["A_true"]
            sit_range = np.arange(num_sits + 1)
            
            ax.plot(sit_range, A_diag[:, 0], color=BLUE, linewidth=1.5, label='Inhale')
            ax.plot(sit_range, A_diag[:, 1], color=ORANGE, linewidth=1.5, label='Exhale')
            ax.axhline(y=A_true[0, 0], color=GRAY, linestyle='--', linewidth=1, alpha=0.7)
            ax.axhline(y=0.5, color=GRAY, linestyle=':', linewidth=0.5, alpha=0.5)
            
            ax.set_ylim(0.45, 1.0)
            ax.set_title(f"lr={lr}, fr={fr}", fontsize=10)
            
            # Add final A value as text
            final_A = (A_diag[-1, 0] + A_diag[-1, 1]) / 2
            ax.text(0.95, 0.05, f"A={final_A:.2f}", transform=ax.transAxes, 
                    ha='right', va='bottom', fontsize=8, 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            if i == n_lr - 1:
                ax.set_xlabel("Sit")
            if j == 0:
                ax.set_ylabel("P(correct)")
    
    # Add legend to first subplot
    axes[0, 0].legend(loc='lower right', framealpha=0.9)
    
    fig.suptitle("A Matrix Learning: lr × fr Sweep", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_dir:
        save_path = os.path.join(save_dir, "figure0_lr_fr_sweep.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor='white')
        pdf_path = save_path.rsplit('.', 1)[0] + '.pdf'
        plt.savefig(pdf_path, bbox_inches="tight", facecolor='white')
        print(f"\nSaved: {save_path}")
        print(f"Saved: {pdf_path}")
    
    return fig, all_results


def run_parameter_sweep_with_precision(
    lr_values: list = [0.5, 1.0, 2.0],
    fr_values: list = [1.0, 0.95, 0.9, 0.85],
    num_sits: int = 200,
    T_per_sit: int = 200,
    seed: int = 42,
    zeta_prior: float = 1.0,
    zeta_step: float = 0.25,
    log_zeta_prior_var: float = 2.0,
    save_dir: str = None,
):
    """
    Run a parameter sweep over lr and fr WITH dynamic precision inference.
    
    This tests whether different A learning parameters work better when
    precision is being inferred dynamically via B.45.
    """
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 9,
        'axes.labelsize': 10,
        'axes.titlesize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 7,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })
    
    n_lr = len(lr_values)
    n_fr = len(fr_values)
    
    # Colors
    BLUE = '#2563eb'
    ORANGE = '#ea580c'
    GRAY = '#6b7280'
    
    fig, axes = plt.subplots(n_lr, n_fr, figsize=(3.5 * n_fr, 3 * n_lr), 
                              sharex=True, sharey=True)
    
    # Ensure axes is 2D
    if n_lr == 1 and n_fr == 1:
        axes = np.array([[axes]])
    elif n_lr == 1:
        axes = axes.reshape(1, -1)
    elif n_fr == 1:
        axes = axes.reshape(-1, 1)
    
    all_results = {}
    
    for i, lr in enumerate(lr_values):
        for j, fr in enumerate(fr_values):
            print(f"\n=== lr={lr}, fr={fr} (with precision) ===")
            results = run_A_learning_with_precision(
                num_sits=num_sits,
                T_per_sit=T_per_sit,
                seed=seed,
                lr=lr,
                fr=fr,
                zeta_prior=zeta_prior,
                log_zeta_prior_var=log_zeta_prior_var,
                zeta_step=zeta_step,
            )
            all_results[(lr, fr)] = results
            
            ax = axes[i, j]
            A_diag = results["A_diagonal_history"]
            A_true = results["A_true"]
            sit_range = np.arange(num_sits + 1)
            
            ax.plot(sit_range, A_diag[:, 0], color=BLUE, linewidth=1.5, label='Inhale')
            ax.plot(sit_range, A_diag[:, 1], color=ORANGE, linewidth=1.5, label='Exhale')
            ax.axhline(y=A_true[0, 0], color=GRAY, linestyle='--', linewidth=1, alpha=0.7)
            ax.axhline(y=0.5, color=GRAY, linestyle=':', linewidth=0.5, alpha=0.5)
            
            ax.set_ylim(0.45, 1.0)
            ax.set_title(f"lr={lr}, fr={fr}", fontsize=10)
            
            # Add final A value and mean zeta as text
            final_A = (A_diag[-1, 0] + A_diag[-1, 1]) / 2
            mean_zeta = results["mean_zeta_history"].mean()
            ax.text(0.95, 0.05, f"A={final_A:.2f}\nζ={mean_zeta:.2f}", 
                    transform=ax.transAxes, ha='right', va='bottom', fontsize=8, 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            if i == n_lr - 1:
                ax.set_xlabel("Sit")
            if j == 0:
                ax.set_ylabel("P(correct)")
    
    # Add legend to first subplot
    axes[0, 0].legend(loc='lower right', framealpha=0.9)
    
    fig.suptitle(f"A Learning with Precision (ζ prior={zeta_prior}, step={zeta_step}, var={log_zeta_prior_var})", 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_dir:
        save_path = os.path.join(save_dir, "figure0_lr_fr_sweep_precision.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor='white')
        pdf_path = save_path.rsplit('.', 1)[0] + '.pdf'
        plt.savefig(pdf_path, bbox_inches="tight", facecolor='white')
        print(f"\nSaved: {save_path}")
        print(f"Saved: {pdf_path}")
    
    return fig, all_results


def run_precision_param_sweep_for_A_learning(
    zeta_step_values: list = [0.1, 0.5, 1.0, 1.5],
    var_values: list = [0.5, 1.0, 2.0, 4.0],
    num_sits: int = 200,
    T_per_sit: int = 200,
    seed: int = 42,
    lr: float = 1.0,
    fr: float = 0.9,
    save_dir: str = None,
):
    """
    Sweep over precision parameters (zeta_step, log_zeta_prior_var) to find
    the best combination for A matrix learning.
    
    A learning parameters are fixed at lr=1.0, fr=0.9.
    Compare against fixed zeta=1 baseline.
    """
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 9,
        'axes.labelsize': 10,
        'axes.titlesize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 7,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })
    
    # First run fixed zeta=1 baseline
    print("=== Baseline: Fixed ζ=1 ===")
    baseline = run_A_learning(
        num_sits=num_sits, T_per_sit=T_per_sit, seed=seed, lr=lr, fr=fr, zeta=1.0
    )
    baseline_A = (baseline["A_diagonal_history"][-1, 0] + baseline["A_diagonal_history"][-1, 1]) / 2
    print(f"Baseline Final A: {baseline_A:.3f}")
    
    n_step = len(zeta_step_values)
    n_var = len(var_values)
    
    # Colors
    BLUE = '#2563eb'
    ORANGE = '#ea580c'
    GRAY = '#6b7280'
    GREEN = '#16a34a'
    
    fig, axes = plt.subplots(n_step, n_var, figsize=(3.5 * n_var, 3 * n_step), 
                              sharex=True, sharey=True)
    
    # Ensure axes is 2D
    if n_step == 1 and n_var == 1:
        axes = np.array([[axes]])
    elif n_step == 1:
        axes = axes.reshape(1, -1)
    elif n_var == 1:
        axes = axes.reshape(-1, 1)
    
    all_results = {}
    results_matrix = np.zeros((n_step, n_var))
    
    for i, zeta_step in enumerate(zeta_step_values):
        for j, var in enumerate(var_values):
            print(f"\n=== step={zeta_step}, var={var} ===")
            results = run_A_learning_with_precision(
                num_sits=num_sits,
                T_per_sit=T_per_sit,
                seed=seed,
                lr=lr,
                fr=fr,
                zeta_prior=1.0,
                log_zeta_prior_var=var,
                zeta_step=zeta_step,
            )
            all_results[(zeta_step, var)] = results
            
            ax = axes[i, j]
            A_diag = results["A_diagonal_history"]
            A_true = results["A_true"]
            sit_range = np.arange(num_sits + 1)
            
            ax.plot(sit_range, A_diag[:, 0], color=BLUE, linewidth=1.5, label='Inhale')
            ax.plot(sit_range, A_diag[:, 1], color=ORANGE, linewidth=1.5, label='Exhale')
            ax.axhline(y=A_true[0, 0], color=GRAY, linestyle='--', linewidth=1, alpha=0.7, label='True')
            ax.axhline(y=baseline_A, color=GREEN, linestyle=':', linewidth=1.5, alpha=0.8, label=f'Fixed ζ=1')
            ax.axhline(y=0.5, color=GRAY, linestyle=':', linewidth=0.5, alpha=0.5)
            
            ax.set_ylim(0.45, 1.0)
            ax.set_title(f"step={zeta_step}, var={var}", fontsize=10)
            
            # Add final A value and comparison
            final_A = (A_diag[-1, 0] + A_diag[-1, 1]) / 2
            results_matrix[i, j] = final_A
            mean_zeta = results["mean_zeta_history"].mean()
            
            # Color code: green if beats baseline, red if worse
            color = GREEN if final_A > baseline_A else '#dc2626'
            ax.text(0.95, 0.05, f"A={final_A:.2f}\nζ={mean_zeta:.2f}", 
                    transform=ax.transAxes, ha='right', va='bottom', fontsize=8, 
                    bbox=dict(boxstyle='round', facecolor='white', edgecolor=color, linewidth=2, alpha=0.9))
            
            if i == n_step - 1:
                ax.set_xlabel("Sit")
            if j == 0:
                ax.set_ylabel("P(correct)")
    
    # Add legend to first subplot
    axes[0, 0].legend(loc='lower right', framealpha=0.9, fontsize=6)
    
    fig.suptitle(f"A Learning: Precision Parameter Sweep (A lr={lr}, fr={fr})\n"
                 f"Green border = beats fixed ζ=1 ({baseline_A:.2f})", 
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_dir:
        save_path = os.path.join(save_dir, "precision_param_sweep_A_learning.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor='white')
        print(f"\nSaved: {save_path}")
    
    # Also create a summary heatmap
    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6))
    
    im = ax2.imshow(results_matrix, cmap='RdYlGn', aspect='auto', 
                    vmin=min(results_matrix.min(), baseline_A - 0.05),
                    vmax=max(results_matrix.max(), baseline_A + 0.05))
    
    ax2.set_xticks(range(n_var))
    ax2.set_xticklabels([f"{v}" for v in var_values])
    ax2.set_yticks(range(n_step))
    ax2.set_yticklabels([f"{s}" for s in zeta_step_values])
    ax2.set_xlabel("Prior Variance (log_zeta_prior_var)")
    ax2.set_ylabel("Zeta Step Size")
    ax2.set_title(f"Final A Matrix Value\n(Fixed ζ=1 baseline: {baseline_A:.3f})")
    plt.colorbar(im, ax=ax2, label='Final A')
    
    # Add text annotations
    for i in range(n_step):
        for j in range(n_var):
            val = results_matrix[i, j]
            color = 'white' if abs(val - 0.85) > 0.1 else 'black'
            fontweight = 'bold' if val > baseline_A else 'normal'
            ax2.text(j, i, f"{val:.2f}", ha='center', va='center', 
                    fontsize=10, color=color, fontweight=fontweight)
    
    # Add baseline line
    ax2.axhline(y=-0.5, color='black', linestyle='-', linewidth=0)  # dummy for colorbar reference
    
    plt.tight_layout()
    
    if save_dir:
        save_path2 = os.path.join(save_dir, "precision_param_sweep_A_learning_summary.png")
        plt.savefig(save_path2, dpi=300, bbox_inches="tight", facecolor='white')
        print(f"Saved: {save_path2}")
    
    # Print best result
    best_idx = np.unravel_index(np.argmax(results_matrix), results_matrix.shape)
    best_step = zeta_step_values[best_idx[0]]
    best_var = var_values[best_idx[1]]
    best_A = results_matrix[best_idx]
    
    print(f"\n=== SUMMARY ===")
    print(f"Fixed ζ=1 baseline: A = {baseline_A:.3f}")
    print(f"Best dynamic: step={best_step}, var={best_var} → A = {best_A:.3f}")
    if best_A > baseline_A:
        print(f"✓ Dynamic precision BEATS fixed ζ=1 by {best_A - baseline_A:.3f}")
    else:
        print(f"✗ Fixed ζ=1 wins by {baseline_A - best_A:.3f}")
    
    return fig, fig2, all_results, baseline_A


def run_zeta_comparison(
    zeta_values: list = [0.5, 1.0, 2.0],
    num_sits: int = 200,
    T_per_sit: int = 200,
    lr: float = 1.0,
    fr: float = 0.9,
    seed: int = 42,
    save_dir: str = None,
    p_correct: float = None,
):
    """
    Compare A matrix learning across different attention levels (zeta).
    
    Creates a two-panel figure:
    A) A matrix convergence for each zeta
    B) Accuracy for each zeta
    
    Parameters
    ----------
    p_correct : float or None
        Override the environment's p_correct. If None, uses env default (0.98).
    """
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })
    
    # Colors for different zeta values - gradient from red (low) to green (high)
    import matplotlib.cm as cm
    cmap = cm.get_cmap('RdYlGn')
    zeta_min, zeta_max = min(zeta_values), max(zeta_values)
    colors = {}
    labels = {}
    for zeta in zeta_values:
        # Normalize zeta to [0, 1] for colormap
        norm = (zeta - zeta_min) / (zeta_max - zeta_min) if zeta_max > zeta_min else 0.5
        colors[zeta] = cmap(norm)
        if zeta < 1.0:
            labels[zeta] = f'ζ={zeta} (distracted)'
        elif zeta == 1.0:
            labels[zeta] = f'ζ={zeta} (baseline)'
        else:
            labels[zeta] = f'ζ={zeta} (focused)'
    
    all_results = {}
    
    for zeta in zeta_values:
        print(f"\n=== zeta={zeta} ===")
        results = run_A_learning(
            num_sits=num_sits,
            T_per_sit=T_per_sit,
            seed=seed,
            lr=lr,
            fr=fr,
            zeta=zeta,
            p_correct=p_correct,
        )
        all_results[zeta] = results
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                              gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.25})
    
    GRAY = '#6b7280'
    
    # Get true A value from any result
    A_true = list(all_results.values())[0]["A_true"]
    
    # Panel A: A matrix learning
    ax = axes[0]
    for zeta in zeta_values:
        results = all_results[zeta]
        A_diag = results["A_diagonal_history"]
        sit_range = np.arange(results["num_sits"] + 1)
        # Plot average of diagonal
        A_avg = (A_diag[:, 0] + A_diag[:, 1]) / 2
        ax.plot(sit_range, A_avg, color=colors[zeta], linewidth=2, 
                label=labels[zeta])
    
    ax.axhline(y=A_true[0, 0], color=GRAY, linestyle='--', linewidth=1.5, 
               alpha=0.7, label=f'True ({A_true[0,0]:.2f})')
    ax.axhline(y=0.5, color=GRAY, linestyle=':', linewidth=1, alpha=0.5)
    ax.set_ylabel("P(correct observation)")
    ax.set_ylim(0.55, 1.0)
    ax.legend(loc='lower right', framealpha=0.9)
    ax.text(-0.06, 1.05, 'A', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')
    ax.set_title("Observation Likelihood Learning", fontsize=12)
    
    # Panel B: Accuracy
    ax = axes[1]
    for zeta in zeta_values:
        results = all_results[zeta]
        accuracy = results["accuracy_history"]
        sit_range = np.arange(1, results["num_sits"] + 1)
        ax.plot(sit_range, accuracy, color=colors[zeta], linewidth=2, 
                label=labels[zeta])
    
    ax.set_ylabel("Inference Accuracy")
    ax.set_xlabel("Meditation Sit")
    ax.set_ylim(0.5, 1.05)
    ax.legend(loc='lower right', framealpha=0.9)
    ax.text(-0.06, 1.05, 'B', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')
    ax.set_title("Breath State Inference Accuracy", fontsize=12)
    
    # Get actual p_correct used
    actual_p_correct = A_true[0, 0]
    fig.suptitle(f"Effect of Attention (ζ) on Learning (lr={lr}, fr={fr}, p_correct={actual_p_correct:.2f})", 
                 fontsize=14, fontweight='bold', y=1.02)
    fig.align_ylabels(axes)
    plt.tight_layout()
    
    if save_dir:
        p_str = f"_p{actual_p_correct:.2f}".replace(".", "")
        save_path = os.path.join(save_dir, f"figure0_zeta_comparison{p_str}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor='white')
        pdf_path = save_path.rsplit('.', 1)[0] + '.pdf'
        plt.savefig(pdf_path, bbox_inches="tight", facecolor='white')
        print(f"\nSaved: {save_path}")
        print(f"Saved: {pdf_path}")
    
    return fig, all_results


def run_zeta_prior_comparison(
    zeta_prior_values: list = [0.95, 1.0, 1.05, 1.1, 1.15],
    num_sits: int = 200,
    T_per_sit: int = 200,
    lr: float = 1.0,
    fr: float = 0.9,
    log_zeta_prior_var: float = 2.0,
    zeta_step: float = 0.25,
    seed: int = 42,
    save_dir: str = None,
    p_correct: float = None,
):
    """
    Compare A matrix learning with precision inference across different zeta priors.
    Figure 0.2: Dynamic precision version.
    
    Parameters
    ----------
    p_correct : float or None
        Override the environment's p_correct. If None, uses env default (0.98).
    """
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.size': 11, 'axes.labelsize': 12,
        'axes.titlesize': 13, 'legend.fontsize': 9, 'figure.dpi': 150,
        'axes.spines.top': False, 'axes.spines.right': False,
    })
    
    import matplotlib.cm as cm
    cmap = plt.get_cmap('RdYlGn')
    zeta_min, zeta_max = min(zeta_prior_values), max(zeta_prior_values)
    colors, labels = {}, {}
    for zp in zeta_prior_values:
        norm = (zp - zeta_min) / (zeta_max - zeta_min) if zeta_max > zeta_min else 0.5
        colors[zp] = cmap(norm)
        labels[zp] = f'ζ prior={zp}'
    
    all_results = {}
    for zp in zeta_prior_values:
        print(f"\n=== zeta_prior={zp} ===")
        results = run_A_learning_with_precision(
            num_sits=num_sits, T_per_sit=T_per_sit, seed=seed,
            lr=lr, fr=fr, zeta_prior=zp,
            log_zeta_prior_var=log_zeta_prior_var, zeta_step=zeta_step,
            p_correct=p_correct)
        all_results[zp] = results
    
    # Create 3-panel figure
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True,
                              gridspec_kw={'height_ratios': [1, 1, 1], 'hspace': 0.25})
    GRAY = '#6b7280'
    A_true = list(all_results.values())[0]["A_true"]
    
    # Panel A: A matrix learning
    ax = axes[0]
    for zp in zeta_prior_values:
        A_diag = all_results[zp]["A_diagonal_history"]
        sit_range = np.arange(all_results[zp]["num_sits"] + 1)
        A_avg = (A_diag[:, 0] + A_diag[:, 1]) / 2
        ax.plot(sit_range, A_avg, color=colors[zp], linewidth=2, label=labels[zp])
    ax.axhline(y=A_true[0, 0], color=GRAY, linestyle='--', linewidth=1.5, alpha=0.7, label=f'True ({A_true[0,0]:.2f})')
    ax.set_ylabel("P(correct)")
    ax.set_ylim(0.55, 1.0)
    ax.legend(loc='lower right', framealpha=0.9, ncol=2)
    ax.text(-0.06, 1.05, 'A', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')
    ax.set_title("Observation Likelihood Learning", fontsize=12)
    
    # Panel B: Mean zeta during sits
    ax = axes[1]
    for zp in zeta_prior_values:
        mz = all_results[zp]["mean_zeta_history"]
        sit_range = np.arange(1, all_results[zp]["num_sits"] + 1)
        ax.plot(sit_range, mz, color=colors[zp], linewidth=2, label=labels[zp])
        ax.axhline(y=zp, color=colors[zp], linestyle=':', linewidth=1, alpha=0.5)
    ax.set_ylabel("Mean ζ")
    ax.set_ylim(0.5, 2.0)
    ax.legend(loc='upper right', framealpha=0.9, ncol=2)
    ax.text(-0.06, 1.05, 'B', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')
    ax.set_title("Dynamic Precision (mean per sit)", fontsize=12)
    
    # Panel C: Accuracy
    ax = axes[2]
    for zp in zeta_prior_values:
        acc = all_results[zp]["accuracy_history"]
        sit_range = np.arange(1, all_results[zp]["num_sits"] + 1)
        ax.plot(sit_range, acc, color=colors[zp], linewidth=2, label=labels[zp])
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Meditation Sit")
    ax.set_ylim(0.5, 1.05)
    ax.legend(loc='lower right', framealpha=0.9, ncol=2)
    ax.text(-0.06, 1.05, 'C', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')
    ax.set_title("Inference Accuracy", fontsize=12)
    
    actual_p_correct = A_true[0, 0]
    fig.suptitle(f"Figure 0.2: A Learning with Dynamic Precision (lr={lr}, fr={fr}, p_correct={actual_p_correct:.2f})", 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_dir:
        p_str = f"_p{actual_p_correct:.2f}".replace(".", "")
        save_path = os.path.join(save_dir, f"figure0.2_zeta_prior_comparison{p_str}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor='white')
        plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches="tight", facecolor='white')
        print(f"\nSaved: {save_path}")
    
    return fig, all_results


def run_A_learning_with_noise(
    num_sits: int = 200,
    T_per_sit: int = 200,
    seed: int = 42,
    lr: float = 1.0,
    fr: float = 0.9,
    zeta: float = 1.0,
    use_precision: bool = False,
    noise_start: int = None,
    noise_end: int = None,
    initial_pA_strength: float = 2.0,
    zeta_step: float = 0.25,
    log_zeta_prior_var: float = 2.0,
):
    """
    Run A matrix learning with optional noise injection and precision updating.
    
    Parameters
    ----------
    use_precision : bool
        If True, use dynamic precision updating. If False, use fixed zeta.
    noise_start, noise_end : int or None
        If set, inject random observations during [noise_start, noise_end) in each sit.
    zeta_step : float
        Learning rate for precision updates (when use_precision=True)
    log_zeta_prior_var : float
        Prior variance for log(zeta) (when use_precision=True)
    """
    np.random.seed(seed)
    
    env = BreathEnv(seed=seed)
    p_correct = float(env.p_correct)
    A_true = np.array([[p_correct, 1-p_correct], [1-p_correct, p_correct]])
    
    stay_p = 1.0 - 1.0 / max(1, np.mean(env.inhale_range))
    B = np.array([[stay_p, 1-stay_p], [1-stay_p, stay_p]])
    
    pA = np.array([[0.6, 0.4], [0.4, 0.6]]) * initial_pA_strength
    A = normalize_A(pA)
    
    mode = "dynamic precision" if use_precision else "fixed ζ"
    noise_info = f", noise [{noise_start}-{noise_end}]" if noise_start is not None else ""
    print(f"A Learning ({mode}, ζ={zeta}{noise_info})")
    
    A_diagonal_history = np.zeros((num_sits + 1, 2))
    A_diagonal_history[0] = [A[0,0], A[1,1]]
    accuracy_history = np.zeros(num_sits)
    mean_zeta_history = np.zeros(num_sits)
    
    for sit in range(num_sits):
        env = BreathEnv(seed=seed + sit)
        
        if use_precision:
            obs, beliefs, _, acc, mz, _, _ = run_single_sit_with_precision_inference(
                env, A, B, T_per_sit, 
                zeta_prior=zeta,
                log_zeta_prior_var=log_zeta_prior_var,
                zeta_step=zeta_step,
                noise_start=noise_start, noise_end=noise_end
            )
        else:
            obs, beliefs, _, acc = run_single_sit(
                env, A, B, T_per_sit, zeta=zeta,
                noise_start=noise_start, noise_end=noise_end
            )
            mz = zeta  # Fixed
        
        accuracy_history[sit] = acc
        mean_zeta_history[sit] = mz
        pA, A = update_A_after_sit(pA, A, obs, beliefs, lr=lr, fr=fr)
        A_diagonal_history[sit + 1] = [A[0,0], A[1,1]]
        
        if (sit + 1) % 20 == 0 or sit == 0:
            print(f"  Sit {sit+1:3d}: A=[{A[0,0]:.3f},{A[1,1]:.3f}], ζ={mz:.3f}")
    
    print(f"Final: A=[{A[0,0]:.3f},{A[1,1]:.3f}]")
    
    return {
        "A_diagonal_history": A_diagonal_history,
        "accuracy_history": accuracy_history,
        "mean_zeta_history": mean_zeta_history,
        "A_true": A_true,
        "num_sits": num_sits,
        "lr": lr,
        "fr": fr,
        "noise_start": noise_start,
        "noise_end": noise_end,
    }


def run_fixed_vs_dynamic_comparison(
    num_sits: int = 200,
    T_per_sit: int = 200,
    lr: float = 1.0,
    fr: float = 0.9,
    seed: int = 42,
    noise_start: int = None,
    noise_end: int = None,
    zeta_step: float = 0.25,
    log_zeta_prior_var: float = 2.0,
    save_dir: str = None,
):
    """
    Compare fixed zeta=1 vs dynamic precision with zeta_prior=1.
    
    This tests whether dynamic precision inference adds anything when 
    the prior is already at the optimal value.
    
    If noise_start/noise_end are set, random observations are injected
    during [noise_start, noise_end) in each sit.
    """
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.size': 11, 'axes.labelsize': 12,
        'axes.titlesize': 13, 'legend.fontsize': 10, 'figure.dpi': 150,
        'axes.spines.top': False, 'axes.spines.right': False,
    })
    
    noise_str = "" if noise_start is None else f" (noise t={noise_start}-{noise_end})"
    
    # Run fixed zeta=1 with noise
    print(f"\n=== Fixed ζ=1 (no precision updating){noise_str} ===")
    results_fixed = run_A_learning_with_noise(
        num_sits=num_sits, T_per_sit=T_per_sit, seed=seed,
        lr=lr, fr=fr, zeta=1.0, use_precision=False,
        noise_start=noise_start, noise_end=noise_end
    )
    
    # Run dynamic precision with zeta_prior=1 with noise
    print(f"\n=== Dynamic precision (ζ prior=1, lr={zeta_step}){noise_str} ===")
    results_dynamic = run_A_learning_with_noise(
        num_sits=num_sits, T_per_sit=T_per_sit, seed=seed,
        lr=lr, fr=fr, zeta=1.0, use_precision=True,
        noise_start=noise_start, noise_end=noise_end,
        zeta_step=zeta_step, log_zeta_prior_var=log_zeta_prior_var
    )
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                              gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.25})
    
    noise_title = f", noise t={noise_start}-{noise_end}" if noise_start is not None else ""
    fig.suptitle(f"Fixed ζ=1 vs Dynamic Precision (ζ prior=1)\n(lr={lr}, fr={fr}{noise_title})", 
                 fontsize=14, fontweight='bold', y=1.02)
    
    # Colors
    BLUE = '#2563eb'
    ORANGE = '#ea580c'
    GRAY = '#6b7280'
    
    sit_range = np.arange(num_sits + 1)
    sit_range_acc = np.arange(1, num_sits + 1)
    A_true = results_fixed["A_true"][0, 0]
    
    # Panel A: A matrix learning
    ax = axes[0]
    A_fixed = results_fixed["A_diagonal_history"]
    A_dynamic = results_dynamic["A_diagonal_history"]
    
    ax.plot(sit_range, (A_fixed[:, 0] + A_fixed[:, 1]) / 2, 
            color=BLUE, linewidth=2, label='Fixed ζ=1')
    ax.plot(sit_range, (A_dynamic[:, 0] + A_dynamic[:, 1]) / 2,
            color=ORANGE, linewidth=2, linestyle='--', label='Dynamic (ζ prior=1)')
    ax.axhline(y=A_true, color=GRAY, linestyle='--', linewidth=1.5, 
               alpha=0.7, label=f'True ({A_true:.2f})')
    
    ax.set_ylabel("P(correct)")
    ax.set_ylim(0.55, 1.02)
    ax.legend(loc='lower right')
    ax.text(-0.06, 1.05, 'A', transform=ax.transAxes, fontsize=14, fontweight='bold')
    ax.set_title("Observation Likelihood Learning", fontsize=12)
    
    # Panel B: Inference accuracy
    ax = axes[1]
    ax.plot(sit_range_acc, results_fixed["accuracy_history"],
            color=BLUE, linewidth=2, label='Fixed ζ=1')
    ax.plot(sit_range_acc, results_dynamic["accuracy_history"],
            color=ORANGE, linewidth=2, linestyle='--', label='Dynamic (ζ prior=1)')
    
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Meditation Sit")
    ax.set_ylim(0.55, 1.02)
    ax.legend(loc='lower right')
    ax.text(-0.06, 1.05, 'B', transform=ax.transAxes, fontsize=14, fontweight='bold')
    ax.set_title("Breath Inference Accuracy", fontsize=12)
    
    plt.tight_layout()
    
    # Print summary
    print("\n=== Summary ===")
    print(f"Fixed ζ=1:      Final A = [{A_fixed[-1, 0]:.3f}, {A_fixed[-1, 1]:.3f}]")
    print(f"Dynamic ζ prior=1: Final A = [{A_dynamic[-1, 0]:.3f}, {A_dynamic[-1, 1]:.3f}]")
    print(f"Mean ζ (dynamic): {np.mean(results_dynamic['mean_zeta_history']):.3f}")
    
    if save_dir:
        noise_suffix = f"_noise_{noise_start}_{noise_end}" if noise_start is not None else ""
        save_path = f"{save_dir}/figure_fixed_vs_dynamic{noise_suffix}.png"
        plt.savefig(save_path, bbox_inches="tight", facecolor='white')
        plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches="tight", facecolor='white')
        print(f"\nSaved: {save_path}")
    
    return fig, {"fixed": results_fixed, "dynamic": results_dynamic}


# =============================================================================
# Diagnostic: Within-Sit Precision Dynamics
# =============================================================================

def run_within_sit_diagnostic(
    sit_indices: list = [1, 20, 60, 100],
    T_per_sit: int = 200,
    seed: int = 42,
    lr: float = 1.0,
    fr: float = 0.9,
    noise_start: int = None,
    noise_end: int = None,
    zeta_step: float = 0.25,
    log_zeta_prior_var: float = 2.0,
    save_dir: str = None,
    use_true_A: bool = False,
):
    """
    Run specific sits and visualize within-sit precision dynamics.
    
    This is a diagnostic to verify precision dynamics are working correctly
    during noisy periods.
    
    Parameters
    ----------
    use_true_A : bool
        If True, use the true environment A matrix (like Figure 1) instead of
        learning A. This isolates precision dynamics from A learning issues.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    
    # Initialize A matrix
    p_correct = 0.9
    A_true = np.array([[p_correct, 1.0 - p_correct],
                       [1.0 - p_correct, p_correct]])
    
    if use_true_A:
        # Use true A matrix (like Figure 1) - no learning
        A = A_true.copy()
        pA = None  # No learning
    else:
        # Initialize slightly informative prior (will learn)
        initial_diagonal_bias = 0.6
        pA = np.ones((2, 2)) * 2.0
        pA[0, 0] = pA[1, 1] = 2.0 * initial_diagonal_bias / 0.5
        pA[0, 1] = pA[1, 0] = 2.0 * (1 - initial_diagonal_bias) / 0.5
        A = normalize_A(pA.copy())
    
    # Transition matrix
    B = np.array([[0.95, 0.15], [0.05, 0.85]])
    
    max_sit = max(sit_indices)
    sit_data = {}
    
    print(f"Running {max_sit} sits to collect data for sits {sit_indices}...")
    print(f"  Noise period: t={noise_start}-{noise_end}")
    print(f"  Precision params: lr={zeta_step}, var={log_zeta_prior_var}")
    print(f"  Using {'TRUE' if use_true_A else 'LEARNED'} A matrix")
    
    for sit in range(max_sit):
        env = BreathEnv(seed=seed + sit)
        
        obs, beliefs, true_states, acc, mz, zeta_hist, is_noise = run_single_sit_with_precision_inference(
            env, A, B, T_per_sit,
            zeta_prior=1.0,
            log_zeta_prior_var=log_zeta_prior_var,
            zeta_step=zeta_step,
            noise_start=noise_start, noise_end=noise_end
        )
        
        # Store data for requested sits (1-indexed in user request)
        if (sit + 1) in sit_indices:
            sit_data[sit + 1] = {
                'zeta_history': np.array(zeta_hist),
                'is_noise': np.array(is_noise),
                'observations': np.array(obs),
                'beliefs': np.array([b[0] for b in beliefs]),  # P(inhale)
                'true_states': np.array(true_states),
                'accuracy': acc,
                'mean_zeta': mz,
                'A_diag': [A[0, 0], A[1, 1]],
            }
            print(f"  Sit {sit+1}: acc={acc:.3f}, mean_ζ={mz:.3f}, A=[{A[0,0]:.3f},{A[1,1]:.3f}]")
        
        # Update A for next sit (only if learning)
        if not use_true_A:
            pA, A = update_A_after_sit(pA, A, obs, beliefs, lr=lr, fr=fr)
    
    # Plot
    n_sits = len(sit_indices)
    fig, axes = plt.subplots(n_sits, 2, figsize=(14, 3 * n_sits), sharex='col')
    
    colors = {'blue': '#2563eb', 'orange': '#ea580c', 'gray': '#6b7280', 'red': '#dc2626'}
    
    for i, sit_num in enumerate(sit_indices):
        data = sit_data[sit_num]
        t_range = np.arange(len(data['zeta_history']))
        
        # Left panel: Breath inference
        ax = axes[i, 0]
        ax.plot(t_range, data['beliefs'], color=colors['blue'], linewidth=1.5, label='P(Inhaling)')
        true_binary = 1 - data['true_states']  # INHALE=0 -> 1.0
        ax.scatter(t_range, true_binary, s=10, color=colors['gray'], alpha=0.6, label='True')
        
        # Shade noise region
        if noise_start is not None:
            ax.axvspan(noise_start, noise_end, alpha=0.15, color=colors['red'], label='Noise')
        
        ax.set_ylabel(f"Sit {sit_num}")
        ax.set_ylim(-0.05, 1.05)
        if i == 0:
            ax.set_title("Breath Inference")
            ax.legend(loc='upper right', fontsize=8)
        if i == n_sits - 1:
            ax.set_xlabel("Time step")
        
        # Right panel: Precision dynamics
        ax = axes[i, 1]
        ax.plot(t_range, data['zeta_history'], color=colors['orange'], linewidth=1.5)
        ax.axhline(y=1.0, color=colors['gray'], linestyle='--', linewidth=1, alpha=0.6, label='Prior (ζ=1)')
        
        # Shade noise region
        if noise_start is not None:
            ax.axvspan(noise_start, noise_end, alpha=0.15, color=colors['red'], label='Noise')
        
        # Add text annotation
        mean_z = data['mean_zeta']
        ax.text(0.98, 0.95, f"mean ζ={mean_z:.3f}", transform=ax.transAxes,
                ha='right', va='top', fontsize=9, color=colors['gray'])
        
        ax.set_ylabel("Precision (ζ)")
        ax.set_ylim(0, 3.0)
        if i == 0:
            ax.set_title("Precision Dynamics")
            ax.legend(loc='upper right', fontsize=8)
        if i == n_sits - 1:
            ax.set_xlabel("Time step")
    
    a_label = "trueA" if use_true_A else "learnedA"
    plt.suptitle(f"Within-Sit Precision Dynamics (noise t={noise_start}-{noise_end}, {a_label})", fontsize=12)
    plt.tight_layout()
    
    if save_dir:
        save_path = os.path.join(save_dir, f"diagnostic_within_sit_noise_{noise_start}_{noise_end}_{a_label}.png")
        plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor='white')
        print(f"\nSaved: {save_path}")
    
    return fig, sit_data


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="A matrix learning across meditation sits")
    parser.add_argument("--num-sits", type=int, default=30, help="Number of meditation sits")
    parser.add_argument("--T-per-sit", type=int, default=200, help="Timesteps per sit")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--lr", type=float, default=1.0, help="Learning rate")
    parser.add_argument("--fr", type=float, default=1.0, help="Forgetting rate [0,1]")
    parser.add_argument("--zeta", type=float, default=1.0, help="Likelihood precision (attention)")
    parser.add_argument("--initial-pA", type=float, default=2.0, 
                        help="Initial pA pseudo-count strength")
    parser.add_argument("--save", action="store_true", help="Save plot")
    parser.add_argument("--sweep", action="store_true", help="Run lr/fr parameter sweep (fixed zeta)")
    parser.add_argument("--sweep-precision", action="store_true", help="Run lr/fr parameter sweep WITH precision inference")
    parser.add_argument("--sweep-precision-params", action="store_true", help="Sweep precision params (step, var) for A learning")
    parser.add_argument("--zeta-sweep", action="store_true", help="Run zeta (attention) comparison")
    parser.add_argument("--zeta-prior-sweep", action="store_true", help="Run zeta prior comparison with precision inference")
    parser.add_argument("--nonstationary", action="store_true", help="Run non-stationary environment comparison")
    parser.add_argument("--attention-sweep", action="store_true", help="Run attention comparison (Figure 0.3)")
    parser.add_argument("--gap-sweep", action="store_true", help="Run attention gap sweep (Figure 0.4)")
    parser.add_argument("--fixed-vs-dynamic", action="store_true", help="Compare fixed zeta=1 vs dynamic precision")
    parser.add_argument("--diagnostic", action="store_true", help="Show within-sit precision dynamics for specific sits")
    parser.add_argument("--use-true-A", action="store_true", help="Use true A matrix (no learning) for diagnostic")
    parser.add_argument("--noise-start", type=int, default=None, help="Start of noise period within each sit")
    parser.add_argument("--noise-end", type=int, default=None, help="End of noise period within each sit")
    parser.add_argument("--zeta-step", type=float, default=0.5, help="Zeta update step size (B.45)")
    parser.add_argument("--zeta-var", type=float, default=8.0, help="Log zeta prior variance")
    parser.add_argument("--p-correct", type=float, default=None, help="Override environment's p_correct (default 0.98)")
    args = parser.parse_args()
    
    here = os.path.dirname(__file__)
    outdir = os.path.join(here, "outputs")
    os.makedirs(outdir, exist_ok=True)
    
    if args.diagnostic:
        # Run within-sit diagnostic for specific sits
        fig, sit_data = run_within_sit_diagnostic(
            sit_indices=[1, 20, 60, 100],
            T_per_sit=args.T_per_sit,
            seed=args.seed,
            lr=args.lr,
            fr=args.fr,
            noise_start=args.noise_start,
            noise_end=args.noise_end,
            zeta_step=args.zeta_step,
            log_zeta_prior_var=args.zeta_var,
            save_dir=outdir if args.save else None,
            use_true_A=args.use_true_A,
        )
        if not args.save:
            plt.show()
    elif args.fixed_vs_dynamic:
        # Compare fixed zeta=1 vs dynamic precision
        fig, all_results = run_fixed_vs_dynamic_comparison(
            num_sits=args.num_sits,
            T_per_sit=args.T_per_sit,
            lr=args.lr,
            fr=args.fr,
            seed=args.seed,
            noise_start=args.noise_start,
            noise_end=args.noise_end,
            zeta_step=args.zeta_step,
            log_zeta_prior_var=args.zeta_var,
            save_dir=outdir if args.save else None,
        )
        if not args.save:
            plt.show()
    elif args.gap_sweep:
        # Run attention multiplier sweep (Figure 0.4) - multiplicative symmetry
        p_correct = args.p_correct if args.p_correct is not None else 0.98
        fig, all_results = run_attention_gap_sweep(
            multiplier_values=[1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0],
            num_sits=args.num_sits,
            T_per_sit=args.T_per_sit,
            lr=args.lr,
            fr=args.fr,
            seed=args.seed,
            save_dir=outdir if args.save else None,
            p_correct=p_correct,
        )
        if not args.save:
            plt.show()
    elif args.attention_sweep:
        # Run attention comparison (Figure 0.3)
        fig, all_results = run_attention_comparison(
            zeta_focused_values=[1.0, 1.25, 1.5, 2.0],
            zeta_distracted=0.8,
            num_sits=args.num_sits,
            T_per_sit=args.T_per_sit,
            lr=args.lr,
            fr=args.fr,
            seed=args.seed,
            save_dir=outdir if args.save else None,
        )
        if not args.save:
            plt.show()
    elif args.zeta_prior_sweep:
        # Run zeta prior comparison with precision inference (Figure 0.2)
        fig, all_results = run_zeta_prior_comparison(
            zeta_prior_values=[0.95, 1.0, 1.05, 1.1, 1.15],
            num_sits=args.num_sits,
            T_per_sit=args.T_per_sit,
            lr=args.lr,
            fr=args.fr,
            seed=args.seed,
            save_dir=outdir if args.save else None,
            p_correct=args.p_correct,
        )
        if not args.save:
            plt.show()
    elif args.nonstationary:
        # Run non-stationary environment comparison
        # Use --p-correct to set phase2 value (default 0.9)
        phase2 = args.p_correct if args.p_correct is not None else 0.90
        fig, all_results = run_nonstationary_comparison(
            zeta_prior_values=[0.95, 1.0, 1.05, 1.1, 1.15],
            num_sits=args.num_sits,
            T_per_sit=args.T_per_sit,
            lr=args.lr,
            fr=args.fr,
            seed=args.seed,
            p_correct_phase1=0.98,
            p_correct_phase2=phase2,
            switch_sit=args.num_sits // 2,
            save_dir=outdir if args.save else None,
        )
        if not args.save:
            plt.show()
    elif args.zeta_sweep:
        # Run zeta comparison - fine sweep near 1.0 to find optimal
        fig, all_results = run_zeta_comparison(
            zeta_values=[0.95, 1.0, 1.05, 1.1, 1.15],
            num_sits=args.num_sits,
            T_per_sit=args.T_per_sit,
            lr=args.lr,
            fr=args.fr,
            seed=args.seed,
            save_dir=outdir if args.save else None,
            p_correct=args.p_correct,
        )
        if not args.save:
            plt.show()
    elif args.sweep:
        # Run parameter sweep - finer sweep around fr=0.9
        fig, all_results = run_parameter_sweep(
            lr_values=[1.0],  # lr doesn't matter much, fix at 1.0
            fr_values=[1.0, 0.95, 0.92, 0.90, 0.88, 0.85],
            num_sits=args.num_sits,
            T_per_sit=args.T_per_sit,
            seed=args.seed,
            save_dir=outdir if args.save else None,
        )
        if not args.save:
            plt.show()
    
    elif args.sweep_precision:
        # Run parameter sweep WITH precision inference
        fig, all_results = run_parameter_sweep_with_precision(
            lr_values=[0.5, 1.0, 2.0],
            fr_values=[1.0, 0.95, 0.9, 0.85, 0.8],
            num_sits=args.num_sits,
            T_per_sit=args.T_per_sit,
            seed=args.seed,
            zeta_prior=1.0,
            zeta_step=args.zeta_step,
            log_zeta_prior_var=args.zeta_var,
            save_dir=outdir if args.save else None,
        )
        if not args.save:
            plt.show()
    
    elif args.sweep_precision_params:
        # Sweep precision parameters to find best for A learning
        fig1, fig2, all_results, baseline = run_precision_param_sweep_for_A_learning(
            zeta_step_values=[0.1, 0.25, 0.5, 1.0],
            var_values=[0.5, 1.0, 2.0, 4.0, 8.0],
            num_sits=args.num_sits,
            T_per_sit=args.T_per_sit,
            seed=args.seed,
            lr=args.lr,
            fr=args.fr,
            save_dir=outdir if args.save else None,
        )
        if not args.save:
            plt.show()
    else:
        # Single run
        results = run_A_learning(
            num_sits=args.num_sits,
            T_per_sit=args.T_per_sit,
            seed=args.seed,
            lr=args.lr,
            fr=args.fr,
            zeta=args.zeta,
            initial_pA_strength=args.initial_pA,
        )
        
        filename = f"figure0_A_learning_lr{args.lr}_fr{args.fr}_zeta{args.zeta}.png"
        save_path = os.path.join(outdir, filename) if args.save else None
        
        fig = plot_figure0(results, save_path=save_path)
        
        if not args.save:
            plt.show()

