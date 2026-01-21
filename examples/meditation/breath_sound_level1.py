#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Level 1: Dual Perceptual Inference (Breath + Sound)

Two parallel perceptual modalities, each with dynamic likelihood precision (B.45):
    - Breath: Oscillating interoceptive state (INHALE/EXHALE)
    - Sound: Stable exteroceptive state with surprise (QUIET/SOUND_PRESENT)

This establishes the foundation for hierarchical attention control.
At this stage, both modalities run with full precision (no attention modulation yet).

Figure 2: Dual modality perception with precision dynamics
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pymdp.envs import BreathEnv
from pymdp.maths import EPS_VAL


# =============================================================================
# Constants
# =============================================================================

# Breath states
INHALE = 0
EXHALE = 1

# Sound states
QUIET = 0
SOUND_PRESENT = 1


# =============================================================================
# Helper Functions
# =============================================================================

def scale_likelihood(A_base: np.ndarray, zeta: float) -> np.ndarray:
    """Scale likelihood matrix with precision parameter ζ."""
    log_A = np.log(A_base + EPS_VAL) * zeta
    A_scaled = np.exp(log_A)
    return A_scaled / A_scaled.sum(axis=0, keepdims=True)


def compute_pe(A, obs, qs_prior):
    """Compute squared prediction error."""
    expected_obs = A @ qs_prior
    actual_obs = np.zeros(A.shape[0])
    actual_obs[int(obs)] = 1.0
    return np.sum((actual_obs - expected_obs) ** 2)


def update_precision_B45(zeta, A, obs, qs_prior,
                          log_zeta_prior_mean=0.0, log_zeta_prior_var=4.0,
                          lr=1.5, min_zeta=0.1, max_zeta=5.0):
    """
    Update likelihood precision via B.45.
    
    Precision increases when predictions are accurate (low PE),
    decreases when predictions are poor (high PE).
    """
    log_zeta = np.log(zeta + EPS_VAL)
    pe = compute_pe(A, obs, qs_prior)
    uniform = np.ones(len(qs_prior)) / len(qs_prior)
    baseline_pe = compute_pe(A, obs, uniform)
    precision_weighted_error = zeta * pe
    expected_error = np.exp(log_zeta_prior_mean) * baseline_pe
    error_drive = 0.5 * (expected_error - precision_weighted_error)
    prior_term = (log_zeta - log_zeta_prior_mean) / log_zeta_prior_var
    log_zeta_new = log_zeta + lr * (error_drive - prior_term)
    zeta_new = np.exp(log_zeta_new)
    return np.clip(zeta_new, min_zeta, max_zeta), pe


def bayesian_update(A, obs, prior):
    """Simple Bayesian state inference."""
    likelihood = A[obs, :]
    posterior = likelihood * prior
    return posterior / posterior.sum()


# =============================================================================
# Sound Environment
# =============================================================================

class SoundEnv:
    """
    Simple sound environment with surprise.
    
    States: QUIET (0), SOUND_PRESENT (1)
    Stays quiet until t_surprise, then sound appears.
    """
    
    def __init__(self, t_surprise=75, sound_duration=30, p_correct=0.95, seed=None):
        self.t_surprise = t_surprise
        self.sound_duration = sound_duration
        self.p_correct = p_correct
        self.rng = np.random.default_rng(seed)
        self.state = QUIET
        self.t = 0
        
    def reset(self):
        self.state = QUIET
        self.t = 0
        return self._get_observation()
    
    def step(self):
        self.t += 1
        if self.t_surprise <= self.t < self.t_surprise + self.sound_duration:
            self.state = SOUND_PRESENT
        else:
            self.state = QUIET
        return self._get_observation()
    
    def _get_observation(self):
        if self.rng.random() < self.p_correct:
            return self.state
        else:
            return 1 - self.state


# =============================================================================
# Main Simulation
# =============================================================================

def run_dual_perception(
    T: int = 200,
    seed: int = 42,
    t_surprise: int = 75,
    sound_duration: int = 30,
):
    """
    Run dual perceptual inference (breath + sound) with dynamic precision.
    
    Both modalities run independently with their own precision dynamics.
    """
    np.random.seed(seed)
    
    # =========================================================================
    # Build Breath Environment & Model
    # =========================================================================
    env_breath = BreathEnv(seed=seed)
    
    p_correct_breath = float(env_breath.p_correct)
    A_breath = np.array([
        [p_correct_breath, 1 - p_correct_breath],
        [1 - p_correct_breath, p_correct_breath]
    ])
    
    stay_p_inhale = 1.0 - 1.0 / max(1, np.mean(env_breath.inhale_range))
    stay_p_exhale = 1.0 - 1.0 / max(1, np.mean(env_breath.exhale_range))
    B_breath = np.array([
        [stay_p_inhale, 1 - stay_p_exhale],
        [1 - stay_p_inhale, stay_p_exhale]
    ])
    
    # =========================================================================
    # Build Sound Environment & Model
    # =========================================================================
    env_sound = SoundEnv(t_surprise=t_surprise, sound_duration=sound_duration, seed=seed+1)
    
    p_correct_sound = 0.95
    A_sound = np.array([
        [p_correct_sound, 1 - p_correct_sound],
        [1 - p_correct_sound, p_correct_sound]
    ])
    
    # Sound mostly stays in current state
    B_sound = np.array([
        [0.98, 0.2],   # to QUIET
        [0.02, 0.8]    # to SOUND_PRESENT
    ])
    
    # =========================================================================
    # Data Logs
    # =========================================================================
    # Breath
    true_breath = np.zeros(T, dtype=int)
    posterior_breath = np.zeros((T, 2))
    zeta_breath = np.zeros(T)
    
    # Sound
    true_sound = np.zeros(T, dtype=int)
    posterior_sound = np.zeros((T, 2))
    zeta_sound = np.zeros(T)
    
    # =========================================================================
    # Initialize
    # =========================================================================
    # Breath
    prior_breath = np.array([0.5, 0.5])
    z_breath = 1.0
    obs_breath = int(env_breath.reset())
    
    # Sound
    prior_sound = np.array([0.95, 0.05])  # Expect quiet initially
    z_sound = 1.0
    obs_sound = int(env_sound.reset())
    
    print(f"Running dual perception simulation:")
    print(f"  T={T}, t_surprise={t_surprise}, sound_duration={sound_duration}")
    print()
    
    # =========================================================================
    # Main Loop
    # =========================================================================
    for t in range(T):
        # Store true states
        true_breath[t] = env_breath.state
        true_sound[t] = env_sound.state
        
        # --- Breath perception ---
        # 1. Update precision (before inference)
        z_breath, pe_b = update_precision_B45(z_breath, A_breath, obs_breath, prior_breath)
        zeta_breath[t] = z_breath
        
        # 2. State inference
        A_b_scaled = scale_likelihood(A_breath, z_breath)
        post_breath = bayesian_update(A_b_scaled, obs_breath, prior_breath)
        posterior_breath[t] = post_breath
        
        # 3. Prior for next timestep
        prior_breath = B_breath @ post_breath
        
        # --- Sound perception ---
        # 1. Update precision (before inference)
        z_sound, pe_s = update_precision_B45(z_sound, A_sound, obs_sound, prior_sound)
        zeta_sound[t] = z_sound
        
        # 2. State inference
        A_s_scaled = scale_likelihood(A_sound, z_sound)
        post_sound = bayesian_update(A_s_scaled, obs_sound, prior_sound)
        posterior_sound[t] = post_sound
        
        # 3. Prior for next timestep
        prior_sound = B_sound @ post_sound
        
        # --- Advance environments ---
        obs_breath = int(env_breath.step(None))
        obs_sound = int(env_sound.step())
    
    # =========================================================================
    # Summary
    # =========================================================================
    breath_acc = (np.argmax(posterior_breath, axis=1) == true_breath).mean()
    sound_acc = (np.argmax(posterior_sound, axis=1) == true_sound).mean()
    
    print("Results:")
    print(f"  Breath inference accuracy: {breath_acc:.3f}")
    print(f"  Sound inference accuracy: {sound_acc:.3f}")
    print(f"  Mean breath precision: {zeta_breath.mean():.3f}")
    print(f"  Mean sound precision: {zeta_sound.mean():.3f}")
    
    return {
        "true_breath": true_breath,
        "posterior_breath": posterior_breath,
        "zeta_breath": zeta_breath,
        "true_sound": true_sound,
        "posterior_sound": posterior_sound,
        "zeta_sound": zeta_sound,
        "t_surprise": t_surprise,
        "sound_duration": sound_duration,
    }


# =============================================================================
# Plotting - Figure 2
# =============================================================================

def plot_figure2(results: dict, save_path: str = None):
    """
    Generate Figure 2: Dual Modality Perception with Dynamic Precision.
    
    Four-panel figure:
    A) Breath state inference
    B) Breath precision (ζ_breath)
    C) Sound state inference
    D) Sound precision (ζ_sound)
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
    
    T = len(results["true_breath"])
    t_range = np.arange(T)
    t_surprise = results["t_surprise"]
    sound_duration = results["sound_duration"]
    
    # Colors
    BLUE = '#2563eb'
    ORANGE = '#ea580c'
    GREEN = '#16a34a'
    TEAL = '#0d9488'
    GRAY = '#6b7280'
    RED = '#dc2626'
    
    fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True,
                              gridspec_kw={'height_ratios': [1, 0.8, 1, 0.8], 'hspace': 0.2})
    
    def shade_surprise(ax):
        ax.axvspan(t_surprise, t_surprise + sound_duration, 
                   alpha=0.15, color=RED)
    
    # =========================================================================
    # Panel A: Breath State Inference
    # =========================================================================
    ax = axes[0]
    p_inhale = results["posterior_breath"][:, INHALE]
    ax.plot(t_range, p_inhale, color=BLUE, linewidth=1.5, label='P(Inhaling)')
    true_breath_binary = 1.0 - results["true_breath"]  # INHALE=0 → 1.0
    ax.scatter(t_range, true_breath_binary, s=12, color=GRAY, alpha=0.5)
    shade_surprise(ax)
    ax.set_ylabel("Probability")
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks([0, 0.5, 1])
    legend = [Line2D([0], [0], color=BLUE, linewidth=1.5, label='P(Inhaling)'),
              Line2D([0], [0], marker='o', color='w', markerfacecolor=GRAY, 
                     markersize=5, label='True state', alpha=0.5)]
    ax.legend(handles=legend, loc='upper right', framealpha=0.9)
    ax.text(-0.06, 1.05, 'A', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')
    ax.set_title("Breath Perception", fontsize=12)
    
    # =========================================================================
    # Panel B: Breath Precision
    # =========================================================================
    ax = axes[1]
    ax.plot(t_range, results["zeta_breath"], color=ORANGE, linewidth=1.5)
    ax.axhline(y=1.0, color=GRAY, linestyle='--', linewidth=1, alpha=0.6, label='Prior mean')
    shade_surprise(ax)
    ax.set_ylabel("Precision (ζ)")
    ax.set_ylim(0, 2.5)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.text(-0.06, 1.05, 'B', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')
    
    # =========================================================================
    # Panel C: Sound State Inference
    # =========================================================================
    ax = axes[2]
    p_sound = results["posterior_sound"][:, SOUND_PRESENT]
    ax.plot(t_range, p_sound, color=GREEN, linewidth=1.5, label='P(Sound present)')
    true_sound = results["true_sound"]
    ax.scatter(t_range, true_sound, s=12, color=GRAY, alpha=0.5)
    shade_surprise(ax)
    ax.set_ylabel("Probability")
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks([0, 0.5, 1])
    legend = [Line2D([0], [0], color=GREEN, linewidth=1.5, label='P(Sound present)'),
              Line2D([0], [0], marker='o', color='w', markerfacecolor=GRAY, 
                     markersize=5, label='True state', alpha=0.5)]
    ax.legend(handles=legend, loc='upper right', framealpha=0.9)
    ax.text(-0.06, 1.05, 'C', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')
    ax.set_title("Sound Perception", fontsize=12)
    
    # =========================================================================
    # Panel D: Sound Precision
    # =========================================================================
    ax = axes[3]
    ax.plot(t_range, results["zeta_sound"], color=TEAL, linewidth=1.5)
    ax.axhline(y=1.0, color=GRAY, linestyle='--', linewidth=1, alpha=0.6, label='Prior mean')
    shade_surprise(ax)
    ax.set_ylabel("Precision (ζ)")
    ax.set_xlabel("Time step")
    ax.set_ylim(0, 2.5)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.text(-0.06, 1.05, 'D', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')
    
    fig.align_ylabels(axes)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor='white')
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
    
    parser = argparse.ArgumentParser(description="Dual perceptual inference (breath + sound)")
    parser.add_argument("--T", type=int, default=200, help="Number of timesteps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--t-surprise", type=int, default=75, help="Timestep of surprising sound")
    parser.add_argument("--sound-duration", type=int, default=30, help="Duration of sound")
    parser.add_argument("--save", action="store_true", help="Save plot")
    args = parser.parse_args()
    
    results = run_dual_perception(
        T=args.T,
        seed=args.seed,
        t_surprise=args.t_surprise,
        sound_duration=args.sound_duration,
    )
    
    here = os.path.dirname(__file__)
    outdir = os.path.join(here, "outputs")
    os.makedirs(outdir, exist_ok=True)
    
    filename = "figure2_dual_perception.png"
    save_path = os.path.join(outdir, filename) if args.save else None
    
    fig = plot_figure2(results, save_path=save_path)
    
    if not args.save:
        plt.show()

