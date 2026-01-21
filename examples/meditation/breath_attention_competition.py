#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Hierarchical active inference model of attention competition during meditation.

Architecture:
    Level 1a: Breath perception (interoceptive)
        - States: INHALE, EXHALE
        - Precision ζ_breath modulated by attention
        
    Level 1b: Sound perception (exteroceptive)
        - States: QUIET, SOUND_PRESENT
        - Precision ζ_sound modulated by attention
        
    Level 2: Attentional control (mental action)
        - States: ATTEND_BREATH, ATTEND_SOUND
        - Actions: MAINTAIN, SWITCH
        - Policy selection via EFE roll-out to Level 1

Key mechanism: Attention allocates a precision bonus to one modality.
When a surprising sound occurs, the epistemic value of attending sound
increases (uncertainty to resolve), pulling attention away from breath.

References:
    - Sandved-Smith et al. (2021) "Towards a computational phenomenology 
      of mental action: modelling meta-awareness and attentional control 
      with deep parametric active inference"
    - Parr et al. (2022) "Active Inference" Appendix B
"""

import numpy as np
from pymdp.envs import BreathEnv
from pymdp.agent import Agent
from pymdp import utils
from pymdp.maths import EPS_VAL, softmax


# =============================================================================
# Constants
# =============================================================================

# Level 1a: Breath states
INHALE = 0
EXHALE = 1

# Level 1b: Sound states
QUIET = 0
SOUND_PRESENT = 1

# Level 2: Attentional states
ATTEND_BREATH = 0
ATTEND_SOUND = 1

# Level 2: Actions
MAINTAIN = 0
SWITCH = 1

# Precision allocation parameters
ZETA_BASE = 0.3      # Minimum precision (unattended modality)
ZETA_BONUS = 1.5     # Additional precision when attended


# =============================================================================
# Helper Functions
# =============================================================================

def scale_likelihood(A_base: np.ndarray, zeta: float) -> np.ndarray:
    """Scale likelihood matrix with precision parameter ζ."""
    log_A = np.log(A_base + EPS_VAL) * zeta
    A_scaled = np.exp(log_A)
    return A_scaled / A_scaled.sum(axis=0, keepdims=True)


def compute_entropy(qs: np.ndarray) -> float:
    """Compute entropy of a probability distribution."""
    return -np.sum(qs * np.log(qs + EPS_VAL))


def compute_pe(A, obs, qs_prior):
    """Compute squared prediction error."""
    expected_obs = A @ qs_prior
    actual_obs = np.zeros(A.shape[0])
    actual_obs[int(obs)] = 1.0
    return np.sum((actual_obs - expected_obs) ** 2)


def update_precision_B45(zeta, A, obs, qs_prior,
                          log_zeta_prior_mean=0.0, log_zeta_prior_var=4.0,
                          lr=1.5, min_zeta=0.1, max_zeta=5.0):
    """Update likelihood precision via B.45."""
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


def allocate_precision(attention_posterior):
    """
    Allocate precision to modalities based on attention posterior.
    
    Returns (ζ_breath, ζ_sound)
    """
    P_breath = attention_posterior[ATTEND_BREATH]
    P_sound = attention_posterior[ATTEND_SOUND]
    
    zeta_breath = ZETA_BASE + ZETA_BONUS * P_breath
    zeta_sound = ZETA_BASE + ZETA_BONUS * P_sound
    
    return zeta_breath, zeta_sound


# =============================================================================
# Level 1 EFE Computation
# =============================================================================

def compute_L1_EFE(qs_prior, qs_posterior, A, B):
    """
    Compute expected free energy for a perceptual modality.
    
    G = ambiguity - epistemic_value
    
    Key insight: Epistemic value considers PREDICTED entropy (what the 
    agent expects to be uncertain about after transition). This captures
    ongoing uncertainty in dynamic modalities like breath.
    
    Lower EFE = better (more informative, less ambiguous)
    """
    # Predict next state distribution
    qs_predicted = B @ qs_posterior
    
    # Ambiguity: expected entropy of observations given state
    ambiguity = 0.0
    for s in range(len(qs_predicted)):
        p_o_given_s = A[:, s]
        H_o_given_s = -np.sum(p_o_given_s * np.log(p_o_given_s + EPS_VAL))
        ambiguity += qs_predicted[s] * H_o_given_s
    
    # PREDICTED entropy: uncertainty about hidden state at NEXT timestep
    # For oscillating modalities (breath), this captures ongoing uncertainty
    # For stable modalities (quiet sound), this stays low
    predicted_entropy = compute_entropy(qs_predicted)
    
    # Also consider current entropy (for immediate uncertainty)
    current_entropy = compute_entropy(qs_posterior)
    
    # Combined uncertainty: average of current and predicted
    # This smooths out instantaneous fluctuations
    combined_entropy = 0.5 * current_entropy + 0.5 * predicted_entropy
    
    # Sensing quality: how informative is this precision level?
    sensing_quality = 1.0 - ambiguity / np.log(A.shape[0])
    
    # Epistemic value: combined uncertainty × ability to resolve it
    epistemic_value = combined_entropy * sensing_quality
    
    return ambiguity - epistemic_value


def compute_L2_policy_EFE(
    action,
    attention_posterior,
    prior_breath, prior_sound,
    posterior_breath, posterior_sound,
    A_breath_base, A_sound_base,
    B_breath, B_sound,
    B_attention,
    breath_preference=0.3
):
    """
    Compute EFE for a Level 2 action by rolling out to Level 1.
    
    EFE_L2(action) = EFE_L1_breath + EFE_L1_sound + pragmatic_term
    
    The pragmatic term encodes a preference for attending breath (the 
    meditation goal). This creates a baseline pull toward breath that
    must be overcome by epistemic value from sound.
    
    Parameters
    ----------
    breath_preference : float
        Strength of preference for attending breath. Higher values make
        it harder for sound to capture attention.
    """
    # 1. Project attention state after this action
    projected_attention = B_attention[:, :, action] @ attention_posterior
    
    # 2. Compute precision allocation under projected attention
    zeta_breath, zeta_sound = allocate_precision(projected_attention)
    
    # 3. Scale likelihoods
    A_breath_scaled = scale_likelihood(A_breath_base, zeta_breath)
    A_sound_scaled = scale_likelihood(A_sound_base, zeta_sound)
    
    # 4. Compute EFE at Level 1 for each modality (epistemic term)
    EFE_breath = compute_L1_EFE(prior_breath, posterior_breath, A_breath_scaled, B_breath)
    EFE_sound = compute_L1_EFE(prior_sound, posterior_sound, A_sound_scaled, B_sound)
    
    # 5. Pragmatic term: preference for attending breath
    # projected_attention[0] = P(ATTEND_BREATH) after this action
    # Higher P(ATTEND_BREATH) → lower pragmatic cost
    pragmatic_cost = -breath_preference * projected_attention[ATTEND_BREATH]
    
    # 6. Total EFE = epistemic + pragmatic
    return EFE_breath + EFE_sound + pragmatic_cost


# =============================================================================
# Sound Environment
# =============================================================================

class SoundEnv:
    """
    Simple sound environment with surprise injection.
    
    States: QUIET (0), SOUND_PRESENT (1)
    Most of the time stays QUIET, with surprise at t_surprise.
    
    Observations are reliable (p_correct=0.99) to ensure the agent
    has clear evidence about sound state.
    """
    
    def __init__(self, t_surprise=75, sound_duration=30, p_correct=0.99, seed=None):
        """
        Parameters
        ----------
        t_surprise : int
            Timestep when surprising sound occurs
        sound_duration : int
            How long the sound lasts
        p_correct : float
            Observation accuracy (high by default for clear evidence)
        seed : int
            Random seed
        """
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
        
        # Deterministic state transition based on time
        if self.t_surprise <= self.t < self.t_surprise + self.sound_duration:
            self.state = SOUND_PRESENT
        else:
            self.state = QUIET
            
        return self._get_observation()
    
    def _get_observation(self):
        """Generate observation of sound state (mostly reliable)."""
        if self.rng.random() < self.p_correct:
            return self.state
        else:
            return 1 - self.state


# =============================================================================
# Main Simulation
# =============================================================================

def run_attention_competition(
    T: int = 200,
    seed: int = 42,
    t_surprise: int = 75,
    sound_duration: int = 30,
    action_temperature: float = 1.0,
    breath_preference: float = 0.3,
):
    """
    Run the attention competition simulation.
    
    Parameters
    ----------
    T : int
        Total timesteps
    seed : int
        Random seed
    t_surprise : int
        When surprising sound occurs
    sound_duration : int
        How long sound lasts
    action_temperature : float
        Temperature for action selection softmax
    """
    np.random.seed(seed)
    
    # =========================================================================
    # Build environments
    # =========================================================================
    env_breath = BreathEnv(seed=seed)
    env_sound = SoundEnv(t_surprise=t_surprise, sound_duration=sound_duration, seed=seed+1)
    
    # =========================================================================
    # Build Level 1a: Breath perception
    # =========================================================================
    p_correct_breath = float(env_breath.p_correct)
    A_breath_base = np.array([
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
    # Build Level 1b: Sound perception
    # =========================================================================
    # Use fixed high accuracy for sound (reliable sensing)
    p_correct_sound = 0.95
    A_sound_base = np.array([
        [p_correct_sound, 1 - p_correct_sound],
        [1 - p_correct_sound, p_correct_sound]
    ])
    
    # Sound transitions: mostly stays in current state
    B_sound = np.array([
        [0.95, 0.3],   # to QUIET
        [0.05, 0.7]    # to SOUND_PRESENT
    ])
    
    # =========================================================================
    # Build Level 2: Attentional control
    # =========================================================================
    # A_attention: P(obs | attention_state)
    # Observations: [breath_precision_obs, sound_precision_obs]
    # For simplicity, we'll handle this in the loop based on actual precision
    
    # B_attention: Transition dynamics for attention under actions
    # Shape: (new_state, old_state, action)
    B_attention = np.zeros((2, 2, 2))
    
    # MAINTAIN action: deterministically stay in current state
    B_attention[:, :, MAINTAIN] = np.array([
        [1.0, 0.0],   # to ATTEND_BREATH
        [0.0, 1.0]    # to ATTEND_SOUND
    ])
    
    # SWITCH action: deterministically move to other state
    B_attention[:, :, SWITCH] = np.array([
        [0.0, 1.0],    # to ATTEND_BREATH
        [1.0, 0.0]     # to ATTEND_SOUND
    ])
    
    # =========================================================================
    # Data logs
    # =========================================================================
    # Level 1a: Breath
    true_breath_states = np.zeros(T, dtype=int)
    breath_posteriors = np.zeros((T, 2))
    zeta_breath_history = np.zeros(T)
    
    # Level 1b: Sound
    true_sound_states = np.zeros(T, dtype=int)
    sound_posteriors = np.zeros((T, 2))
    zeta_sound_history = np.zeros(T)
    
    # Level 2: Attention
    attention_posteriors = np.zeros((T, 2))
    actions_taken = np.zeros(T, dtype=int)
    EFE_maintain_history = np.zeros(T)
    EFE_switch_history = np.zeros(T)
    
    # =========================================================================
    # Initialize
    # =========================================================================
    # Attention starts on breath
    attention_posterior = np.array([0.9, 0.1])  # [P(ATTEND_BREATH), P(ATTEND_SOUND)]
    
    # Level 1 priors
    prior_breath = np.array([0.5, 0.5])
    prior_sound = np.array([0.9, 0.1])  # Expect quiet initially
    
    # Level 1 posteriors (start same as priors)
    posterior_breath = prior_breath.copy()
    posterior_sound = prior_sound.copy()
    
    # Get initial observations
    obs_breath = int(env_breath.reset())
    obs_sound = int(env_sound.reset())
    
    # Initial precision allocation
    zeta_breath, zeta_sound = allocate_precision(attention_posterior)
    
    print(f"Running attention competition simulation:")
    print(f"  T={T}, t_surprise={t_surprise}, sound_duration={sound_duration}")
    print(f"  Precision: base={ZETA_BASE}, bonus={ZETA_BONUS}")
    print()
    
    # =========================================================================
    # Main loop
    # =========================================================================
    for t in range(T):
        # Store true states
        true_breath_states[t] = env_breath.state
        true_sound_states[t] = env_sound.state
        
        # ---------------------------------------------------------------------
        # LEVEL 2: Attention control with EFE roll-out
        # ---------------------------------------------------------------------
        
        # Compute EFE for each action
        # Uses current posteriors to evaluate epistemic value
        EFE_maintain = compute_L2_policy_EFE(
            action=MAINTAIN,
            attention_posterior=attention_posterior,
            prior_breath=prior_breath,
            prior_sound=prior_sound,
            posterior_breath=posterior_breath,
            posterior_sound=posterior_sound,
            A_breath_base=A_breath_base,
            A_sound_base=A_sound_base,
            B_breath=B_breath,
            B_sound=B_sound,
            B_attention=B_attention,
            breath_preference=breath_preference
        )
        
        EFE_switch = compute_L2_policy_EFE(
            action=SWITCH,
            attention_posterior=attention_posterior,
            prior_breath=prior_breath,
            prior_sound=prior_sound,
            posterior_breath=posterior_breath,
            posterior_sound=posterior_sound,
            A_breath_base=A_breath_base,
            A_sound_base=A_sound_base,
            B_breath=B_breath,
            B_sound=B_sound,
            B_attention=B_attention,
            breath_preference=breath_preference
        )
        
        EFE_maintain_history[t] = EFE_maintain
        EFE_switch_history[t] = EFE_switch
        
        # Select action (softmax over negative EFE)
        action_logits = -np.array([EFE_maintain, EFE_switch]) / action_temperature
        action_probs = softmax(action_logits)
        action = np.random.choice([MAINTAIN, SWITCH], p=action_probs)
        actions_taken[t] = action
        
        # Update attention posterior based on action
        attention_posterior = B_attention[:, :, action] @ attention_posterior
        attention_posterior = attention_posterior / attention_posterior.sum()  # Normalize
        attention_posteriors[t] = attention_posterior
        
        # ---------------------------------------------------------------------
        # LEVEL 1: Perceptual inference (both modalities)
        # ---------------------------------------------------------------------
        
        # Precision allocation from current attention
        zeta_breath, zeta_sound = allocate_precision(attention_posterior)
        zeta_breath_history[t] = zeta_breath
        zeta_sound_history[t] = zeta_sound
        
        # --- Breath perception ---
        # Update precision via B.45
        zeta_breath_updated, pe_breath = update_precision_B45(
            zeta_breath, A_breath_base, obs_breath, prior_breath
        )
        
        # State inference with current precision
        A_breath_scaled = scale_likelihood(A_breath_base, zeta_breath)
        # Simple Bayesian update
        likelihood_breath = A_breath_scaled[obs_breath, :]
        new_posterior_breath = likelihood_breath * prior_breath
        new_posterior_breath = new_posterior_breath / new_posterior_breath.sum()
        breath_posteriors[t] = new_posterior_breath
        
        # Update for next iteration
        prior_breath = B_breath @ new_posterior_breath
        posterior_breath = new_posterior_breath  # For next EFE computation
        
        # --- Sound perception ---
        # Update precision via B.45
        zeta_sound_updated, pe_sound = update_precision_B45(
            zeta_sound, A_sound_base, obs_sound, prior_sound
        )
        
        # State inference with current precision
        A_sound_scaled = scale_likelihood(A_sound_base, zeta_sound)
        likelihood_sound = A_sound_scaled[obs_sound, :]
        new_posterior_sound = likelihood_sound * prior_sound
        new_posterior_sound = new_posterior_sound / new_posterior_sound.sum()
        sound_posteriors[t] = new_posterior_sound
        
        # Update for next iteration
        prior_sound = B_sound @ new_posterior_sound
        posterior_sound = new_posterior_sound  # For next EFE computation
        
        # ---------------------------------------------------------------------
        # ENVIRONMENT: Advance both modalities
        # ---------------------------------------------------------------------
        obs_breath = int(env_breath.step(None))
        obs_sound = int(env_sound.step())
    
    # =========================================================================
    # Print summary
    # =========================================================================
    print("Results:")
    breath_acc = (np.argmax(breath_posteriors, axis=1) == true_breath_states).mean()
    sound_acc = (np.argmax(sound_posteriors, axis=1) == true_sound_states).mean()
    print(f"  Breath inference accuracy: {breath_acc:.3f}")
    print(f"  Sound inference accuracy: {sound_acc:.3f}")
    print(f"  Mean P(ATTEND_BREATH): {attention_posteriors[:, ATTEND_BREATH].mean():.3f}")
    print(f"  Switch actions: {(actions_taken == SWITCH).sum()}")
    
    return {
        # Level 1a: Breath
        "true_breath_states": true_breath_states,
        "breath_posteriors": breath_posteriors,
        "zeta_breath": zeta_breath_history,
        # Level 1b: Sound
        "true_sound_states": true_sound_states,
        "sound_posteriors": sound_posteriors,
        "zeta_sound": zeta_sound_history,
        # Level 2: Attention
        "attention_posteriors": attention_posteriors,
        "actions": actions_taken,
        "EFE_maintain": EFE_maintain_history,
        "EFE_switch": EFE_switch_history,
        # Metadata
        "t_surprise": t_surprise,
        "sound_duration": sound_duration,
    }


# =============================================================================
# Plotting
# =============================================================================

def plot_attention_competition(results: dict, save_path: str = None):
    """Plot results from attention competition simulation."""
    import matplotlib.pyplot as plt
    
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })
    
    T = len(results["true_breath_states"])
    t_range = np.arange(T)
    t_surprise = results["t_surprise"]
    sound_duration = results["sound_duration"]
    
    fig, axes = plt.subplots(5, 1, figsize=(12, 12), sharex=True,
                              gridspec_kw={'hspace': 0.3})
    
    # Colors
    BLUE = '#2563eb'
    ORANGE = '#ea580c'
    PURPLE = '#7c3aed'
    GREEN = '#16a34a'
    RED = '#dc2626'
    GRAY = '#6b7280'
    
    # Surprise period shading
    def shade_surprise(ax):
        ax.axvspan(t_surprise, t_surprise + sound_duration, 
                   alpha=0.15, color=RED, label='Sound present')
    
    # =========================================================================
    # Panel A: Breath perception
    # =========================================================================
    ax = axes[0]
    p_inhale = results["breath_posteriors"][:, INHALE]
    ax.plot(t_range, p_inhale, color=BLUE, linewidth=1.5, label='P(Inhaling)')
    true_breath = 1.0 - results["true_breath_states"]
    ax.scatter(t_range, true_breath, s=8, color=GRAY, alpha=0.5)
    shade_surprise(ax)
    ax.set_ylabel("Probability")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Level 1a: Breath Perception", fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.text(-0.05, 1.05, 'A', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')
    
    # =========================================================================
    # Panel B: Sound perception
    # =========================================================================
    ax = axes[1]
    p_sound = results["sound_posteriors"][:, SOUND_PRESENT]
    ax.plot(t_range, p_sound, color=GREEN, linewidth=1.5, label='P(Sound present)')
    true_sound = results["true_sound_states"]
    ax.scatter(t_range, true_sound, s=8, color=GRAY, alpha=0.5)
    shade_surprise(ax)
    ax.set_ylabel("Probability")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Level 1b: Sound Perception", fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.text(-0.05, 1.05, 'B', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')
    
    # =========================================================================
    # Panel C: Precision allocation
    # =========================================================================
    ax = axes[2]
    ax.plot(t_range, results["zeta_breath"], color=BLUE, linewidth=1.5, 
            label='ζ_breath', alpha=0.8)
    ax.plot(t_range, results["zeta_sound"], color=GREEN, linewidth=1.5, 
            label='ζ_sound', alpha=0.8)
    ax.axhline(y=ZETA_BASE, color=GRAY, linestyle=':', alpha=0.5, label=f'ζ_base={ZETA_BASE}')
    ax.axhline(y=ZETA_BASE + ZETA_BONUS, color=GRAY, linestyle='--', alpha=0.5)
    shade_surprise(ax)
    ax.set_ylabel("Precision (ζ)")
    ax.set_ylim(0, 2.2)
    ax.set_title("Precision Allocation (from attention)", fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.text(-0.05, 1.05, 'C', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')
    
    # =========================================================================
    # Panel D: Attention posterior
    # =========================================================================
    ax = axes[3]
    p_attend_breath = results["attention_posteriors"][:, ATTEND_BREATH]
    ax.plot(t_range, p_attend_breath, color=PURPLE, linewidth=2, label='P(Attend breath)')
    ax.fill_between(t_range, 0, p_attend_breath, alpha=0.2, color=PURPLE)
    ax.axhline(y=0.5, color=GRAY, linestyle='--', alpha=0.5)
    shade_surprise(ax)
    ax.set_ylabel("Probability")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Level 2: Attentional State", fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.text(-0.05, 1.05, 'D', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')
    
    # =========================================================================
    # Panel E: EFE comparison
    # =========================================================================
    ax = axes[4]
    ax.plot(t_range, results["EFE_maintain"], color=BLUE, linewidth=1.5, 
            label='EFE(MAINTAIN)', alpha=0.8)
    ax.plot(t_range, results["EFE_switch"], color=ORANGE, linewidth=1.5, 
            label='EFE(SWITCH)', alpha=0.8)
    shade_surprise(ax)
    ax.set_ylabel("Expected Free Energy")
    ax.set_xlabel("Time step")
    ax.set_title("Level 2: Policy EFE (lower = preferred)", fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.text(-0.05, 1.05, 'E', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')
    
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
    
    parser = argparse.ArgumentParser(description="Attention competition during meditation")
    parser.add_argument("--T", type=int, default=200, help="Number of timesteps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--t-surprise", type=int, default=75, help="Timestep of surprising sound")
    parser.add_argument("--sound-duration", type=int, default=30, help="Duration of sound")
    parser.add_argument("--temperature", type=float, default=1.0, help="Action selection temperature")
    parser.add_argument("--breath-pref", type=float, default=0.3, help="Preference strength for attending breath")
    parser.add_argument("--save", action="store_true", help="Save plot")
    args = parser.parse_args()
    
    results = run_attention_competition(
        T=args.T,
        seed=args.seed,
        t_surprise=args.t_surprise,
        sound_duration=args.sound_duration,
        action_temperature=args.temperature,
        breath_preference=args.breath_pref,
    )
    
    here = os.path.dirname(__file__)
    outdir = os.path.join(here, "outputs")
    os.makedirs(outdir, exist_ok=True)
    
    filename = f"attention_competition_surprise_{args.t_surprise}.png"
    save_path = os.path.join(outdir, filename) if args.save else None
    
    fig = plot_attention_competition(results, save_path=save_path)
    
    if not args.save:
        import matplotlib.pyplot as plt
        plt.show()

