#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Plotting script for Level-1 breath inference (2-state: Inhale/Exhale).

Visualizes:
  - Line plot of P(Inhaling) in [0, 1]
  - True states as black dots at y=1 (Inhaling) and y=0 (Exhaling)
  - (Dynamic mode) Precision zeta and prediction error history
  - (Figure 1 mode) Publication-quality 2-panel figure
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from matplotlib.lines import Line2D

import sys
from pymdp.envs import BreathEnv
from pymdp.agent import Agent
from pymdp import utils
from pymdp.maths import (
    scale_likelihood,
    update_likelihood_precision,
)

# Make local directory importable regardless of CWD
_HERE = os.path.dirname(__file__)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from breath_level1 import run_session, run_session_with_precision_updating  # noqa: E402


# =============================================================================
# Publication Figure Style Settings
# =============================================================================

PUBLICATION_RCPARAMS = {
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
}

# Color palette
COLORS = {
    'blue': '#2563eb',
    'orange': '#ea580c',
    'gray': '#6b7280',
    'green': '#2E7D32',
}


# =============================================================================
# Simulation Functions
# =============================================================================

def run_figure1_simulation(T=150, seed=42, zeta_step=0.25, log_zeta_prior_var=2.0):
    """
    Run Level 1 breath perception with dynamic precision for Figure 1.
    
    Uses the principled B.45 precision update with tuned parameters.
    
    Parameters
    ----------
    T : int
        Number of timesteps
    seed : int
        Random seed
    zeta_step : float
        Step size for precision updates
    log_zeta_prior_var : float
        Prior variance for log(zeta)
    """
    np.random.seed(seed)
    
    env = BreathEnv(seed=seed)
    p_correct = float(env.p_correct)
    A_base = np.array([[p_correct, 1.0 - p_correct], 
                       [1.0 - p_correct, p_correct]])
    
    stay_p_inhale = 1.0 - 1.0 / max(1, np.mean(env.inhale_range))
    stay_p_exhale = 1.0 - 1.0 / max(1, np.mean(env.exhale_range))
    
    B_base = np.zeros((2, 2, 1))
    B_base[:, 0, 0] = [stay_p_inhale, 1.0 - stay_p_inhale]
    B_base[:, 1, 0] = [1.0 - stay_p_exhale, stay_p_exhale]
    
    B_obj = utils.obj_array(1)
    B_obj[0] = B_base
    agent = Agent(A=A_base, B=B_obj, save_belief_hist=True)
    
    # Data logs
    true_states = np.zeros(T, dtype=int)
    posteriors = np.zeros((T, 2))
    zeta_history = np.zeros(T)
    
    zeta = 1.0
    prior = np.array([0.5, 0.5])
    obs = int(env.reset())
    
    for t in range(T):
        true_states[t] = env.state
        
        # 1. Update precision FIRST: compare observation to prior prediction
        #    This determines how much to trust the incoming sensory evidence
        zeta, _, _ = update_likelihood_precision(
            zeta, A_base, obs, prior,
            log_zeta_prior_var=log_zeta_prior_var,
            zeta_step=zeta_step
        )
        zeta_history[t] = zeta
        
        # 2. State inference: use updated precision to weight likelihood
        A_scaled = scale_likelihood(A_base, zeta)
        agent.A = utils.to_obj_array(A_scaled)
        qs = agent.infer_states([obs])
        posteriors[t] = qs[0]
        
        # 3. Prepare for next timestep
        prior = B_base[:, :, 0] @ qs[0]
        obs = int(env.step(None))
    
    return {
        "true_states": true_states,
        "posteriors": posteriors,
        "zeta_history": zeta_history,
        "zeta_step": zeta_step,
        "log_zeta_prior_var": log_zeta_prior_var,
    }


# =============================================================================
# Plotting Functions
# =============================================================================

def _plot_panel(ax, results: dict, title_suffix: str = ""):
    """Basic breath inference panel (used for exploration)."""
    true_states = results["true_states"]
    posteriors = results["posteriors"]  # shape (T, 2)
    T = len(true_states)

    env = BreathEnv()
    p_inhale = posteriors[:, 0]

    ax.plot(np.arange(T), p_inhale, color=COLORS['blue'], linewidth=2.0, label='P(Inhaling)')

    true_binary = np.array([1.0 if s == env.INHALE else 0.0 for s in true_states])
    ax.scatter(np.arange(T), true_binary, s=12, color='black', alpha=0.85, label='true state')

    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0, T - 1)
    ax.set_yticks([0.0, 1.0])
    ax.set_yticklabels(["Exhaling", "Inhaling"])
    if title_suffix:
        ax.set_title(f"Breath inference ({title_suffix})")

    legend_handles = [
        Line2D([0], [0], color=COLORS['blue'], linewidth=2.5, label='P(Inhaling)'),
        Line2D([0], [0], color='black', marker='o', linestyle='None', label='true state'),
    ]
    leg = ax.legend(
        handles=legend_handles,
        loc="upper right",
        frameon=True,
        fancybox=True,
        borderpad=0.6,
        labelspacing=0.5,
        handlelength=1.6,
        handleheight=1.0,
        fontsize=9,
    )
    leg.get_frame().set_facecolor('white')
    leg.get_frame().set_alpha(1.0)
    leg.get_frame().set_edgecolor('0.8')


def plot_session(results: dict, title_suffix: str = ""):
    """Single panel breath inference plot."""
    fig = plt.figure(figsize=(12, 5))
    ax = plt.subplot(1, 1, 1)
    _plot_panel(ax, results, title_suffix=title_suffix)
    plt.tight_layout()
    return fig


def plot_compare(T: int = 200, seed: int = 7, zetas: tuple = (0.5, 2.0)):
    """Runs simulations with different zeta values and plots them in subplots."""
    num_zetas = len(zetas)
    fig, axes = plt.subplots(num_zetas, 1, figsize=(12, 5 * num_zetas), sharex=True)
    axes = np.atleast_1d(axes)

    for ax, z in zip(axes, zetas):
        results = run_session(T=T, seed=seed, zeta=z)
        _plot_panel(ax, results, title_suffix=f"ζ={z}")

    plt.tight_layout()
    plt.show()


def plot_session_with_precision(results: dict, title_suffix: str = ""):
    """
    Plot session results including precision history.
    
    Creates a 3-panel figure:
    1. Breath inference (posteriors and true states)
    2. Precision (zeta) over time
    3. Prediction error over time
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # Panel 1: Breath inference
    _plot_panel(axes[0], results, title_suffix=title_suffix)
    
    # Panel 2: Precision history
    T = len(results["true_states"])
    zeta_history = results.get("zeta_history", np.ones(T))
    axes[1].plot(np.arange(T), zeta_history, color=COLORS['orange'], linewidth=2.0)
    axes[1].set_ylabel("Precision (ζ)")
    axes[1].set_title("Likelihood Precision Over Time")
    axes[1].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='prior mean')
    axes[1].legend(loc='upper right')
    
    # Panel 3: Prediction error
    prediction_errors = results.get("prediction_errors", np.zeros(T))
    axes[2].plot(np.arange(T), prediction_errors, color=COLORS['green'], linewidth=2.0)
    axes[2].set_ylabel("Prediction Error")
    axes[2].set_xlabel("Timestep")
    axes[2].set_title("Sensory Prediction Error Over Time")
    
    # Mark phase transitions (high PE moments)
    true_states = results["true_states"]
    transitions = np.where(np.diff(true_states) != 0)[0]
    for t in transitions:
        for ax in axes:
            ax.axvline(x=t, color='gray', linestyle=':', alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_figure1(results: dict, save_path: str = None, save_pdf: bool = False):
    """
    Generate publication-quality Figure 1: Level 1 Breath Perception with Dynamic Precision.
    
    Two-panel figure:
    A) Breath state inference (true vs inferred)
    B) Evolution of likelihood precision (ζ)
    
    Parameters
    ----------
    results : dict
        Output from run_figure1_simulation()
    save_path : str, optional
        Path to save PNG. If None, figure is displayed.
    save_pdf : bool, default=False
        Also save PDF version for publication
        
    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    # Apply publication style
    plt.rcParams.update(PUBLICATION_RCPARAMS)
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True, 
                              gridspec_kw={'height_ratios': [1.2, 1], 'hspace': 0.15})
    
    T = len(results["true_states"])
    t_range = np.arange(T)
    
    # =========================================================================
    # Panel A: Breath State Inference
    # =========================================================================
    ax = axes[0]
    
    # True state (INHALE=0 maps to 1.0, EXHALE=1 maps to 0.0)
    true_binary = 1.0 - results["true_states"]
    ax.scatter(t_range, true_binary, s=15, color=COLORS['gray'], alpha=0.7, 
               label='True state', zorder=2)
    
    # Posterior belief P(Inhaling)
    p_inhale = results["posteriors"][:, 0]
    ax.plot(t_range, p_inhale, color=COLORS['blue'], linewidth=1.5, 
            label='P(Inhaling)', zorder=3)
    
    ax.set_ylabel("Probability")
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(['0', '0.5', '1'])
    
    # Legend
    legend_elements = [
        Line2D([0], [0], color=COLORS['blue'], linewidth=1.5, label='P(Inhaling)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['gray'], 
               markersize=6, label='True state', alpha=0.7)
    ]
    ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9)
    
    # Panel label
    ax.text(-0.08, 1.05, 'A', transform=ax.transAxes, fontsize=14, 
            fontweight='bold', va='top')
    
    # =========================================================================
    # Panel B: Likelihood Precision
    # =========================================================================
    ax = axes[1]
    
    # Get precision parameters from results (with defaults for backward compatibility)
    zeta_step = results.get("zeta_step", 0.25)
    log_zeta_prior_var = results.get("log_zeta_prior_var", 2.0)
    
    ax.plot(t_range, results["zeta_history"], color=COLORS['orange'], linewidth=1.5)
    ax.axhline(y=1.0, color=COLORS['gray'], linestyle='--', linewidth=1, alpha=0.6,
               label='Prior mean (ζ=1)')
    
    ax.set_xlabel("Time step")
    ax.set_ylabel("Precision (ζ)")
    ax.set_ylim(0, 2.5)
    ax.set_yticks([0, 0.5, 1, 1.5, 2, 2.5])
    
    # Legend with parameters
    ax.legend(loc='upper right', framealpha=0.9, 
              title=f"step={zeta_step}, var={log_zeta_prior_var}")
    
    # Panel label
    ax.text(-0.08, 1.05, 'B', transform=ax.transAxes, fontsize=14, 
            fontweight='bold', va='top')
    
    # Final adjustments
    fig.align_ylabels(axes)
    
    # Save or show
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor='white')
        print(f"Saved: {save_path}")
        
        if save_pdf:
            pdf_path = save_path.rsplit('.', 1)[0] + '.pdf'
            fig.savefig(pdf_path, bbox_inches="tight", facecolor='white')
            print(f"Saved: {pdf_path}")
    
    return fig


# =============================================================================
# Parameter Sweep for Precision Dynamics
# =============================================================================

def run_precision_param_sweep(
    lr_values: list = [0.5, 1.0, 1.5, 2.0],
    var_values: list = [1.0, 2.0, 4.0, 8.0],
    T: int = 100,
    seed: int = 42,
    save_dir: str = None,
):
    """
    Sweep learning rate and prior variance to find optimal precision dynamics.
    
    Creates a grid showing precision range and mean for each parameter combo.
    """
    np.random.seed(seed)
    
    # Run all combinations
    results_grid = {}
    
    for lr in lr_values:
        for var in var_values:
            print(f"Running lr={lr}, var={var}...")
            
            # Run simulation with these params
            env = BreathEnv(seed=seed)
            p_correct = float(env.p_correct)
            A_base = np.array([[p_correct, 1.0 - p_correct], 
                               [1.0 - p_correct, p_correct]])
            
            stay_p = 1.0 - 1.0 / max(1, np.mean(env.inhale_range))
            B_base = np.array([[stay_p, 1-stay_p], [1-stay_p, stay_p]])
            
            zeta = 1.0
            prior = np.array([0.5, 0.5])
            obs = int(env.reset())
            
            zeta_history = []
            true_states = []
            posteriors = []
            
            for t in range(T):
                true_states.append(env.state)
                
                # Precision update with current params
                zeta, _, _ = update_likelihood_precision(
                    zeta, A_base, obs, prior,
                    log_zeta_prior_mean=0.0,
                    log_zeta_prior_var=var,
                    zeta_step=lr
                )
                zeta_history.append(zeta)
                
                # State inference
                A_scaled = scale_likelihood(A_base, zeta)
                log_lik = np.log(A_scaled[obs, :] + 1e-16)
                qs = np.exp(log_lik) * prior
                qs = qs / qs.sum()
                posteriors.append(qs.copy())
                
                # Next timestep
                prior = B_base @ qs
                obs = int(env.step(None))
            
            zeta_history = np.array(zeta_history)
            posteriors = np.array(posteriors)
            true_states = np.array(true_states)
            
            accuracy = (np.argmax(posteriors, axis=1) == true_states).mean()
            
            results_grid[(lr, var)] = {
                'zeta_history': zeta_history,
                'mean_zeta': zeta_history.mean(),
                'min_zeta': zeta_history.min(),
                'max_zeta': zeta_history.max(),
                'range_zeta': zeta_history.max() - zeta_history.min(),
                'std_zeta': zeta_history.std(),
                'accuracy': accuracy,
            }
    
    # Create visualization
    fig = plt.figure(figsize=(16, 12))
    
    # Grid of precision traces
    n_lr = len(lr_values)
    n_var = len(var_values)
    
    for i, lr in enumerate(lr_values):
        for j, var in enumerate(var_values):
            ax = fig.add_subplot(n_lr, n_var, i * n_var + j + 1)
            
            res = results_grid[(lr, var)]
            t_range = np.arange(len(res['zeta_history']))
            
            ax.plot(t_range, res['zeta_history'], color='#ea580c', linewidth=1.5)
            ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
            
            ax.set_ylim(0, 3.0)
            ax.set_xlim(0, T)
            
            # Add stats as text
            ax.text(0.05, 0.95, f"range: {res['range_zeta']:.2f}\nacc: {res['accuracy']:.2f}", 
                    transform=ax.transAxes, fontsize=8, va='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            if i == 0:
                ax.set_title(f"var={var}", fontsize=10)
            if j == 0:
                ax.set_ylabel(f"lr={lr}", fontsize=10)
            if i == n_lr - 1:
                ax.set_xlabel("Time")
    
    fig.suptitle("Precision Dynamics: Learning Rate vs Prior Variance\n"
                 "(range = max - min ζ, acc = inference accuracy)", 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_dir:
        save_path = os.path.join(save_dir, "precision_param_sweep.png")
        plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor='white')
        print(f"\nSaved: {save_path}")
    
    # Also create a summary heatmap
    fig2, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Heatmap of precision range
    range_matrix = np.zeros((n_lr, n_var))
    acc_matrix = np.zeros((n_lr, n_var))
    
    for i, lr in enumerate(lr_values):
        for j, var in enumerate(var_values):
            range_matrix[i, j] = results_grid[(lr, var)]['range_zeta']
            acc_matrix[i, j] = results_grid[(lr, var)]['accuracy']
    
    im1 = axes[0].imshow(range_matrix, cmap='YlOrRd', aspect='auto')
    axes[0].set_xticks(range(n_var))
    axes[0].set_xticklabels([f"{v}" for v in var_values])
    axes[0].set_yticks(range(n_lr))
    axes[0].set_yticklabels([f"{lr}" for lr in lr_values])
    axes[0].set_xlabel("Prior Variance (log_zeta_prior_var)")
    axes[0].set_ylabel("Learning Rate")
    axes[0].set_title("Precision Range (max - min ζ)")
    plt.colorbar(im1, ax=axes[0])
    
    # Add text annotations
    for i in range(n_lr):
        for j in range(n_var):
            axes[0].text(j, i, f"{range_matrix[i,j]:.2f}", ha='center', va='center', fontsize=9)
    
    im2 = axes[1].imshow(acc_matrix, cmap='RdYlGn', aspect='auto', vmin=0.8, vmax=1.0)
    axes[1].set_xticks(range(n_var))
    axes[1].set_xticklabels([f"{v}" for v in var_values])
    axes[1].set_yticks(range(n_lr))
    axes[1].set_yticklabels([f"{lr}" for lr in lr_values])
    axes[1].set_xlabel("Prior Variance (log_zeta_prior_var)")
    axes[1].set_ylabel("Learning Rate")
    axes[1].set_title("Inference Accuracy")
    plt.colorbar(im2, ax=axes[1])
    
    for i in range(n_lr):
        for j in range(n_var):
            axes[1].text(j, i, f"{acc_matrix[i,j]:.2f}", ha='center', va='center', fontsize=9)
    
    fig2.suptitle("Precision Parameter Sweep Summary", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_dir:
        save_path2 = os.path.join(save_dir, "precision_param_sweep_summary.png")
        plt.savefig(save_path2, dpi=200, bbox_inches="tight", facecolor='white')
        print(f"Saved: {save_path2}")
    
    return fig, fig2, results_grid


# =============================================================================
# Main CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot Level-1 breath inference with optional dynamic precision.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fixed precision exploration
  python plot_breath_level1.py --mode fixed --zeta 1.0 --save
  
  # Dynamic precision exploration  
  python plot_breath_level1.py --mode dynamic --T 150 --save
  
  # Generate publication Figure 1
  python plot_breath_level1.py --mode figure1 --T 150 --seed 42 --save
        """
    )
    parser.add_argument("--zeta", type=float, default=1.0, 
                        help="Likelihood precision (fixed mode) or initial precision (dynamic mode)")
    parser.add_argument("--T", type=int, default=100, help="Number of timesteps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save", action="store_true", help="Save the plot instead of displaying it")
    parser.add_argument("--outdir", type=str, default=None, 
                        help="Output directory (defaults to examples/meditation/outputs)")
    parser.add_argument("--mode", choices=["fixed", "dynamic", "figure1", "sweep"], default="fixed",
                        help="Mode: 'fixed' (constant precision), 'dynamic' (updates precision), 'figure1' (publication figure), 'sweep' (parameter sweep)")
    parser.add_argument("--prior-mean", dest="prior_mean", type=float, default=0.0,
                        help="Prior mean for log-precision (dynamic mode). Default 0.0 → ζ≈1")
    parser.add_argument("--prior-var", dest="prior_var", type=float, default=8.0,
                        help="Prior variance for log-precision (dynamic mode)")
    parser.add_argument("--lr", type=float, default=0.5,
                        help="Learning rate for precision updates (dynamic mode)")
    args = parser.parse_args()

    # Set up output directory
    here = os.path.dirname(__file__)
    outdir = args.outdir or os.path.join(here, "outputs")
    
    if args.mode == "fixed":
        results = run_session(T=args.T, seed=args.seed, zeta=args.zeta)
        fig = plot_session(results, title_suffix=f"ζ={args.zeta}")
        filename = f"breath_inference_zeta_{args.zeta}.png"
        
    elif args.mode == "dynamic":
        results = run_session_with_precision_updating(
            T=args.T, 
            seed=args.seed, 
            zeta_init=args.zeta,
            log_zeta_prior_mean=args.prior_mean,
            log_zeta_prior_var=args.prior_var,
            precision_lr=args.lr
        )
        fig = plot_session_with_precision(results, title_suffix="Dynamic Precision")
        filename = f"breath_inference_dynamic_lr{args.lr}.png"
        
    elif args.mode == "figure1":
        results = run_figure1_simulation(
            T=args.T, 
            seed=args.seed,
            zeta_step=args.lr,  # Using --lr for zeta_step
            log_zeta_prior_var=args.prior_var
        )
        
        # Print summary stats
        print(f"\nFigure 1 simulation summary:")
        print(f"  T = {args.T}, seed = {args.seed}")
        print(f"  Precision params: step={args.lr}, var={args.prior_var}")
        accuracy = (np.argmax(results['posteriors'], axis=1) == results['true_states']).mean()
        print(f"  Inference accuracy: {accuracy:.3f}")
        print(f"  Mean precision (ζ): {results['zeta_history'].mean():.3f}")
        print(f"  Precision range: [{results['zeta_history'].min():.3f}, {results['zeta_history'].max():.3f}]")
        
        # Include parameters in filename
        filename = f"figure1_breath_precision_step{args.lr}_var{args.prior_var}.png"
        
        if args.save:
            os.makedirs(outdir, exist_ok=True)
            save_path = os.path.join(outdir, filename)
            fig = plot_figure1(results, save_path=save_path, save_pdf=False)
        else:
            fig = plot_figure1(results)
            plt.show()
        
        # Exit early since plot_figure1 handles its own saving
        sys.exit(0)
    
    elif args.mode == "sweep":
        os.makedirs(outdir, exist_ok=True)
        fig1, fig2, results_grid = run_precision_param_sweep(
            lr_values=[0.1, 0.5, 1.0, 1.5, 2.0],
            var_values=[0.5, 1.0, 2.0, 4.0, 8.0],
            T=args.T,
            seed=args.seed,
            save_dir=outdir if args.save else None,
        )
        if not args.save:
            plt.show()
        sys.exit(0)

    # Handle saving for fixed/dynamic modes
    if args.save:
        os.makedirs(outdir, exist_ok=True)
        outfile = os.path.join(outdir, filename)
        fig.savefig(outfile, dpi=200, bbox_inches="tight")
        print(f"Saved plot to {outfile}")
    else:
        plt.show()
