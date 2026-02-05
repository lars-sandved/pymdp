#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Publication-quality plotting functions for meditation figures.

Figure sequence:
    1. Breath Perception with Dynamic Precision
    2. Attention Modulates Precision
    3. Precision Dynamics Improve Learning
    4. The Attention Trap (Non-Meditator)
    5. Meditation Instruction Breaks the Cycle
    6. Learning Across Sits
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from typing import Dict, Any, Optional
import os

try:
    from .models import FOCUSED, DISTRACTED, INHALE, EXHALE, STAY, SWITCH
except ImportError:
    from examples.meditation.models import FOCUSED, DISTRACTED, INHALE, EXHALE, STAY, SWITCH


# =============================================================================
# Style Configuration
# =============================================================================

# Color palette
COLORS = {
    "breath": "#2563eb",       # Blue
    "precision": "#7c3aed",    # Purple
    "attention": "#ea580c",    # Orange
    "awareness": "#16a34a",    # Green
    "focused": "#16a34a",      # Green
    "distracted": "#dc2626",   # Red
    "true_state": "#6b7280",   # Gray
    "switch": "#f59e0b",       # Amber
    "stay": "#6b7280",         # Gray
    "shading": "#fecaca",      # Light red for distraction periods
}


def setup_style():
    """Configure matplotlib for publication-quality figures."""
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
        'lines.linewidth': 1.5,
    })


def add_panel_label(ax, label: str, x: float = -0.08, y: float = 1.08):
    """Add panel label (A, B, C, etc.) to axis."""
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=14, fontweight='bold', va='top')


def shade_distraction_period(ax, start: int, end: int, alpha: float = 0.15):
    """Add shaded region indicating distraction period."""
    ax.axvspan(start, end, alpha=alpha, color=COLORS["distracted"], zorder=0)


def save_figure(fig, save_path: str):
    """Save figure as PDF only."""
    # Convert any path to PDF
    pdf_path = save_path.rsplit('.', 1)[0] + '.pdf'
    fig.savefig(pdf_path, bbox_inches="tight", facecolor='white')
    print(f"Saved: {pdf_path}")


# =============================================================================
# Figure 1: Breath Perception with Dynamic Precision
# =============================================================================

def plot_figure1(results: Dict[str, Any], save_path: Optional[str] = None):
    """
    Figure 1: Breath perception with dynamic precision (B.45).

    Two panels:
    A) Breath state inference (posterior + true state)
    B) Dynamic precision (zeta)
    """
    setup_style()

    T = results["T"]
    t_range = np.arange(T)

    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True,
                              gridspec_kw={'height_ratios': [1, 0.8], 'hspace': 0.35})

    # Panel A: Breath state inference
    ax = axes[0]
    p_inhale = results["posteriors"][:, INHALE]
    ax.plot(t_range, p_inhale, color=COLORS["breath"], label='P(Inhaling)')

    # True state as scatter (same color as line)
    true_inhale = 1.0 - results["true_states"]  # INHALE=0 -> 1.0
    ax.scatter(t_range, true_inhale, s=8, color=COLORS["breath"], alpha=0.4, label='True state')

    ax.set_ylabel("Probability")
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks([0, 0.5, 1])
    ax.legend(loc='upper right', framealpha=0.9)
    add_panel_label(ax, 'A')
    ax.set_title("Breath State Inference", fontsize=12)

    # Panel B: Dynamic precision
    ax = axes[1]
    ax.plot(t_range, results["zeta_history"], color=COLORS["precision"])
    ax.axhline(y=1.0, color=COLORS["true_state"], linestyle='--', linewidth=1, alpha=0.6, label='Prior mean (ζ=1)')
    ax.set_ylabel("Precision (ζ)")
    ax.set_xlabel("Time step")
    ax.set_ylim(0, 2.5)
    ax.set_xlim(0, T)
    ax.legend(loc='upper right', framealpha=0.9)
    add_panel_label(ax, 'B')
    ax.set_title("Likelihood Precision Inference", fontsize=12)

    fig.align_ylabels(axes)

    if save_path:
        save_figure(fig, save_path)

    return fig


# =============================================================================
# Figure 2: Attention Modulates Precision
# =============================================================================

def plot_figure2(results: Dict[str, Any], save_path: Optional[str] = None):
    """
    Figure 2: Attention modulates precision.

    Three panels:
    A) Attention state inference (posterior line + true state dots)
    B) Precision (zeta) with dynamic updating and descending prior from attention
    C) Breath state inference
    """
    setup_style()

    T = results["T"]
    t_range = np.arange(T)

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True,
                              gridspec_kw={'height_ratios': [0.8, 0.8, 1], 'hspace': 0.35})

    # Panel A: Attention state inference
    ax = axes[0]
    # Posterior P(Focused) as line
    p_focused = results["attention_posteriors"][:, FOCUSED]
    ax.plot(t_range, p_focused, color=COLORS["attention"], label='P(Focused)')
    # True state as dots (same color as line)
    true_attention = 1.0 - results["true_attention_states"]
    ax.scatter(t_range, true_attention, s=8, color=COLORS["attention"], alpha=0.4, label='True state')
    ax.set_ylabel("Probability")
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlim(0, T)
    ax.legend(loc='upper right', framealpha=0.9)
    add_panel_label(ax, 'A')
    ax.set_title("Attention State Inference", fontsize=12)

    # Panel B: Precision (dynamic with descending prior from attention)
    ax = axes[1]
    ax.plot(t_range, results["zeta_history"], color=COLORS["precision"], label='ζ posterior')
    ax.plot(t_range, results["zeta_prior_history"], color=COLORS["precision"], linestyle='--', alpha=0.6, label='ζ prior (↓)')
    ax.set_ylabel("Precision (ζ)")
    ax.set_ylim(0, 2.5)
    ax.set_xlim(0, T)
    ax.legend(loc='upper right', framealpha=0.9)
    add_panel_label(ax, 'B')
    ax.set_title("Likelihood Precision Inference", fontsize=12)

    # Panel C: Breath inference
    ax = axes[2]
    p_inhale = results["posteriors"][:, INHALE]
    ax.plot(t_range, p_inhale, color=COLORS["breath"], label='P(Inhaling)')
    true_inhale = 1.0 - results["true_states"]
    ax.scatter(t_range, true_inhale, s=8, color=COLORS["breath"], alpha=0.4, label='True state')
    ax.set_ylabel("Probability")
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlim(0, T)
    ax.set_xlabel("Time step")
    ax.legend(loc='upper right', framealpha=0.9)
    add_panel_label(ax, 'C')
    ax.set_title("Breath State Inference", fontsize=12)

    fig.align_ylabels(axes)

    if save_path:
        save_figure(fig, save_path)

    return fig


# =============================================================================
# Figure 2.1: Attention with Mental Action
# =============================================================================

def plot_figure2_1(results: Dict[str, Any], save_path: Optional[str] = None):
    """
    Figure 2.1: Attention with mental action and natural transitions.

    Four panels:
    A) Action selection (policy inference + selected actions)
    B) Attention state inference (posterior + true state)
    C) Likelihood precision (zeta posterior + prior)
    D) Breath state inference
    """
    setup_style()

    T = results["T"]
    t_range = np.arange(T)
    params = results["params"]

    fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True,
                              gridspec_kw={'height_ratios': [0.7, 0.8, 0.8, 1], 'hspace': 0.35})

    # Panel A: Action Selection
    ax = axes[0]
    # P(Stay) as line
    p_stay = results["q_pi_history"][:, STAY]
    ax.plot(t_range, p_stay, color=COLORS["awareness"], label='P(Stay)')

    # Switch actions as dots at bottom
    actions = results["actions"]
    switch_mask = actions == SWITCH
    ax.scatter(t_range[switch_mask], np.ones(switch_mask.sum()) * 0.05,
               s=15, color=COLORS["switch"], alpha=0.8, marker='o', label='Switch')

    ax.set_ylabel("P(Stay)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlim(0, T)
    ax.legend(loc='right', framealpha=0.9, fontsize=9)
    add_panel_label(ax, 'A')
    ax.set_title("Mental Action Selection", fontsize=12)

    # Panel B: Attention State Inference
    ax = axes[1]
    p_focused = results["attention_posteriors"][:, FOCUSED]
    ax.plot(t_range, p_focused, color=COLORS["attention"], label='P(Focused)')
    # True state as horizontal segments
    true_attention = results["true_attention_states"]
    true_focused = 1.0 - true_attention  # FOCUSED=0 -> 1.0, DISTRACTED=1 -> 0.0
    ax.scatter(t_range, true_focused, s=8, color=COLORS["attention"], alpha=0.4, label='True state')

    ax.set_ylabel("P(Focused)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlim(0, T)
    ax.legend(loc='upper right', framealpha=0.9)
    add_panel_label(ax, 'B')
    ax.set_title("Attention State Inference", fontsize=12)

    # Panel C: Likelihood Precision
    ax = axes[2]
    ax.plot(t_range, results["zeta_history"], color=COLORS["precision"], label='ζ posterior')
    ax.plot(t_range, results["zeta_prior_history"], color=COLORS["precision"],
            linestyle='--', alpha=0.6, label='ζ prior (↓)')

    ax.set_ylabel("Precision (ζ)")
    ax.set_ylim(0, max(2.5, params.zeta_focused + 0.5))
    ax.set_xlim(0, T)
    ax.legend(loc='upper right', framealpha=0.9, fontsize=9)
    add_panel_label(ax, 'C')
    ax.set_title("Likelihood Precision", fontsize=12)

    # Panel D: Breath Perception
    ax = axes[3]
    p_inhale = results["posteriors"][:, INHALE]
    ax.plot(t_range, p_inhale, color=COLORS["breath"], label='P(Inhaling)')
    true_inhale = 1.0 - results["true_states"]
    ax.scatter(t_range, true_inhale, s=8, color=COLORS["breath"], alpha=0.4, label='True state')

    ax.set_ylabel("P(Inhaling)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlim(0, T)
    ax.set_xlabel("Time step")
    ax.legend(loc='upper right', framealpha=0.9)
    add_panel_label(ax, 'D')
    ax.set_title("Breath Perception", fontsize=12)

    fig.align_ylabels(axes)

    if save_path:
        save_figure(fig, save_path)

    return fig


# =============================================================================
# Figure 3: Precision Dynamics Improve Learning
# =============================================================================

def plot_figure3(
    results_fixed: Dict[str, Any],
    results_dynamic: Dict[str, Any],
    results_attention: Dict[str, Any] = None,
    save_path: Optional[str] = None
):
    """
    Figure 3: Precision dynamics improve A1 learning across sits.

    Single panel showing A1 diagonal (accuracy) over sits.
    """
    setup_style()

    num_sits = results_fixed["num_sits"]
    sit_range = np.arange(num_sits)

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))

    # True A1 diagonal as dashed line
    ax.axhline(y=results_fixed["A1_true_diagonal"], color=COLORS["true_state"],
               linestyle='--', linewidth=1.5, alpha=0.8, label='True likelihood accuracy')

    ax.plot(sit_range, results_fixed["A1_diagonal"], color=COLORS["true_state"],
            label='Fixed precision ζ=1', alpha=0.7)
    ax.plot(sit_range, results_dynamic["A1_diagonal"], color=COLORS["precision"],
            label='Dynamic ζ')

    if results_attention is not None:
        ax.plot(sit_range, results_attention["A1_diagonal"], color=COLORS["attention"],
                label='Dynamic ζ with attention inference')

    ax.set_xlabel("Sit")
    ax.set_ylabel("Likelihood accuracy")
    ax.set_title("Likelihood Learning Convergence", fontsize=12)
    ax.legend(loc='lower right', framealpha=0.9)
    ax.set_ylim(0.5, 1.0)
    ax.set_xlim(0, num_sits)

    if save_path:
        save_figure(fig, save_path)

    return fig


# =============================================================================
# Figure 4: The Attention Trap
# =============================================================================

def plot_figure4(results: Dict[str, Any], save_path: Optional[str] = None):
    """
    Figure 4: The attention trap (non-meditator).

    Five panels showing hierarchical dynamics.
    """
    setup_style()

    T = results["T"]
    t_range = np.arange(T)
    distraction_onset = results["distraction_onset"]

    fig, axes = plt.subplots(5, 1, figsize=(10, 10), sharex=True,
                              gridspec_kw={'height_ratios': [1, 0.8, 0.8, 0.8, 0.6], 'hspace': 0.15})

    # Shade distraction period
    for ax in axes:
        shade_distraction_period(ax, distraction_onset, T)

    # Panel A: Breath inference
    ax = axes[0]
    p_inhale = results["posterior_breath"][:, INHALE]
    ax.plot(t_range, p_inhale, color=COLORS["breath"], label='P(Inhaling)')
    true_inhale = 1.0 - results["true_breath"]
    ax.scatter(t_range, true_inhale, s=8, color=COLORS["true_state"], alpha=0.4)
    ax.set_ylabel("P(Inhaling)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks([0, 0.5, 1])
    add_panel_label(ax, 'A')
    ax.set_title("Breath Perception", fontsize=12)

    # Panel B: Precision
    ax = axes[1]
    ax.plot(t_range, results["zeta_history"], color=COLORS["precision"])
    ax.axhline(y=1.0, color=COLORS["true_state"], linestyle='--', linewidth=1, alpha=0.6)
    ax.set_ylabel("Precision (ζ)")
    add_panel_label(ax, 'B')

    # Panel C: Attention inference
    ax = axes[2]
    p_focused = results["posterior_attention"][:, FOCUSED]
    ax.plot(t_range, p_focused, color=COLORS["attention"], label='P(Focused)')
    true_focused = 1.0 - results["true_attention"]
    ax.scatter(t_range, true_focused, s=8, color=COLORS["true_state"], alpha=0.4)
    ax.set_ylabel("P(Focused)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks([0, 0.5, 1])
    add_panel_label(ax, 'C')
    ax.set_title("Attention Inference", fontsize=12)

    # Panel D: Awareness inference
    ax = axes[3]
    p_aware = results["posterior_awareness"][:, 0]  # AWARE=0
    ax.plot(t_range, p_aware, color=COLORS["awareness"], label='P(Aware)')
    ax.set_ylabel("P(Aware)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks([0, 0.5, 1])
    add_panel_label(ax, 'D')

    # Panel E: Actions
    ax = axes[4]
    actions = results["actions"]
    ax.fill_between(t_range, 0, 1, where=(actions == STAY),
                    color=COLORS["stay"], alpha=0.6, label='STAY')
    ax.fill_between(t_range, 0, 1, where=(actions == SWITCH),
                    color=COLORS["switch"], alpha=0.8, label='SWITCH')
    ax.set_ylabel("Action")
    ax.set_xlabel("Time step")
    ax.set_yticks([])
    ax.legend(loc='upper right', framealpha=0.9, ncol=2)
    add_panel_label(ax, 'E')

    fig.align_ylabels(axes)

    # Summary stats
    fig.text(0.99, 0.01,
             f"Time distracted: {results['time_distracted']:.1%} | "
             f"Switch rate: {results['switch_rate']:.1%} | "
             f"Attention accuracy: {results['attention_accuracy']:.1%}",
             ha='right', va='bottom', fontsize=10, color=COLORS["true_state"])

    if save_path:
        save_figure(fig, save_path)

    return fig


# =============================================================================
# Figure 5: Meditation Instruction
# =============================================================================

def plot_figure5(results: Dict[str, Any], save_path: Optional[str] = None):
    """
    Figure 5: Meditation instruction breaks the cycle.

    Same layout as Figure 4, but showing successful recovery.
    """
    # Use same plotting function as Figure 4
    fig = plot_figure4(results, save_path=None)

    # Update title
    fig.axes[0].set_title("Breath Perception (with Meditation Instruction)", fontsize=12)
    fig.axes[2].set_title("Attention Inference (with Meditation Instruction)", fontsize=12)

    if save_path:
        save_figure(fig, save_path)

    return fig


# =============================================================================
# Figure 6: Learning Across Sits
# =============================================================================

def plot_figure6(
    results_baseline: Dict[str, Any],
    results_meditation: Dict[str, Any],
    save_path: Optional[str] = None
):
    """
    Figure 6: Learning across sits.

    Compares non-meditator (baseline) vs meditator (with instruction).

    Four panels:
    A) Time distracted per sit
    B) Attention accuracy per sit
    C) A2 diagonal learning
    D) B2 learning (stay transitions)
    """
    setup_style()

    num_sits = results_baseline["num_sits"]
    sit_range = np.arange(num_sits)

    meditation_start = results_meditation.get("meditation_start_sit", 0)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8),
                              gridspec_kw={'hspace': 0.25, 'wspace': 0.25})
    axes = axes.flatten()

    # Shade meditation period
    def shade_meditation(ax):
        if meditation_start > 0:
            ax.axvspan(meditation_start, num_sits, alpha=0.1, color=COLORS["awareness"])
            ax.axvline(x=meditation_start, color=COLORS["awareness"], linestyle='--',
                       linewidth=1, alpha=0.7)

    # Panel A: Time distracted
    ax = axes[0]
    ax.plot(sit_range, results_baseline["time_distracted"],
            color=COLORS["distracted"], alpha=0.7, label='Non-meditator')
    ax.plot(sit_range, results_meditation["time_distracted"],
            color=COLORS["focused"], alpha=0.9, label='Meditator')
    shade_meditation(ax)
    ax.set_ylabel("Proportion distracted")
    ax.set_xlabel("Sit number")
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', framealpha=0.9)
    add_panel_label(ax, 'A')
    ax.set_title("Time Spent Distracted", fontsize=12)

    # Panel B: Attention accuracy
    ax = axes[1]
    ax.plot(sit_range, results_baseline["attention_accuracy"],
            color=COLORS["distracted"], alpha=0.7, label='Non-meditator')
    ax.plot(sit_range, results_meditation["attention_accuracy"],
            color=COLORS["focused"], alpha=0.9, label='Meditator')
    shade_meditation(ax)
    ax.set_ylabel("Attention accuracy")
    ax.set_xlabel("Sit number")
    ax.set_ylim(0, 1)
    ax.legend(loc='lower right', framealpha=0.9)
    add_panel_label(ax, 'B')
    ax.set_title("Attention Inference Accuracy", fontsize=12)

    # Panel C: A2 diagonal
    ax = axes[2]
    # True value line
    params = results_baseline["params"]
    ax.axhline(y=0.9, color=COLORS["true_state"], linestyle='--',
               linewidth=1, alpha=0.6, label='True A2')

    ax.plot(sit_range, results_baseline["A2_diagonal"],
            color=COLORS["distracted"], alpha=0.7, label='Non-meditator')
    ax.plot(sit_range, results_meditation["A2_diagonal"],
            color=COLORS["focused"], alpha=0.9, label='Meditator')
    shade_meditation(ax)
    ax.set_ylabel("A2 diagonal")
    ax.set_xlabel("Sit number")
    ax.set_ylim(0.4, 1.0)
    ax.legend(loc='lower right', framealpha=0.9)
    add_panel_label(ax, 'C')
    ax.set_title("A2 Learning (Precision → Attention)", fontsize=12)

    # Panel D: B2 learning
    ax = axes[3]
    # True values
    ax.axhline(y=params.p_stay_focused, color=COLORS["focused"],
               linestyle='--', linewidth=1, alpha=0.4, label=f'True B2[foc,foc,STAY]={params.p_stay_focused}')
    ax.axhline(y=params.p_stay_distracted, color=COLORS["distracted"],
               linestyle='--', linewidth=1, alpha=0.4, label=f'True B2[dist,dist,STAY]={params.p_stay_distracted}')

    # Learned values - meditation only (baseline doesn't learn much)
    ax.plot(sit_range, results_meditation["B2_stay_focused"],
            color=COLORS["focused"], alpha=0.9, label='Meditator: B2[foc,foc,STAY]')
    ax.plot(sit_range, results_meditation["B2_stay_distracted"],
            color=COLORS["distracted"], alpha=0.9, label='Meditator: B2[dist,dist,STAY]')

    shade_meditation(ax)
    ax.set_ylabel("B2 transition probability")
    ax.set_xlabel("Sit number")
    ax.set_ylim(0, 1.1)
    ax.legend(loc='right', framealpha=0.9, fontsize=8)
    add_panel_label(ax, 'D')
    ax.set_title("B2 Learning (Attention Transitions)", fontsize=12)

    if save_path:
        save_figure(fig, save_path)

    return fig


# =============================================================================
# Combined Figures 4+5 (Side by Side)
# =============================================================================

def plot_figures4_5_combined(
    results_trap: Dict[str, Any],
    results_meditation: Dict[str, Any],
    save_path: Optional[str] = None
):
    """
    Combined figure comparing non-meditator (trap) vs meditator (escape).

    Two columns, showing key panels side by side.
    """
    setup_style()

    T = results_trap["T"]
    t_range = np.arange(T)
    distraction_onset = results_trap["distraction_onset"]

    fig, axes = plt.subplots(4, 2, figsize=(14, 10), sharex='col',
                              gridspec_kw={'hspace': 0.2, 'wspace': 0.15})

    results_list = [results_trap, results_meditation]
    titles = ["Non-Meditator", "With Meditation Instruction"]

    for col, (results, title) in enumerate(zip(results_list, titles)):
        # Shade distraction period
        for row in range(4):
            shade_distraction_period(axes[row, col], distraction_onset, T)

        # Row 0: Breath
        ax = axes[0, col]
        p_inhale = results["posterior_breath"][:, INHALE]
        ax.plot(t_range, p_inhale, color=COLORS["breath"])
        true_inhale = 1.0 - results["true_breath"]
        ax.scatter(t_range, true_inhale, s=6, color=COLORS["true_state"], alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
        ax.set_yticks([0, 0.5, 1])
        if col == 0:
            ax.set_ylabel("P(Inhaling)")
            add_panel_label(ax, 'A')
        ax.set_title(title, fontsize=12)

        # Row 1: Precision
        ax = axes[1, col]
        ax.plot(t_range, results["zeta_history"], color=COLORS["precision"])
        ax.axhline(y=1.0, color=COLORS["true_state"], linestyle='--', linewidth=1, alpha=0.6)
        if col == 0:
            ax.set_ylabel("Precision (ζ)")
            add_panel_label(ax, 'B')

        # Row 2: Attention
        ax = axes[2, col]
        p_focused = results["posterior_attention"][:, FOCUSED]
        ax.plot(t_range, p_focused, color=COLORS["attention"])
        true_focused = 1.0 - results["true_attention"]
        ax.scatter(t_range, true_focused, s=6, color=COLORS["true_state"], alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
        ax.set_yticks([0, 0.5, 1])
        if col == 0:
            ax.set_ylabel("P(Focused)")
            add_panel_label(ax, 'C')

        # Row 3: Actions
        ax = axes[3, col]
        actions = results["actions"]
        ax.fill_between(t_range, 0, 1, where=(actions == STAY),
                        color=COLORS["stay"], alpha=0.6)
        ax.fill_between(t_range, 0, 1, where=(actions == SWITCH),
                        color=COLORS["switch"], alpha=0.8)
        ax.set_yticks([])
        ax.set_xlabel("Time step")
        if col == 0:
            ax.set_ylabel("Action")
            add_panel_label(ax, 'D')

        # Add stats
        stats_text = (f"Distracted: {results['time_distracted']:.0%}\n"
                      f"Switch rate: {results['switch_rate']:.0%}")
        axes[3, col].text(0.98, 0.95, stats_text, transform=axes[3, col].transAxes,
                          ha='right', va='top', fontsize=9,
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Legend for actions
    legend_elements = [Patch(facecolor=COLORS["stay"], alpha=0.6, label='STAY'),
                       Patch(facecolor=COLORS["switch"], alpha=0.8, label='SWITCH')]
    axes[3, 1].legend(handles=legend_elements, loc='upper left', framealpha=0.9)

    if save_path:
        save_figure(fig, save_path)

    return fig


# =============================================================================
# Figure 4: The Distraction Trap - Learning Heatmap
# =============================================================================

def plot_figure4_heatmap(
    results: Dict[str, Any],
    save_path: Optional[str] = None,
    show_contours: bool = True,
):
    """
    Plot heatmap showing the distraction trap.

    X-axis: ω (B2 precision) - transition model precision
    Y-axis: ζ (A2 precision) - likelihood model precision
    Color: Final % time focused after learning

    Precision parameters scale the true model:
    - 0 = completely flat (no knowledge)
    - 1 = matches true generative process

    Shows that without sufficient initial structure, agents cannot
    learn their way out of distraction.
    """
    setup_style()

    fig, ax = plt.subplots(figsize=(8, 7))

    zeta_range = results["zeta_range"]
    omega_range = results["omega_range"]
    data = results["final_time_focused"]

    # Create heatmap
    # Note: imshow expects (rows, cols) where rows are Y (zeta) and cols are X (omega)
    im = ax.imshow(
        data,
        origin='lower',
        aspect='auto',
        extent=[omega_range[0], omega_range[-1], zeta_range[0], zeta_range[-1]],
        cmap='RdYlGn',  # Red (trapped) -> Yellow -> Green (focused)
        vmin=0,
        vmax=1,
    )

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, label='Final % Time Focused', shrink=0.8)
    cbar.ax.tick_params(labelsize=10)

    # Labels
    ax.set_xlabel('ω (Transition Model Precision)', fontsize=12)
    ax.set_ylabel('ζ (Likelihood Model Precision)', fontsize=12)
    ax.set_title('The Distraction Trap: Learning Requires Initial Structure', fontsize=14)

    # Ticks
    ax.set_xticks(np.linspace(omega_range[0], omega_range[-1], 5))
    ax.set_yticks(np.linspace(zeta_range[0], zeta_range[-1], 5))

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)

    return fig


def plot_figure4_diagnostic(
    results_list: list,
    save_path: Optional[str] = None,
):
    """
    Plot diagnostic within-sit dynamics for Figure 4 debugging.

    Shows attention state, posterior, observations, and actions for
    multiple (zeta, omega) parameterizations at different sit numbers.

    Parameters
    ----------
    results_list : list
        List of results from run_figure4_diagnostic, each with different params
    """
    setup_style()

    n_params = len(results_list)
    n_sits = len(results_list[0]["capture_sits"])

    fig, axes = plt.subplots(n_params, n_sits, figsize=(5 * n_sits, 3.5 * n_params))
    if n_params == 1:
        axes = axes.reshape(1, -1)
    if n_sits == 1:
        axes = axes.reshape(-1, 1)

    for row, results in enumerate(results_list):
        zeta = results["zeta_A2"]
        omega = results["omega_B2"]
        captured = results["captured_dynamics"]

        for col, sit in enumerate(results["capture_sits"]):
            ax = axes[row, col]
            data = captured[sit]

            T = len(data["true_attention"])
            t = np.arange(T)

            # True attention state (background shading)
            for i in range(T):
                if data["true_attention"][i] == 1:  # DISTRACTED
                    ax.axvspan(i - 0.5, i + 0.5, alpha=0.2, color='red', linewidth=0)

            # Posterior P(focused)
            ax.plot(t, data["qs_focused"], 'b-', linewidth=1.5, label='P(focused)')

            # Reference line at 0.5
            ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=0.5)

            # Observations as scatter
            obs_colors = ['green' if o == 0 else 'orange' for o in data["obs_precision"]]
            ax.scatter(t, data["obs_precision"] * 0.1 + 0.05, c=obs_colors, s=10, alpha=0.7)

            # Switch actions as vertical lines
            switch_times = t[data["action"] == 1]
            for st in switch_times:
                ax.axvline(x=st, color='purple', alpha=0.5, linewidth=1)

            ax.set_ylim(-0.05, 1.05)
            ax.set_xlim(-1, T)

            if row == 0:
                ax.set_title(f'Sit {sit + 1}', fontsize=11)
            if col == 0:
                ax.set_ylabel(f'ζ={zeta:.1f}, ω={omega:.1f}\nP(focused)', fontsize=10)
            if row == n_params - 1:
                ax.set_xlabel('Timestep', fontsize=10)

            # Add A2/B2 diagonal info
            ax.text(0.02, 0.98, f"A2d={data['A2_diagonal']:.2f}\nB2d={data['B2_stay_diag']:.2f}",
                    transform=ax.transAxes, fontsize=8, va='top', ha='left',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Legend in first panel
    axes[0, 0].legend(loc='upper right', fontsize=8)

    plt.suptitle('Figure 4 Diagnostic: Within-Sit Dynamics\n'
                 '(Red shading = distracted, Purple lines = SWITCH, Green/Orange dots = precise/imprecise obs)',
                 fontsize=12)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)

    if save_path:
        save_figure(fig, save_path)

    return fig


def plot_A2_precision_comparison(
    results_list: list,
    save_path: Optional[str] = None,
):
    """
    Plot comparison of Figure 2.1 dynamics across different A2 precision values.

    Shows how meta-awareness precision affects ability to maintain focus.
    Uses Figure 2.1-style layout but with multiple columns for different A2 values.

    Parameters
    ----------
    results_list : list
        List of results from run_figure2_1_with_A2_precision, each with different A2 precision
    """
    setup_style()

    n_conditions = len(results_list)
    fig, axes = plt.subplots(4, n_conditions, figsize=(4 * n_conditions, 9), sharex='col',
                              gridspec_kw={'height_ratios': [0.7, 0.8, 0.8, 1], 'hspace': 0.25})

    if n_conditions == 1:
        axes = axes.reshape(-1, 1)

    for col, results in enumerate(results_list):
        T = results["T"]
        t_range = np.arange(T)
        params = results["params"]
        A2_prec = results["A2_precision"]
        time_dist = results["time_distracted"]

        # Panel A: Action Selection
        ax = axes[0, col]
        p_stay = results["q_pi_history"][:, STAY]
        ax.plot(t_range, p_stay, color=COLORS["awareness"], label='P(Stay)')
        actions = results["actions"]
        switch_mask = actions == SWITCH
        ax.scatter(t_range[switch_mask], np.ones(switch_mask.sum()) * 0.05,
                   s=15, color=COLORS["switch"], alpha=0.8, marker='o', label='Switch')
        ax.set_ylim(-0.05, 1.05)
        ax.set_yticks([0, 0.5, 1])
        ax.set_xlim(0, T)
        if col == 0:
            ax.set_ylabel("P(Stay)")
            ax.legend(loc='right', framealpha=0.9, fontsize=8)
            add_panel_label(ax, 'A')
        ax.set_title(f"A2 prec = {A2_prec:.2f}\n({time_dist:.0%} distracted)", fontsize=11)

        # Panel B: Attention State Inference
        ax = axes[1, col]
        p_focused = results["attention_posteriors"][:, FOCUSED]
        ax.plot(t_range, p_focused, color=COLORS["attention"], label='P(Focused)')
        true_attention = results["true_attention_states"]
        # Background shading for distraction
        for i in range(T):
            if true_attention[i] == DISTRACTED:
                ax.axvspan(i - 0.5, i + 0.5, alpha=0.15, color='red', linewidth=0)
        ax.set_ylim(-0.05, 1.05)
        ax.set_yticks([0, 0.5, 1])
        ax.set_xlim(0, T)
        if col == 0:
            ax.set_ylabel("P(Focused)")
            ax.legend(loc='upper right', framealpha=0.9, fontsize=8)
            add_panel_label(ax, 'B')

        # Panel C: Likelihood Precision
        ax = axes[2, col]
        ax.plot(t_range, results["zeta_history"], color=COLORS["precision"], label='ζ posterior')
        ax.plot(t_range, results["zeta_prior_history"], color=COLORS["precision"],
                linestyle='--', alpha=0.6, label='ζ prior')
        ax.set_ylim(0, max(2.5, params.zeta_focused + 0.5))
        ax.set_xlim(0, T)
        if col == 0:
            ax.set_ylabel("Precision (ζ)")
            ax.legend(loc='upper right', framealpha=0.9, fontsize=8)
            add_panel_label(ax, 'C')

        # Panel D: Breath Perception
        ax = axes[3, col]
        p_inhale = results["posteriors"][:, INHALE]
        ax.plot(t_range, p_inhale, color=COLORS["breath"], label='P(Inhaling)')
        true_inhale = 1.0 - results["true_states"]
        ax.scatter(t_range, true_inhale, s=5, color=COLORS["breath"], alpha=0.3, label='True')
        ax.set_ylim(-0.05, 1.05)
        ax.set_yticks([0, 0.5, 1])
        ax.set_xlim(0, T)
        ax.set_xlabel("Time step")
        if col == 0:
            ax.set_ylabel("P(Inhaling)")
            ax.legend(loc='upper right', framealpha=0.9, fontsize=8)
            add_panel_label(ax, 'D')

    plt.suptitle('Effect of Meta-Awareness Precision (A2) on Attention Maintenance\n'
                 '(Red shading = truly distracted)', fontsize=13)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)

    if save_path:
        save_figure(fig, save_path)

    return fig
