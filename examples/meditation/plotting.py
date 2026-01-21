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
    "precision": "#ea580c",    # Orange
    "attention": "#16a34a",    # Green
    "awareness": "#7c3aed",    # Purple
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
    """Save figure in PNG and PDF formats."""
    fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor='white')
    pdf_path = save_path.rsplit('.', 1)[0] + '.pdf'
    fig.savefig(pdf_path, bbox_inches="tight", facecolor='white')
    print(f"Saved: {save_path}")
    print(f"Saved: {pdf_path}")


# =============================================================================
# Figure 1: Breath Perception with Dynamic Precision
# =============================================================================

def plot_figure1(results: Dict[str, Any], save_path: Optional[str] = None):
    """
    Figure 1: Breath perception with dynamic precision (B.45).

    Three panels:
    A) Breath state inference (posterior + true state)
    B) Dynamic precision (zeta)
    C) Prediction error
    """
    setup_style()

    T = results["T"]
    t_range = np.arange(T)

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True,
                              gridspec_kw={'height_ratios': [1, 0.8, 0.8], 'hspace': 0.15})

    # Panel A: Breath state inference
    ax = axes[0]
    p_inhale = results["posteriors"][:, INHALE]
    ax.plot(t_range, p_inhale, color=COLORS["breath"], label='P(Inhaling)')

    # True state as scatter
    true_inhale = 1.0 - results["true_states"]  # INHALE=0 -> 1.0
    ax.scatter(t_range, true_inhale, s=8, color=COLORS["true_state"], alpha=0.4, label='True state')

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
    ax.set_ylim(0, max(3.0, results["zeta_history"].max() * 1.1))
    ax.legend(loc='upper right', framealpha=0.9)
    add_panel_label(ax, 'B')

    # Panel C: Prediction error
    ax = axes[2]
    ax.plot(t_range, results["prediction_errors"], color=COLORS["attention"], alpha=0.7)
    ax.set_ylabel("Prediction Error")
    ax.set_xlabel("Time step")
    ax.set_ylim(0, None)
    add_panel_label(ax, 'C')

    fig.align_ylabels(axes)

    # Add accuracy annotation
    acc = results["accuracy"]
    fig.text(0.99, 0.01, f"Accuracy: {acc:.1%}", ha='right', va='bottom',
             fontsize=10, color=COLORS["true_state"])

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
    A) Breath inference quality
    B) Precision (zeta) - showing focused vs distracted
    C) Attention state (true)
    """
    setup_style()

    T = results["T"]
    t_range = np.arange(T)
    distraction_onset = results["distraction_onset"]

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True,
                              gridspec_kw={'height_ratios': [1, 0.8, 0.6], 'hspace': 0.15})

    # Shade distraction period on all panels
    for ax in axes:
        shade_distraction_period(ax, distraction_onset, T)

    # Panel A: Breath inference
    ax = axes[0]
    p_inhale = results["posteriors"][:, INHALE]
    ax.plot(t_range, p_inhale, color=COLORS["breath"], label='P(Inhaling)')
    true_inhale = 1.0 - results["true_states"]
    ax.scatter(t_range, true_inhale, s=8, color=COLORS["true_state"], alpha=0.4, label='True state')
    ax.set_ylabel("Probability")
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks([0, 0.5, 1])
    ax.legend(loc='upper right', framealpha=0.9)
    add_panel_label(ax, 'A')
    ax.set_title("Breath State Inference", fontsize=12)

    # Panel B: Precision
    ax = axes[1]
    ax.plot(t_range, results["zeta_history"], color=COLORS["precision"])
    ax.axhline(y=1.0, color=COLORS["true_state"], linestyle='--', linewidth=1, alpha=0.6)
    ax.set_ylabel("Precision (ζ)")

    # Add annotations
    params = results["params"]
    ax.annotate(f'Focused: ζ={params.zeta_focused}',
                xy=(distraction_onset/4, params.zeta_focused),
                fontsize=9, color=COLORS["focused"])
    ax.annotate(f'Distracted: ζ={params.zeta_distracted}',
                xy=(distraction_onset + T/8, params.zeta_distracted),
                fontsize=9, color=COLORS["distracted"])

    add_panel_label(ax, 'B')

    # Panel C: Attention state
    ax = axes[2]
    attention = results["attention_states"]
    ax.fill_between(t_range, 0, 1, where=(attention == FOCUSED),
                    color=COLORS["focused"], alpha=0.6, label='Focused')
    ax.fill_between(t_range, 0, 1, where=(attention == DISTRACTED),
                    color=COLORS["distracted"], alpha=0.6, label='Distracted')
    ax.set_ylabel("Attention")
    ax.set_xlabel("Time step")
    ax.set_yticks([])
    ax.legend(loc='upper right', framealpha=0.9, ncol=2)
    add_panel_label(ax, 'C')

    fig.align_ylabels(axes)

    # Add accuracy annotations
    fig.text(0.99, 0.01,
             f"Focused accuracy: {results['accuracy_focused']:.1%} | "
             f"Distracted accuracy: {results['accuracy_distracted']:.1%}",
             ha='right', va='bottom', fontsize=10, color=COLORS["true_state"])

    if save_path:
        save_figure(fig, save_path)

    return fig


# =============================================================================
# Figure 3: Precision Dynamics Improve Learning
# =============================================================================

def plot_figure3(
    results_fixed: Dict[str, Any],
    results_dynamic: Dict[str, Any],
    results_attention: Dict[str, Any],
    save_path: Optional[str] = None
):
    """
    Figure 3: Precision dynamics improve A1 learning.

    Two panels:
    A) A1 error over time (three curves)
    B) Final A1 matrices comparison
    """
    setup_style()

    T = results_fixed["T"]
    t_range = np.arange(T)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4),
                              gridspec_kw={'width_ratios': [2, 1], 'wspace': 0.3})

    # Panel A: Learning curves
    ax = axes[0]
    ax.plot(t_range, results_fixed["A1_error"], color=COLORS["true_state"],
            label='Fixed ζ=1', linestyle='--')
    ax.plot(t_range, results_dynamic["A1_error"], color=COLORS["precision"],
            label='Dynamic ζ (B.45)')
    ax.plot(t_range, results_attention["A1_error"], color=COLORS["focused"],
            label='+ Attention inference')

    ax.set_xlabel("Time step")
    ax.set_ylabel("Mean |A1 - A1_true|")
    ax.set_title("Likelihood Learning Convergence", fontsize=12)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_ylim(0, None)
    add_panel_label(ax, 'A')

    # Panel B: Final A1 comparison (bar chart)
    ax = axes[1]

    A1_true = results_fixed["A1_true"]
    A1_fixed = results_fixed["A1_history"][-1]
    A1_dynamic = results_dynamic["A1_history"][-1]
    A1_attention = results_attention["A1_history"][-1]

    # Show diagonal elements (correct observation probabilities)
    x = np.arange(3)
    width = 0.2

    true_diag = (A1_true[0, 0] + A1_true[1, 1]) / 2
    fixed_diag = (A1_fixed[0, 0] + A1_fixed[1, 1]) / 2
    dynamic_diag = (A1_dynamic[0, 0] + A1_dynamic[1, 1]) / 2
    attention_diag = (A1_attention[0, 0] + A1_attention[1, 1]) / 2

    bar_colors = [COLORS["true_state"], COLORS["true_state"],
                  COLORS["precision"], COLORS["focused"]]
    bar_alphas = [1.0, 0.5, 0.8, 0.8]
    bar_values = [true_diag, fixed_diag, dynamic_diag, attention_diag]
    bar_labels = ['True', 'Fixed ζ', 'Dynamic ζ', '+ Attention']

    for i, (label, val, color, alpha) in enumerate(zip(bar_labels, bar_values, bar_colors, bar_alphas)):
        ax.bar(i, val, color=color, alpha=alpha)

    ax.set_xticks(range(len(bar_labels)))
    ax.set_xticklabels(bar_labels)
    ax.set_ylabel("A1 diagonal (accuracy)")
    ax.set_title("Final Learned A1", fontsize=12)
    ax.set_ylim(0.5, 1.0)
    add_panel_label(ax, 'B', x=-0.15)

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
