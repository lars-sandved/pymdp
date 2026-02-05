#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate publication figures for focused attention meditation paper.

Usage:
    python run_figures.py --figure 1
    python run_figures.py --figure 2
    python run_figures.py --figure 3
    python run_figures.py --figure 4
    python run_figures.py --figure 5
    python run_figures.py --figure 6
    python run_figures.py --figure 4-5  # Combined comparison
    python run_figures.py --all

Figure sequence:
    1. Breath Perception with Dynamic Precision
    2. Attention Modulates Precision
    3. Precision Dynamics Improve Learning
    4. The Attention Trap (Non-Meditator)
    5. Meditation Instruction Breaks the Cycle
    6. Learning Across Sits
"""

import argparse
import os
import sys
import matplotlib.pyplot as plt

# Add parent directories to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from examples.meditation.models import ModelParams
from examples.meditation.simulation import (
    run_figure1, run_figure2, run_figure2_1, run_figure3,
    run_figure4, run_figure5, run_figure6,
    run_figure4_learning_sweep, run_figure4_diagnostic,
    run_figure2_1_with_A2_precision,
)
from examples.meditation.plotting import (
    plot_figure1, plot_figure2, plot_figure2_1, plot_figure3,
    plot_figure4, plot_figure5, plot_figure6,
    plot_figures4_5_combined, plot_figure4_heatmap, plot_figure4_diagnostic,
    plot_A2_precision_comparison,
)


def get_output_dir():
    """Get or create output directory."""
    here = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(here, "outputs")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def generate_figure1(save: bool = True, show: bool = False):
    """Generate Figure 1: Breath Perception with Dynamic Precision."""
    print("\n" + "="*60)
    print("Figure 1: Breath Perception with Dynamic Precision")
    print("="*60)

    params = ModelParams(zeta_prior_var=4.0, zeta_step=1.0, A1_precision=0.8)
    results = run_figure1(T=100, seed=42, params=params)

    print(f"  Accuracy: {results['accuracy']:.1%}")
    print(f"  Mean precision: {results['zeta_history'].mean():.2f}")

    save_path = os.path.join(get_output_dir(), "figure1_breath_precision.png") if save else None
    fig = plot_figure1(results, save_path=save_path)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return results


def generate_figure2(save: bool = True, show: bool = False):
    """Generate Figure 2: Attention Modulates Precision."""
    print("\n" + "="*60)
    print("Figure 2: Attention Modulates Precision")
    print("="*60)

    params = ModelParams(
        zeta_prior_var=2.0,
        zeta_step=0.5,
        A1_precision=0.8,
        A2_precision_obs=0.85,  # Strong enough for attention inference
        A2_true_precision=1.0   # Deterministic: obs fully determined by attention state
    )
    results = run_figure2(T=100, seed=42, params=params, distraction_onset=50)

    print(f"  Focused accuracy: {results['accuracy_focused']:.1%}")
    print(f"  Distracted accuracy: {results['accuracy_distracted']:.1%}")

    save_path = os.path.join(get_output_dir(), "figure2_attention_precision.png") if save else None
    fig = plot_figure2(results, save_path=save_path)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return results


def generate_figure2_1(save: bool = True, show: bool = False):
    """Generate Figure 2.1: Attention with Mental Action."""
    print("\n" + "="*60)
    print("Figure 2.1: Attention with Mental Action")
    print("="*60)

    params = ModelParams(
        zeta_prior_var=2.0,
        zeta_step=0.25,
        A1_precision=0.75,
        A2_precision_obs=0.6,  # Agent's attention inference (weaker than truth)
        A2_true_precision=0.95,  # True precision observation model
        B2_stay_prob=0.97,  # Agent's B2 aligned with p_stay_focused
        B2_switch_prob=0.8,  # Agent's uncertain belief about switch effect
        zeta_focused=2.0,
        zeta_distracted=0.5,
        gamma=16.0,
        E_stay=0.99,  # Strong habit of staying
        C_precision_precise=2.0,  # Preference for focused (precise) observations
        C_precision_imprecise=0.0,
        p_stay_focused=0.97,  # 3% drift probability per step
        p_stay_distracted=1.0,  # Absorbing once distracted
        p_switch_success=1.0,  # True switch is deterministic
    )
    results = run_figure2_1(T=200, seed=123, params=params)

    print(f"  Time distracted: {results['time_distracted']:.1%}")
    print(f"  Switch rate: {results['switch_rate']:.1%}")
    print(f"  Attention accuracy: {results['attention_accuracy']:.1%}")

    save_path = os.path.join(get_output_dir(), "figure2_1_attention_action.png") if save else None
    fig = plot_figure2_1(results, save_path=save_path)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return results


def generate_figure3(save: bool = True, show: bool = False):
    """Generate Figure 3: Precision Dynamics Improve Learning."""
    print("\n" + "="*60)
    print("Figure 3: Precision Dynamics Improve Learning")
    print("="*60)

    params = ModelParams(
        zeta_prior_var=2.0,
        zeta_step=0.25,
        A1_precision=0.9,  # True A1 (environment)
        A_learning_rate=0.1,
        forgetting_rate=0.9,
        zeta_focused=1.25,
        zeta_distracted=0.8,  # 1/1.25
    )

    print("  Running fixed precision mode...")
    results_fixed = run_figure3(num_sits=300, T_per_sit=100, seed=42, params=params, mode="fixed", A1_init_precision=0.52)
    print(f"    Final A1 diagonal: {results_fixed['A1_diagonal'][-1]:.4f} (true: {results_fixed['A1_true_diagonal']:.4f})")

    print("  Running dynamic precision mode...")
    results_dynamic = run_figure3(num_sits=300, T_per_sit=100, seed=42, params=params, mode="dynamic", A1_init_precision=0.52)
    print(f"    Final A1 diagonal: {results_dynamic['A1_diagonal'][-1]:.4f} (true: {results_dynamic['A1_true_diagonal']:.4f})")

    print("  Running with attention inference...")
    results_attention = run_figure3(num_sits=300, T_per_sit=100, seed=42, params=params, mode="attention", A1_init_precision=0.52)
    print(f"    Final A1 diagonal: {results_attention['A1_diagonal'][-1]:.4f} (true: {results_attention['A1_true_diagonal']:.4f})")

    save_path = os.path.join(get_output_dir(), "figure3_precision_learning.png") if save else None
    fig = plot_figure3(results_fixed, results_dynamic, results_attention, save_path=save_path)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return {"fixed": results_fixed, "dynamic": results_dynamic, "attention": results_attention}


def generate_figure4(save: bool = True, show: bool = False):
    """Generate Figure 4: The Distraction Trap - Learning Heatmap."""
    print("\n" + "="*60)
    print("Figure 4: The Distraction Trap (Learning Sweep)")
    print("="*60)

    import numpy as np

    # Use Figure 2.1 parameters for full hierarchical stack
    params = ModelParams(
        # Breath model (Level 1)
        A1_precision=0.75,
        # Precision dynamics
        zeta_prior_var=2.0,
        zeta_step=0.25,
        zeta_focused=2.0,
        zeta_distracted=0.5,
        # Attention environment
        A2_true_precision=0.95,
        p_stay_focused=0.97,  # 3% drift rate (same as Fig 2.1)
        p_stay_distracted=1.0,  # Absorbing
        p_switch_success=1.0,  # Deterministic switch
        # Policy selection
        gamma=16.0,
        E_stay=0.99,  # Strong habit (same as Fig 2.1)
        C_precision_precise=2.0,
        C_precision_imprecise=0.0,
        # Learning
        A_learning_rate=0.1,
        B_learning_rate=0.1,
        forgetting_rate=0.9,
    )

    # Sweep over initial precision parameters
    # zeta: A2 (likelihood) precision, omega: B2 (transition) precision
    # 0 = completely flat, 1 = matches true model
    zeta_range = np.linspace(0.0, 1.0, 11)
    omega_range = np.linspace(0.0, 1.0, 11)

    results = run_figure4_learning_sweep(
        zeta_range=zeta_range,
        omega_range=omega_range,
        num_sits=100,
        T_per_sit=100,  # Same as Fig 3
        seed=42,
        params=params,
    )

    # Summary stats
    data = results["final_time_focused"]
    print(f"  Min time focused: {data.min():.1%}")
    print(f"  Max time focused: {data.max():.1%}")
    print(f"  Mean time focused: {data.mean():.1%}")

    # Count trap region (< 50% focused)
    trap_fraction = (data < 0.5).mean()
    print(f"  Fraction trapped: {trap_fraction:.1%}")

    save_path = os.path.join(get_output_dir(), "figure4_distraction_trap.png") if save else None
    fig = plot_figure4_heatmap(results, save_path=save_path)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return results


def generate_figure4_diagnostic(save: bool = True, show: bool = False):
    """Generate Figure 4 diagnostic: within-sit dynamics for low-zeta cases."""
    print("\n" + "="*60)
    print("Figure 4 Diagnostic: Within-Sit Dynamics")
    print("="*60)

    params = ModelParams(
        A2_true_precision=0.95,
        p_stay_focused=0.95,
        p_stay_distracted=1.0,
        p_switch_success=1.0,
        gamma=16.0,
        E_stay=0.9,
        C_precision_precise=2.0,
        C_precision_imprecise=0.0,
        A_learning_rate=0.1,
        B_learning_rate=0.1,
        forgetting_rate=0.9,
    )

    # Test cases: ζ=0 with varying ω
    test_cases = [
        (0.0, 0.2),
        (0.0, 0.3),
        (0.0, 0.5),
        (0.1, 0.3),  # Compare to ζ=0.1
    ]

    capture_sits = [0, 49, 99]  # Sit 1, 50, 100

    results_list = []
    for zeta, omega in test_cases:
        print(f"  Running ζ={zeta:.1f}, ω={omega:.1f}...")
        result = run_figure4_diagnostic(
            zeta_A2=zeta,
            omega_B2=omega,
            num_sits=100,
            T_per_sit=100,
            seed=42,
            params=params,
            capture_sits=capture_sits,
        )
        results_list.append(result)

    save_path = os.path.join(get_output_dir(), "figure4_diagnostic.pdf") if save else None
    fig = plot_figure4_diagnostic(results_list, save_path=save_path)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return results_list


def generate_A2_precision_diagnostic(save: bool = True, show: bool = False):
    """Generate A2 precision diagnostic: full stack with varying meta-awareness."""
    print("\n" + "="*60)
    print("A2 Precision Diagnostic: Effect of Meta-Awareness")
    print("="*60)

    # Use same params as Figure 2.1 but vary A2 precision
    params = ModelParams(
        zeta_prior_var=2.0,
        zeta_step=0.25,
        A1_precision=0.75,
        A2_true_precision=0.95,
        B2_stay_prob=0.97,
        B2_switch_prob=0.8,
        zeta_focused=2.0,
        zeta_distracted=0.5,
        gamma=16.0,
        E_stay=0.99,
        C_precision_precise=2.0,
        C_precision_imprecise=0.0,
        p_stay_focused=0.97,
        p_stay_distracted=1.0,
        p_switch_success=1.0,
    )

    # Test different A2 precision values
    A2_precisions = [0.5, 0.6, 0.75, 0.9]

    results_list = []
    for A2_prec in A2_precisions:
        print(f"  Running A2_precision={A2_prec:.2f}...")
        result = run_figure2_1_with_A2_precision(
            T=200,
            seed=123,  # Same seed as Figure 2.1
            params=params,
            A2_precision_override=A2_prec,
        )
        results_list.append(result)
        print(f"    Time distracted: {result['time_distracted']:.1%}")

    save_path = os.path.join(get_output_dir(), "A2_precision_diagnostic.pdf") if save else None
    fig = plot_A2_precision_comparison(results_list, save_path=save_path)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return results_list


def generate_figure5(save: bool = True, show: bool = False):
    """Generate Figure 5: Meditation Instruction."""
    print("\n" + "="*60)
    print("Figure 5: Meditation Instruction Breaks the Cycle")
    print("="*60)

    params = ModelParams()
    results = run_figure5(T=100, seed=42, params=params, distraction_onset=20)

    print(f"  Time distracted: {results['time_distracted']:.1%}")
    print(f"  Switch rate: {results['switch_rate']:.1%}")
    print(f"  Attention accuracy: {results['attention_accuracy']:.1%}")

    save_path = os.path.join(get_output_dir(), "figure5_meditation_instruction.png") if save else None
    fig = plot_figure5(results, save_path=save_path)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return results


def generate_figures4_5_combined(save: bool = True, show: bool = False):
    """Generate combined Figure 4+5 comparison."""
    print("\n" + "="*60)
    print("Figure 4-5: Non-Meditator vs Meditator Comparison")
    print("="*60)

    params = ModelParams()

    print("  Running non-meditator simulation...")
    results_trap = run_figure4(T=100, seed=42, params=params, distraction_onset=20)
    print(f"    Time distracted: {results_trap['time_distracted']:.1%}")

    print("  Running meditator simulation...")
    results_meditation = run_figure5(T=100, seed=42, params=params, distraction_onset=20)
    print(f"    Time distracted: {results_meditation['time_distracted']:.1%}")

    save_path = os.path.join(get_output_dir(), "figure4_5_comparison.png") if save else None
    fig = plot_figures4_5_combined(results_trap, results_meditation, save_path=save_path)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return {"trap": results_trap, "meditation": results_meditation}


def generate_figure6(save: bool = True, show: bool = False):
    """Generate Figure 6: Learning Across Sits."""
    print("\n" + "="*60)
    print("Figure 6: Learning Across Sits")
    print("="*60)

    params = ModelParams()

    print("  Running non-meditator baseline (200 sits)...")
    results_baseline = run_figure6(
        num_sits=200, T_per_sit=100, seed=42, params=params,
        meditation_start_sit=None  # No instruction
    )
    print(f"    Mean time distracted: {results_baseline['time_distracted'].mean():.1%}")
    print(f"    Final A2 diagonal: {results_baseline['A2_diagonal'][-1]:.3f}")

    print("  Running meditator (instruction at sit 100)...")
    results_meditation = run_figure6(
        num_sits=200, T_per_sit=100, seed=42, params=params,
        meditation_start_sit=100  # Instruction starts at sit 100
    )
    print(f"    Mean time distracted (after instruction): {results_meditation['time_distracted'][100:].mean():.1%}")
    print(f"    Final A2 diagonal: {results_meditation['A2_diagonal'][-1]:.3f}")

    save_path = os.path.join(get_output_dir(), "figure6_learning_across_sits.png") if save else None
    fig = plot_figure6(results_baseline, results_meditation, save_path=save_path)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return {"baseline": results_baseline, "meditation": results_meditation}


def generate_all(save: bool = True, show: bool = False):
    """Generate all figures."""
    print("\n" + "="*60)
    print("Generating All Figures")
    print("="*60)

    generate_figure1(save=save, show=False)
    generate_figure2(save=save, show=False)
    generate_figure3(save=save, show=False)
    generate_figure4(save=save, show=False)
    generate_figure5(save=save, show=False)
    generate_figures4_5_combined(save=save, show=False)
    generate_figure6(save=save, show=False)

    print("\n" + "="*60)
    print("All figures generated successfully!")
    print(f"Output directory: {get_output_dir()}")
    print("="*60)

    if show:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Generate publication figures for meditation paper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--figure", "-f",
        type=str,
        default="all",
        help="Figure to generate: 1, 2, 2.1, 3, 4, 5, 6, 4-5, or 'all'"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save figures to disk"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures interactively"
    )

    args = parser.parse_args()
    save = not args.no_save

    if args.figure == "1":
        generate_figure1(save=save, show=args.show)
    elif args.figure == "2":
        generate_figure2(save=save, show=args.show)
    elif args.figure == "2.1":
        generate_figure2_1(save=save, show=args.show)
    elif args.figure == "3":
        generate_figure3(save=save, show=args.show)
    elif args.figure == "4":
        generate_figure4(save=save, show=args.show)
    elif args.figure == "4d":
        generate_figure4_diagnostic(save=save, show=args.show)
    elif args.figure == "A2d":
        generate_A2_precision_diagnostic(save=save, show=args.show)
    elif args.figure == "5":
        generate_figure5(save=save, show=args.show)
    elif args.figure == "4-5":
        generate_figures4_5_combined(save=save, show=args.show)
    elif args.figure == "6":
        generate_figure6(save=save, show=args.show)
    elif args.figure.lower() == "all":
        generate_all(save=save, show=args.show)
    else:
        print(f"Unknown figure: {args.figure}")
        print("Valid options: 1, 2, 2.1, 3, 4, 4d, 5, 6, 4-5, all")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
