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
    run_figure1, run_figure2, run_figure3,
    run_figure4, run_figure5, run_figure6,
)
from examples.meditation.plotting import (
    plot_figure1, plot_figure2, plot_figure3,
    plot_figure4, plot_figure5, plot_figure6,
    plot_figures4_5_combined,
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

    params = ModelParams()
    results = run_figure1(T=200, seed=42, params=params)

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

    params = ModelParams()
    results = run_figure2(T=200, seed=42, params=params, distraction_onset=100)

    print(f"  Focused accuracy: {results['accuracy_focused']:.1%}")
    print(f"  Distracted accuracy: {results['accuracy_distracted']:.1%}")

    save_path = os.path.join(get_output_dir(), "figure2_attention_precision.png") if save else None
    fig = plot_figure2(results, save_path=save_path)

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

    params = ModelParams()

    print("  Running fixed precision mode...")
    results_fixed = run_figure3(T=500, seed=42, params=params, mode="fixed")
    print(f"    Final A1 error: {results_fixed['A1_error'][-1]:.4f}")

    print("  Running dynamic precision mode...")
    results_dynamic = run_figure3(T=500, seed=42, params=params, mode="dynamic")
    print(f"    Final A1 error: {results_dynamic['A1_error'][-1]:.4f}")

    print("  Running with attention inference...")
    results_attention = run_figure3(T=500, seed=42, params=params, mode="attention")
    print(f"    Final A1 error: {results_attention['A1_error'][-1]:.4f}")

    save_path = os.path.join(get_output_dir(), "figure3_precision_learning.png") if save else None
    fig = plot_figure3(results_fixed, results_dynamic, results_attention, save_path=save_path)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return {"fixed": results_fixed, "dynamic": results_dynamic, "attention": results_attention}


def generate_figure4(save: bool = True, show: bool = False):
    """Generate Figure 4: The Attention Trap."""
    print("\n" + "="*60)
    print("Figure 4: The Attention Trap (Non-Meditator)")
    print("="*60)

    params = ModelParams()
    results = run_figure4(T=300, seed=42, params=params, distraction_onset=50)

    print(f"  Time distracted: {results['time_distracted']:.1%}")
    print(f"  Switch rate: {results['switch_rate']:.1%}")
    print(f"  Attention accuracy: {results['attention_accuracy']:.1%}")

    save_path = os.path.join(get_output_dir(), "figure4_attention_trap.png") if save else None
    fig = plot_figure4(results, save_path=save_path)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return results


def generate_figure5(save: bool = True, show: bool = False):
    """Generate Figure 5: Meditation Instruction."""
    print("\n" + "="*60)
    print("Figure 5: Meditation Instruction Breaks the Cycle")
    print("="*60)

    params = ModelParams()
    results = run_figure5(T=300, seed=42, params=params, distraction_onset=50)

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
    results_trap = run_figure4(T=300, seed=42, params=params, distraction_onset=50)
    print(f"    Time distracted: {results_trap['time_distracted']:.1%}")

    print("  Running meditator simulation...")
    results_meditation = run_figure5(T=300, seed=42, params=params, distraction_onset=50)
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
        help="Figure to generate: 1, 2, 3, 4, 5, 6, 4-5, or 'all'"
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
    elif args.figure == "3":
        generate_figure3(save=save, show=args.show)
    elif args.figure == "4":
        generate_figure4(save=save, show=args.show)
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
        print("Valid options: 1, 2, 3, 4, 5, 6, 4-5, all")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
