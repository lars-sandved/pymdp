"""
Focused Attention Meditation Simulation

A hierarchical active inference model of meditation and breath perception.

Modules:
    models.py      - Generative model builders (A, B, C matrices)
    simulation.py  - Core simulation functions
    plotting.py    - Publication-quality figure generation
    run_figures.py - CLI entry point for generating figures

Usage:
    python run_figures.py --all
    python run_figures.py --figure 1

Figure sequence:
    1. Breath Perception with Dynamic Precision
    2. Attention Modulates Precision
    3. Precision Dynamics Improve Learning
    4. The Attention Trap (Non-Meditator)
    5. Meditation Instruction Breaks the Cycle
    6. Learning Across Sits
"""

from .models import (
    ModelParams,
    build_breath_model,
    build_attention_model,
    build_awareness_model,
    build_environment,
    INHALE, EXHALE,
    FOCUSED, DISTRACTED,
    AWARE, UNAWARE,
    STAY, SWITCH,
)

from .simulation import (
    run_figure1,
    run_figure2,
    run_figure3,
    run_figure4,
    run_figure5,
    run_figure6,
)

from .plotting import (
    plot_figure1,
    plot_figure2,
    plot_figure3,
    plot_figure4,
    plot_figure5,
    plot_figure6,
)

__all__ = [
    # Models
    "ModelParams",
    "build_breath_model",
    "build_attention_model",
    "build_awareness_model",
    "build_environment",
    # Constants
    "INHALE", "EXHALE",
    "FOCUSED", "DISTRACTED",
    "AWARE", "UNAWARE",
    "STAY", "SWITCH",
    # Simulation
    "run_figure1",
    "run_figure2",
    "run_figure3",
    "run_figure4",
    "run_figure5",
    "run_figure6",
    # Plotting
    "plot_figure1",
    "plot_figure2",
    "plot_figure3",
    "plot_figure4",
    "plot_figure5",
    "plot_figure6",
]
