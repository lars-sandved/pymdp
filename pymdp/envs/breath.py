#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Breath cycle environment (2-state)

Two-phase breathing cycle with variable phase durations:
    - 0: INHALE
    - 1: EXHALE

Observations (single modality):
    - 0: EXPANSION (maps to INHALE)
    - 1: CONTRACTION (maps to EXHALE)

The environment advances through a semi-Markov cycle with variable durations
for each phase. By default, inhale/exhale last ~4-7 steps; the cycle repeats.
"""

from typing import Optional
import numpy as np

from .env import Env


class BreathEnv(Env):
    """Two-stage breath cycle with variable durations (no explicit pauses)."""

    # Hidden state labels
    INHALE = 0
    EXHALE = 1

    # Observation labels
    EXPANSION = 0
    CONTRACTION = 1

    def __init__(
        self,
        inhale_range: tuple[int, int] = (4, 7),
        exhale_range: tuple[int, int] = (4, 7),
        p_correct: float = 0.98,
        seed: Optional[int] = None,
    ) -> None:
        """
        Parameters
        ----------
        inhale_range: (low, high)
            Inclusive range to sample inhale duration in timesteps.
        exhale_range: (low, high)
            Inclusive range to sample exhale duration in timesteps.
        p_correct: float
            Probability of emitting the 'intended' observation for INHALE/EXHALE.
        seed: Optional[int]
            RNG seed for reproducibility.
        """

        self.inhale_range = inhale_range
        self.exhale_range = exhale_range
        self.p_correct = float(p_correct)

        self._rng = np.random.default_rng(seed)

        # Discrete space sizes
        self.n_states = 2
        self.n_observations = 2

        # Internal state
        self._phase_sequence = []  # list of phase labels for the current cycle
        self._phase_index = 0      # index into the current sequence
        self._state = None         # current phase label (0..3)

        # initialize first cycle
        self._regenerate_cycle()

    def _sample_duration(self, low: int, high: int) -> int:
        # inclusive integer range
        return int(self._rng.integers(low, high + 1))

    def _regenerate_cycle(self) -> None:
        inhale_len = self._sample_duration(*self.inhale_range)
        exhale_len = self._sample_duration(*self.exhale_range)

        seq = ([self.INHALE] * inhale_len) + ([self.EXHALE] * exhale_len)
        self._phase_sequence = seq
        self._phase_index = 0
        self._state = self._phase_sequence[self._phase_index]

    def _advance(self) -> None:
        self._phase_index += 1
        if self._phase_index >= len(self._phase_sequence):
            self._regenerate_cycle()
        else:
            self._state = self._phase_sequence[self._phase_index]

    def reset(self, state: Optional[int] = None):
        """Reset the breath cycle. If state is provided, start from that phase; otherwise
        start from the beginning of a freshly sampled cycle.

        Returns
        -------
        observation: int
            Observation index for the current state.
        """
        self._regenerate_cycle()
        if state is not None:
            if state < 0 or state >= self.n_states:
                raise ValueError("`state` must be in [0, 1]")
            # place the first occurrence of the requested state at the start if present
            try:
                first_idx = self._phase_sequence.index(state)
                # rotate sequence so that it starts at first_idx
                self._phase_sequence = self._phase_sequence[first_idx:] + self._phase_sequence[:first_idx]
                self._phase_index = 0
                self._state = self._phase_sequence[self._phase_index]
            except ValueError:
                # fallback: set current state directly and rebuild a minimal sequence from there
                self._state = state
                self._phase_sequence = [state]
                self._phase_index = 0
        return self._emit_observation()

    def step(self, action):
        """Advance the breath cycle by one timestep.

        Parameters
        ----------
        action: ignored (no control in this environment)

        Returns
        -------
        observation: int
            Observation index for the new state.
        """
        self._advance()
        return self._emit_observation()

    def _emit_observation(self) -> int:
        s = self._state
        if s == self.INHALE:
            # emit EXPANSION with probability p_correct, else CONTRACTION
            if self._rng.random() < self.p_correct:
                return self.EXPANSION
            else:
                return self.CONTRACTION
        else:  # EXHALE
            if self._rng.random() < self.p_correct:
                return self.CONTRACTION
            else:
                return self.EXPANSION

    def render(self):
        pass

    def get_transition_dist(self):
        """Return a nominal transition model B (shape [states, states, 1]).

        Note: The actual GP is semi-Markov with sampled durations, so this B is an
        approximation capturing expected self-transitions in phases and progression order.
        """
        stay_p_inhale = 1.0 - 1.0 / max(1, np.mean(self.inhale_range))
        stay_p_exhale = 1.0 - 1.0 / max(1, np.mean(self.exhale_range))

        B = np.zeros((self.n_states, self.n_states, 1))
        # INHALE -> INHALE or EXHALE
        B[self.INHALE, self.INHALE, 0] = stay_p_inhale
        B[self.EXHALE, self.INHALE, 0] = 1.0 - stay_p_inhale
        # EXHALE -> EXHALE or INHALE
        B[self.EXHALE, self.EXHALE, 0] = stay_p_exhale
        B[self.INHALE, self.EXHALE, 0] = 1.0 - stay_p_exhale
        return B

    def get_likelihood_dist(self):
        """Return an observation model A (shape [obs, states]).

        INHALE  -> EXPANSION with high probability
        EXHALE  -> CONTRACTION with high probability
        """
        A = np.zeros((self.n_observations, self.n_states))
        eps = 1.0 - self.p_correct

        # INHALE column
        A[self.EXPANSION, self.INHALE] = self.p_correct
        A[self.CONTRACTION, self.INHALE] = eps

        # EXHALE column
        A[self.CONTRACTION, self.EXHALE] = self.p_correct
        A[self.EXPANSION, self.EXHALE] = eps

        return A

    @property
    def state(self) -> int:
        return int(self._state)


