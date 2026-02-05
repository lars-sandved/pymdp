# Focused Attention Meditation: An Active Inference Account

## Document Purpose
This document provides comprehensive context for the figures and simulations developed for a paper on focused attention meditation using active inference. It is intended to accompany a draft manuscript and provide sufficient detail for completing the paper.

---

## 1. Overview and Core Thesis

### The Central Claim
Focused attention meditation can be understood as a form of hierarchical active inference where:
1. **Meta-awareness** (awareness of one's attention state) is formalized as inference over a hidden state variable
2. **Meditation instruction** provides prior structure that enables this inference
3. **Without this structure**, agents become trapped in distraction because they cannot detect when attention has wandered

### Why Active Inference?
Active inference provides a principled framework for understanding meditation because:
- It unifies perception, action, and learning under a single objective (minimizing free energy)
- It naturally accommodates hierarchical processing (breath perception → attention state → meta-awareness)
- Precision weighting provides a formal account of how attention modulates perception
- The distinction between model structure and model parameters maps onto the distinction between meditation instruction and meditation practice

---

## 2. The Hierarchical Model Architecture

### Level 1: Breath Perception (Interoceptive Inference)
The foundation is a simple two-state model of breath perception:
- **Hidden states**: INHALE, EXHALE
- **Observations**: EXPANSION, CONTRACTION (bodily sensations)
- **A1 matrix**: P(observation | breath_state) - the likelihood mapping
- **B1 matrix**: P(breath_state' | breath_state) - transition dynamics based on breath phase durations

The agent performs Bayesian inference to track the breath:
```
posterior ∝ likelihood × prior
qs_breath = A1[obs, :] * (B1 @ qs_breath_prev)
```

### Level 2: Attention State
Attention is modeled as a hidden state that affects the quality of breath perception:
- **Hidden states**: FOCUSED, DISTRACTED
- **Observations**: PRECISE, IMPRECISE (precision of breath perception)
- **A2 matrix**: P(precision_obs | attention_state) - the likelihood mapping
- **B2 matrix**: P(attention' | attention, action) - transition dynamics under mental actions

Key asymmetry in the environment (B2_true):
- **Distraction is absorbing**: P(distracted → distracted | STAY) = 1.0
- **Focus drifts naturally**: P(focused → distracted | STAY) = 0.03 per timestep
- **SWITCH action can restore focus**: P(distracted → focused | SWITCH) = 1.0

### Level 3: Meta-Awareness (Implicit)
While not explicitly modeled as a separate level in most figures, meta-awareness emerges from the agent's ability to infer its own attention state and select appropriate actions.

### The Hierarchical Message Passing
Two directions of information flow:

**Ascending (bottom-up):**
1. Breath observations inform breath perception
2. Quality of breath perception generates precision observations
3. Precision observations inform attention state inference

**Descending (top-down):**
1. Attention state sets prior on precision (zeta_prior)
2. Precision modulates the likelihood used for breath inference

---

## 3. Precision Dynamics (Equation B.45)

### The Core Mechanism
Precision (ζ) modulates how much observations influence beliefs. Following Parr, Pezzulo & Friston (2022) Appendix B, Equation B.45, precision is updated to minimize free energy:

```python
# Gradient of free energy with respect to log-precision
# Balances prediction error against prior expectations
zeta_new = zeta_prior + zeta_step * gradient
```

### How Precision Scaling Works
The `scale_likelihood` function raises the likelihood matrix to a power:
```python
A_scaled = A^zeta  # then renormalize columns
```

Effects:
- **ζ = 0**: Flat likelihood, observations provide no information
- **ζ = 1**: Original likelihood, normal inference
- **ζ > 1**: Sharpened likelihood, observations dominate over priors
- **ζ < 1**: Flattened likelihood, priors dominate over observations

### Attention Modulates Precision
When focused:
- zeta_focused = 2.0 (high precision, clear breath perception)

When distracted:
- zeta_distracted = 0.5 (low precision, unclear breath perception)

This creates a bidirectional link:
- Attention state determines precision of breath perception
- Precision observations reveal attention state

---

## 4. The Distraction Trap

### Why Distraction is Absorbing
In the model (reflecting reality), distraction has an asymmetric structure:
- Focus naturally drifts toward distraction (mind-wandering)
- Distraction doesn't naturally return to focus
- Active intervention (SWITCH action) is required to restore focus

### The Meta-Awareness Requirement
To escape distraction, an agent must:
1. **Detect** that they are distracted (infer attention state)
2. **Decide** to switch attention (action selection)
3. **Execute** the switch (which succeeds deterministically in the model)

The critical bottleneck is step 1: without accurate inference of attention state, the agent doesn't know to switch.

### The Precision of Meta-Awareness (A2)
The agent's A2 matrix determines how well precision observations inform attention inference:
- **A2_precision = 0.5 (flat)**: Observations provide no information about attention
- **A2_precision = 0.6**: Weak signal, often insufficient
- **A2_precision = 0.75**: Threshold for effective meta-awareness
- **A2_precision = 0.9+**: Reliable attention inference

This is the key insight: **meta-awareness precision determines whether you can escape the distraction trap**.

---

## 5. Meditation Instruction as Prior Structure

### What Meditation Instruction Provides
In the model, "meditation instruction" is formalized as:
1. **Knowledge that attention wanders** (structure in A2)
2. **Knowledge that you can redirect attention** (structure in B2 for SWITCH action)
3. **Preference for focused states** (C2 preferences)

### The Non-Meditator vs Meditator Distinction

**Non-Meditator (weak A2):**
- Flat or weak A2: can't reliably detect distraction
- Uncertain B2: doesn't know SWITCH redirects attention
- Gets trapped in distraction

**Meditator (structured A2):**
- Precise A2: can detect when attention has wandered
- Structured B2: knows SWITCH action restores focus
- Can maintain focus through active monitoring and correction

### Why Structure Can't Be Learned Without Structure
This is the core paradox the paper addresses:
- To learn A2 (attention → precision mapping), you need informative posteriors over attention
- To have informative posteriors, you need structured A2
- Without initial structure, learning signals are too weak/noisy

This is why meditation instruction is necessary - it provides the initial structure that enables learning.

---

## 6. Figure Narratives

### Figure 1: Breath Perception with Dynamic Precision
**Purpose**: Demonstrate basic breath tracking with precision dynamics.

**Setup**:
- Single agent tracking breath state over 100 timesteps
- Precision (ζ) modulates likelihood via B.45 updates
- No attention level, just breath perception

**Key observations**:
- Agent accurately tracks breath state (high accuracy)
- Precision fluctuates based on prediction errors
- Demonstrates the basic precision updating mechanism

**Message**: Precision dynamics work at the interoceptive level.

---

### Figure 2: Attention Modulates Precision
**Purpose**: Show how attention state affects breath perception quality.

**Setup**:
- Two-level hierarchy: breath + attention
- Distraction forced at timestep 50
- Compare precision/accuracy before and after distraction

**Key observations**:
- Before distraction: high precision, good breath tracking
- After distraction: low precision, degraded breath tracking
- Precision observations reveal attention state

**Message**: Attention and precision are bidirectionally linked.

---

### Figure 2.1: Mental Action Selection
**Purpose**: Show complete hierarchical inference with action selection.

**Setup**:
- Full stack: breath perception, attention inference, action selection
- Natural attention transitions (3% drift rate, absorbing distraction)
- Agent can take STAY or SWITCH actions

**Parameters**:
```python
A2_precision_obs = 0.6      # Agent's meta-awareness precision
A2_true_precision = 0.95    # Environment's true precision mapping
B2_stay_prob = 0.97         # Agent believes states persist
B2_switch_prob = 0.8        # Agent believes SWITCH mostly works
E_stay = 0.99               # Strong habit of staying
p_stay_focused = 0.97       # 3% drift toward distraction
p_stay_distracted = 1.0     # Distraction is absorbing
p_switch_success = 1.0      # SWITCH deterministically restores focus
```

**Panels**:
- A: Mental Action Selection - P(Stay) and SWITCH actions
- B: Attention State Inference - P(Focused) vs true state
- C: Likelihood Precision - ζ posterior and prior (descending message)
- D: Breath Perception - P(Inhaling) vs true state

**Key observations**:
- Agent detects distraction (P(Focused) drops)
- Agent selects SWITCH when confident about distraction
- Precision drops during distraction, recovers after switch
- Breath perception degrades during distraction

**Message**: The full hierarchical loop enables attention maintenance through active inference.

---

### Figure 3: Precision Dynamics Improve Learning
**Purpose**: Show that precision modulation improves learning of breath model.

**Setup**:
- Learning A1 (breath likelihood) over 300 sits
- Compare three modes:
  1. Fixed precision (ζ = 1.0)
  2. Dynamic precision (B.45 updates)
  3. Attention-gated precision (ζ depends on attention state)

**Key observations**:
- Dynamic precision learns faster than fixed
- Attention-gated precision learns best
- Precision weighting emphasizes informative observations

**Message**: Precision dynamics serve a functional role in learning.

---

### Figure 4: The Distraction Trap (Learning Heatmap)
**Purpose**: Demonstrate that without initial meta-awareness structure, agents cannot learn their way out of distraction.

**Setup**:
- Sweep over initial A2 precision (ζ: 0→1) and B2 precision (ω: 0→1)
- ζ/ω = 0: flat (no knowledge), ζ/ω = 1: true model
- 100 sits per parameterization, 100 timesteps per sit
- Full Figure 2.1 hierarchical stack
- Learning updates for A2 and B2 at end of each sit
- 50/50 chance of starting each sit focused or distracted

**Parameters**:
```python
# Matches Figure 2.1 for full stack
A1_precision = 0.75
zeta_prior_var = 2.0
zeta_step = 0.25
zeta_focused = 2.0
zeta_distracted = 0.5
A2_true_precision = 0.95
p_stay_focused = 0.97
p_stay_distracted = 1.0
p_switch_success = 1.0
gamma = 16.0
E_stay = 0.99
A_learning_rate = 0.1
B_learning_rate = 0.1
forgetting_rate = 0.9
```

**Interpretation**:
- X-axis: ω (transition model precision) - how well agent knows dynamics
- Y-axis: ζ (likelihood model precision) - how well agent can detect attention state
- Color: Average % time focused across all sits

**Key observations**:
- **Low ζ (bottom rows)**: Trapped regardless of ω - can't detect distraction
- **High ζ (top rows)**: Maintain focus - can detect and correct distraction
- **ζ is dominant**: Horizontal bands clearer than vertical
- **Threshold around ζ ≈ 0.4**: Below this, agents struggle

**Message**: Meta-awareness precision (ζ) is the critical factor. Without it, even learning over many sits cannot bootstrap sufficient structure to escape the trap.

---

### Figure 5: Meditation Instruction Breaks the Cycle
**Purpose**: Show that meditation instruction (prior structure) enables escape from distraction.

**Setup**:
- Compare non-meditator (weak A2) vs meditator (structured A2)
- Same environment, different agent beliefs

**Key observations**:
- Non-meditator gets trapped
- Meditator detects distraction and switches back

**Message**: Instruction provides the structure that enables meta-awareness.

---

### Figure 6: Learning Across Sits
**Purpose**: Show long-term learning trajectories with and without meditation instruction.

**Setup**:
- 200 sits, instruction introduced at sit 100 for one condition
- Track A2 precision and time distracted over sits

**Key observations**:
- Without instruction: stuck at high distraction
- With instruction: rapid improvement after sit 100

**Message**: Instruction catalyzes learning that couldn't happen otherwise.

---

### A2 Precision Diagnostic
**Purpose**: Directly show how meta-awareness precision affects ability to maintain focus.

**Setup**:
- Full Figure 2.1 stack
- Vary A2_precision: 0.5, 0.6, 0.75, 0.9
- Same seed, same environment

**Results**:
- A2 = 0.50: 66% distracted (trapped, P(Focused) stuck at 0.5)
- A2 = 0.60: 66% distracted (still trapped, same as flat!)
- A2 = 0.75: 12% distracted (threshold crossed, can maintain focus)
- A2 = 0.90: 7% distracted (reliable meta-awareness)

**Message**: There's a sharp threshold in meta-awareness precision. Below ~0.75, agents cannot reliably detect distraction and get trapped. This threshold cannot be crossed through learning alone.

---

## 7. Technical Implementation Details

### Precision Scaling
```python
def scale_likelihood(A, zeta):
    """Scale likelihood by precision parameter."""
    A_scaled = A ** zeta
    return A_scaled / A_scaled.sum(axis=0)  # renormalize columns
```

### Bayesian Update
```python
def bayesian_update(A, obs, prior):
    """Simple Bayesian state inference."""
    likelihood = A[obs, :]
    posterior = likelihood * prior
    return posterior / posterior.sum()
```

### Action Selection (Expected Free Energy)
Uses pymdp's `update_posterior_policies`:
- Computes expected free energy G for each action
- Balances utility (preferences) and epistemic value (information gain)
- Softmax with precision γ to get policy distribution
- Policy prior E biases toward STAY (habit)

### Learning Updates
Uses pymdp's Dirichlet learning:
- `update_obs_likelihood_dirichlet`: Updates pA from (obs, posterior) pairs
- `update_state_likelihood_dirichlet`: Updates pB from (qs_prev, qs_curr, action) tuples
- Batch learning at end of each sit
- Forgetting: `pA = forgetting_rate * pA + (1 - forgetting_rate) * uniform`

---

## 8. Key Theoretical Points

### The Bootstrap Problem
Learning requires:
1. Informative posteriors over hidden states
2. Which require informative likelihoods (A matrices)
3. Which are what we're trying to learn

Without prior structure, this is circular. Meditation instruction breaks the circle by providing initial structure.

### Precision as Attention
The model implements the "precision as attention" hypothesis:
- Attention modulates precision of sensory processing
- This is formalized through the ζ parameter scaling likelihoods
- Higher precision = more influence of sensory evidence
- Lower precision = more influence of prior beliefs

### Active Inference Account of Meditation
1. **Meditation object** (breath): Hidden state being inferred at Level 1
2. **Attention state**: Hidden state at Level 2, affects Level 1 precision
3. **Mental action**: STAY/SWITCH actions at Level 2
4. **Meditation instruction**: Prior structure enabling Level 2 inference
5. **Practice effects**: Learning that refines A2/B2 over sits

### Why Meditation Works (in the model)
1. Instruction provides A2 structure enabling attention state inference
2. With inference, agent can detect distraction
3. Detecting distraction triggers SWITCH action
4. SWITCH restores focus
5. Learning refines A2/B2 with each successful detection/switch
6. Virtuous cycle: better inference → more switching → more learning

### Why Meditation is Hard Without Instruction
1. No A2 structure → can't detect distraction
2. Can't detect → never switch
3. Never switch → stay trapped in distraction
4. Trapped → no informative learning signal
5. No learning → A2 stays flat
6. Vicious cycle: poor inference → no switching → no learning

---

## 9. Relation to Empirical Meditation Research

### Mind-Wandering Literature
- ~50% of waking life spent mind-wandering (Killingsworth & Gilbert, 2010)
- Meta-awareness is key to catching mind-wandering (Schooler et al., 2011)
- Model captures: distraction is default, meta-awareness required to escape

### Focused Attention Meditation
- Core practice: sustain attention on object (breath), notice wandering, return
- Model captures: STAY on focus, detect distraction, SWITCH back

### Expert vs Novice Differences
- Experts show reduced mind-wandering, faster detection
- Model: higher A2 precision, more frequent/faster switching

### Neural Correlates
- Salience network activity during mind-wandering detection
- Model: A2 inference process

---

## 10. Limitations and Future Directions

### Current Limitations
1. Binary attention state (focused/distracted) is simplistic
2. No graded attention or partial distraction
3. SWITCH action is deterministic in environment
4. No modeling of emotional/motivational factors
5. No account of different meditation styles (open monitoring, etc.)

### Future Extensions
1. Continuous attention state
2. Multiple distraction types
3. Stochastic action effects
4. Affective modulation of precision
5. Different meditation practices as different model structures

---

## 11. Summary

The paper presents an active inference account of focused attention meditation where:

1. **Breath perception** is interoceptive inference with precision dynamics
2. **Attention state** is a hidden variable that modulates precision
3. **Meta-awareness** is inference over attention state
4. **Meditation instruction** provides prior structure for this inference
5. **Without instruction**, agents are trapped in distraction
6. **With instruction**, agents can detect distraction and restore focus
7. **Learning** refines this ability over practice sessions

The key insight is that **meta-awareness precision is the critical factor** that determines whether an agent can escape the distraction trap. This precision cannot be learned without prior structure - hence the necessity of meditation instruction. The model provides a formal account of why meditation instruction is necessary and how it enables the development of attentional skills.

---

## Appendix: File Locations

- **Models**: `/examples/meditation/models.py` - Model builders and constants
- **Simulations**: `/examples/meditation/simulation.py` - All run_figure functions
- **Plotting**: `/examples/meditation/plotting.py` - All plot functions
- **Runner**: `/examples/meditation/run_figures.py` - CLI for generating figures
- **Outputs**: `/examples/meditation/outputs/` - Generated figures

## Appendix: Running the Figures

```bash
cd /Users/lars/repos/pymdp-fork
source .venv/bin/activate

# Individual figures
python examples/meditation/run_figures.py --figure 1
python examples/meditation/run_figures.py --figure 2
python examples/meditation/run_figures.py --figure 2.1
python examples/meditation/run_figures.py --figure 3
python examples/meditation/run_figures.py --figure 4
python examples/meditation/run_figures.py --figure 5
python examples/meditation/run_figures.py --figure 6

# Diagnostics
python examples/meditation/run_figures.py --figure 4d   # Figure 4 within-sit dynamics
python examples/meditation/run_figures.py --figure A2d  # A2 precision comparison

# All figures
python examples/meditation/run_figures.py --all

# With display
python examples/meditation/run_figures.py --figure 2.1 --show
```
