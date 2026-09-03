# High-Dimensional Dynamic Programming Notebooks

These notebooks compare three ways to solve nonlinear dynamic models with
occasionally binding constraints or sovereign default. Each notebook is meant
to be read from top to bottom: it begins with learning objectives and a
model-to-code roadmap, develops the numerical method in stages, and ends with a
compact two-panel economic summary.

## Notebook map

| Model | Method | Language | Final diagnostics |
| --- | --- | --- | --- |
| Occasionally binding collateral constraint | Dense time iteration with JAX | Python | Debt policy and Kuhn–Tucker multiplier |
| Sovereign default with long-term debt | Smoothed value-function iteration with JAX | Python | Expected debt policy and default-risk heatmap |
| Occasionally binding collateral constraint | Adaptive sparse grid | Julia | Debt policy and adaptive nodes |
| Occasionally binding collateral constraint | DDSG/Tasmanian adaptive sparse grid | Python | Debt policy and multiplier-colored nodes |
| Sovereign default with long-term debt | Adaptive sparse grid | Julia | Debt policy and repayment-value nodes |
| Occasionally binding collateral constraint | Neural network with Flux.jl | Julia | Training history and learned policy |
| Occasionally binding collateral constraint | Neural network with PyTorch | Python | Training history and learned policy |

## Suggested reading order

1. Start with [`JAX/obc_JAX.ipynb`](JAX/obc_JAX.ipynb). The state space is small,
   and the multiplier makes the binding region easy to diagnose.
2. Compare the same model across the two adaptive sparse-grid notebooks. Focus
   on where the algorithms add nodes relative to the policy kink.
3. Compare the Flux and PyTorch notebooks. The economic residual is the same;
   the implementation details and training interfaces differ.
4. Finish with the long-term sovereign-default notebooks, where values, choice
   probabilities, and endogenous bond prices must converge jointly.

## How to judge a solution

- **Convergence:** the reported policy/value change should satisfy the stated
  tolerance. For neural networks, inspect a fixed validation grid in addition
  to the stochastic training loss.
- **Economic consistency:** the collateral multiplier should become positive
  where the policy bends at the borrowing limit. Default probability should
  rise with debt and fall with income.
- **Approximation allocation:** adaptive-grid nodes should concentrate near
  kinks, curvature, or the default boundary—not uniformly across smooth areas.
- **Robustness:** vary grid density, refinement tolerance, smoothing, or network
  capacity and confirm that the economic objects are stable.

The saved outputs are reference runs. Hardware, package versions, and stochastic
optimization can lead to small numerical differences when the notebooks are
executed again.
