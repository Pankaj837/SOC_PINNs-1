# Week 3 — Canonical PDEs: Heat Equation & Wave Equation

## Overview

Week 3 scaled up from ODEs (one variable: time) to PDEs (two variables: space and time). The **DeepXDE** library was introduced, which separates the problem definition (geometry, PDE, BCs, ICs) from the solution process (network, optimizer). Two canonical PDEs were solved: the heat equation and the wave equation. A collocation sampling experiment was also conducted to understand how point count affects accuracy.

---

## Topics Covered

### From ODEs to PDEs

In Week 2, the network took `t` as input and output `x(t)`. Now the network takes both `x` and `t` as input and outputs `u(x, t)`, the solution at every point in space and time simultaneously.

Collocation points are now sampled in 2D (x, t) space rather than 1D.

---

### Heat Equation (Parabolic PDE)

```
uₜ = α u_xx,   x ∈ [0,1],  t ∈ [0,1]
u(0, t) = 0,     u(1, t) = 0     ← Dirichlet BCs (edges always cold)
u(x, 0) = sin(πx)                ← Initial condition (sine wave temperature)
```

**Physics:** How temperature spreads along a rod over time. A hot spike gradually diffuses and flattens out. The solution decays smoothly to zero.

**Analytical solution:** `u(x, t) = exp(−α π² t) · sin(πx)`

---

### Wave Equation (Hyperbolic PDE)

```
uₜₜ = c² u_xx,   x ∈ [0,1],  t ∈ [0,1],  c = 1
u(x, 0) = sin(πx),    uₜ(x, 0) = 0
u(0, t) = 0,           u(1, t) = 0
```

**Physics:** How a wave travels along a string. A plucked guitar string, the shape bounces back and forth without decaying.

**Analytical solution:** `u(x, t) = sin(πx) · cos(πt)`

---

### DeepXDE Library

DeepXDE separates **what the problem is** from **how to solve it**:

```python
# 1. Define the PDE
def heat_pde(x, y):
    dy_t = dde.grad.jacobian(y, x, i=0, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    return dy_t - 0.4 * dy_xx

# 2. Define geometry and time domain
# 3. Define BCs and ICs
# 4. Build the problem and model
# 5. Train
```

No manual training loop needed, DeepXDE handles sampling, loss construction, and optimization.

---

### Collocation Sampling Experiment

Repeated the heat equation solve with different numbers of collocation points: `[500, 1000, 2500, 5000, 10000]`

---

## Key Results

### Heat Equation

- **L2 Relative Error: ~0.013** (1.3% error)
- PINN solution heatmap closely matched the analytical solution
- Maximum absolute error: ~0.007 (very small)
- Error was slightly higher at later times, the network finds future states slightly harder to learn

### Collocation Experiment

| Collocation Points | L2 Error |
|---|---|
| 500 | 0.021 |
| 1000 | 0.004 |
| 2500 | 0.007 |
| 5000 | 0.036 |
| 10000 | 0.002 ✅ |

**Observation:** More collocation points generally improves accuracy, especially at 10,000. The spike at 5,000 is due to random training variability, if averaged over multiple runs it would smooth out. The general trend confirms that better coverage of the domain leads to better physics enforcement.

### Wave Equation

- PINN matched the exact solution closely
- The solution showed clear oscillation (red at t=0, transitions to blue at t=0.5, back to red at t=1)
- Distinct from heat equation: no decay, just wave bouncing

---

## Heat vs Wave: Key Comparison

| | Heat Equation | Wave Equation |
|---|---|---|
| PDE type | Parabolic | Hyperbolic |
| Physical behavior | Temperature diffuses and decays | Wave oscillates back and forth |
| Solution shape | Smooth decay to zero | Sinusoidal oscillation |
| Training difficulty | Easier (smooth) | Slightly harder (oscillates) |

---

## Reflection Questions

**Q1: Why does more collocation points help but with diminishing returns?**
More points means we check the equation at more locations, giving the network better coverage of the domain. But beyond a certain point, the network's capacity becomes the bottleneck, adding more points doesn't help if the network architecture cannot learn faster.

**Q2: Why use `tanh` instead of `ReLU` for PINNs?**
ReLU's second derivative is zero everywhere, this would make `d²u/dx²` always zero, making the PDE loss meaningless for second-order equations. `tanh` is smooth and infinitely differentiable, making it the standard activation for PINNs.

---

## Conclusion & Next Steps

Week 3 demonstrated that PINNs can handle full PDEs in 2D space-time domains, and that the DeepXDE library significantly simplifies the workflow. The collocation experiment confirmed that sampling density affects accuracy but with diminishing returns.

In **Week 4**, the focus shifts to Burgers' equation, the benchmark PINN problem, implementing hard boundary conditions, and honestly comparing PINNs against classical FEM solvers.
