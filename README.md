# DevsHouse26-Submission-Qtorchx: High-Fidelity Quantum Simulation Engine

![Downloads](https://img.shields.io/pypi/dm/qtorchx)

**QtorchX** is a research-grade quantum simulation framework engineered to bridge the gap between ideal computational models and the stochastic realities of physical hardware.

Instead of injecting memoryless noise as independent coin flips, QtorchX models the quantum processor as a dynamical physical environment governed by a hidden **7D noise manifold**. Built entirely on **PyTorch**, the platform provides a fully differentiable pipeline from gate execution to measurement statistics.

In short:

> QtorchX doesn’t sprinkle noise on circuits. It evolves it.

This enables a powerful inverse workflow: backtracking from empirical hardware histograms to the latent physical parameters of the noise field—creating a high-precision **Digital Twin** of the quantum device.

---

# The QNaF Physical Formalism

At the core of QtorchX lies the **QNaF (Quantum Noise as Fields) Framework**. Rather than assigning static Pauli error probabilities, QNaF defines a structured, time-evolving manifold:

$$\Phi(t) \in \mathbb{R}^{7 \times Q \times T}$$

Where:

* $Q$ = number of qubits
* $T$ = circuit depth
* $7$ = independent decoherence channels

The manifold evolves according to a nonlinear, topology-aware dynamical system:

$$\Phi(t+1) = \tanh\Big( \mathcal{M}\Phi(t) + \beta L \Phi(t) + \kappa L D(t) + \mathcal{N}(\Phi(t)) + \Sigma(t) \Big)$$

Where:

* $L$ = graph Laplacian derived from circuit connectivity
* $D(t)$ = gate-induced disturbance field
* $\mathcal{M}$ = temporal memory operator
* $\mathcal{N}$ = nonlinear coupling term
* $\Sigma(t)$ = stochastic environmental fluctuations

The $\tanh$ activation ensures bounded thermodynamic stability: $\Phi \in [-1, 1]$. Noise is no longer independent per gate—it diffuses, accumulates, saturates, and interacts across qubits and time.

---

## Learnable Decoherence Projections

The hidden manifold is projected into Pauli error space via two trainable tensors:

$$P(t) = W \Phi(t) + B$$

Where:

* $W \in \mathbb{R}^{3 \times 7}$ → **Decoherence Projection Matrix (DPM)**
* $B \in \mathbb{R}^{3}$ → **Baseline Pauli Offset (BPO)**

The resulting Pauli channel strengths are mapped via a sigmoid function: $p_{X,Y,Z}(q,t) = \sigma(P_{X,Y,Z}(q,t))$. Both **DPM** and **BPO** are fully differentiable, enabling hardware fingerprint discovery and device-specific dephasing modeling.

---

# Technical Architecture

QtorchX is structured into three execution layers optimized for GPU-native performance and zero-copy tensor flow.

---

## 1️⃣ Core Simulation Engine (`qtorchx.core`)

The backend is a statevector engine designed to avoid the $2^n \times 2^n$ matrix expansion bottleneck. By utilizing a path of $|\psi\rangle \rightarrow \text{reshape} \rightarrow \text{permute} \rightarrow U \rightarrow \text{inverse-permute}$, the system guarantees:

* **$O(2^n)$** memory scaling
* Linear depth scaling
* Native GPU acceleration via PyTorch

### Intelligent Caching

* **Static gates**: $O(1)$ fixed lookup.
* **Parametric gates**: LRU cache with angle quantization to maximize hit rates.

---

## 2️⃣ Physics Layer (`qtorchx.noise`)

The **PhiManifoldExtractor** evolves the hidden noise field at each time slice. Each gate modifies the disturbance field $D_i(t) = a G_i(t) + b M_i(t)$, where $G_i(t)$ represents gate activity and $M_i(t)$ represents measurement activity. Noise becomes a direct function of the circuit's spatio-temporal structure.

---

## 3️⃣ Calibration Layer (`qtorchx.noise.calibrator`)

The **NoiseCalibrator** utilizes the PyTorch autograd engine to minimize the loss between simulation and hardware:

$$\mathcal{L} = \text{MSE}(p_{\text{sim}}, \hat{p}) + \lambda \cdot \text{StabilityPenalty}$$

Gradients flow back to the **DPM** ($W$) and **BPO** ($B$) tensors, enabling direct-to-hardware calibration.

---

# 🛠 API Implementation Guide

### High-Fidelity Simulation

```python
import qtorchx as qtx

circ = qtx.Circuit(num_qubits=5)
circ.add('H', [0]).add('CNOT', [0, 1])

# Fetch pre-calibrated hardware preset
preset = qtx.PresetManager.fetch("qtorch_local")

backend = qtx.QtorchBackend(circuit=circ, simulate_with_noise=True)
results = backend.get_histogram_data(shots=1000)

```

### Real Hardware Calibration

```python
# Backtrack from empirical bitstring probabilities
calibrator = qtx.NoiseCalibrator(circ, preset_name="qtorch_standard")

history = calibrator.calibrate(target_data, epochs=100)

# Export the Digital Twin
calibrator.export_qnaf("calibrated_twin_v1")

```

---

# Deployment & Credits

**Version**: 1.0.0

**Lead Engineer**: Phani Kumar

**Status**: Stable Release (Patent Pending)

**Installation**:

```bash
pip install qtorchx

```

QtorchX is licensed under the MIT License. The QNaF formalism and associated differentiable architectures are protected research assets.

---
