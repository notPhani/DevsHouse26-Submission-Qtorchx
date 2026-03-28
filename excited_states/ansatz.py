"""
ansatz.py
---------
Differentiable hardware-efficient ansatz that hooks into QtorchX's GateLibrary
for gate matrices, but applies them via pure torch autograd-compatible ops.

QtorchBackend._apply_k_qubit() uses in-place statevector mutations which break
autograd — so we build a standalone differentiable forward pass here using the
same gate matrices from GateLibrary (converted to complex128).

Architecture (hardware-efficient ansatz):
    For n_layers layers:
        RY(θ) on every qubit           ← parameterised
        CNOT cascade  q[i]→q[i+1]     ← fixed
    Final RY layer on every qubit      ← parameterised

Total parameters: (n_layers + 1) * n_qubits
"""

import sys
import os

import torch
import numpy as np

# ── Import GateLibrary from the existing QtorchX package ────────────────────
_REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, _REPO)

from qtorchx.core.primitives import GateLibrary


# ── Utility: embed a 2×2 gate into the full 2^n × 2^n Hilbert space ─────────

def _embed_1q(gate_2x2: torch.Tensor, qubit: int, n_qubits: int) -> torch.Tensor:
    """
    Tensor-product embed a 2×2 gate acting on `qubit` into 2^n × 2^n space.

    Uses I ⊗ ... ⊗ gate ⊗ ... ⊗ I (qubit 0 = leftmost = most-significant bit).
    """
    I2 = torch.eye(2, dtype=torch.complex128, device=gate_2x2.device)
    ops = [I2] * n_qubits
    ops[qubit] = gate_2x2.to(dtype=torch.complex128)

    result = ops[0]
    for op in ops[1:]:
        result = torch.kron(result, op)
    return result


def _build_cnot(ctrl: int, tgt: int, n_qubits: int,
                device: torch.device) -> torch.Tensor:
    """
    Build CNOT (ctrl → tgt) embedded in 2^n × 2^n space.
    Uses |0⟩⟨0|_ctrl ⊗ I_tgt  +  |1⟩⟨1|_ctrl ⊗ X_tgt  with I elsewhere.
    """
    I2 = torch.eye(2, dtype=torch.complex128, device=device)
    X  = torch.tensor([[0., 1.], [1., 0.]], dtype=torch.complex128, device=device)
    P0 = torch.tensor([[1., 0.], [0., 0.]], dtype=torch.complex128, device=device)
    P1 = torch.tensor([[0., 0.], [0., 1.]], dtype=torch.complex128, device=device)

    ops0 = [I2] * n_qubits
    ops0[ctrl] = P0
    ops1 = [I2] * n_qubits
    ops1[ctrl] = P1
    ops1[tgt]  = X

    mat0 = ops0[0]
    for op in ops0[1:]:
        mat0 = torch.kron(mat0, op)

    mat1 = ops1[0]
    for op in ops1[1:]:
        mat1 = torch.kron(mat1, op)

    return mat0 + mat1


# ── Differentiable RY matrix ──────────────────────────────────────────────────

def _ry_mat(theta: torch.Tensor) -> torch.Tensor:
    """
    RY(θ) = [[cos(θ/2), -sin(θ/2)],
              [sin(θ/2),  cos(θ/2)]]
    Returns 2×2 complex128 tensor; autograd flows through θ.
    """
    c = torch.cos(theta / 2).to(torch.complex128)
    s = torch.sin(theta / 2).to(torch.complex128)
    zeros = torch.zeros_like(c)
    row0 = torch.stack([c,  -s], dim=-1)
    row1 = torch.stack([s,   c], dim=-1)
    return torch.stack([row0, row1], dim=-2)  # (2, 2)


# ── Pre-built CNOT layer (fixed, no gradient) ────────────────────────────────

def _precompute_cnots(n_qubits: int, device: torch.device) -> list:
    """Return list of CNOT matrices: q[0]→q[1], q[1]→q[2], ..."""
    return [
        _build_cnot(i, i + 1, n_qubits, device)
        for i in range(n_qubits - 1)
    ]


# ── Main differentiable ansatz ────────────────────────────────────────────────

class HEAnsatz(torch.nn.Module):
    """
    Hardware-Efficient Ansatz (HEA) with RY rotations + CNOT entanglement.

    Parameters
    ----------
    n_qubits : int
        Number of qubits (4 for H2/STO-3G).
    n_layers : int
        Number of (RY + CNOT) blocks before the final RY layer.

    Total trainable parameters: (n_layers + 1) * n_qubits
    """

    def __init__(self, n_qubits: int = 4, n_layers: int = 2):
        super().__init__()
        self.n_qubits  = n_qubits
        self.n_layers  = n_layers
        self.n_params  = (n_layers + 1) * n_qubits

        # Trainable parameters (initialised uniformly in [0, 2π])
        self.params = torch.nn.Parameter(
            torch.rand(self.n_params) * 2 * torch.pi
        )

    def forward(self) -> torch.Tensor:
        """
        Return the normalised statevector |ψ(θ)⟩ as a complex128 tensor
        of shape (2^n_qubits,).  Autograd flows through all RY angles.
        """
        device = self.params.device
        n = self.n_qubits

        # Pre-compute fixed CNOT layer matrices (no gradient needed)
        cnots = _precompute_cnots(n, device)

        # Initial state |0...0⟩
        state = torch.zeros(2 ** n, dtype=torch.complex128, device=device)
        state[0] = 1.0 + 0.0j

        idx = 0
        for layer in range(self.n_layers):
            # ── RY block ──────────────────────────────────────────────────
            for q in range(n):
                ry = _ry_mat(self.params[idx].to(torch.float64))
                U  = _embed_1q(ry, q, n)
                state = U @ state
                idx += 1

            # ── CNOT cascade ──────────────────────────────────────────────
            for cnot in cnots:
                state = cnot @ state

        # ── Final RY block (no CNOT after) ────────────────────────────────
        for q in range(n):
            ry = _ry_mat(self.params[idx].to(torch.float64))
            U  = _embed_1q(ry, q, n)
            state = U @ state
            idx += 1

        # Renormalise (numerical safety)
        norm = torch.linalg.norm(state)
        return state / norm


# ── Expectation value ─────────────────────────────────────────────────────────

def expectation(H: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
    """
    ⟨ψ|H|ψ⟩  — real part only (H is Hermitian so imaginary part ≈ 0).

    Parameters
    ----------
    H   : (2^n, 2^n) complex128 Hamiltonian
    psi : (2^n,)     complex128 statevector

    Returns
    -------
    Scalar real tensor.
    """
    return torch.real(torch.conj(psi) @ (H @ psi))


# ── VQD penalty: β |⟨ψ_k|ψ⟩|² ───────────────────────────────────────────────

def overlap_penalty(
    psi: torch.Tensor,
    prev_states: list,          # list of previously optimised statevectors
    beta: float = 8.0,
) -> torch.Tensor:
    """
    Σ_k  β * |⟨ψ_k|ψ⟩|²

    This penalises the ansatz for having overlap with previously found states,
    forcing VQD to converge to orthogonal (higher-energy) eigenstates.
    """
    penalty = torch.zeros(1, dtype=torch.float64, device=psi.device)
    for phi in prev_states:
        overlap = torch.abs(torch.conj(phi.detach()) @ psi) ** 2
        penalty = penalty + beta * overlap
    return penalty.squeeze()
