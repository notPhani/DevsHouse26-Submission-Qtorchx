from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
import torch
import numpy as np
import sys, os
from qtorchx.core.primitives import GateLibrary, Gate, Circuit


class QtorchBackend:
    """
    This is the main entry point for the Qtorch quantum computing backend.
    This will contain functions to configure the qubits, statevectors etc
    
    Then will have three main flags:
        . simulate_with_noise: bool - whether to simulate with noise or not
        . persistant_data: bool - whether to store persistant data or not
        . fusion_optimizations: bool - whether to use fusion optimizations or not 
          (this is optional for noise retained simulations)
    """
    
    def __init__(
        self, 
        simulate_with_noise: bool = False,
        persistant_data: bool = True,
        fusion_optimizations: bool = False,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        circuit: Circuit = None,
        quantized_angle_precision: float = 0.001,
        parametric_cache_size: int = 1024,
        verbose: bool = False
    ):
        self.simulate_with_noise = simulate_with_noise
        self.persistant_data = persistant_data
        self.fusion_optimizations = fusion_optimizations
        self.device = device
        self.circuit = circuit
        self.num_qubits = circuit.num_qubits if circuit else 0
        self.verbose = verbose
        
        if self.num_qubits > 24:
            raise ValueError(
                f"QtorchBackend supports up to 24 qubits but given {self.num_qubits}"
            )
        
        # Initialize statevector to |0...0⟩
        self.statevector = torch.zeros(
            (2**self.num_qubits,), 
            dtype=torch.complex64, 
            device=self.device
        )
        self.statevector[0] = 1.0 + 0.0j
        
        # Caching configuration
        self.angle_precision = quantized_angle_precision if persistant_data else None
        self.parametric_cache_size = parametric_cache_size if persistant_data else 0
        
        # Cache storage
        self.fixed_cache = {} if persistant_data else None
        self.parametric_cache = None  # Will be set by _setup_parametric_cache()
        
        # Classical register for measurement outcomes
        self.classical_register = {}
        
        # Initialize caches if persistent_data is enabled
        if self.persistant_data:
            self._precompute_fixed_gates()
            self._setup_parametric_cache(self.parametric_cache_size)
    
    def _precompute_fixed_gates(self) -> None:
        """
        Precompute and cache all non-parametric gate matrices.
        Only runs if persistent_data=True.
        
        Caches 28 static gates from the 43-gate library.
        """
        # All static (non-parametric) gates in your library
        fixed_gate_names = [
            # === Single-qubit gates (14) ===
            'I',           # Identity
            'X', 'Y', 'Z', # Pauli gates
            'H',           # Hadamard
            'S', 'SDG',    # Phase gates
            'T', 'TDG',    # π/8 gates
            'SX', 'SY', 'SZ',  # Square root gates
            'V', 'VDG',    # V gates
            
            # === Two-qubit gates (10) ===
            'CNOT', 'CX',  # Controlled-NOT (aliases)
            'CY', 'CZ',    # Controlled Pauli
            'SWAP',        # SWAP
            'ISWAP',       # iSWAP
            'SQRT_SWAP',   # √SWAP
            'CH',          # Controlled-Hadamard
            'ECR',         # Echoed Cross-Resonance
            'DCX',         # Double CNOT
            
            # === Three-qubit gates (4) ===
            'TOFFOLI', 'CCNOT',  # Toffoli (aliases)
            'FREDKIN', 'CSWAP',  # Fredkin (aliases)
        ]
        
        cached_count = 0
        failed_gates = []
        
        for gate_name in fixed_gate_names:
            try:
                # Get matrix from GateLibrary
                matrix = GateLibrary.get_gate(gate_name, [])
                
                if matrix is None:
                    failed_gates.append((gate_name, "returned None"))
                    continue
                
                # Cache with uppercase key
                self.fixed_cache[gate_name.upper()] = matrix.to(
                    dtype=torch.complex64,
                    device=self.device
                )
                cached_count += 1
                
            except Exception as e:
                failed_gates.append((gate_name, str(e)))
                if self.verbose:
                    print(f"[Warning] Failed to cache {gate_name}: {e}")
        
        if self.verbose and cached_count > 0:
            print(f"[Backend] Precomputed {cached_count} fixed gates (persistent_data=True)")
            if failed_gates:
                print(f"[Backend] Skipped {len(failed_gates)} gates:")
                for name, reason in failed_gates[:3]:  # Show first 3
                    print(f"  - {name}: {reason}")
    
    def _setup_parametric_cache(self, maxsize: int) -> None:
        """
        Setup LRU cache for parametric gates with angle quantization.
        Only runs if persistent_data=True.
        
        Parametric gates (15):
            Single-qubit: RX, RY, RZ, P, U1, U2, U3
            Two-qubit: CRX, CRY, CRZ, RXX, RYY, RZZ
        
        Args:
            maxsize: Maximum cache entries (typically 128-1024)
        """
        from functools import lru_cache
        
        @lru_cache(maxsize=maxsize)
        def _cached_parametric_matrix(gate_name: str, quantized_params: tuple) -> torch.Tensor:
            """
            Cached computation of parametric gate matrices.
            
            Uses quantized parameters as cache key for high hit rates.
            Example: RX(0.1234567) and RX(0.1230001) both map to RX(0.123)
            
            Args:
                gate_name: Gate name (uppercase)
                quantized_params: Tuple of quantized angles
                
            Returns:
                Gate matrix on correct device
            """
            # Unpack quantized params back to list
            params_list = list(quantized_params) if quantized_params else []
            
            # Get matrix from GateLibrary
            matrix = GateLibrary.get_gate(gate_name, params_list)
            
            if matrix is None:
                raise ValueError(f"Unknown parametric gate: {gate_name}")
            
            return matrix.to(dtype=torch.complex64, device=self.device)
        
        self.parametric_cache = _cached_parametric_matrix
        
        if self.verbose:
            print(f"[Backend] LRU cache enabled (size={maxsize}, precision={self.angle_precision:.4f} rad)")
    
    def _quantize_params(self, params: Optional[List[float]]) -> Optional[tuple]:
        """
        Quantize angle parameters to nearest precision step.
        
        Increases cache hit rate by collapsing nearby angles to same key.
        
        Example with angle_precision=0.001:
            0.1234567 → 0.123
            0.9998    → 1.000
            π/4       → 0.785 (quantized)
        
        Args:
            params: List of gate parameters (angles in radians)
            
        Returns:
            Tuple of quantized parameters (hashable for cache key)
        """
        if params is None or len(params) == 0:
            return None
        
        quantized = tuple(
            round(p / self.angle_precision) * self.angle_precision
            for p in params
        )
        
        return quantized
    
    def _get_gate_matrix_cached(self, gate: Gate) -> torch.Tensor:
        """
        Get gate matrix using 2-tier caching system.
        
        Tier 1: Fixed cache (28 static gates) - O(1) lookup
        Tier 2: LRU cache (15 parametric gates) - quantized params
        Tier 3: Direct computation (fallback)
        
        Args:
            gate: Gate object
            
        Returns:
            Gate matrix (2^k, 2^k) on correct device
        """
        gate_name = gate.name.upper()
        
        if self.persistant_data:
            # ================================================================
            # TIER 1: Fixed cache (static gates)
            # ================================================================
            if gate_name in self.fixed_cache:
                return self.fixed_cache[gate_name]
            
            # ================================================================
            # TIER 2: LRU cache (parametric gates with quantization)
            # ================================================================
            if gate.params and self.parametric_cache is not None:
                # Quantize parameters for better cache hits
                quantized_params = self._quantize_params(gate.params)
                
                try:
                    return self.parametric_cache(gate_name, quantized_params)
                except Exception as e:
                    raise ValueError(f"Failed to get matrix for {gate_name}: {e}")
            
            # ================================================================
            # TIER 3: Direct computation (not in cache)
            # ================================================================
            matrix = GateLibrary.get_gate(gate_name, gate.params)
            
            if matrix is None:
                raise ValueError(f"Unknown gate: {gate_name}")
            
            matrix = matrix.to(dtype=torch.complex64, device=self.device)
            
            # If non-parametric, add to fixed cache for future use
            if not gate.params:
                self.fixed_cache[gate_name] = matrix
                if self.verbose:
                    print(f"[Cache] Added {gate_name} to fixed cache dynamically")
            
            return matrix
        
        else:
            # ================================================================
            # NO CACHING: Fresh computation every time
            # ================================================================
            matrix = GateLibrary.get_gate(gate_name, gate.params)
            
            if matrix is None:
                raise ValueError(f"Unknown gate: {gate_name}")
            
            return matrix.to(dtype=torch.complex64, device=self.device)
    
    def get_cache_stats(self) -> dict:
        """
        Get detailed statistics about cache usage.
        
        Returns:
            Dictionary with:
                - persistent_data: Whether caching is enabled
                - fixed_cache_size: Number of static gates cached
                - lru_cache: Hit rate, misses, current size
                - angle_precision: Quantization step
        """
        stats = {
            'persistent_data': self.persistant_data,
            'fixed_cache_size': len(self.fixed_cache) if self.fixed_cache else 0,
            'angle_precision': self.angle_precision,
        }
        
        # Show gate names if verbose
        if self.verbose and self.fixed_cache:
            stats['fixed_cache_gates'] = sorted(list(self.fixed_cache.keys()))
        
        # LRU cache statistics
        if self.persistant_data and self.parametric_cache is not None:
            cache_info = self.parametric_cache.cache_info()
            total_requests = cache_info.hits + cache_info.misses
            
            stats['lru_cache'] = {
                'hits': cache_info.hits,
                'misses': cache_info.misses,
                'current_size': cache_info.currsize,
                'max_size': cache_info.maxsize,
                'hit_rate': (cache_info.hits / total_requests * 100) 
                           if total_requests > 0 else 0.0,
                'total_requests': total_requests
            }
        
        return stats
    
    def clear_lru_cache(self) -> None:
        """Clear LRU cache (useful for benchmarking different quantization levels)"""
        if self.parametric_cache is not None:
            self.parametric_cache.cache_clear()
            
            if self.verbose:
                print("[Backend] LRU cache cleared")
    
    def set_statevector(self, statevector: torch.Tensor) -> None:
        """
        Set custom statevector.
        
        Args:
            statevector: Complex tensor of shape (2^n,)
            
        Raises:
            ValueError: If shape or dtype is incorrect
        """
        if statevector.shape != (2**self.num_qubits,):
            raise ValueError(
                f"Statevector must have shape {(2**self.num_qubits,)}, "
                f"got {statevector.shape}"
            )
        
        if statevector.dtype not in [torch.complex64, torch.complex128]:
            raise ValueError(
                f"Statevector must have dtype torch.complex64 or torch.complex128, "
                f"got {statevector.dtype}"
            )
        
        self.statevector = statevector.to(
            dtype=torch.complex64,  # Standardize to complex64
            device=self.device
        )
        
        # Normalize
        norm = torch.sqrt(torch.sum(torch.abs(self.statevector)**2))
        if norm < 1e-10:
            raise ValueError("Statevector has zero norm (invalid state)")
        
        self.statevector = self.statevector / norm
    
    def apply_gate(self, gate: Gate) -> None:
        """
        Apply gate to statevector with optional noise.
        
        Args:
            gate: Gate instance from Circuit
            
        Raises:
            ValueError: If gate is unknown or qubits are invalid
        """
        # ====================================================================
        # VALIDATION
        # ====================================================================
        
        # Validate qubit indices
        for q in gate.qubits:
            if q < 0 or q >= self.num_qubits:
                raise ValueError(
                    f"Gate {gate.name} uses qubit {q} but circuit only has "
                    f"{self.num_qubits} qubits (indices 0-{self.num_qubits-1})"
                )
        
        # Check for duplicate qubits
        if len(gate.qubits) != len(set(gate.qubits)):
            raise ValueError(
                f"Gate {gate.name} has duplicate qubits: {gate.qubits}"
            )
        
        # ====================================================================
        # HANDLE SPECIAL GATES
        # ====================================================================
        
        # Measurement gate
        if gate.name.upper() == 'M':
            for q in gate.qubits:
                self._apply_measure(q)
            return
        
        # Classical control gates (for teleportation)
        if gate.name.upper() == 'XC':
            if gate.depends_on and len(gate.depends_on) >= 2:
                self._apply_classical_pauli(gate.qubits[0], 'X', gate.depends_on)
            return
        
        if gate.name.upper() == 'ZC':
            if gate.depends_on and len(gate.depends_on) >= 2:
                self._apply_classical_pauli(gate.qubits[0], 'Z', gate.depends_on)
            return
        
        # ====================================================================
        # GET GATE MATRIX (WITH CACHING)
        # ====================================================================
        
        U = self._get_gate_matrix_cached(gate)
        
        # Validate matrix shape
        k = len(gate.qubits)
        expected_size = 1 << k  # 2^k
        if U.shape != (expected_size, expected_size):
            raise ValueError(
                f"Gate '{gate.name}' matrix has shape {U.shape}, expected "
                f"{(expected_size, expected_size)} for {k}-qubit gate"
            )
        
        # ====================================================================
        # APPLY IDEAL GATE
        # ====================================================================
        
        self._apply_k_qubit(U, gate.qubits)
        
        # ====================================================================
        # APPLY NOISE (IF ENABLED)
        # ====================================================================
        
        if self.simulate_with_noise and 'noise_model' in gate.metadata:
            self._apply_noise_from_metadata(gate)
    
    def _apply_k_qubit(self, U: torch.Tensor, targets: List[int]) -> None:
        """
        Apply k-qubit gate U on target qubits using efficient tensor reshaping.
        
        This avoids building the full 2^n × 2^n matrix, instead using tensor
        operations for O(2^n) complexity instead of O(2^{2n}).
        
        Args:
            U: Gate matrix of shape (2^k, 2^k)
            targets: List of target qubit indices
        """
        n = self.num_qubits
        k = len(targets)
        expected_size = 1 << k
        
        # Validate matrix size
        assert U.shape == (expected_size, expected_size), \
            f"Matrix shape {U.shape} doesn't match {k}-qubit gate"
        
        # Reshape statevector to n-dimensional tensor: (2, 2, 2, ..., 2)
        psi = self.statevector.view([2] * n)
        
        # Move target qubits to the end via permutation
        # Example: n=5, targets=[1,3] → perm=[0,2,4,1,3]
        targets = list(targets)
        perm = [i for i in range(n) if i not in targets] + targets
        psi = psi.permute(perm)
        
        # Reshape to (batch_size, 2^k) where batch_size = 2^(n-k)
        batch = psi.numel() // expected_size
        psi = psi.reshape(batch, expected_size)
        
        # Apply gate via matrix multiplication
        # (batch, 2^k) @ (2^k, 2^k)^T = (batch, 2^k)
        psi = psi @ U.t()
        
        # Reshape back to n-dimensional tensor
        psi = psi.view([2] * n)
        
        # Inverse permutation to restore original qubit order
        inv = [0] * n
        for i, p in enumerate(perm):
            inv[p] = i
        psi = psi.permute(inv)
        
        # Flatten back to statevector
        self.statevector = psi.reshape(-1)
    
    def _apply_noise_from_metadata(self, gate: Gate) -> None:
        """
        Apply Pauli noise based on phi manifold annotation in gate metadata.
        
        For each qubit the gate touches, sample error type from probabilities
        [p_i, p_x, p_y, p_z] and apply corresponding Pauli operator.
        
        Args:
            gate: Gate with 'noise_model' in metadata
        """
        noise = gate.metadata['noise_model']
        pauli_probs = noise['pauli_probs']
        
        # Apply noise to each qubit this gate touched
        for q, probs in pauli_probs.items():
            p_i, p_x, p_y, p_z = probs
            
            # Sample which error occurs
            r = torch.rand(1, device=self.device).item()
            
            # Apply sampled Pauli error
            if r < p_x:
                # X error (bit flip)
                self._apply_single_pauli('X', q)
                
            elif r < p_x + p_y:
                # Y error (bit + phase flip)
                self._apply_single_pauli('Y', q)
                
            elif r < p_x + p_y + p_z:
                # Z error (phase flip)
                self._apply_single_pauli('Z', q)
            
            # else: no error (identity with probability p_i)
    
    def _apply_single_pauli(self, pauli_name: str, qubit: int) -> None:
        """
        Apply single Pauli gate (X, Y, or Z) to one qubit.
        
        Optimized for noise application - uses cached Pauli matrices.
        
        Args:
            pauli_name: 'X', 'Y', or 'Z'
            qubit: Target qubit index (0 to n-1)
        """
        # Get Pauli matrix (will hit fixed cache)
        pauli_name_upper = pauli_name.upper()
        
        if self.persistant_data and pauli_name_upper in self.fixed_cache:
            U = self.fixed_cache[pauli_name_upper]
        else:
            U = GateLibrary.get_gate(pauli_name, [])
            if U is None:
                raise ValueError(f"Cannot get Pauli matrix for '{pauli_name}'")
            U = U.to(dtype=torch.complex64, device=self.device)
        
        # Apply to single qubit
        self._apply_k_qubit(U, [qubit])
    
    def _apply_measure(self, q: int) -> int:
        """
        Perform Z-basis measurement of a single qubit with state collapse.
        
        Samples outcome from |ψ|² probability distribution, then projects
        the statevector onto the measured eigenspace.
        
        Args:
            q: Qubit index to measure (0 to n-1)
            
        Returns:
            Measurement outcome (0 or 1)
            
        Raises:
            ValueError: If qubit index is invalid
            RuntimeError: If state has zero norm
        """
        n = self.num_qubits
        
        # Validate qubit index
        if q < 0 or q >= n:
            raise ValueError(
                f"Cannot measure qubit {q} (circuit has {n} qubits)"
            )
        
        # Reshape statevector to tensor
        psi = self.statevector.view([2] * n)
        
        # Move measured qubit to last dimension
        perm = [i for i in range(n) if i != q] + [q]
        psi = psi.permute(perm)
        
        # Reshape to (batch, 2) where last dim is measured qubit
        psi = psi.reshape(-1, 2)
        
        # Compute probabilities for |0⟩ and |1⟩
        probs = (psi.conj() * psi).sum(dim=0).real
        p0 = float(probs[0])
        p1 = float(probs[1])
        
        # Normalize (handle numerical errors)
        total = p0 + p1
        if total < 1e-10:
            raise RuntimeError(
                f"Measurement of qubit {q} has zero probability "
                "(invalid quantum state)"
            )
        p0 /= total
        p1 /= total
        
        # Sample measurement outcome
        r = torch.rand((), device=self.device).item()
        outcome = 0 if r < p0 else 1
        
        # Collapse state: project onto |outcome⟩ subspace
        mask = torch.zeros_like(psi)
        mask[:, outcome] = 1.0
        psi = psi * mask
        
        # Renormalize collapsed state
        norm = torch.linalg.norm(psi)
        if norm > 0:
            psi = psi / norm
        
        # Reshape back to tensor
        psi = psi.view([2] * n)
        
        # Inverse permutation to restore qubit order
        inv = [0] * n
        for i, p in enumerate(perm):
            inv[p] = i
        psi = psi.permute(inv)
        
        # Flatten back to statevector
        self.statevector = psi.reshape(-1)
        
        # Store measurement result in classical register
        self.classical_register[q] = outcome
        
        return outcome
    
    def _apply_classical_pauli(
        self, 
        target_q: int, 
        pauli: str, 
        depends_on: List[Gate]
    ) -> None:
        """
        Apply X or Z gate conditioned on classical measurement outcomes.
        
        Used for quantum teleportation protocol where Bob applies corrections
        based on Alice's measurement results.
        
        Args:
            target_q: Qubit to apply gate on
            pauli: 'X' or 'Z'
            depends_on: List of measurement gates (should be length 2)
        """
        if len(depends_on) != 2:
            return
        
        # Extract measurement gate info
        m0_gate, m1_gate = depends_on
        q0 = m0_gate.qubits[0]
        q1 = m1_gate.qubits[0]
        
        # Get measurement outcomes from classical register
        b0 = self.classical_register.get(q0, None)
        b1 = self.classical_register.get(q1, None)
        
        # If measurements haven't happened yet, skip
        if b0 is None or b1 is None:
            return
        
        # Determine if we should apply the gate
        fire = False
        if pauli == 'Z':
            # Z correction depends on first measurement
            fire = (b0 == 1)
        elif pauli == 'X':
            # X correction depends on second measurement
            fire = (b1 == 1)
        else:
            raise ValueError(f"Unknown classical Pauli: '{pauli}'")
        
        # Apply gate if condition is met
        if fire:
            self._apply_single_pauli(pauli, target_q)
    
    def reset(self) -> None:
        """Reset statevector to |0...0⟩ and clear classical register."""
        self.statevector.zero_()
        self.statevector[0] = 1.0 + 0.0j
        self.classical_register.clear()
    
    def measure_all(self) -> str:
        """
        Measure all qubits (sampling from probability distribution).
        
        Returns:
            Bitstring representing measurement outcome (e.g., "01101")
        """
        # Sample from probability distribution |ψ|²
        probs = torch.abs(self.statevector)**2
        
        # Sample one outcome
        outcome_idx = torch.multinomial(probs, 1).item()
        
        # Convert to binary string
        bitstring = format(outcome_idx, f'0{self.num_qubits}b')
        
        return bitstring
    
    def get_bloch_sphere(self, qubit_index: int) -> Dict[str, float]:
        """
        Get Bloch sphere coordinates for a single qubit.
        
        Computes expectation values ⟨X⟩, ⟨Y⟩, ⟨Z⟩ for the qubit.
        
        Args:
            qubit_index: Qubit to compute Bloch vector for
            
        Returns:
            Dictionary with 'x', 'y', 'z' coordinates
        """
        if qubit_index < 0 or qubit_index >= self.num_qubits:
            raise ValueError(f"Qubit index {qubit_index} out of range")
        
        n = self.num_qubits
        psi = self.statevector
        
        # Compute ⟨Z⟩
        z_exp = 0.0
        for i in range(2**n):
            bit = (i >> (n - 1 - qubit_index)) & 1
            prob = torch.abs(psi[i])**2
            z_exp += prob.item() * (1 if bit == 0 else -1)
        
        # Save current state
        original_state = psi.clone()
        
        # Compute ⟨X⟩
        if self.persistant_data and 'X' in self.fixed_cache:
            X = self.fixed_cache['X']
        else:
            X = GateLibrary.get_gate('X', []).to(device=self.device)
        
        self._apply_k_qubit(X, [qubit_index])
        x_exp = 2 * torch.real(torch.vdot(original_state, self.statevector)).item()
        
        # Restore and compute ⟨Y⟩
        self.statevector = original_state.clone()
        
        if self.persistant_data and 'Y' in self.fixed_cache:
            Y = self.fixed_cache['Y']
        else:
            Y = GateLibrary.get_gate('Y', []).to(device=self.device)
        
        self._apply_k_qubit(Y, [qubit_index])
        y_exp = 2 * torch.real(torch.vdot(original_state, self.statevector)).item()
        
        # Restore original state
        self.statevector = original_state
        
        return {
            'x': float(x_exp),
            'y': float(y_exp),
            'z': float(z_exp)
        }
    def get_significant_states(self, threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Extract significant computational basis states from statevector.
        Maps each state to a Bloch-like visualization using amplitude and phase.
        
        Args:
            threshold: Minimum probability to include state
            
        Returns:
            List of dicts with: {state, probability, theta, phi, x, y, z}
        """
        statevector = self.statevector.cpu()
        probs = torch.abs(statevector) ** 2
        significant_states = []
        
        for idx in range(len(statevector)):
            prob = probs[idx].item()
            if prob >= threshold:
                # Convert index to binary string (e.g., 8 -> "1000" for 4 qubits)
                state_label = format(idx, f'0{self.num_qubits}b')
                
                # Get complex amplitude
                amp = statevector[idx]
                re = float(torch.real(amp))
                im = float(torch.imag(amp))
                r = float(torch.abs(amp))  # sqrt(probability)
                
                # Compute Bloch angles
                # theta: map probability to polar angle (0 = high prob, π = low prob)
                theta = np.arccos(2 * prob - 1)  # Maps [0,1] → [π, 0]
                
                # phi: azimuthal angle from complex phase
                phi = float(torch.angle(amp))  # Phase in [-π, π]
                
                # Cartesian coordinates (standard spherical conversion)
                x = np.sin(theta) * np.cos(phi)
                y = np.sin(theta) * np.sin(phi)
                z = np.cos(theta)
                
                significant_states.append({
                    'state': state_label,
                    'probability': prob,
                    'theta': theta,
                    'phi': phi,
                    'x': x,
                    'y': y,
                    'z': z
                })
        
        return significant_states


    def get_all_bloch_sphere(self) -> List[Dict[str, float]]:
        """
        Get Bloch sphere coordinates for all qubits.
        
        Returns:
            List of dictionaries with 'x', 'y', 'z' for each qubit
        """
        return [self.get_bloch_sphere(i) for i in range(self.num_qubits)]
    
    def execute_circuit(self, shots: int = 1) -> List[str]:
        """
        Execute circuit multiple times and return measurement outcomes.
        
        Args:
            shots: Number of times to execute the circuit
            
        Returns:
            List of measurement outcome bitstrings
        """
        if self.circuit is None:
            raise ValueError("No circuit loaded in backend")
        
        results = []
        
        for _ in range(shots):
            # Reset to initial state
            self.reset()
            
            # Apply all gates in circuit
            for gate in self.circuit.gates:
                self.apply_gate(gate)
            
            # Measure all qubits
            outcome = self.measure_all()
            results.append(outcome)
        
        return results
    
    def get_final_statevector(self) -> torch.Tensor:
        """
        Get current statevector (returns a copy on CPU).
        
        Returns:
            Complex tensor of shape (2^n,)
        """
        return self.statevector.clone().cpu()
    
    def get_histogram_data(self, shots: int = 1024) -> Dict[str, int]:
        """
        Execute circuit and return histogram of measurement outcomes.
        
        Args:
            shots: Number of circuit executions
            
        Returns:
            Dictionary mapping bitstrings to counts
        """
        # Execute circuit
        results = self.execute_circuit(shots=shots)
        
        # Count occurrences
        histogram = {}
        for outcome in results:
            histogram[outcome] = histogram.get(outcome, 0) + 1
        
        return histogram
