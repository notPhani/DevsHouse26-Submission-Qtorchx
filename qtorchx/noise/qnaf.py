from typing import Optional, Dict
import torch
import sys, os
from qtorchx.core.backend import Circuit

class PhiManifoldExtractor:
    """
    Extracts 6-channel phi manifold from quantum circuit execution.
    Output shape: (6, num_qubits, max_depth)
    Channels:
        [0] Memory: (alpha-λ)φ_i(t)
        [1] Spatial Diffusion: β[Lφ(t)]_i
        [2] Disturbance Diffusion: κ[LD(t)]_i
        [3] Nonlocal Bleed: ε Σ_j exp(-gamma_d_ij)φ_j(t)
        [4] Nonlinear Saturation: rho φ_i(t)/(1+φ_i²(t))
        [5] Stochastic Kicks: alpha_i(t)(G_i(t) + M_i(t))η_i(t)
    """
    
    def __init__(
        self, 
        circuit: Circuit, 
        DecoherenceProjectionMatrix: torch.Tensor, 
        BaselinePauliOffset: torch.Tensor, 
        alpha: float = 0.9, 
        lam: float = 0.05, 
        beta: float = 0.15, 
        kappa: float = 0.1, 
        epsilon: float = 0.002, 
        gamma: float = 1.0, 
        rho: float = 0.08, 
        sigma: float = 0.05, 
        a: float = 1.0,  # Gate disturbance amplification
        b: float = 2.0,  # Measurement disturbance amplification (typically 2x gates)
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ) -> None:
        self.circuit = circuit
        self.DecoherenceProjectionMatrix = DecoherenceProjectionMatrix.to(device)
        self.BaselinePauliOffset = BaselinePauliOffset.to(device)
        
        self.num_qubits = circuit.num_qubits
        self.max_time = circuit.depth
        self.device = device
        
        # Hyperparameters
        self.alpha = alpha
        self.lam = lam
        self.beta = beta
        self.kappa = kappa
        self.epsilon = epsilon
        self.gamma = gamma
        self.rho = rho
        self.sigma = sigma
        self.a = a  # Gate burst amplification
        self.b = b  # Measurement burst amplification
        
        # Storage for phi manifold (6, num_qubits, max_time)
        self.PhiManifold = torch.zeros(
            (6, self.num_qubits, self.max_time),
            dtype=torch.float32,
            device=device
        )

        self.PauliChannelField = torch.zeros(
            (3, self.num_qubits, self.max_time),
            dtype = torch.float32,
            device = device
        )
        
        # Precomputed graph structures (lazy initialization)
        self._laplacian: Optional[torch.Tensor] = None
        self._distance_matrix: Optional[torch.Tensor] = None
        self._adjacency: Optional[torch.Tensor] = None
    
    # ========================================================================
    # GRAPH STRUCTURE METHODS
    # ========================================================================
    
    def _get_laplacian(self) -> torch.Tensor:
        """
        Build graph Laplacian L = D - W from circuit connectivity.
        Edges exist between qubits connected by multi-qubit gates.
        
        Returns:
            Laplacian matrix (num_qubits, num_qubits)
        """
        if self._laplacian is not None:
            return self._laplacian
        
        n = self.num_qubits
        W = torch.zeros((n, n), device=self.device)
        
        # Build adjacency from multi-qubit gates
        for gate in self.circuit.gates:
            if len(gate.qubits) >= 2:
                # Create edges for all pairs in multi-qubit gate
                for i in range(len(gate.qubits)):
                    for j in range(i + 1, len(gate.qubits)):
                        q1, q2 = gate.qubits[i], gate.qubits[j]
                        W[q1, q2] = 1.0
                        W[q2, q1] = 1.0
        
        # Degree matrix
        D = torch.diag(W.sum(dim=1))
        
        # Laplacian L = D - W
        self._laplacian = D - W
        self._adjacency = W
        
        return self._laplacian
    
    def _get_distance_matrix(self) -> torch.Tensor:
        """
        Compute all-pairs shortest path distances using Floyd-Warshall.
        Distance is measured in number of hops on circuit graph.
        
        Returns:
            Distance matrix (num_qubits, num_qubits)
        """
        if self._distance_matrix is not None:
            return self._distance_matrix
        
        n = self.num_qubits
        
        # Initialize with infinity (unreachable)
        dist = torch.full((n, n), float('inf'), device=self.device)
        
        # Self-loops (distance 0)
        for i in range(n):
            dist[i, i] = 0.0
        
        # Add edges from circuit (distance 1 for neighbors)
        for gate in self.circuit.gates:
            if len(gate.qubits) >= 2:
                for i in range(len(gate.qubits)):
                    for j in range(i + 1, len(gate.qubits)):
                        q1, q2 = gate.qubits[i], gate.qubits[j]
                        dist[q1, q2] = 1.0
                        dist[q2, q1] = 1.0
        
        # Floyd-Warshall algorithm
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i, k] + dist[k, j] < dist[i, j]:
                        dist[i, j] = dist[i, k] + dist[k, j]
        
        # Replace unreachable (inf) with large distance
        dist[dist == float('inf')] = 10.0
        
        self._distance_matrix = dist
        return self._distance_matrix
    
    def _get_disturbance_field(self, time_step: int) -> torch.Tensor:
        """
        Compute disturbance field D_i(t) = a*G_i(t)*w_gate + b*M_i(t)*w_meas
        
        Where w_gate and w_meas are hardware-calibrated burst weights.
        
        Args:
            time_step: Current time step
            
        Returns:
            Disturbance vector (num_qubits,)
        """
        D = torch.zeros(self.num_qubits, device=self.device)
        
        gates_at_t = self.circuit.get_time_slice(time_step)
        
        for gate in gates_at_t:
            # Get hardware-calibrated burst weight
            burst = gate.get_burst_weight()
            
            for q in gate.qubits:
                if gate.name.upper() == 'M':
                    # Measurement: amplified by factor b
                    D[q] += self.b * burst
                else:
                    # Gate: amplified by factor a
                    D[q] += self.a * burst
        
        return D
    
    # ========================================================================
    # FEATURE COMPUTATION METHODS
    # ========================================================================
    
    def _compute_memory_term(self, phi_prev: torch.Tensor) -> torch.Tensor:
        """
        Channel [0]: Memory term (α - λ)φ_i(t)
        
        Implements non-Markovian persistence with decay.
        
        Args:
            phi_prev: Previous phi state (num_qubits,)
            
        Returns:
            Memory contribution (num_qubits,)
        """
        return (self.alpha - self.lam) * phi_prev
    
    def _compute_spatial_diffusion(self, phi_prev: torch.Tensor) -> torch.Tensor:
        """
        Channel [1]: Spatial diffusion β[Lφ(t)]_i
        
        Noise spreads along circuit topology via graph Laplacian.
        
        Args:
            phi_prev: Previous phi state (num_qubits,)
            
        Returns:
            Diffusion contribution (num_qubits,)
        """
        L = self._get_laplacian()
        return self.beta * torch.matmul(L, phi_prev)
    
    def _compute_disturbance_diffusion(self, time_step: int) -> torch.Tensor:
        """
        Channel [2]: Disturbance diffusion κ[LD(t)]_i
        
        Gate/measurement disturbances propagate along circuit graph.
        
        Args:
            time_step: Current time step
            
        Returns:
            Disturbance contribution (num_qubits,)
        """
        L = self._get_laplacian()
        D_t = self._get_disturbance_field(time_step)
        return self.kappa * torch.matmul(L, D_t)
    
    def _compute_nonlocal_bleed(self, phi_prev: torch.Tensor) -> torch.Tensor:
        """
        Channel [3]: Nonlocal exponential bleed ε Σ_j exp(-γd_ij)φ_j(t)
        
        Long-range coupling with exponential distance decay.
        Creates smooth gradients in manifold.
        
        Args:
            phi_prev: Previous phi state (num_qubits,)
            
        Returns:
            Nonlocal contribution (num_qubits,)
        """
        dist_matrix = self._get_distance_matrix()
        
        # Compute exponential decay matrix exp(-γ*d_ij)
        decay_matrix = torch.exp(-self.gamma * dist_matrix)
        
        # Zero out diagonal (no self-interaction)
        decay_matrix.fill_diagonal_(0.0)
        
        # Weighted sum over neighbors
        return self.epsilon * torch.matmul(decay_matrix, phi_prev)
    
    def _compute_nonlinear_saturation(self, phi_prev: torch.Tensor) -> torch.Tensor:
        """
        Channel [4]: Nonlinear saturation ρ φ_i(t)/(1 + φ_i²(t))
        
        Prevents runaway growth via soft saturation function.
        
        Args:
            phi_prev: Previous phi state (num_qubits,)
            
        Returns:
            Nonlinear contribution (num_qubits,)
        """
        return self.rho * phi_prev / (1.0 + phi_prev**2)
    
    def _compute_stochastic_kicks(self, time_step: int) -> torch.Tensor:
        """
        Channel [5]: Stochastic kicks σ(G_i(t) + M_i(t))η_i(t)
        
        Activity-modulated Gaussian noise. Idle qubits have minimal noise,
        active qubits (gates/measurements) have proportional noise.
        
        Args:
            time_step: Current time step
            
        Returns:
            Stochastic contribution (num_qubits,)
        """
        # Get disturbance field (activity indicator)
        D_t = self._get_disturbance_field(time_step)
        
        # Gaussian white noise
        eta = torch.randn(self.num_qubits, device=self.device)
        
        # Activity-modulated noise
        return self.sigma * D_t * eta
    
    # ========================================================================
    # MAIN EXTRACTION METHOD
    # ========================================================================
    
    def GetManifold(self) -> torch.Tensor:
        """
        Extract complete phi manifold by simulating coupled dynamics.
        
        Simulates the equation:
        φ_i(t+1) = tanh[(α-λ)φ_i(t) + β[Lφ(t)]_i + κ[LD(t)]_i 
                        + ε Σ_j exp(-γd_ij)φ_j(t) + ρH(φ_i(t)) 
                        + σ(G_i(t) + M_i(t))η_i(t)]
        
        The tanh ensures phi stays bounded in [-1, 1] range.
        
        Returns:
            PhiManifold: Tensor of shape (6, num_qubits, max_time)
        """
        # Initialize phi(0) with small random noise
        phi = torch.randn(self.num_qubits, device=self.device) * 0.01
        
        # Time evolution loop
        for t in range(self.max_time):
            # Compute all 6 feature channels independently
            
            # [0] Memory
            memory = self._compute_memory_term(phi)
            self.PhiManifold[0, :, t] = memory
            
            # [1] Spatial diffusion
            diffusion = self._compute_spatial_diffusion(phi)
            self.PhiManifold[1, :, t] = diffusion
            
            # [2] Disturbance diffusion
            disturbance = self._compute_disturbance_diffusion(t)
            self.PhiManifold[2, :, t] = disturbance
            
            # [3] Nonlocal bleed
            nonlocal_term = self._compute_nonlocal_bleed(phi)
            self.PhiManifold[3, :, t] = nonlocal_term
            
            # [4] Nonlinear saturation
            nonlinear = self._compute_nonlinear_saturation(phi)
            self.PhiManifold[4, :, t] = nonlinear
            
            # [5] Stochastic kicks
            stochastic = self._compute_stochastic_kicks(t)
            self.PhiManifold[5, :, t] = stochastic
            
            # Sum all contributions
            phi_next_raw = memory + diffusion + disturbance + nonlocal_term + nonlinear + stochastic
            
            # Apply tanh soft clamping to keep phi bounded
            phi = torch.tanh(phi_next_raw)
        
        return self.PhiManifold

    def get_pauli_channel(self) -> torch.Tensor:
        """
        Project 6-channel phi manifold into 3-channel Pauli error space.
        
        Formula:
            PauliChannel[p, q, t] = Σ_f W[p, f] * Φ[f, q, t] + B[p]
            
        Where:
            - Φ: PhiManifold (6, num_qubits, max_time)
            - W: DecoherenceProjectionMatrix (3, 6) 
            - B: BaselinePauliOffset (3,)
            
        Returns:
            PauliChannel: (3, num_qubits, max_time)
        """
        # Φ shape: (6, num_qubits, max_time)
        # W shape: (3, 6)
        # Want: (3, num_qubits, max_time)
        
        # Reshape for matmul: (6, num_qubits * max_time)
        num_qubits = self.PhiManifold.shape[1]
        max_time = self.PhiManifold.shape[2]
        
        phi_reshaped = self.PhiManifold.reshape(6, -1)  # (6, num_qubits * max_time)
        
        # Project: (3, 6) @ (6, num_qubits * max_time) -> (3, num_qubits * max_time)
        pauli_flat = torch.matmul(self.DecoherenceProjectionMatrix, phi_reshaped)
        
        # Reshape back: (3, num_qubits, max_time)
        pauli_channel = pauli_flat.reshape(3, num_qubits, max_time)
        
        # Add baseline offset (broadcast over qubits and time)
        pauli_channel = pauli_channel + self.BaselinePauliOffset[:, None, None]
        
        return pauli_channel
    #---Star Function :) Annotating the ideal circuit with noise channels at appropriate time steps---#
    # once this is complete the circuit can be simulated with noise at each time step
    def annotate_circuit(self) -> Circuit:
        """
        Annotate circuit gates with Pauli error probabilities from phi manifold.
        
        For each gate at time t on qubits Q, extract error probabilities
        from the manifold at locations (q, t) and store in gate.metadata.
        
        The backend can then apply noise by sampling from these probabilities
        during circuit execution.
        
        Formula for each qubit q at time t:
            p_x(q,t) = sigmoid(PauliChannel[0, q, t] - bias)
            p_y(q,t) = sigmoid(PauliChannel[1, q, t] - bias)
            p_z(q,t) = sigmoid(PauliChannel[2, q, t] - bias)
            
            Normalize: if p_total > 1, scale by (1 / p_total)
            p_i = 1 - (p_x + p_y + p_z)  # No error probability
        
        Returns:
            Circuit: Same circuit with noise_model added to gate.metadata
        """
        # Get Pauli channel (3, num_qubits, max_time)
        pauli_channel = self.get_pauli_channel().cpu()
        
        # Shift to get realistic error rates (1-5%)
        pauli_channel = pauli_channel - 3.0
        
        # Sigmoid activation
        def sigmoid(x):
            return 1.0 / (1.0 + torch.exp(-x))
        
        # Convert to probabilities
        p_x_all = sigmoid(pauli_channel[0])  # (num_qubits, max_time)
        p_y_all = sigmoid(pauli_channel[1])
        p_z_all = sigmoid(pauli_channel[2])
        
        # Statistics tracking
        total_gates_annotated = 0
        max_error_prob = 0.0
        error_distribution = {'X': 0.0, 'Y': 0.0, 'Z': 0.0}
        
        # ========================================================================
        # ANNOTATE EACH GATE
        # ========================================================================
        for gate in self.circuit.gates:
            if gate.t is None:
                # Skip unscheduled gates
                continue
            
            t = gate.t
            pauli_probs = {}
            gate_max_error = 0.0
            dominant_error = 'I'
            
            # For each qubit this gate touches
            for q in gate.qubits:
                # Extract probabilities at this specific (qubit, time) location
                p_x = p_x_all[q, t].item()
                p_y = p_y_all[q, t].item()
                p_z = p_z_all[q, t].item()
                
                # Normalize to ensure sum ≤ 1
                p_total = p_x + p_y + p_z
                
                if p_total > 1.0:
                    # Scale down proportionally
                    scale = 1.0 / p_total
                    p_x *= scale
                    p_y *= scale
                    p_z *= scale
                    p_total = 1.0
                
                # Compute no-error probability
                p_i = 1.0 - p_total
                
                # Store as [p_i, p_x, p_y, p_z]
                pauli_probs[q] = [p_i, p_x, p_y, p_z]
                
                # Track dominant error for this gate
                if p_total > gate_max_error:
                    gate_max_error = p_total
                    dominant_error = max(
                        [('X', p_x), ('Y', p_y), ('Z', p_z)],
                        key=lambda x: x[1]
                    )[0]
                
                # Update global distribution
                error_distribution['X'] += p_x
                error_distribution['Y'] += p_y
                error_distribution['Z'] += p_z
            
            # Store noise model in gate metadata
            gate.metadata['noise_model'] = {
                'pauli_probs': pauli_probs,
                'source': 'phi_manifold',
                'time_step': t,
                'gate_name': gate.name,
                'dominant_error': dominant_error,
                'max_error_prob': gate_max_error,
                'burst_weight': gate.get_burst_weight()
            }
            
            total_gates_annotated += 1
            max_error_prob = max(max_error_prob, gate_max_error)
        
        # ========================================================================
        # STORE CIRCUIT-LEVEL STATISTICS
        # ========================================================================
        total_error = sum(error_distribution.values())
        
        self.circuit.metadata['noise_annotation'] = {
            'source': 'phi_manifold',
            'gates_annotated': total_gates_annotated,
            'max_error_probability': max_error_prob,
            'error_distribution': {
                'X': (error_distribution['X'] / total_error * 100) if total_error > 0 else 0,
                'Y': (error_distribution['Y'] / total_error * 100) if total_error > 0 else 0,
                'Z': (error_distribution['Z'] / total_error * 100) if total_error > 0 else 0
            },
            'hardware_preset': getattr(self, '_hardware_preset', 'superconducting'),
            'phi_manifold_shape': tuple(self.PhiManifold.shape),
            'hyperparameters': {
                'alpha': self.alpha,
                'beta': self.beta,
                'kappa': self.kappa,
                'epsilon': self.epsilon,
                'sigma': self.sigma,
                'a': self.a,
                'b': self.b
            }
        }
        
        return self.circuit
 
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def get_composite_manifold(self) -> torch.Tensor:
        """
        Get composite manifold (sum over all 6 feature channels).
        
        Returns:
            Composite: (num_qubits, max_time)
        """
        return self.PhiManifold.sum(dim=0)
    
    def get_feature_channel(self, channel_idx: int) -> torch.Tensor:
        """
        Get specific feature channel.
        
        Args:
            channel_idx: Index 0-5
            
        Returns:
            Feature: (num_qubits, max_time)
        """
        if not 0 <= channel_idx < 6:
            raise ValueError(f"channel_idx must be 0-5, got {channel_idx}")
        
        return self.PhiManifold[channel_idx]
    
    def get_stats(self) -> Dict[str, float]:
        """Get summary statistics of composite manifold"""
        composite = self.get_composite_manifold()
        
        return {
            'max': composite.max().item(),
            'min': composite.min().item(),
            'mean': composite.mean().item(),
            'std': composite.std().item(),
            'total_activity': torch.abs(composite).sum().item(),
            'peak_time': composite.abs().sum(dim=0).argmax().item(),
            'peak_qubit': composite.abs().sum(dim=1).argmax().item()
        }
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Compute contribution of each feature to total activity.
        
        Returns:
            Dict mapping feature index to percentage contribution
        """
        # Total absolute contribution per feature
        feature_totals = torch.abs(self.PhiManifold).sum(dim=(1, 2))
        
        # Normalize to percentages
        total = feature_totals.sum()
        if total == 0:
            return {f"feature_{i}": 0.0 for i in range(6)}
        
        percentages = (feature_totals / total * 100).cpu().numpy()
        
        feature_names = [
            'Memory',
            'Spatial Diffusion',
            'Disturbance',
            'Nonlocal Bleed',
            'Nonlinear Saturation',
            'Stochastic Kicks'
        ]
        
        return {name: float(pct) for name, pct in zip(feature_names, percentages)}
    
    def __repr__(self) -> str:
        return (
            f"PhiManifoldExtractor(\n"
            f"  shape=(6, {self.num_qubits}, {self.max_time})\n"
            f"  α={self.alpha:.3f}, λ={self.lam:.3f}\n"
            f"  β={self.beta:.3f}, κ={self.kappa:.3f}\n"
            f"  ε={self.epsilon:.4f}, γ={self.gamma:.3f}\n"
            f"  ρ={self.rho:.3f}, σ={self.sigma:.3f}\n"
            f"  a={self.a:.2f}, b={self.b:.2f}\n"
            f"  device={self.device}\n"
            f")"
        )
    