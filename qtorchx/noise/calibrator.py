import sys, os
import torch
import torch.nn as nn
import torch.optim as optim
import json
from typing import Optional, Dict, Tuple, List
import sys, os
# Internal QtorchX Imports
from qtorchx.core.primitives import Circuit, Gate
from qtorchx.noise.presets import Preset, PresetManager
from qtorchx.noise.qnaf import PhiManifoldExtractor
from qtorchx.core.backend import QtorchBackend

class NoiseCalibrator(nn.Module):
    """
    A Differentiable Digital Twin Generator.
    Converges on hardware counts using physics-constrained optimization.
    """
    def __init__(
        self, 
        circuit: Circuit, 
        preset_name: str = "qtorch_standard", 
        config_path: Optional[str] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super().__init__()
        self.circuit = circuit
        self.device = device
        
        # 1. Load Hardware Fingerprint (.qnaf) with Fallback
        try:
            preset = PresetManager.fetch(preset_name, device=device)
        except FileNotFoundError:
            print(f"⚠️  Preset {preset_name} not found. Loading 'qtorch_local' fallback.")
            preset = PresetManager.fetch("qtorch_local", device=device)
        
        # 2. Register Trainable Parameters
        self.dpm = nn.Parameter(preset.dpm.clone().detach().requires_grad_(True))
        self.bpo = nn.Parameter(preset.bpo.clone().detach().requires_grad_(True))
        
        # 3. Load Physics Engine Configuration (.config)
        self.physics_config = self._load_config(config_path)

    def _load_config(self, path: Optional[str]) -> Dict:
        default_config = {
            'alpha': 0.9, 'lam': 0.05, 'beta': 0.15,
            'kappa': 0.1, 'epsilon': 0.002, 'a': 2.0, 'b': 1.6
        }
        if path and os.path.exists(path):
            with open(path, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        return default_config

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Differentiable Forward Pass:
        Physics Engine -> 7D Manifold -> Pauli Probabilities
        """
        extractor = PhiManifoldExtractor(
            circuit=self.circuit,
            DecoherenceProjectionMatrix=self.dpm,
            BaselinePauliOffset=self.bpo,
            device=self.device,
            **self.physics_config
        )
        
        phi = extractor.GetManifold() 
        pauli_logits = extractor.get_pauli_channel() 
        
        # Convert to differentiable probabilities via Sigmoid
        error_probs = torch.sigmoid(pauli_logits)
        
        return error_probs, phi

    def compute_loss(self, predicted: torch.Tensor, target: torch.Tensor, phi: torch.Tensor):
        """
        MSE(Counts) + Physicality Constraints (RL-inspired)
        """
        # Accuracy Loss
        acc_loss = nn.MSELoss()(predicted, target)
        
        # Physicality Penalty: Prevents field runaway
        stability_penalty = torch.mean(torch.relu(torch.abs(phi) - 0.98))
        
        return acc_loss + (2.5 * stability_penalty)

    def calibrate(self, target_data: torch.Tensor, epochs: int = 100, lr: float = 0.05):
        optimizer = optim.Adam(self.parameters(), lr=lr)
        history = []
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            pred_probs, phi = self.forward()
            loss = self.compute_loss(pred_probs, target_data, phi)
            loss.backward()
            optimizer.step()
            history.append(loss.item())
            
        return history

    def export_qnaf(self, name: str):
        new_preset = Preset(name=name, dpm=self.dpm.data, bpo=self.bpo.data)
        new_preset.save(PresetManager.PRESET_DIR)