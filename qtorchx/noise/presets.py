from dataclasses import dataclass, field
import torch
import json
import os
from typing import Dict
import sys, os

@dataclass
class Preset:
    """
    The .qnaf file structure. 
    Encapsulates a specific hardware's noise fingerprint (DPM and BPO).
    """
    name: str
    dpm: torch.Tensor
    bpo: torch.Tensor
    is_trainable: bool = True
    metadata: Dict = field(default_factory=dict)

    def save(self, folder_path: str):
        """Saves the preset as a .qnaf (JSON-based) file."""
        os.makedirs(folder_path, exist_ok=True)
        file_path = os.path.join(folder_path, f"{self.name}.qnaf")
        
        data = {
            "name": self.name,
            "dpm": self.dpm.detach().cpu().tolist(),
            "bpo": self.bpo.detach().cpu().tolist(),
            "is_trainable": self.is_trainable,
            "metadata": self.metadata or {}
        }
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Preset saved: {file_path}")

    @classmethod
    def load(cls, file_path: str, device: str = 'cpu'):
        """Loads a .qnaf file back into a Preset object."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No preset found at {file_path}")
            
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        return cls(
            name=data["name"],
            dpm=torch.tensor(data["dpm"], device=device),
            bpo=torch.tensor(data["bpo"], device=device),
            is_trainable=data["is_trainable"],
            metadata=data.get("metadata", {})
        )

class PresetManager:
    """
    Handles the global registry and retrieval of .qnaf files.
    """
    # Locates the internal presets folder relative to this file
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PRESET_DIR = os.path.join(BASE_DIR, "presets")

    @classmethod
    def get_available(cls):
        """Lists all bundled .qnaf files."""
        if not os.path.exists(cls.PRESET_DIR):
            return []
        return [f.replace(".qnaf", "") for f in os.listdir(cls.PRESET_DIR) if f.endswith(".qnaf")]

    @classmethod
    def fetch(cls, name: str, device: str = 'cpu'):
        """Fetches a bundled preset by name."""
        file_path = os.path.join(cls.PRESET_DIR, f"{name}.qnaf")
        if not os.path.exists(file_path):
            # Fallback for generic or custom paths
            if os.path.exists(name):
                return Preset.load(name, device=device)
            raise FileNotFoundError(f"Preset '{name}' not found in bundled library or path.")
        return Preset.load(file_path, device=device)