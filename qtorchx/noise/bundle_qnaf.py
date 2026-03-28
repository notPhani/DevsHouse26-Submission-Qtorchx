import torch
import sys, os
from qtorchx.noise.presets import Preset, PresetManager

def bundle_optimized_matrices():
    preset_dir = PresetManager.PRESET_DIR

    # --- Version 1: Global Circuit Validation ---
    w1 = torch.tensor([
        [0.15, 0.08, 0.25, 0.03, 0.08, 0.35], # X
        [0.12, 0.12, 0.20, 0.05, 0.10, 0.25], # Y
        [0.40, 0.25, 0.45, 0.02, 0.08, 0.15]  # Z
    ], dtype=torch.float32)
    b1 = torch.tensor([0.0005, 0.0003, 0.0025], dtype=torch.float32)

    v1 = Preset(name="qtorch_standard", dpm=w1, bpo=b1, metadata={"version": "global_v1"})

    # --- Version 2: Local Validation Fallback ---
    w2 = torch.tensor([
        [0.15, 0.08, 0.25, 0.03, 0.08, 0.38], # X (Stochastic-heavy)
        [0.12, 0.12, 0.20, 0.05, 0.10, 0.28], # Y
        [0.45, 0.25, 0.50, 0.02, 0.06, 0.18]  # Z (Memory-heavy)
    ], dtype=torch.float32)
    # Z has higher baseline (-3.0) vs X (-4.0) -> Dephasing dominant!
    b2 = torch.tensor([-4.0, -4.5, -3.0], dtype=torch.float32)

    v2 = Preset(name="qtorch_local", dpm=w2, bpo=b2, metadata={"version": "local_v2"})

    v1.save(preset_dir)
    v2.save(preset_dir)

if __name__ == "__main__":
    bundle_optimized_matrices()