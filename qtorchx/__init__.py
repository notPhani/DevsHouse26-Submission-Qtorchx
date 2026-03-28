"""
qtorchx namespace shim.

The published wheel installs `core` and `noise` as *top-level* packages but
their source files internally do `from qtorchx.core.xxx import ...`.
This file bootstraps the full `qtorchx.*` namespace by direct-loading every
sub-module via its file path (bypassing __init__ triggers) in dependency order:

  core/primitives.py   (no cross-deps)
  core/backend.py      (needs qtorchx.core.primitives)
  noise/presets.py     (no cross-deps)
  noise/bundle_qnaf.py (no cross-deps)
  noise/calibrator.py  (needs qtorchx.noise.presets)
  noise/qnaf.py        (needs qtorchx.core.backend)
  core   __init__      (all sub-modules already in sys.modules)
  noise  __init__      (all sub-modules already in sys.modules)
"""
import sys, os, types, importlib.util as _ilu

# Register this package itself immediately
sys.modules.setdefault('qtorchx', sys.modules[__name__])

_SP = os.path.dirname(os.path.abspath(__file__))  # qtorchx package dir


def _load(dotted_name: str, rel_path: str):
    """Load a .py file directly and register under both `dotted_name` and
    the qtorchx.* mirror in sys.modules."""
    path = os.path.join(_SP, rel_path)
    spec = _ilu.spec_from_file_location(dotted_name, path)
    mod = _ilu.module_from_spec(spec)
    # Register under the real name AND the qtorchx.* alias BEFORE exec
    # so any further cross-imports inside the file can find both.
    sys.modules[dotted_name] = mod
    qtx_name = 'qtorchx.' + dotted_name
    sys.modules[qtx_name] = mod
    spec.loader.exec_module(mod)
    return mod


# ── core ─────────────────────────────────────────────────────────────────────
_prims   = _load('core.primitives',  os.path.join('core', 'primitives.py'))
_backend = _load('core.backend',     os.path.join('core', 'backend.py'))

# Manufacture a synthetic `core` package that looks like the real __init__
_core_pkg = types.ModuleType('core')
_core_pkg.__path__ = [os.path.join(_SP, 'core')]
_core_pkg.__package__ = 'core'
_core_pkg.Circuit        = _prims.Circuit
_core_pkg.Gate           = _prims.Gate
_core_pkg.GateLibrary    = _prims.GateLibrary
_core_pkg.QtorchBackend  = _backend.QtorchBackend
sys.modules.setdefault('core', _core_pkg)
sys.modules.setdefault('qtorchx.core', _core_pkg)

# ── noise ─────────────────────────────────────────────────────────────────────
# presets has no cross-deps
_presets    = _load('noise.presets',     os.path.join('noise', 'presets.py'))
# qnaf needs qtorchx.core.backend (Circuit)
_qnaf       = _load('noise.qnaf',        os.path.join('noise', 'qnaf.py'))
# bundle_qnaf needs qtorchx.noise.presets
_bundle     = _load('noise.bundle_qnaf', os.path.join('noise', 'bundle_qnaf.py'))
# calibrator needs qtorchx.core.*, qtorchx.noise.presets + qtorchx.noise.qnaf
_calibrator = _load('noise.calibrator',  os.path.join('noise', 'calibrator.py'))

_noise_pkg = types.ModuleType('noise')
_noise_pkg.__path__ = [os.path.join(_SP, 'noise')]
_noise_pkg.__package__ = 'noise'
_noise_pkg.PhiManifoldExtractor = _qnaf.PhiManifoldExtractor
_noise_pkg.NoiseCalibrator      = _calibrator.NoiseCalibrator
_noise_pkg.Preset               = _presets.Preset
_noise_pkg.PresetManager        = _presets.PresetManager
sys.modules.setdefault('noise', _noise_pkg)
sys.modules.setdefault('qtorchx.noise', _noise_pkg)

# ── top-level re-exports ──────────────────────────────────────────────────────
Circuit             = _prims.Circuit
Gate                = _prims.Gate
GateLibrary         = _prims.GateLibrary
QtorchBackend       = _backend.QtorchBackend
PhiManifoldExtractor = _qnaf.PhiManifoldExtractor
NoiseCalibrator     = _calibrator.NoiseCalibrator
Preset              = _presets.Preset
PresetManager       = _presets.PresetManager

__all__ = [
    "Circuit", "Gate", "GateLibrary", "QtorchBackend",
    "PhiManifoldExtractor", "NoiseCalibrator", "Preset", "PresetManager",
]
