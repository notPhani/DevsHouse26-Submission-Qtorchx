"""
QtorchX FastAPI Backend
~~~~~~~~~~~~~~~~~~~~~~~
Wires the qtorchx pip package (v0.0.0) to the existing frontend.
Serves static files and exposes /simulate endpoint.
"""

from typing import List, Optional, Dict, Any
import time
import logging
import threading
from contextlib import asynccontextmanager
import torch

# ── qtorchx package imports ──────────────────────────────────────────────────
from qtorchx.core.primitives import Circuit, Gate, GateLibrary
from qtorchx.core.backend import QtorchBackend
from qtorchx.noise.qnaf import PhiManifoldExtractor

# ── Qiskit / Aer ────────────────────────────────────────────────────────────
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error
from qiskit.compiler import transpile
import math

# ── FastAPI / Pydantic ───────────────────────────────────────────────────────
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

logger = logging.getLogger("qtorchx.warmup")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")


# ============================================================================
# CPU WARMUP  (non-blocking — server starts instantly)
# ============================================================================
#
# Strategy
# --------
# Warmup runs kernel-cache priming in a background daemon thread so the
# server accepts requests immediately.  Three representative ops are used:
#
#   • complex64 matmul   → gate / unitary application
#   • linalg.norm        → state normalisation
#   • fft (real)         → noise φ-field projection
#   • QtorchBackend      → full simulation JIT path (Bell-state circuit)
#
# _warmup_done is set once the background thread finishes.
# ─────────────────────────────────────────────────────────────────────────────

_warmup_done   = threading.Event()
_warmup_device = "cpu"



def _background_kernel_warmup() -> None:
    """
    CPU kernel-cache priming — runs in a daemon thread so it never blocks
    the server from accepting requests.

    Operations mirror exactly what /simulate uses:
      • complex64 matmul   → gates & unitary application
      • linalg.norm        → state normalisation
      • fft (real)         → noise φ-field projection
      • QtorchBackend      → full quantum simulation JIT path
    """
    t0 = time.perf_counter()
    # Three cycles is the empirical sweet-spot: cycle 1 compiles, 2–3 verify
    # the cache is warm.  More cycles add time without additional benefit.
    CYCLES = 3
    DIM    = 128   # 128×128 complex64 ≈ 256 KB — covers all gate matrix ops

    try:
        for i in range(CYCLES):
            a = torch.randn(DIM, DIM, dtype=torch.complex64)
            b = torch.randn(DIM, DIM, dtype=torch.complex64)
            _ = torch.matmul(a, b)
            _ = torch.linalg.norm(a)
            _ = torch.fft.fft(a.real.flatten())
            del a, b
            logger.info("[warmup] kernel cycle %d/%d done", i + 1, CYCLES)

        # Pre-exercise the full simulation code-path (Bell state, 2 qubits).
        # This JIT-compiles the QtorchBackend dispatch table and gate kernels.
        bell = Circuit(num_qubits=2)
        bell.add(Gate(name="H",    qubits=[0], params=[], t=0))
        bell.add(Gate(name="CNOT", qubits=[0, 1], params=[], t=1))

        wb = QtorchBackend(
            simulate_with_noise=False,
            persistant_data=False,
            fusion_optimizations=False,
            circuit=bell,
            verbose=False,
        )
        _ = wb.get_histogram_data(shots=128)
        wb.reset()
        for g in bell.gates:
            wb.apply_gate(g)
        _ = wb.get_final_statevector()
        logger.info("[warmup] QtorchBackend gate kernels pre-compiled")

    except Exception as exc:
        logger.warning("[warmup] background kernel warmup partial failure: %s", exc)
    finally:
        elapsed = round(time.perf_counter() - t0, 3)
        logger.info("[warmup] background warmup finished in %.3fs", elapsed)
        _warmup_done.set()


@asynccontextmanager
async def lifespan(application: FastAPI):
    """
    Startup sequence (non-blocking, CPU-only):
      1. Log that the server is starting on CPU.
      2. Background thread kicks off kernel-cache priming on CPU.
    """
    global _warmup_device
    _warmup_device = "cpu"
    logger.info("[warmup] running on CPU — launching background kernel warmup")

    threading.Thread(
        target=_background_kernel_warmup,
        daemon=True,
        name="qtorchx-warmup",
    ).start()

    yield

    logger.info("[shutdown] QtorchX server stopping.")


# ============================================================================
# PYDANTIC MODELS  (must match what the frontend sends/expects)
# ============================================================================

class GateInput(BaseModel):
    name: str
    qubits: List[int]
    t: int


class SimRequest(BaseModel):
    num_qubits: int
    shots: int = 64
    noise_enabled: bool = False
    persistent_mode: bool = True
    show_phi: bool = True
    gates: List[GateInput]


class BlochState(BaseModel):
    state: str           # e.g. "0000", "1000"
    probability: float   # |amplitude|²
    x: float
    y: float
    z: float
    theta: float
    phi: float


class SimResponse(BaseModel):
    statevector: List[str]
    histogram_ideal: Dict[str, float]
    histogram_noisy: Optional[Dict[str, float]] = None
    bloch_states: List[BlochState]
    phi_manifold: Optional[List[List[float]]] = None
    metadata: Dict


# ============================================================================
# FASTAPI APP + CORS + STATIC FILES
# ============================================================================

app = FastAPI(title="QtorchX Quantum Simulator API", version="0.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the frontend at /static
app.mount("/static", StaticFiles(directory="static", html=True), name="static")


@app.get("/")
async def root():
    """Health-check & redirect hint."""
    return {
        "status": "ok",
        "message": "QtorchX API is running. Open /static/index.html for the playground.",
    }


# ============================================================================
# MAIN ENDPOINT: /simulate
# ============================================================================

@app.post("/simulate", response_model=SimResponse)
async def simulate(req: SimRequest) -> SimResponse:
    """
    Execute quantum circuit with optional noise simulation and phi manifold
    extraction, powered by the qtorchx pip package.
    """
    device = "cpu"
    start_time = time.time()

    # ── 1. BUILD CIRCUIT ────────────────────────────────────────────────────
    circuit = Circuit(num_qubits=req.num_qubits)

    non_measurement_gates = [g for g in req.gates if g.name.upper() != "M"]
    measurement_gates = [g for g in req.gates if g.name.upper() == "M"]

    for g in sorted(non_measurement_gates, key=lambda x: x.t):
        gate = Gate(
            name=g.name,
            qubits=g.qubits,
            params=[],
            t=g.t,
        )
        circuit.add(gate)

    for g in measurement_gates:
        gate = Gate(name="M", qubits=g.qubits, params=[], t=g.t)
        circuit.add(gate)

    circuit_build_time = time.time() - start_time

    # ── 2. IDEAL SIMULATION ────────────────────────────────────────────────
    ideal_start = time.time()

    backend_ideal = QtorchBackend(
        simulate_with_noise=False,
        persistant_data=req.persistent_mode,
        fusion_optimizations=False,
        circuit=circuit,
        verbose=False,
    )

    ideal_hist_counts = backend_ideal.get_histogram_data(shots=req.shots)
    histogram_ideal = {k: v / req.shots for k, v in ideal_hist_counts.items()}

    # Single deterministic run for statevector & Bloch data
    backend_ideal.reset()
    for gate in circuit.gates:
        if gate.name.upper() != "M":
            backend_ideal.apply_gate(gate)

    final_state = backend_ideal.get_final_statevector()
    statevector_strs = [
        f"{float(torch.real(amp)):+.6f}{float(torch.imag(amp)):+.6f}i"
        for amp in final_state
    ]

    significant_states = backend_ideal.get_significant_states(threshold=0.01)
    bloch_states = [
        BlochState(
            state=s["state"],
            probability=s["probability"],
            theta=s["theta"],
            phi=s["phi"],
            x=s["x"],
            y=s["y"],
            z=s["z"],
        )
        for s in significant_states
    ]

    ideal_time = time.time() - ideal_start

    # ── 3. NOISY SIMULATION (if enabled) ────────────────────────────────────
    histogram_noisy = None
    phi_manifold_out = None
    noisy_time = 0.0
    phi_time = 0.0

    if req.noise_enabled:
        noisy_start = time.time()

        # --- Phi Manifold Extraction ---
        phi_start = time.time()

        DecoherenceProjectionMatrix = torch.eye(
            3, 6, device=device, dtype=torch.float32
        )
        BaselinePauliOffset = torch.zeros(3, device=device, dtype=torch.float32)

        extractor = PhiManifoldExtractor(
            circuit=circuit,
            DecoherenceProjectionMatrix=DecoherenceProjectionMatrix,
            BaselinePauliOffset=BaselinePauliOffset,
            device=device,
            a=1.0,
            b=2.0,
        )

        phi_manifold_tensor = extractor.GetManifold()           # (6, Q, T)
        circuit_with_noise = extractor.annotate_circuit()

        phi_time = time.time() - phi_start

        # --- Run Noisy Simulation ---
        backend_noisy = QtorchBackend(
            simulate_with_noise=True,
            persistant_data=req.persistent_mode,
            fusion_optimizations=False,
            circuit=circuit_with_noise,
            verbose=False,
        )

        noisy_hist_counts = backend_noisy.get_histogram_data(shots=req.shots)
        histogram_noisy = {k: v / req.shots for k, v in noisy_hist_counts.items()}

        noisy_time = time.time() - noisy_start

        # --- Export Phi Manifold (if requested) ---
        if req.show_phi:
            composite = phi_manifold_tensor.sum(dim=0)          # (Q, T)
            phi_manifold_out = composite.cpu().tolist()

    # ── 4. METADATA ─────────────────────────────────────────────────────────
    total_time = time.time() - start_time
    cache_stats = backend_ideal.get_cache_stats()

    metadata = {
        "circuit_depth": circuit.depth,
        "circuit_size": circuit.size,
        "timing": {
            "total_seconds": round(total_time, 4),
            "circuit_build_seconds": round(circuit_build_time, 4),
            "ideal_simulation_seconds": round(ideal_time, 4),
            "noisy_simulation_seconds": round(noisy_time, 4) if req.noise_enabled else 0.0,
            "phi_extraction_seconds": round(phi_time, 4) if req.noise_enabled else 0.0,
        },
        "cache_stats": cache_stats,
        "shots": req.shots,
        "noise_enabled": req.noise_enabled,
        "persistent_mode": req.persistent_mode,
        "device": device,
    }

    # ── 5. RETURN ────────────────────────────────────────────────────────────
    return SimResponse(
        statevector=statevector_strs,
        histogram_ideal=histogram_ideal,
        histogram_noisy=histogram_noisy,
        bloch_states=bloch_states,
        phi_manifold=phi_manifold_out,
        metadata=metadata,
    )


# ============================================================================
# /compare  —  QtorchX φ-Manifold  vs  Qiskit Aer  side-by-side
# ============================================================================

class CompareRequest(BaseModel):
    num_qubits: int
    shots: int = 64
    gates: List[dict]
    noise_level: float = 0.5   # 0.0 = near-ideal,  1.0 = heavy noise


QISKIT_GATE_MAP = {
    "i": "id", "x": "x", "y": "y", "z": "z",
    "h": "h", "s": "s", "sdg": "sdg", "sx": "sx", "t": "t", "tdg": "tdg",
    "rx": "rx", "ry": "ry", "rz": "rz",
    "p": "p", "u1": "p", "u2": "u2", "u3": "u3",
    "cnot": "cx", "cx": "cx", "cy": "cy", "cz": "cz",
    "swap": "swap", "ch": "ch",
    "crx": "crx", "cry": "cry", "crz": "crz",
    "rxx": "rxx", "ryy": "ryy", "rzz": "rzz",
    "ccx": "ccx", "toffoli": "ccx", "cswap": "cswap",
}


def _build_qiskit_circuit(num_qubits, gates):
    qc = QuantumCircuit(num_qubits, num_qubits)
    sorted_gates = sorted(
        [g for g in gates if g["name"].lower() != "m"],
        key=lambda g: g.get("t", 0)
    )
    for g in sorted_gates:
        method_name = QISKIT_GATE_MAP.get(g["name"].lower())
        if not method_name:
            continue
        method = getattr(qc, method_name, None)
        if not method:
            continue
        params = g.get("params", [])
        qubits = g["qubits"]
        if params:
            method(*params, *qubits)
        else:
            method(*qubits)
    qc.measure(list(range(num_qubits)), list(range(num_qubits)))
    return qc


def _build_noise_model(num_qubits, noise_level):
    nm = NoiseModel()
    p1  = 0.001 + noise_level * 0.009    # 1q depolarizing:  0.1% – 1.0%
    p2  = 0.005 + noise_level * 0.045    # 2q depolarizing:  0.5% – 5.0%
    t1  = max(10e-6, 100e-6 * (1 - noise_level * 0.8))
    t2  = max(5e-6,   80e-6 * (1 - noise_level * 0.8))
    dt1 = 50e-9    # 1q gate time
    dt2 = 300e-9   # 2q gate time

    one_q = ["id","x","y","z","h","s","sdg","sx","t","tdg","rx","ry","rz","p","u2","u3"]
    two_q = ["cx","cy","cz","swap","ch"]

    dep1 = depolarizing_error(p1, 1)
    dep2 = depolarizing_error(p2, 2)

    for q in range(num_qubits):
        err1 = thermal_relaxation_error(t1, t2, dt1).compose(dep1)
        nm.add_quantum_error(err1, one_q, [q])

    for q0 in range(num_qubits):
        for q1 in range(num_qubits):
            if q0 != q1:
                relax2 = thermal_relaxation_error(t1, t2, dt2).expand(
                    thermal_relaxation_error(t1, t2, dt2)
                )
                err2 = relax2.compose(dep2)
                nm.add_quantum_error(err2, two_q, [q0, q1])
    return nm


def _normalize(counts, shots, n_qubits):
    """Qiskit counts (little-endian) → probability dict (big-endian, zero-padded)."""
    out = {}
    for bits, cnt in counts.items():
        key = bits.replace(" ", "")[::-1].zfill(n_qubits)
        out[key] = cnt / shots
    return out


def _tvd(p, q):
    keys = set(p) | set(q)
    return round(0.5 * sum(abs(p.get(k, 0) - q.get(k, 0)) for k in keys), 6)


def _fidelity(p, q):
    keys = set(p) | set(q)
    bc = sum(math.sqrt(p.get(k, 0) * q.get(k, 0)) for k in keys)
    return round(bc ** 2, 6)


@app.post("/compare")
async def compare(req: CompareRequest):
    t0 = time.perf_counter()
    out = {}

    # ── QtorchX ──────────────────────────────────────────────────────────────
    try:
        t_qtx = time.perf_counter()
        circuit = Circuit(num_qubits=req.num_qubits)
        for g in sorted([x for x in req.gates if x["name"].lower() != "m"],
                        key=lambda x: x.get("t", 0)):
            circuit.add(Gate(g["name"], g["qubits"], g.get("params", []), g.get("t", 0)))

        # ideal
        b_ideal = QtorchBackend(
            circuit=circuit, simulate_with_noise=False, persistant_data=True
        )
        ideal_counts = b_ideal.get_histogram_data(shots=req.shots)
        ideal_probs  = {k: v / req.shots for k, v in ideal_counts.items()}

        # noisy
        device = "cpu"
        DPM = torch.eye(3, 6, device=device, dtype=torch.float32)
        BPO = torch.zeros(3, device=device, dtype=torch.float32)
        ext = PhiManifoldExtractor(
            circuit=circuit,
            DecoherenceProjectionMatrix=DPM,
            BaselinePauliOffset=BPO,
            device=device,
        )
        ext.GetManifold()
        ann = ext.annotate_circuit()
        b_noisy = QtorchBackend(
            circuit=ann, simulate_with_noise=True, persistant_data=True
        )
        noisy_counts = b_noisy.get_histogram_data(shots=req.shots)
        noisy_probs  = {k: v / req.shots for k, v in noisy_counts.items()}

        out["qtorchx"] = {
            "ideal_probs":  ideal_probs,
            "noisy_probs":  noisy_probs,
            "time_seconds": round(time.perf_counter() - t_qtx, 4),
            "engine":       "QtorchX  φ-Manifold",
            "noise_model":  "6-channel spatiotemporal φ-field",
        }
    except Exception as e:
        out["qtorchx"] = {"error": str(e)}

    # ── Qiskit Aer ───────────────────────────────────────────────────────────
    try:
        t_aer = time.perf_counter()
        qc = _build_qiskit_circuit(req.num_qubits, req.gates)
        nm = _build_noise_model(req.num_qubits, req.noise_level)

        ideal_sim = AerSimulator()
        noisy_sim = AerSimulator(noise_model=nm)

        # ideal
        ideal_job = ideal_sim.run(transpile(qc, ideal_sim), shots=req.shots)
        ideal_aer = _normalize(ideal_job.result().get_counts(), req.shots, req.num_qubits)

        # noisy
        noisy_job = noisy_sim.run(transpile(qc, noisy_sim), shots=req.shots)
        noisy_aer = _normalize(noisy_job.result().get_counts(), req.shots, req.num_qubits)

        out["qiskit_aer"] = {
            "ideal_probs":  ideal_aer,
            "noisy_probs":  noisy_aer,
            "time_seconds": round(time.perf_counter() - t_aer, 4),
            "engine":       "Qiskit Aer",
            "noise_model":  f"Depolarizing + Thermal Relaxation  (level={req.noise_level:.1f})",
        }
    except Exception as e:
        out["qiskit_aer"] = {"error": str(e)}

    # ── Cross-engine metrics ──────────────────────────────────────────────────
    metrics = {}
    qtx = out.get("qtorchx", {})
    aer = out.get("qiskit_aer", {})
    if "noisy_probs" in qtx and "noisy_probs" in aer:
        metrics["fidelity_between_engines"]  = _fidelity(qtx["noisy_probs"], aer["noisy_probs"])
        metrics["tvd_between_engines"]       = _tvd(qtx["noisy_probs"], aer["noisy_probs"])
        metrics["qtorchx_noise_impact_tvd"]  = _tvd(qtx.get("ideal_probs", {}), qtx["noisy_probs"])
        metrics["qiskit_noise_impact_tvd"]   = _tvd(aer.get("ideal_probs", {}), aer["noisy_probs"])

    return {
        "results": out,
        "metrics": metrics,
        "total_time_seconds": round(time.perf_counter() - t0, 4),
        "shots": req.shots,
        "num_qubits": req.num_qubits,
    }


# ============================================================================
# UVICORN RUNNER
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("entry:app", host="0.0.0.0", port=8888, reload=True)
