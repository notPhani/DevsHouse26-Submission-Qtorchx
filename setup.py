from setuptools import setup, find_packages

setup(
    name="qtorchx",
    version="1.0.0",
    author="Phani Kumar",
    description="A Differentiable Quantum Noise Simulator powered by the QNaF 7D Manifold",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.20.0",
        "fastapi",
        "uvicorn[standard]",
        "pydantic",
        "qiskit",
        "qiskit-aer",
        "scipy>=1.7.0",
    ],
)
